# Experiments

This document describes the experiments conducted to evaluate the LLM-based parser generation pipeline. All experiments measure the quality of automatically generated parsers for four RCPSP file formats (`.sm`, `.mm`, `.rcp`, `.msrcp`) against manually annotated ground truth.

---

## Setup

### Data

Benchmark instances from three dataset families were used (see [DATASETS.md](DATASETS.md)):

- **PSPLIB j30** - single-mode RCPSP, `.sm` and `.rcp` formats
- **MMLIB** - multi-mode RCPSP, `.mm` format
- **MSLIB** - multi-skilled RCPSP, `.msrcp` format

A subset of instances with available ground truth was used for validation during generation. The same instances were used across all experiments for comparability.

### Evaluation

Each generated parser is evaluated on the validation set using four metrics: Duration Accuracy, Dependencies F1, Resources F1, and Requirements F1 (see [METRICS_AND_ANNOTATION.md](METRICS_AND_ANNOTATION.md)). The aggregate score is the sum of all four.

### LLM

All experiments used the same model and API endpoint. Temperature was set to 0.2 for all generation and repair calls.

---

## Experiment 1 — Baseline Pipeline (`generate_parsers.py`)

### Description

The baseline pipeline generates a parser for each format in a single prompt containing the DSL schema description and a format-specific hint. If the generated parser fails validation (structurally or by metrics), the errors are passed back to the LLM and a corrected version is requested. This loop repeats for up to `n_attempts` iterations.

### Running

```bash
python generate_parsers.py \
    --source data/benchmark/1_raw_data \
    --gt data/benchmark/2_ground_truth \
    --output data/benchmark/4_generated_parsers_v1 \
    --attempts 5
```

### Results

| Format | Duration Acc. | Deps F1 | Resources F1 | Requirements F1 | Best attempt | Total tokens |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| `.sm` | **0.969** | **1.000** | **1.000** | **0.967** | 1 | 31146 |
| `.mm` | **0.985** | **0.000** | **1.000** | **0.532** | 5 | 30247 |
| `.rcp` | **1.000** | **1.000** | **1.000** | **1.000** | 1 | 2745 |
| `.msrcp` | **0.000** | **0.000** | **-** | **-** | - | 23206 |

**Total tokens used:** 87344 (prompt: 47304, completion: 40040)

### Observations

- The baseline pipeline relies on a single fixed schema description and format hints embedded in the prompt.
- Common failure modes include: incorrect capacity extraction from `RESOURCEAVAILABILITIES`, empty `requirements[]` arrays, and missing `START`/`END` dummy tasks.
- Errors are passed back verbatim; the LLM does not receive a structured analysis of which fields are wrong.

---

## Experiment 2 — PARSE Pipeline, No Grammar Induction (`generate_parsers_v2.py`)

### Description

The v2 pipeline introduces two improvements over the baseline:

**ARCHITECT step:** Before generating parser code, the LLM is asked to rewrite the DSL schema description to make it more suitable for the specific format, using an example file as context. The enriched schema is then used in all subsequent generation and repair prompts.

**SCOPE validation:** Instead of passing raw error messages back, a structured validator (`scope_validate`) analyses the parser output and produces targeted, human-readable error descriptions for each detected structural problem (missing dummy tasks, zero capacities, empty requirements, etc.).

### Running

```bash
python generate_parsers_v2.py \
    --source data/benchmark/1_raw_data \
    --gt data/benchmark/2_ground_truth \
    --output data/benchmark/4_generated_parsers_v2 \
    --attempts 5
```

### Results

| Format | Duration Acc. | Deps F1 | Resources F1 | Requirements F1 | Best attempt | Total tokens |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| `.sm` | **0.969** | **1.000** | **1.000** | **0.967** | 1 | 11636 |
| `.mm` | **0.985** | **1.000** | **1.000** | **0.513** | 1 | 15129 |
| `.rcp` | **1.000** | **1.000** | **1.000** | **1.000** | 1 | 11009 |
| `.msrcp` | **1.000** | **1.000** | **1.000** | **1.000** | 1 | 12028 |

**Total tokens used:** 49802 (prompt: 28959, completion: 20843)

### Observations

- All four formats produced a passing parser on the first attempt, indicating that the ARCHITECT enrichment significantly reduces the number of repair iterations needed.
- Requirements F1 for `.mm` (0.513) is notably lower than for other formats. This is consistent with the structural complexity of multi-mode requirement parsing, where requirements must be extracted per mode, and modes are encoded across continuation lines in the source file.
- Token usage includes one additional ARCHITECT call per format (enriching the schema), which adds overhead compared to the baseline but reduces the number of repair iterations.

---

## Comparison

| Pipeline | Avg. Duration Acc. | Avg. Deps F1 | Avg. Resources F1 | Avg. Requirements F1 | Total tokens |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline (v1) | 0.739 | 0.500 | 0.750 | 0.625 | 87344 |
| PARSE (v2) | 0.989 | 1.000 | 1.000 | 0.870 | 49802 |

The key result of Experiment 2 is that the PARSE pipeline achieves high quality across three out of four metrics with a single generation attempt per format, while spending roughly 50 000 tokens total. The remaining gap in Requirements F1 for `.mm` is the main target for further improvement.

---

## Output Structure

All experiment outputs follow the same directory layout:

```
data/benchmark/
├── 1_raw_data/             # Source files (.sm, .mm, .rcp, .msrcp)
├── 2_ground_truth/         # GT JSON files
├── 3_parser_output/        # Output of deterministic parsers (evaluate_parsers.py)
├── 4_generated_parsers_v1/ # Baseline pipeline output
│   ├── sm/
│   │   ├── attempts/       # Parser code per attempt
│   │   ├── best_parser.py  # Best parser selected by aggregate score
│   │   └── generation_summary.json
│   └── global_summary.json
└── 4_generated_parsers_v2/ # PARSE pipeline output (same structure)
    └── sm/
        ├── enriched_schema.txt   # ARCHITECT output
        ├── induced_grammar.lark  # Grammar Induction output (if enabled)
        ├── attempts/
        ├── best_parser.py
        └── generation_summary.json
```