# RCPSP DSL — Structured Project Data Extraction Library

**RCPSP** (Resource-Constrained Project Scheduling Problem) is the problem of constructing an optimal project schedule under limited resource availability. It is one of the core problems in project scheduling and exists in many variants: classical single-mode RCPSP, multi-mode RCPSP (MRCPSP), multi-skilled RCPSP (MSRCPSP), location-constrained scheduling, and others.

This library addresses an infrastructure problem common to all these variants: **automatic extraction of a structured project description from arbitrary input files** and conversion into a unified **L0 format** suitable for downstream optimization.

The L0 format follows a compositional design. The core schema describes a generic scheduling graph, while domain-specific information is encapsulated in the `extensions` field. The current implementation supports two extensions:

* **`rcpsp`** — classical project scheduling (single-mode and multi-mode, renewable and non-renewable resources)
* **`cluster`** — scheduling computational workloads on GPU clusters

---

## Features

* **Deterministic parsers** for `.sm`, `.mm`, `.rcp`, and `.msrcp` formats with accurate and reproducible extraction
* **LLM-based parser generation** for arbitrary input formats with iterative validation and automatic refinement
* **Pydantic-based L0 schema** with built-in validation of both inputs and outputs
* **Extraction quality evaluation** via `evaluate_run()` using four metrics against annotated ground truth
* **Ground truth generation** from raw files using deterministic parsers

---

## Installation

```bash
git clone https://github.com/AISheduling/dsl_kernel_for_rcpsp
cd dsl_kernel_for_rcpsp
pip install -r requirements.txt
```

---

## Quick Start

### Parse a file into the L0 format

```python
from parsers import parse_sm, parse_mm, parse_rcp, parse_msrcp

# Single-mode RCPSP (.sm)
problem = parse_sm("data/j302_1.sm")
print(problem.model_dump_json(indent=2, exclude_none=True))

# Multi-mode RCPSP (.mm)
problem = parse_mm("data/J501_1.mm")

# Compact Patterson format (.rcp)
problem = parse_rcp("data/EV1.rcp")

# Multi-skilled RCPSP (.msrcp)
problem = parse_msrcp("data/MSLIB_Set1_1.msrcp")
```

### Generate ground truth data

```bash
python generate_gt.py \
    --source data/benchmark/1_raw_data \
    --output data/benchmark/2_ground_truth
```

### Evaluate parser quality

```python
from evaluate_parsers import evaluate_run
import json

gt = json.load(open("data/benchmark/2_ground_truth/j3010_1.json"))
pred = json.load(open("my_parser_output.json"))

metrics = evaluate_run(gt, pred)
# {
#   "duration_accuracy":  0.969,
#   "dependencies_f1": 1.000,
#   "resources_f1": 1.000,
#   "requirements_f1": 0.967
# }
```

Or evaluate all files in a directory:

```bash
python evaluate_parsers.py \
    --source data/benchmark/1_raw_data \
    --gt data/benchmark/2_ground_truth \
    --output data/benchmark/3_parser_output
```

### Generate a parser for a new format using an LLM

```bash
# Baseline pipeline
python generate_parsers.py \
    --source data/benchmark/1_raw_data \
    --gt data/benchmark/2_ground_truth \
    --output data/benchmark/4_generated_parsers \
    --attempts 5

# PARSE pipeline with ARCHITECT + SCOPE (recommended)
python generate_parsers_v2.py \
    --source data/benchmark/1_raw_data \
    --gt data/benchmark/2_ground_truth \
    --output data/benchmark/4_generated_parsers_v2 \
    --attempts 5
```

---

## Repository Structure

```text
.
├── src/
│   ├── dsl_schema.py          # Pydantic schema of the L0 format
│   ├── parsers.py             # Deterministic parsers (.sm/.mm/.rcp/.msrcp)
│   ├── evaluate_parsers.py    # Evaluation metrics (evaluate_run)
│   ├── generate_gt.py         # Ground truth generation
│   ├── generate_parsers.py    # LLM pipeline v1 (baseline)
│   └── generate_parsers_v2.py # LLM pipeline v2 (PARSE: ARCHITECT + SCOPE)
├── tests/
│   ├── conftest.py
│   ├── test_evaluate_parsers.py
│   └── test_generate_gt.py
├── docs/
│   ├── FORMAT_L0.md
│   ├── DATASETS.md
│   ├── METRICS_AND_ANNOTATION.md
│   └── EXPERIMENTS.md
├── data/
│   ├── datasets/
│   │   ├── base_dataset/
│   │   ├── extension_1/
│   │   └── extension_2/
│   ├── benchmark/
│   │   ├── 1_raw_data/
│   │   └── 2_ground_truth/
│   ├── results_v1/
│   └── results_v2/
├── requirements.txt
└── README.md
```

---

## L0 Format

A complete specification is available in `docs/FORMAT_L0.md`.

Example output:

```json
{
  "schema_version": "0.1",
  "problem_id": "j3010_1",
  "domain": "rcpsp",
  "project": { "name": "j3010_1", "objective": "minimize_makespan" },
  "resources": [
    {
      "id": "R1",
      "capacity": 9,
      "extensions": {
        "rcpsp": {
          "type": "renewable"
        }
      }
    }
  ],
  "tasks": [
    {
      "id": "START",
      "dependencies": [],
      "extensions": {
        "rcpsp": {
          "modes": [
            {
              "mode_id": "M1",
              "duration": 0,
              "requirements": []
            }
          ]
        }
      }
    },
    {
      "id": "T2",
      "dependencies": [
        {
          "task_id": "START",
          "type": "FS"
        }
      ],
      "extensions": {
        "rcpsp": {
          "modes": [
            {
              "mode_id": "M1",
              "duration": 6,
              "requirements": [
                {
                  "resource_id": "R1",
                  "quantity": 3
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

Key conventions:

* The first and last tasks are always named `START` and `END`
* The `dependencies` field contains **predecessors**, not successors
* Only resources with `quantity > 0` are included in `requirements`
* Multi-mode activities store all execution modes in the `modes` array

---

## Metrics

`evaluate_run(gt_data, pred_data)` returns four evaluation metrics.

See `docs/METRICS_AND_ANNOTATION.md` for details.

| Metric              | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `duration_accuracy` | Fraction of tasks with correctly extracted durations            |
| `dependencies_f1`   | Micro-averaged F1 score over predecessor relations              |
| `resources_f1`      | F1 score over `(resource_id, capacity)` pairs                   |
| `requirements_f1`   | F1 score over task resource requirements within execution modes |

---

## Experimental Results

Experiments were conducted on the PSPLIB j30, MMLIB, and MSLIB benchmark datasets. A detailed description is available in `docs/EXPERIMENTS.md`.

**Best results obtained with the PARSE v2 pipeline:**

| Format   | Duration Acc. | Deps F1 | Resources F1 | Requirements F1 | Attempt | Tokens |
| -------- | :-----------: | :-----: | :----------: | :-------------: | :-----: | :----: |
| `.sm`    |     0.969     |  1.000  |     1.000    |      0.967      |    1    |  11636 |
| `.mm`    |     0.985     |  1.000  |     1.000    |      0.513      |    1    |  15129 |
| `.rcp`   |     1.000     |  1.000  |     1.000    |      1.000      |    1    |  11009 |
| `.msrcp` |     1.000     |  1.000  |     1.000    |      1.000      |    1    |  12028 |

All formats achieved acceptable quality in **a single generation attempt**.

Total token consumption across all four formats: **49,802 tokens**.

---

## Tests

```bash
pytest tests/ -v
```

---

## Datasets

The benchmark data is not stored directly in the repository. It can be downloaded from:

https://drive.google.com/drive/folders/1vjWRgPVkmWCHlmVbrD2DNJWBDSP_qd1Z?usp=sharing

The downloaded benchmark instances and their corresponding ground-truth annotations are expected to be placed in the `data/benchmark/` directory.

The `data/datasets/` directory contains example scheduling problems manually created by the authors and represented directly in the L0 schema.

See `docs/DATASETS.md` for details.

| Dataset    | Format        | Problem Type |
| ---------- | ------------- | ------------ |
| PSPLIB j30 | `.sm`, `.rcp` | RCPSP        |
| MMLIB50    | `.mm`         | MRCPSP       |
| MSLIB1     | `.msrcp`      | MSRCPSP      |

```
