# Metrics and Annotation

This document describes the evaluation metrics used to assess parser quality, how the ground truth data was generated, and how to run evaluation on new data.

---

## Ground Truth Generation

### How GT Files Are Produced

Ground truth (GT) files are JSON documents that follow the L0 schema (see [FORMAT_L0.md](FORMAT_L0.md)). Each GT file corresponds to one benchmark instance and is generated automatically from the source file using the deterministic hand-written parsers (`parse_sm`, `parse_mm`, `parse_rcp`, `parse_msrcp`).

The script `generate_gt.py` runs each parser on every source file in the raw data directory and writes the resulting L0-formatted JSON to the ground truth directory:

```bash
python generate_gt.py \
    --source data/benchmark/1_raw_data \
    --output data/benchmark/2_ground_truth
```

Each output file has the same stem as the source file and a `.json` extension. The parsers are deterministic: running the script twice on the same input produces identical output.

### Why Deterministic Parsers Are Used as GT

The deterministic parsers (`parsers.py`) implement a precise, rule-based reading of each format according to its specification. They are validated manually against known benchmark properties (number of activities, number of resources, network topology, best-known solutions) and serve as the reference implementation of the L0 extraction logic. Using them to produce GT ensures that the ground truth is consistent, reproducible, and free of annotation subjectivity.

The LLM-based pipeline (`generate_parsers.py`, `generate_parsers_v2.py`) is then evaluated against this GT: its task is to replicate what the deterministic parsers produce, but without any format-specific hand-coded rules.

### GT File Format

Each GT file is a valid L0 JSON document. Example structure:

```json
{
  "schema_version": "0.1",
  "problem_id": "j3010_1",
  "domain": "rcpsp",
  "project": { "name": "j3010_1", "objective": "minimize_makespan" },
  "resources": [
    { "id": "R1", "capacity": 9, "extensions": { "rcpsp": { "type": "renewable" } } }
  ],
  "tasks": [
    {
      "id": "START",
      "dependencies": [],
      "extensions": { "rcpsp": { "modes": [{ "mode_id": "M1", "duration": 0, "requirements": [] }] } }
    },
    {
      "id": "T2",
      "dependencies": [{ "task_id": "START", "type": "FS" }],
      "extensions": {
        "rcpsp": {
          "modes": [{
            "mode_id": "M1",
            "duration": 6,
            "requirements": [{ "resource_id": "R1", "quantity": 3 }]
          }]
        }
      }
    }
  ]
}
```

### Adding New Instances to the Benchmark

1. Place the source file in `data/benchmark/1_raw_data/`.
2. Run `generate_gt.py` — it will automatically pick up the new file and write the GT.
3. Verify the output JSON is structurally correct and matches known benchmark properties.
4. If the deterministic parser fails on the new file, fix the parser first, then regenerate.

---

## Evaluation Metrics

Four metrics are computed for each parsed instance by comparing the LLM-generated parser output against the GT. All metrics are computed at the task or resource level and averaged across instances in the validation set.

The evaluation logic is implemented in `evaluate_parsers.py`, function `evaluate_run(gt_data, pred_data)`.

### Task Alignment

Before computing task-level metrics, GT and predicted tasks are aligned. Alignment is attempted first by matching task `id` values directly. For tasks whose IDs do not match (e.g. the LLM used a different prefix), positional alignment is used as a fallback, after filtering out dummy tasks with `duration=0` from the predicted output.

An ID normalisation map is built from the aligned pairs so that dependency IDs in the predicted output can be compared correctly against GT dependency IDs even when prefixes differ.

### Duration Accuracy

Measures the fraction of tasks for which the parser correctly extracted the duration. For single-mode instances each task has one mode; for multi-mode instances the first mode is used for comparison.

$$\text{Duration Accuracy} = \frac{|\{t \in T : \text{duration}(t)_{\text{pred}} = \text{duration}(t)_{\text{gt}}\}|}{|T|}$$

where $T$ is the set of aligned task pairs.

**Range:** [0, 1]. A value of 1.0 means every task duration was extracted correctly.

### Dependencies F1

Measures how accurately the parser reconstructed the predecessor graph. For each aligned task pair the predicted predecessor ID set is compared to the GT set. Counts are accumulated across all tasks and a single micro-averaged F1 is computed.

$$\text{Precision} = \frac{TP_{\text{deps}}}{TP_{\text{deps}} + FP_{\text{deps}}}, \quad \text{Recall} = \frac{TP_{\text{deps}}}{TP_{\text{deps}} + FN_{\text{deps}}}$$

$$F1_{\text{deps}} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Common failure mode:** Parsers that read successor lists without inverting them score near 0.

### Resources F1

Measures how accurately the parser extracted the set of resources and their capacities. Each resource is represented as the string key `"<id>:<capacity>"`; both fields must match for a true positive.

$$F1_{\text{res}} = f1\!\left(\{\text{"id:cap"}\}_{\text{pred}},\ \{\text{"id:cap"}\}_{\text{gt}}\right)$$

**Common failure mode:** Parsers that return sequential indices (1, 2, 3, …) instead of the actual values from the `RESOURCEAVAILABILITIES` section score 0.

### Requirements F1

Measures how accurately the parser extracted resource requirements inside task modes. For each aligned task pair, requirements from all modes are collected into a set of `"<resource_id>:<quantity>"` strings and compared against the GT set. Counts are accumulated across all tasks.

$$F1_{\text{req}} = f1\!\left(\{\text{"res:qty"}\}_{\text{pred}},\ \{\text{"res:qty"}\}_{\text{gt}}\right)$$

Only requirements with `quantity > 0` are included. This metric captures whether the parser correctly reads the `REQUESTS/DURATIONS` section and maps quantities to the right resources inside `modes[]`.

**Common failure mode:** Requirements placed at the task level instead of inside `modes[]`, or empty `requirements[]` arrays.

---

## Aggregate Score

During LLM parser generation, a single scalar is used to rank candidate parsers and select the best one:

$$\text{Score} = \text{Duration Accuracy} + F1_{\text{deps}} + F1_{\text{res}} + F1_{\text{req}}$$

Maximum possible score is 4.0. Early stopping is triggered when Duration Accuracy ≥ 0.85, Dependencies F1 ≥ 0.80, and Resources F1 ≥ 0.80 simultaneously.

---

## Running Evaluation

**Evaluate all deterministic parsers:**

```bash
python evaluate_parsers.py \
    --source data/benchmark/1_raw_data \
    --gt data/benchmark/2_ground_truth \
    --output data/benchmark/3_parser_output
```

**Evaluate a single output programmatically:**

```python
from evaluate_parsers import evaluate_run
import json

gt = json.load(open("data/benchmark/2_ground_truth/instance.json"))
pred = json.load(open("my_output.json"))
metrics = evaluate_run(gt, pred)
print(metrics)
```

Example output:

```json
{
  "duration_accuracy": 0.969,
  "dependencies_f1": 1.000,
  "resources_f1": 1.000,
  "requirements_f1": 0.967
}
```