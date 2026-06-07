# tests/

Unit tests for the library.

Run all tests:

```bash
pytest tests/ -v
```

Or run individual test files:

```bash
pytest tests/test_evaluate_parsers.py -v
pytest tests/test_generate_gt.py -v
```

## Test Coverage

| Test File                  | Covered Functionality                                                                                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `test_evaluate_parsers.py` | `_f1`, `get_duration`, `get_dep_ids`, `get_requirements`, `get_resource_signature`, `align_tasks`, `build_id_map`, `evaluate_run`         |
| `test_generate_gt.py`      | `generate_gt`: successful generation, skipping unsupported formats, parser error handling, output JSON validation, and directory creation |

## Dependencies

```text
pytest>=7.0
```

Installed automatically with the project's main dependencies. If needed, install manually:

```bash
pip install pytest
```