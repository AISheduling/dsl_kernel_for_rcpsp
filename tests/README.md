# tests/

Юнит-тесты для библиотеки. Запуск:

```bash
pytest tests/ -v
```

Или отдельный файл:

```bash
pytest tests/test_evaluate_parsers.py -v
pytest tests/test_generate_gt.py -v
```

## Покрытие

| Файл теста | Что тестируется |
|---|---|
| `test_evaluate_parsers.py` | `_f1`, `get_duration`, `get_dep_ids`, `get_requirements`, `get_resource_signature`, `align_tasks`, `build_id_map`, `evaluate_run` |
| `test_generate_gt.py` | `generate_gt`: успешная генерация, пропуск неподдерживаемых форматов, обработка ошибок парсера, корректность выходного JSON, создание директорий |

## Зависимости

```
pytest>=7.0
```

Устанавливаются вместе с основными зависимостями проекта. Если нет:

```bash
pip install pytest
```