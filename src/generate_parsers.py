"""
generate_parsers.py — генерация кода парсера через LLM с петлёй валидации.

Подход B: LLM один раз пишет Python-код парсера для формата.
Затем этот код запускается на всех файлах формата без LLM.

Алгоритм для каждого формата:
  1. LLM получает пример файла + схему DSL → генерирует код парсера
  2. Код запускается на валидационных файлах (у которых есть GT)
  3. Считаются метрики vs GT
  4. Если код упал или метрики ниже порога → ошибка передаётся в LLM → регенерация
  5. Повторяем до n_attempts раз, берём лучший парсер по метрикам
  6. Лучший парсер сохраняется, запускается на всех файлах формата

Запуск:
    python generate_parsers.py
    python generate_parsers.py --source data/raw --gt data/gt --output data/results --attempts 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from evaluate_parsers import evaluate_run

load_dotenv()

api_key = os.environ.get("LITELLM_API_KEY")
if not api_key:
    raise ValueError("Не найден LITELLM_API_KEY!")

llm_client = OpenAI(
    api_key=api_key,
    base_url="https://api.duckduck.cloud/v1",
    timeout=120.0,
)

# MODEL = "iairlab/qwen2.5-72b"
MODEL = "openai/gpt-5.4-nano"

# Порог метрик - если оба выше, парсер считается приемлемым досрочно
EARLY_STOP_DURATION = 0.85
EARLY_STOP_DEPS_F1  = 0.80

SUPPORTED_EXTENSIONS = [".sm", ".mm", ".rcp", ".msrcp"]


# DSL-схема для промпта (компактная строковая версия)

DSL_SCHEMA_DESCRIPTION = """
The target format is a JSON object matching this Pydantic schema (SchedulingProblem):

{
  "schema_version": "0.1",
  "problem_id": "<str>",
  "domain": "rcpsp",
  "description": "<str, optional>",
  "project": {"name": "<str>", "objective": "minimize_makespan"},
  "extensions": {"rcpsp": {}},
  "resources": [
    {
      "id": "<str>",           // e.g. "R1", "N1", "SKILL1"
      "capacity": <int>,
      "extensions": {"rcpsp": {"type": "renewable" | "non_renewable"}}
    }
  ],
  "tasks": [
    {
      "id": "<str>",           // e.g. "T2", "J3"
      "dependencies": [{"task_id": "<str>", "type": "FS"}],
      "extensions": {
        "rcpsp": {
          "modes": [
            {
              "mode_id": "M1",
              "duration": <int>,
              "requirements": [{"resource_id": "<str>", "quantity": <int>}]
              // only include resources where quantity > 0
            }
          ]
        }
      }
    }
  ]
}

KEY RULES that apply to ALL formats:
- First job (jobnr 1) and last job are dummy start/finish tasks with duration=0.
  DO NOT include them in the tasks list, but DO reference them in dependencies.
- Dependencies: the file lists SUCCESSORS — you must INVERT to predecessors.
  For task T: find all tasks X that list T as a successor → add dependency on X.
- Task IDs: use the format specified per file type (e.g. "T{N}" or "J{N}").
- Only include resource requirements where quantity > 0.
"""

# Промпты генерации парсера по формату

FORMAT_HINTS = {
    ".sm": """
Format: PSPLIB single-mode (.sm)
- Sections separated by lines of '*'.
- RESOURCES section: counts of renewable (R) and non-renewable (N) resources.
- PRECEDENCE RELATIONS: jobnr  #modes  #successors  succ1 succ2 ...
- REQUESTS/DURATIONS: jobnr  mode  duration  R1  R2  ...  N1  N2  ...
  Single mode per job (mode=1 always).
- RESOURCEAVAILABILITIES: capacities in order R1 R2 ... N1 N2 ...
- Task IDs: use "T{jobnr}" format (e.g. jobnr 2 → "T2").
- Resource IDs: "R1", "R2", ... for renewable; "N1", "N2", ... for non-renewable.
""",
    ".mm": """
Format: PSPLIB multi-mode (.mm)
- Same section structure as .sm but each job has multiple modes.
- REQUESTS/DURATIONS: first line of a job has jobnr; continuation lines (same job, 
  next modes) start with a TAB or whitespace before the mode number.
- Each mode line: [jobnr]  mode  duration  R1  R2  N1  N2
  Continuation:   [TAB]    mode  duration  R1  R2  N1  N2
- Resources: R-prefixed = renewable, N-prefixed = non-renewable.
- Task IDs: use "T{jobnr}".
- Mode IDs: "M1", "M2", "M3", ...
""",
    ".rcp": """
Format: Patterson compact (.rcp)
- No section headers, just numbers.
- Line 1: n_jobs  n_resources
- Line 2: capacity_R1  capacity_R2  ...
- Lines 3..N+2: one line per job in order:
    duration  R1_qty  R2_qty  ...  n_successors  succ1  succ2  ...
- All resources are renewable.
- Task IDs: use "T{N}" where N is the 1-based job index.
- Resource IDs: "R1", "R2", ...
""",
    ".msrcp": """
Format: Multi-Skill RCP (.msrcp)
- Sections marked as \\* Section Name *\\.
- Project Module: first line is [n_jobs  n_skills  n_workers  n_periods].
  Then one or two single-integer lines (horizon, skip them).
  Then n_jobs task lines: duration  n_successors  succ1  succ2  ...
- Workforce Module with Skill Levels: n_workers × n_skills matrix of skill levels.
  Resource capacity for SKILL_k = number of workers with non-zero value in column k.
- Skill Requirements Module: n_jobs × n_skills matrix (one row per job including dummies).
  Map non-zero values as resource requirements.
- Task IDs: use "T{N}".
- Resource IDs: "SKILL1", "SKILL2", ...
""",
}

# Генерация кода парсера через LLM

SYSTEM_GENERATE = """\
You are an expert Python developer. Your task is to write a Python parser function \
that reads a scheduling problem file and converts it to a JSON structure.

Write ONLY a single Python function with this exact signature:
    def parse(file_path: str) -> dict:

Requirements:
- Read the file at file_path using Path(file_path).read_text(encoding="utf-8", errors="replace")
- Return a plain Python dict matching the SchedulingProblem JSON structure described below
- Do NOT use any external libraries except: pathlib, re, json (all stdlib)
- Do NOT include imports at module level — put them inside the function if needed
- Do NOT include any class definitions or other code outside the function
- The function must be self-contained

{schema}

{format_hint}

Return ONLY the Python function code, no explanations, no markdown fences.
"""

SYSTEM_FIX = """\
You are an expert Python developer. You wrote a parser function that has errors.

Here is the function:
```python
{code}
```

Here are the errors encountered when running it on validation files:
{errors}

Fix the function. Return ONLY the corrected Python function code, no explanations, \
no markdown fences. Keep the same signature: def parse(file_path: str) -> dict:
"""


def call_llm(messages: list[dict]) -> tuple[str, dict]:
    """
    Вызов LLM.
    Возвращает (text, token_usage) где token_usage:
      {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    """
    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    usage = response.usage
    token_usage = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }
    return response.choices[0].message.content.strip(), token_usage


def extract_code(llm_output: str) -> str:
    """
    Извлекает Python-код из ответа LLM.
    Убирает markdown-обёртки если LLM их добавила.
    """
    text = llm_output.strip()
    # Убираем ```python ... ``` или ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        # Убираем первую строку (```python или ```)
        lines = lines[1:]
        # Убираем последнюю строку если это ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def generate_parser_code(ext: str, example_file: str) -> tuple[str, dict]:
    """Первая генерация парсера: пример файла + описание формата"""
    example_content = Path(example_file).read_text(encoding="utf-8", errors="replace")
    # Обрезаем пример до ~3000 символов чтобы не перегружать контекст
    if len(example_content) > 3000:
        example_content = example_content[:3000] + "\n... [truncated for brevity]"

    system = SYSTEM_GENERATE.format(
        schema=DSL_SCHEMA_DESCRIPTION,
        format_hint=FORMAT_HINTS.get(ext, ""),
    )
    user = f"Here is an example {ext} file:\n\n{example_content}"

    text, usage = call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    return text, usage


def fix_parser_code(code: str, errors: list[str]) -> tuple[str, dict]:
    """Регенерация с передачей ошибок обратно в LLM. Возвращает (код, токены)."""
    errors_str = "\n\n".join(errors[:5])  # не больше 5 ошибок чтобы не раздувать промпт
    system = SYSTEM_FIX.format(code=code, errors=errors_str)
    text, usage = call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": "Please fix the function."},
    ])
    return text, usage


# Запуск сгенерированного парсера

def load_parse_fn(code: str):
    """
    Динамически загружает функцию parse() из строки кода.
    Возвращает callable или бросает исключение если код не компилируется.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    spec = importlib.util.spec_from_file_location("_generated_parser", tmp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse


def run_parser_on_file(parse_fn, file_path: str) -> tuple[dict | None, str | None]:
    """
    Запускает parse_fn на одном файле.
    Возвращает (result_dict, None) или (None, error_string).
    """
    try:
        result = parse_fn(file_path)
        if not isinstance(result, dict):
            return None, f"parse() returned {type(result).__name__}, expected dict"
        return result, None
    except Exception:
        return None, traceback.format_exc()


# Валидация парсера на GT-файлах

def validate_parser(parse_fn, val_files: list[Path], gt_dir: Path) -> tuple[dict, list[str]]:
    """
    Запускает парсер на валидационных файлах, считает метрики vs GT.

    Возвращает:
      metrics - {"duration_accuracy": float, "dependencies_f1": float, ...}
      errors - список строк с описанием ошибок для передачи в LLM
    """
    all_metrics: dict[str, list[float]] = {
        "duration_accuracy": [],
        "dependencies_f1": [],
        "resources_f1": [],
        "requirements_f1": [],
    }
    errors: list[str] = []

    for file in val_files:
        gt_file = gt_dir / f"{file.stem}.json"
        if not gt_file.exists():
            continue

        pred_dict, error = run_parser_on_file(parse_fn, str(file))

        if error:
            errors.append(f"File {file.name}:\n{error}")
            continue

        gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
        m = evaluate_run(gt_data, pred_dict)

        for key, val in m.items():
            if key in all_metrics:
                all_metrics[key].append(val)

    if not any(all_metrics.values()):
        return {k: 0.0 for k in all_metrics}, errors

    return {
        k: statistics.mean(v) if v else 0.0
        for k, v in all_metrics.items()
    }, errors


# Основной цикл: генерация + валидация + исправление

def generate_and_validate(ext: str, example_file: str, val_files: list[Path], gt_dir: Path, n_attempts: int, output_dir: Path) -> dict:
    """
    Генерирует парсер для формата `ext`, валидирует, исправляет при ошибках.
    Сохраняет все попытки и лучший парсер.

    Возвращает сводку результатов.
    """
    ext_clean = ext.lstrip(".")
    attempts_dir = output_dir / ext_clean / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    best_code: str | None = None
    best_metrics: dict = {"duration_accuracy": 0.0, "dependencies_f1": 0.0}
    best_attempt: int = -1

    run_results: list[dict] = []

    # Счётчики токенов по всем попыткам
    total_tokens: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    print(f"Формат: {ext}  |  Валидация на {len(val_files)} файлах  |  {n_attempts} попыток")

    current_code: str | None = None
    last_errors: list[str] = []

    for attempt in range(1, n_attempts + 1):
        print(f"\n  Попытка {attempt}/{n_attempts}...")

        # Генерация или исправление
        try:
            if current_code is None:
                # Первая генерация
                raw, usage = generate_parser_code(ext, example_file)
            else:
                # Исправление с учётом ошибок
                raw, usage = fix_parser_code(current_code, last_errors)

            # Накапливаем токены
            for k in total_tokens:
                total_tokens[k] += usage.get(k, 0)

            print(f"    Токены: prompt={usage['prompt_tokens']}  "
                  f"completion={usage['completion_tokens']}  "
                  f"total={usage['total_tokens']}")

            current_code = extract_code(raw)
        except Exception as e:
            print(f"    LLM вернула ошибку: {e}")
            run_results.append({"attempt": attempt, "error": str(e), "metrics": None, "tokens": None})
            continue

        # Сохраняем код попытки
        code_file = attempts_dir / f"attempt_{attempt}.py"
        code_file.write_text(current_code, encoding="utf-8")

        # Загрузка функции
        try:
            parse_fn = load_parse_fn(current_code)
        except Exception:
            err = traceback.format_exc()
            print(f"    Код не компилируется:\n{err[:300]}")
            last_errors = [f"The code does not compile:\n{err}"]
            run_results.append({"attempt": attempt, "compile_error": err, "metrics": None, "tokens": usage})
            continue

        # Валидация
        metrics, errors = validate_parser(parse_fn, val_files, gt_dir)
        last_errors = errors

        dur_acc = metrics.get("duration_accuracy", 0.0)
        dep_f1 = metrics.get("dependencies_f1", 0.0)

        print(f" Duration Accuracy: {dur_acc:.3f}  |  Deps F1: {dep_f1:.3f}"
              f"  |  Errors: {len(errors)}")

        run_results.append({
            "attempt": attempt,
            "metrics": metrics,
            "n_errors": len(errors),
            "code_file": str(code_file),
            "tokens": usage,
        })

        # Обновляем лучший результат
        score = dur_acc + dep_f1
        best_score = best_metrics["duration_accuracy"] + best_metrics["dependencies_f1"]
        if score > best_score:
            best_metrics = metrics
            best_code = current_code
            best_attempt = attempt

        # Ранняя остановка
        if dur_acc >= EARLY_STOP_DURATION and dep_f1 >= EARLY_STOP_DEPS_F1:
            print(f"Достигнут порог качества на попытке {attempt}. Останавливаемся.")
            break

        # Если есть ошибки - передадим их в следующую попытку (уже в last_errors)
        if not errors and score < 0.5:
            # Код работает без ошибок, но метрики плохие
            last_errors = [
                f"The parser runs without errors but metrics are poor: "
                f"duration_accuracy={dur_acc:.2f}, dependencies_f1={dep_f1:.2f}. "
                f"Check the dependency inversion logic and task ID format."
            ]

    print(f"\n  Токены итого: prompt={total_tokens['prompt_tokens']}  "
          f"completion={total_tokens['completion_tokens']}  "
          f"total={total_tokens['total_tokens']}")

    # Сохраняем лучший парсер
    summary = {
        "format": ext,
        "best_attempt": best_attempt,
        "best_metrics": best_metrics,
        "total_tokens": total_tokens,
        "all_attempts": run_results,
    }

    if best_code:
        best_path = output_dir / ext_clean / "best_parser.py"
        best_path.write_text(best_code, encoding="utf-8")
        summary["best_parser_path"] = str(best_path)
        print(f"\n  Лучший парсер (попытка {best_attempt}) сохранён → {best_path}")
        print(f"  Метрики: {best_metrics}")
    else:
        print(f"\n НЕ УДАЛОСЬ сгенерировать работающий парсер для {ext}")

    summary_path = output_dir / ext_clean / "generation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return summary

# Применение лучшего парсера ко всем файлам формата

def apply_best_parser(ext: str, source_dir: Path, output_dir: Path) -> None:
    """
    Загружает best_parser.py для формата и прогоняет его на всех файлах.
    Результаты сохраняются в output_dir/{ext}/results/.
    """
    ext_clean = ext.lstrip(".")
    parser_path = output_dir / ext_clean / "best_parser.py"

    if not parser_path.exists():
        print(f"  Нет best_parser.py для {ext}, пропускаем.")
        return

    spec = importlib.util.spec_from_file_location("_best_parser", str(parser_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parse_fn = module.parse

    results_dir = output_dir / ext_clean / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    files = list(source_dir.glob(f"*{ext}"))
    print(f"  Применяем парсер {ext} к {len(files)} файлам...")

    ok, failed = 0, 0
    for file in sorted(files):
        out_file = results_dir / f"{file.stem}.json"
        result, error = run_parser_on_file(parse_fn, str(file))
        if error:
            print(f" ERR {file.name}: {error[:100]}")
            failed += 1
        else:
            out_file.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            ok += 1

    print(f"  Готово: {ok} OK, {failed} ошибок → {results_dir}")


# CLI

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Генерация парсеров через LLM.")
    p.add_argument("--source", default="data/benchmark/1_raw_data",
                   help="Папка с сырыми файлами")
    p.add_argument("--gt", default="data/benchmark/2_ground_truth",
                   help="Папка с GT JSON файлами")
    p.add_argument("--output", default="data/benchmark/4_generated_parsers",
                   help="Папка для сохранения сгенерированных парсеров и результатов")
    p.add_argument("--attempts", type=int, default=5,
                   help="Макс. число попыток генерации на формат")
    p.add_argument("--formats", nargs="+", default=SUPPORTED_EXTENSIONS,
                   help="Форматы для обработки (по умолчанию все)")
    p.add_argument("--apply", action="store_true",
                   help="После генерации применить лучший парсер ко всем файлам")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    source = Path(args.source)
    gt_dir = Path(args.gt)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    all_summaries: dict[str, dict] = {}

    for ext in args.formats:
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"Неизвестный формат: {ext}, пропускаем.")
            continue

        # Файлы этого формата из source
        files = sorted(source.glob(f"*{ext}"))
        if not files:
            print(f"Нет файлов {ext} в {source}, пропускаем.")
            continue

        # Пример для промпта — первый файл
        example_file = str(files[0])

        # Валидационные файлы — все у которых есть GT
        val_files = [f for f in files if (gt_dir / f"{f.stem}.json").exists()]
        if not val_files:
            print(f"Нет GT файлов для валидации формата {ext}. "
                  f"Генерируем парсер без оценки качества.")
            val_files = files[:1]  # хотя бы проверим что код запускается

        summary = generate_and_validate(
            ext=ext,
            example_file=example_file,
            val_files=val_files,
            gt_dir=gt_dir,
            n_attempts=args.attempts,
            output_dir=output,
        )
        all_summaries[ext] = summary

        if args.apply:
            apply_best_parser(ext, source, output)

    # Сводная таблица
    print("ИТОГО:")
    grand_total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for ext, summary in all_summaries.items():
        m = summary.get("best_metrics", {})
        t = summary.get("total_tokens", {})
        for k in grand_total_tokens:
            grand_total_tokens[k] += t.get(k, 0)
        print(f"  {ext:8s}  "
              f"dur_acc={m.get('duration_accuracy', 0):.3f}  "
              f"deps_f1={m.get('dependencies_f1', 0):.3f}  "
              f"best_attempt={summary.get('best_attempt', -1)}  "
              f"tokens={t.get('total_tokens', 0)}"
              f" (prompt={t.get('prompt_tokens', 0)} + completion={t.get('completion_tokens', 0)})")

    print(f"\n  Токены суммарно по всем форматам:")
    print(f"    prompt: {grand_total_tokens['prompt_tokens']}")
    print(f"    completion: {grand_total_tokens['completion_tokens']}")
    print(f"    total: {grand_total_tokens['total_tokens']}")

    global_path = output / "global_summary.json"
    global_path.write_text(
        json.dumps(
            {"summaries": all_summaries, "grand_total_tokens": grand_total_tokens},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nПолная сводка: {global_path}")


if __name__ == "__main__":
    main()