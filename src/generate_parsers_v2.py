"""
generate_parsers_v2.py — генерация парсеров с Grammar Induction + PARSE-подход.

Изменения относительно v1:
  - ARCHITECT: перед генерацией парсера LLM обогащает DSL_SCHEMA_DESCRIPTION
    примерами, описаниями полей и типичными ошибками → LLM-friendly схема
  - SCOPE: validate_parser теперь проверяет не только duration/deps,
    но и resources/requirements, и передаёт конкретные ошибки обратно в LLM
  - Grammar Induction (опционально): LLM сначала выводит EBNF-грамматику формата,
    из неё строится lark-парсер, и только потом генерируется финальный код
  - Все изменения обратно совместимы: CLI и структура output не меняются

Запуск:
    python generate_parsers_v2.py
    python generate_parsers_v2.py --source data/raw --gt data/gt --output data/results --attempts 5
    python generate_parsers_v2.py --grammar-induction   # включает Grammar Induction
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import tempfile
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.evaluate_parsers import evaluate_run

load_dotenv()

api_key = os.environ.get("LITELLM_API_KEY")
if not api_key:
    raise ValueError("Не найден LITELLM_API_KEY!")

llm_client = OpenAI(
    api_key=api_key,
    base_url="https://api.duckduck.cloud/v1",
    timeout=120.0,
)

MODEL = "openai/gpt-5.4-nano"

EARLY_STOP_DURATION = 0.85
EARLY_STOP_DEPS_F1  = 0.80
EARLY_STOP_RES_F1 = 0.80  # парсер должен корректно извлекать capacity ресурсов

SUPPORTED_EXTENSIONS = [".sm", ".mm", ".rcp", ".msrcp"]

# DSL-схема
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
      "id": "<str>",           // "START", "END", or "T{jobnr}" e.g. "T2"
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
- DUMMY TASK NAMING (CRITICAL — must match exactly):
  * First job (jobnr 1): id = "START", duration = 0, requirements = []
  * Last job (jobnr N): id = "END", duration = 0, requirements = []
  * All other jobs: id = "T{jobnr}" (e.g. jobnr 2 → "T2", jobnr 5 → "T5")
  * INCLUDE "START" and "END" in the tasks[] list.
  * When a task depends on dummy start, write: {"task_id": "START", "type": "FS"}
  * When a task is a predecessor of dummy finish, END's dependency is on real task ids.
- Dependencies: the file lists SUCCESSORS — you must INVERT to predecessors.
  For task T: find all tasks X that list T as a successor → add dependency on X.
  Example: if jobnr 1 lists successors [2, 3, 4], then T2/T3/T4 each get
  dependency {"task_id": "START", "type": "FS"}.
- Only include resource requirements where quantity > 0.
"""

FORMAT_HINTS = {
    ".sm": """
Format: PSPLIB single-mode (.sm)
- Sections separated by lines of '*'.
- RESOURCES section: counts of renewable (R) and non-renewable (N) resources.
- PRECEDENCE RELATIONS: jobnr  #modes  #successors  succ1 succ2 ...
- REQUESTS/DURATIONS: jobnr  mode  duration  R1  R2  ...  N1  N2  ...
  Single mode per job (mode=1 always).
- RESOURCEAVAILABILITIES: TWO lines follow the header:
    Line 1 (SKIP): resource name labels, e.g. "  R 1  R 2  R 3  R 4"
    Line 2 (READ): the actual integer capacities, e.g. "    9   11   11   16"
  Example from a real file:
    RESOURCEAVAILABILITIES:
      R 1  R 2  R 3  R 4
        9   11   11   16
  → R1.capacity=9, R2.capacity=11, R3.capacity=11, R4.capacity=16
  WARNING: do NOT read the label line — it contains "1 2 3 4" from "R 1 R 2 R 3 R 4"
  which are NOT the capacities. Always skip the first line, read the second.
- Task IDs: jobnr 1 → "START", last jobnr → "END", others → "T{jobnr}" (e.g. "T2").
- Resource IDs: "R1", "R2", ... for renewable; "N1", "N2", ... for non-renewable.
- Include START and END in tasks[]. START and END have duration=0, requirements=[].
""",
    ".mm": """
Format: PSPLIB multi-mode (.mm)
- Same section structure as .sm but each job has multiple modes.
- REQUESTS/DURATIONS: first line of a job has jobnr; continuation lines (same job,
  next modes) start with a TAB or whitespace before the mode number.
- Each mode line: [jobnr]  mode  duration  R1  R2  N1  N2
  Continuation:   [TAB]    mode  duration  R1  R2  N1  N2
- Resources: R-prefixed = renewable, N-prefixed = non-renewable.
- RESOURCE AVAILABILITIES: In .mm files the section is called "RESOURCE AVAILABILITIES"
  (with a space in the middle), NOT "RESOURCEAVAILABILITIES". Search case-insensitively.
  TWO lines follow the header:
    Line 1 (SKIP): tab-separated resource name labels, e.g. "\tR 1\tR 2\tN 1\tN 2"
    Line 2 (READ): tab-separated integer capacities, e.g. "\t30\t27\t71\t71"
  Real example from a .mm file (tabs shown as spaces here):
     RESOURCE AVAILABILITIES 
        R 1   R 2   N 1   N 2
        30    27    71    71
  → R1.capacity=30, R2.capacity=27, N1.capacity=71, N2.capacity=71
  Parsing tip: after finding the section header line, skip the next non-empty line
  (labels), then parse integers from the following non-empty line (capacities).
  WARNING: "R 1  R 2  N 1  N 2" contains digits 1,2,1,2 — NOT the capacities.
- Task IDs: jobnr 1 → "START", last jobnr → "END", others → "T{jobnr}".
- Mode IDs: "M1", "M2", "M3", ...
- Include START and END in tasks[]. START and END have duration=0, requirements=[].
""",
    ".rcp": """
Format: Patterson compact (.rcp)
- No section headers, just numbers.
- Line 1: n_jobs  n_resources
- Line 2: capacity_R1  capacity_R2  ...
- Lines 3..N+2: one line per job in order:
    duration  R1_qty  R2_qty  ...  n_successors  succ1  succ2  ...
- All resources are renewable.
- Task IDs: job index 1 → "START", last job → "END", others → "T{N}" (e.g. "T2").
- Resource IDs: "R1", "R2", ...
- Include START and END in tasks[]. START and END have duration=0, requirements=[].
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
- Task IDs: job index 1 → "START", last job → "END", others → "T{N}".
- Resource IDs: "SKILL1", "SKILL2", ...
- Include START and END in tasks[]. START and END have duration=0, requirements=[].
""",
}


# ARCHITECT: обогащение схемы для LLM (PARSE-подход)
ARCHITECT_PROMPT = """\
You are optimizing a JSON schema description so that an LLM can extract data from
scheduling files more reliably. The current schema description is often misunderstood:
parsers frequently produce empty requirements lists and miss resource assignments.

Current schema description:
{base_schema}

Format hint for this file type:
{format_hint}

Example file content:
{example_content}

Your task: rewrite the schema description to make it LLM-friendly. For each field add:
1. A concrete description of what it means in this format
2. An extraction example from the example file above
3. Common mistakes to avoid

CRITICAL FIELDS that are most often wrong — give them extra attention:
- resources[].capacity  — must be parsed from RESOURCEAVAILABILITIES section
- resources[].extensions.rcpsp.type  — "renewable" or "non_renewable"
- tasks[].extensions.rcpsp.modes[].requirements  — THIS IS THE MOST CRITICAL FIELD.
  It lives inside modes[], not at the task level. Quantity must be > 0 to be included.
  Parsers often return empty requirements lists — explain exactly how to find these values.
- tasks[].dependencies  — file lists SUCCESSORS, must be INVERTED to predecessors

Return ONLY the improved schema description as plain text (not JSON, not code).
No preamble, no explanation — just the improved description starting with "The target format...".
"""


def architect_enrich_schema(ext: str, example_content: str) -> str:
    """
    ARCHITECT: берёт базовую DSL-схему и обогащает её на основе примера файла.
    Возвращает улучшенную строку описания схемы для использования в промпте генерации.
    """
    prompt = ARCHITECT_PROMPT.format(
        base_schema=DSL_SCHEMA_DESCRIPTION,
        format_hint=FORMAT_HINTS.get(ext, ""),
        example_content=example_content[:2000],
    )

    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a schema optimization expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    enriched = response.choices[0].message.content.strip()
    usage = response.usage

    print(f"    [ARCHITECT] Схема обогащена. "
          f"Токены: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")

    return enriched, {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


# Grammar Induction (опциональный шаг перед генерацией кода)

GRAMMAR_INDUCTION_PROMPT = """\
You are a formal language expert. Your task is to infer a grammar for a scheduling
file format from examples, then use it to guide parser generation.

Format description:
{format_hint}

Example files:
{examples}

Write a formal grammar for this file format in Lark EBNF notation.
The grammar should explicitly cover ALL structural elements:
- Section headers and delimiters
- Job/task lines with all their fields
- Resource capacity lines
- Successor lists (which must be inverted to predecessors)

Focus especially on the resource requirements section — this is most often missed.

Return ONLY the Lark grammar, no explanation.
Start with: start: ...
"""

GRAMMAR_TO_CODE_PROMPT = """\
You are an expert Python developer. You have inferred a grammar for a scheduling file format.

Grammar (Lark EBNF):
{grammar}

Format description:
{format_hint}

Target JSON schema:
{schema}

Using this grammar as a guide, write a Python parser function with signature:
    def parse(file_path: str) -> dict:

The function should follow the grammar structure exactly when extracting data.
Pay special attention to resource requirements — they must appear inside
tasks[].extensions.rcpsp.modes[].requirements[], NOT at the task level.

Requirements:
- Use only stdlib: pathlib, re, json
- No imports at module level — put them inside the function
- No class definitions or other code outside the function
- Return a plain Python dict matching the SchedulingProblem schema

Return ONLY the Python function code, no markdown fences.
"""


def induce_grammar(ext: str, example_files: list[str]) -> tuple[str, dict]:
    """
    Grammar Induction: LLM выводит EBNF-грамматику из примеров файлов.
    Возвращает (grammar_text, token_usage).
    """
    examples_content = []
    for path in example_files[:3]:  # не более 3 примеров
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        examples_content.append(f"--- {Path(path).name} ---\n{content[:1500]}")

    prompt = GRAMMAR_INDUCTION_PROMPT.format(
        format_hint=FORMAT_HINTS.get(ext, ""),
        examples="\n\n".join(examples_content),
    )

    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a formal grammar expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    grammar = response.choices[0].message.content.strip()
    usage = response.usage

    print(f"    [GRAMMAR] Грамматика выведена ({len(grammar)} символов). "
          f"Токены: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")

    return grammar, {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def validate_grammar(grammar_text: str, example_files: list[str]) -> tuple[bool, str]:
    """
    Проверяет грамматику через lark (если установлен).
    Возвращает (is_valid, error_message).
    """
    try:
        import lark
        parser = lark.Lark(grammar_text, parser="earley")

        # Пробуем распарсить хотя бы один пример
        for path in example_files[:1]:
            content = Path(path).read_text(encoding="utf-8", errors="replace")
            try:
                parser.parse(content)
                return True, ""
            except lark.exceptions.LarkError as e:
                return False, f"Grammar cannot parse example file: {e}"

        return True, ""

    except ImportError:
        # lark не установлен — просто проверяем синтаксис грамматики
        print("    [GRAMMAR] lark не установлен, пропускаем валидацию грамматики")
        return True, ""
    except Exception as e:
        return False, str(e)


def generate_parser_from_grammar(
    grammar: str,
    ext: str,
    enriched_schema: str,
) -> tuple[str, dict]:
    """
    Генерирует код парсера используя выведенную грамматику как спецификацию.
    """
    prompt = GRAMMAR_TO_CODE_PROMPT.format(
        grammar=grammar,
        format_hint=FORMAT_HINTS.get(ext, ""),
        schema=enriched_schema,
    )

    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert Python developer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    code = response.choices[0].message.content.strip()
    usage = response.usage

    return code, {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


# SCOPE: улучшенная валидация с детальной диагностикой

def scope_validate(result: dict | None, error: str | None, gt_data: dict | None = None) -> list[str]:
    """
    SCOPE static guardrails: проверяем структуру результата по схеме.
    Возвращает список человекочитаемых ошибок для передачи в LLM.

    Проверяем:
    - Базовая структура (resources, tasks)
    - resources: наличие capacity и extensions.rcpsp.type
    - tasks: наличие START и END dummy-задач
    - tasks: наличие dependencies и extensions.rcpsp.modes
    - modes: наличие duration и requirements (самое частое место ошибки!)
    - requirements: resource_id и quantity > 0
    """
    errors = []

    if error:
        errors.append(f"RUNTIME ERROR: {error[:500]}")
        return errors

    if not isinstance(result, dict):
        errors.append(f"parse() returned {type(result).__name__}, expected dict")
        return errors

    # Верхний уровень
    for field in ("resources", "tasks", "project", "domain", "problem_id"):
        if field not in result:
            errors.append(f"Missing top-level field: '{field}'")

    resources = result.get("resources", [])
    tasks = result.get("tasks", [])

    if not isinstance(resources, list) or len(resources) == 0:
        errors.append(
            "resources[] is empty or not a list. "
            "Parse RESOURCEAVAILABILITIES section to get capacities."
        )

    if not isinstance(tasks, list) or len(tasks) == 0:
        errors.append("tasks[] is empty or not a list.")

    # Проверка dummy-задач START/END
    task_ids = {t.get("id") for t in tasks if isinstance(t, dict)}
    if "START" not in task_ids:
        errors.append(
            "Missing 'START' task. The first dummy job (jobnr 1) must be included in "
            "tasks[] with id='START', duration=0, requirements=[]. "
            "Real tasks that have jobnr 1 as predecessor must reference "
            "{'task_id': 'START', 'type': 'FS'}."
        )
    if "END" not in task_ids:
        errors.append(
            "Missing 'END' task. The last dummy job must be included in "
            "tasks[] with id='END', duration=0, requirements=[]."
        )

    # Ресурсы
    # Проверяем паттерн sequential indices (1,2,3,4) — парсер путает индекс с capacity
    all_caps = [res.get("capacity", 0) for res in resources if isinstance(res, dict)]
    if (all_caps and all_caps == list(range(1, len(all_caps) + 1))
            or all_caps == list(range(len(all_caps)))):
        errors.append(
            f"CRITICAL: Resource capacities are sequential indices {all_caps} — "
            f"you are returning the resource index instead of the actual capacity. "
            f"The RESOURCEAVAILABILITIES line contains the real capacities. "
            f"Example: if it reads '30  27  71  71', then R1.capacity=30, R2.capacity=27, N1.capacity=71, N2.capacity=71."
        )

    for i, res in enumerate(resources):
        rid = res.get("id", f"res[{i}]")

        if "capacity" not in res or res["capacity"] == 0:
            errors.append(
                f"Resource '{rid}': missing or zero capacity. "
                f"capacity must be parsed from RESOURCEAVAILABILITIES."
            )

        ext = res.get("extensions") or {}
        rcpsp_ext = ext.get("rcpsp") or {}
        rtype = rcpsp_ext.get("type")
        if rtype not in ("renewable", "non_renewable"):
            errors.append(
                f"Resource '{rid}': extensions.rcpsp.type must be "
                f"'renewable' or 'non_renewable', got: {rtype!r}"
            )

    # Задачи
    tasks_with_empty_requirements = []
    tasks_with_no_modes = []
    tasks_with_no_deps_field = []
    dummy_ids = {"START", "END"}

    for task in tasks:
        tid = task.get("id", "?")
        is_dummy = tid in dummy_ids

        if "dependencies" not in task:
            tasks_with_no_deps_field.append(tid)

        ext = task.get("extensions") or {}
        rcpsp_ext = ext.get("rcpsp") or {}
        modes = rcpsp_ext.get("modes")

        if not modes:
            tasks_with_no_modes.append(tid)
            continue

        # Проверяем каждый mode (для START/END пустые requirements ок)
        for mode in modes:
            mode_id = mode.get("mode_id", "?")
            requirements = mode.get("requirements", [])

            if not isinstance(requirements, list):
                errors.append(
                    f"Task '{tid}' mode '{mode_id}': requirements must be a list, "
                    f"got {type(requirements).__name__}"
                )
                continue

            if len(requirements) == 0 and not is_dummy:
                tasks_with_empty_requirements.append(f"{tid}/{mode_id}")
            elif not is_dummy:
                for req in requirements:
                    if "resource_id" not in req:
                        errors.append(
                            f"Task '{tid}' mode '{mode_id}': "
                            f"requirement missing 'resource_id'"
                        )
                    qty = req.get("quantity", 0)
                    if not isinstance(qty, int) or qty <= 0:
                        errors.append(
                            f"Task '{tid}' mode '{mode_id}': "
                            f"requirement quantity must be int > 0, got {qty!r}. "
                            f"Only include requirements where quantity > 0."
                        )
    # Агрегированные ошибки — чтобы не спамить на каждую задачу
    if tasks_with_no_deps_field:
        sample = tasks_with_no_deps_field[:5]
        errors.append(
            f"{len(tasks_with_no_deps_field)} tasks missing 'dependencies' field "
            f"(e.g. {sample}). Remember: file lists SUCCESSORS → invert to predecessors."
        )

    if tasks_with_no_modes:
        sample = tasks_with_no_modes[:5]
        errors.append(
            f"{len(tasks_with_no_modes)} tasks missing extensions.rcpsp.modes "
            f"(e.g. {sample}). Every task needs at least one mode with duration."
        )

    frac_empty = (
        len(tasks_with_empty_requirements) / max(len(tasks), 1)
    )
    if frac_empty > 0.8 and len(tasks) > 2:
        errors.append(
            f"CRITICAL: {len(tasks_with_empty_requirements)}/{len(tasks)} tasks have "
            f"empty requirements[]. This means the parser is not extracting resource "
            f"requirements from the REQUESTS/DURATIONS section. "
            f"Requirements live at: tasks[].extensions.rcpsp.modes[].requirements[]. "
            f"Each requirement: {{\"resource_id\": \"R1\", \"quantity\": <int>}}. "
            f"Only include where quantity > 0."
        )

    return errors


# Промпты генерации
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

CRITICAL NAMING RULES (validated automatically — wrong names = 0 score):
1. Dummy start task (jobnr 1)  → id MUST be "START", include in tasks[], duration=0
2. Dummy finish task (last job) → id MUST be "END",   include in tasks[], duration=0
3. All other tasks              → id MUST be "T{{jobnr}}" e.g. "T2", "T3"
4. Dependencies to dummy start  → {{"task_id": "START", "type": "FS"}}
5. Do NOT use "T1" or "T{{N}}" for dummy tasks — only "START" and "END"

{schema}

{format_hint}

Return ONLY the Python function code, no explanations, no markdown fences.
"""

SYSTEM_FIX_TEMPLATE = """\
You are an expert Python developer. You wrote a parser function that has errors.

Here is the function:
```python
PLACEHOLDER_CODE
```

Here are the errors encountered when running it on validation files:
PLACEHOLDER_ERRORS

Fix the function. Pay special attention to these common mistakes:
1. Missing "START" task — jobnr 1 MUST appear in tasks[] with id="START", not "T1"
2. Missing "END" task — last jobnr MUST appear in tasks[] with id="END"
3. Dependencies to dummy start must reference task_id "START", not "T1"
4. resource capacity must come from RESOURCEAVAILABILITIES line — do NOT use index
5. requirements[] is empty → parse from REQUESTS/DURATIONS section
6. requirements live inside modes[], not at task level
7. dependencies must be INVERTED from successors listed in the file
8. Do NOT return resource index as capacity — read the actual integers from RESOURCEAVAILABILITIES

Return ONLY the corrected Python function code, no explanations, \
no markdown fences. Keep the same signature: def parse(file_path: str) -> dict:
"""


def call_llm(messages: list[dict]) -> tuple[str, dict]:
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
    text = llm_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def generate_parser_code(ext: str, example_file: str, enriched_schema: str) -> tuple[str, dict]:
    """Первая генерация парсера с обогащённой схемой"""
    example_content = Path(example_file).read_text(encoding="utf-8", errors="replace")
    if len(example_content) > 3000:
        example_content = example_content[:3000] + "\n... [truncated for brevity]"

    system = SYSTEM_GENERATE.format(
        schema=enriched_schema,
        format_hint=FORMAT_HINTS.get(ext, ""),
    )
    user = f"Here is an example {ext} file:\n\n{example_content}"

    return call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])


def fix_parser_code(code: str, errors: list[str], enriched_schema: str) -> tuple[str, dict]:
    """Регенерация с ошибками и обогащённой схемой"""
    errors_str = "\n\n".join(errors[:5])
    system = (
        SYSTEM_FIX_TEMPLATE
        .replace("PLACEHOLDER_CODE", code)
        .replace("PLACEHOLDER_ERRORS", errors_str)
    )
    return call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": "Please fix the function."},
    ])


# Запуск парсера

def load_parse_fn(code: str):
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
    try:
        result = parse_fn(file_path)
        if not isinstance(result, dict):
            return None, f"parse() returned {type(result).__name__}, expected dict"
        return result, None
    except Exception:
        return None, traceback.format_exc()


# Валидация на GT-файлах

def validate_parser(
    parse_fn,
    val_files: list[Path],
    gt_dir: Path,
) -> tuple[dict, list[str]]:
    """
    Запускает парсер на валидационных файлах.
    Использует scope_validate для детальной диагностики структурных ошибок.
    """
    all_metrics: dict[str, list[float]] = {
        "duration_accuracy": [],
        "dependencies_f1": [],
        "resources_f1": [],
        "requirements_f1": [],
    }
    structural_errors: list[str] = []  # ошибки структуры (SCOPE)
    runtime_errors: list[str] = [] # ошибки выполнения

    for file in val_files:
        gt_file = gt_dir / f"{file.stem}.json"
        if not gt_file.exists():
            continue

        pred_dict, runtime_err = run_parser_on_file(parse_fn, str(file))
        gt_data = json.loads(gt_file.read_text(encoding="utf-8"))

        # SCOPE: структурная валидация
        errs = scope_validate(pred_dict, runtime_err, gt_data)
        if runtime_err:
            runtime_errors.append(f"File {file.name}:\n{runtime_err[:400]}")
        if errs:
            # Добавляем только уникальные ошибки чтобы не дублировать
            for e in errs:
                if e not in structural_errors:
                    structural_errors.append(e)

        if pred_dict is None:
            continue

        # Считаем метрики vs GT
        m = evaluate_run(gt_data, pred_dict)
        for key, val in m.items():
            if key in all_metrics:
                all_metrics[key].append(val)

    # Объединяем: сначала рантайм-ошибки, потом структурные
    all_errors = runtime_errors + structural_errors

    if not any(all_metrics.values()):
        return {k: 0.0 for k in all_metrics}, all_errors

    return {
        k: statistics.mean(v) if v else 0.0
        for k, v in all_metrics.items()
    }, all_errors


# Основной цикл
def generate_and_validate(
    ext: str,
    example_file: str,
    val_files: list[Path],
    gt_dir: Path,
    n_attempts: int,
    output_dir: Path,
    use_grammar_induction: bool = False,
    all_example_files: list[str] | None = None,
) -> dict:
    """
    Генерирует парсер для формата ext с PARSE-подходом (ARCHITECT + SCOPE).
    Опционально: Grammar Induction перед генерацией кода.
    """
    ext_clean = ext.lstrip(".")
    attempts_dir = output_dir / ext_clean / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    best_code: str | None = None
    best_metrics: dict = {"duration_accuracy": 0.0, "dependencies_f1": 0.0}
    best_attempt: int = -1
    run_results: list[dict] = []
    total_tokens: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    print(f"\nФормат: {ext}  |  Валидация на {len(val_files)} файлах  |  {n_attempts} попыток")

    # ARCHITECT — обогащаем схему
    print("  [ARCHITECT] Оптимизирую схему под LLM...")
    example_content = Path(example_file).read_text(encoding="utf-8", errors="replace")
    enriched_schema, arch_usage = architect_enrich_schema(ext, example_content)
    for k in total_tokens:
        total_tokens[k] += arch_usage.get(k, 0)

    # Сохраняем обогащённую схему для диагностики
    schema_path = output_dir / ext_clean / "enriched_schema.txt"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(enriched_schema, encoding="utf-8")
    print(f"    Сохранена -> {schema_path}")

    # Grammar Induction
    induced_grammar: str | None = None
    if use_grammar_induction:
        print("  [GRAMMAR] Вывожу грамматику из примеров...")
        examples_for_grammar = all_example_files or [example_file]
        grammar_text, grammar_usage = induce_grammar(ext, examples_for_grammar)
        for k in total_tokens:
            total_tokens[k] += grammar_usage.get(k, 0)

        is_valid, grammar_err = validate_grammar(grammar_text, examples_for_grammar)
        if is_valid:
            induced_grammar = grammar_text
            grammar_path = output_dir / ext_clean / "induced_grammar.lark"
            grammar_path.write_text(grammar_text, encoding="utf-8")
            print(f"    Грамматика сохранена -> {grammar_path}")
        else:
            print(f"    Грамматика невалидна: {grammar_err}")
            print("    Продолжаем без Grammar Induction.")

    # Основной цикл генерации/исправления
    current_code: str | None = None
    last_errors: list[str] = []

    for attempt in range(1, n_attempts + 1):
        print(f"\n  Попытка {attempt}/{n_attempts}...")

        try:
            if current_code is None:
                if induced_grammar:
                    # Grammar Induction path: генерируем по грамматике
                    print("    [GRAMMAR→CODE] Генерирую парсер из грамматики...")
                    raw, usage = generate_parser_from_grammar(
                        induced_grammar, ext, enriched_schema
                    )
                else:
                    # Стандартный path с обогащённой схемой
                    raw, usage = generate_parser_code(ext, example_file, enriched_schema)
            else:
                # SCOPE feedback: исправляем с детальными ошибками
                raw, usage = fix_parser_code(current_code, last_errors, enriched_schema)

            for k in total_tokens:
                total_tokens[k] += usage.get(k, 0)

            print(f"    Токены: prompt={usage['prompt_tokens']} "
                  f"completion={usage['completion_tokens']} "
                  f"total={usage['total_tokens']}")

            current_code = extract_code(raw)

        except Exception as e:
            print(f"    LLM вернула ошибку: {e}")
            run_results.append({"attempt": attempt, "error": str(e)})
            continue

        # Сохраняем код
        code_file = attempts_dir / f"attempt_{attempt}.py"
        code_file.write_text(current_code, encoding="utf-8")

        # Загрузка функции
        try:
            parse_fn = load_parse_fn(current_code)
        except Exception:
            err = traceback.format_exc()
            print(f"    Код не компилируется:\n{err[:300]}")
            last_errors = [f"The code does not compile:\n{err}"]
            run_results.append({
                "attempt": attempt,
                "compile_error": err[:300],
                "tokens": usage,
            })
            continue

        # Валидация
        metrics, errors = validate_parser(parse_fn, val_files, gt_dir)
        last_errors = errors

        dur_acc = metrics.get("duration_accuracy", 0.0)
        dep_f1 = metrics.get("dependencies_f1", 0.0)
        res_f1 = metrics.get("resources_f1", 0.0)
        req_f1 = metrics.get("requirements_f1", 0.0)

        print(
            f"    dur_acc={dur_acc:.3f}  deps_f1={dep_f1:.3f}  "
            f"res_f1={res_f1:.3f}  req_f1={req_f1:.3f}  "
            f"scope_errors={len(errors)}"
        )
        if errors:
            for e in errors[:3]:
                print(f"      ↳ {e[:120]}")

        run_results.append({
            "attempt": attempt,
            "metrics": metrics,
            "n_scope_errors": len(errors),
            "code_file": str(code_file),
            "tokens": usage,
        })

        # Обновляем лучший результат по сумме всех четырёх метрик
        score = dur_acc + dep_f1 + res_f1 + req_f1
        best_score = sum(best_metrics.get(k, 0.0) for k in
                         ("duration_accuracy", "dependencies_f1", "resources_f1", "requirements_f1"))
        if score > best_score:
            best_metrics = metrics
            best_code = current_code
            best_attempt = attempt

        # Ранняя остановка - все три ключевые метрики должны быть выше порога
        if (dur_acc >= EARLY_STOP_DURATION
                and dep_f1 >= EARLY_STOP_DEPS_F1
                and res_f1 >= EARLY_STOP_RES_F1):
            print(f"    Достигнут порог качества на попытке {attempt}.")
            break

        # Если нет рантайм-ошибок, но метрики плохие - даём явный hint
        if not errors and score < 0.5:
            last_errors = [
                f"Parser runs without errors but metrics are poor: "
                f"duration_accuracy={dur_acc:.2f}, dependencies_f1={dep_f1:.2f}, "
                f"resources_f1={res_f1:.2f}, requirements_f1={req_f1:.2f}. "
                f"Check: 1) dependency inversion (file lists successors → invert), "
                f"2) task ID format, 3) requirements extraction from REQUESTS/DURATIONS."
            ]

        # Отдельный hint если res_f1=0 но остальное хорошее
        if res_f1 == 0.0 and dur_acc > 0.5 and not any("capacity" in e for e in last_errors):
            last_errors = [
                f"CRITICAL: resources_f1=0.0 — all resources have capacity=0. "
                f"You must parse capacity values from the RESOURCEAVAILABILITIES section. "
                f"For .sm/.mm files: find the line 'RESOURCEAVAILABILITIES' and read the "
                f"integers on the NEXT line — those are the capacities in order R1, R2, ..., N1, N2. "
                f"Example: if the line reads '9  11  11  16', then R1.capacity=9, R2.capacity=11, "
                f"R3.capacity=11, R4.capacity=16. Do NOT leave capacity=0."
            ] + last_errors

    print(f"\n  Токены итого: {total_tokens}")

    # Сохранение
    summary = {
        "format": ext,
        "best_attempt": best_attempt,
        "best_metrics": best_metrics,
        "total_tokens": total_tokens,
        "used_grammar_induction": use_grammar_induction and induced_grammar is not None,
        "all_attempts": run_results,
    }

    if best_code:
        best_path = output_dir / ext_clean / "best_parser.py"
        best_path.write_text(best_code, encoding="utf-8")
        summary["best_parser_path"] = str(best_path)
        print(f"\n  Лучший парсер (попытка {best_attempt}) → {best_path}")
        print(f"  Метрики: {best_metrics}")
    else:
        print(f"\n  НЕ УДАЛОСЬ сгенерировать работающий парсер для {ext}")

    summary_path = output_dir / ext_clean / "generation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return summary


# Применение лучшего парсера
def apply_best_parser(ext: str, source_dir: Path, output_dir: Path) -> None:
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
            print(f"    ERR {file.name}: {error[:80]}")
            failed += 1
        else:
            out_file.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            ok += 1

    print(f"  Готово: {ok} OK, {failed} ошибок -> {results_dir}")


# CLI
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Генерация парсеров через LLM (v2: PARSE + Grammar Induction).")
    p.add_argument("--source", default="data/benchmark/1_raw_data")
    p.add_argument("--gt", default="data/benchmark/2_ground_truth")
    p.add_argument("--output", default="data/benchmark/4_generated_parsers")
    p.add_argument("--attempts", type=int, default=5)
    p.add_argument("--formats", nargs="+", default=SUPPORTED_EXTENSIONS)
    p.add_argument("--apply", action="store_true",
                   help="Применить лучший парсер ко всем файлам после генерации")
    p.add_argument("--grammar-induction", action="store_true",
                   help="Включить Grammar Induction (LLM выводит EBNF перед кодом)")
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

        files = sorted(source.glob(f"*{ext}"))
        if not files:
            print(f"Нет файлов {ext} в {source}, пропускаем.")
            continue

        example_file = str(files[0])
        val_files = [f for f in files if (gt_dir / f"{f.stem}.json").exists()]
        if not val_files:
            print(f"Нет GT файлов для {ext}, генерируем без оценки.")
            val_files = files[:1]

        summary = generate_and_validate(
            ext=ext,
            example_file=example_file,
            val_files=val_files,
            gt_dir=gt_dir,
            n_attempts=args.attempts,
            output_dir=output,
            use_grammar_induction=args.grammar_induction,
            all_example_files=[str(f) for f in files[:3]],
        )
        all_summaries[ext] = summary

        if args.apply:
            apply_best_parser(ext, source, output)

    # Итоговая таблица
    print("ИТОГО:")
    grand_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for ext, summary in all_summaries.items():
        m = summary.get("best_metrics", {})
        t = summary.get("total_tokens", {})
        for k in grand_total:
            grand_total[k] += t.get(k, 0)
        print(
            f"  {ext:8s}  "
            f"dur={m.get('duration_accuracy', 0):.3f}  "
            f"deps={m.get('dependencies_f1', 0):.3f}  "
            f"res={m.get('resources_f1', 0):.3f}  "
            f"req={m.get('requirements_f1', 0):.3f}  "
            f"attempt={summary.get('best_attempt', -1)}  "
            f"tokens={t.get('total_tokens', 0)}"
        )

    print(f"\n  Токены суммарно: {grand_total}")

    global_path = output / "global_summary.json"
    global_path.write_text(
        json.dumps(
            {"summaries": all_summaries, "grand_total_tokens": grand_total},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nПолная сводка: {global_path}")


if __name__ == "__main__":
    main()