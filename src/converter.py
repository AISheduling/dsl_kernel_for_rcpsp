import os
import json
import statistics
from pathlib import Path
import instructor
from openai import OpenAI
from dotenv import load_dotenv

from dsl_schema import SchedulingProblem

load_dotenv()

api_key = os.environ.get("LITELLM_API_KEY")
if not api_key:
    raise ValueError("Не найден LITELLM_API_KEY! Установите переменную окружения.")

llm_client = OpenAI(
    api_key=api_key,
    base_url="https://api.duckduck.cloud/v1",
    timeout=120.0
)

client = instructor.from_openai(llm_client, mode=instructor.Mode.MD_JSON)

model = "iairlab/qwen2.5-72b"
# model = "gpt-5.4-nano"

# ПРОМПТЫ ДЛЯ РАЗНЫХ ФОРМАТОВ

PROMPT_SM = """
You are a data extraction expert. Your task is to parse a raw PSPLIB (.sm) file and convert it into a strict JSON structure based on the provided Pydantic schema.

MAPPING RULES:

1. domain: Strictly "rcpsp".

2. project.name: Extract from the 'projects :' line or derive from 'file with basedata'.

3. resources: Locate 'RESOURCEAVAILABILITIES'. Map resources with their capacities.
   - Use IDs exactly as in the file header (e.g., "R1", "R2", "R3", "R4").
   - Set type strictly to "renewable".

4. tasks — CRITICAL RULES:
   a) EXCLUDE DUMMY TASKS AS TASKS: The first task (jobnr. 1, the source) and the last task
      (the sink/finish job) are artificial dummy tasks with duration=0.
      DO NOT output them as task objects.
      BUT: keep their jobnr. numbers for dependency tracking (see rule c).
   b) Task id format: Use "J{jobnr}" — for example, jobnr. 2 becomes id "J2", jobnr. 3 becomes "J3".
      This also applies to dummy tasks when referenced in dependencies (e.g., "J1" for the source).
   c) DEPENDENCY INVERSION: The file lists successors. You must convert them to predecessors.
      Algorithm: for EVERY task T (real and dummy), scan ALL tasks X.
      If X lists T as a successor, then T has a dependency on X.
      Add {"task_id": "J{X}", "type": "FS"} to T's dependencies list.
      IMPORTANT: Include dependencies on dummy jobnr. 1 (the source). If a real task T
      has jobnr. 1 as its only predecessor, its dependencies list must contain
      [{"task_id": "J1", "type": "FS"}], NOT an empty list.
      Only omit a dependency on the dummy sink (last jobnr.) — that one is never a predecessor.

5. modes & durations: Locate 'REQUESTS/DURATIONS'.
   - Create a single mode per task with mode_id: "M1".
   - duration: integer value from the file.
   - requirements: ONLY include resources where quantity > 0. Omit zero-quantity resources entirely.

6. Output format for each task:
   {
     "id": "J2",
     "dependencies": [{"task_id": "J1_predecessor", "type": "FS"}],
     "extensions": {"rcpsp": {"modes": [{"mode_id": "M1", "duration": 3, "requirements": [...]}]}}
   }
"""

PROMPT_MM = """
You are a data extraction expert. Your task is to parse a raw MRCPSP (.mm) file and convert it into a strict JSON structure based on the provided Pydantic schema.

MAPPING RULES:

1. domain: Strictly "rcpsp".

2. project.name: Extract from 'projects :' or 'file with basedata'.

3. resources: Locate 'RESOURCEAVAILABILITIES'.
   - Use IDs exactly as in the file header (e.g., "R1", "R2", "N1", "N2").
   - Resources labeled 'R' must have type: "renewable".
   - Resources labeled 'N' must have type: "non_renewable".

4. tasks — CRITICAL RULES:
   a) EXCLUDE DUMMY TASKS AS TASKS: The first task (jobnr. 1, the source) and the last task
      (the sink) are artificial dummy tasks with duration=0 in all modes.
      DO NOT output them as task objects.
      BUT: keep their jobnr. numbers for dependency tracking.
   b) Task id format: Use "J{jobnr}" — for example, jobnr. 2 becomes id "J2".
      This also applies to dummy tasks when referenced in dependencies.
   c) DEPENDENCY INVERSION: The file lists successors. Convert them to predecessors.
      Algorithm: for EVERY task T (real and dummy), scan ALL tasks X.
      If X lists T as a successor, then T depends on X.
      Add {"task_id": "J{X}", "type": "FS"} to T's dependencies.
      IMPORTANT: Include dependencies on dummy jobnr. 1 (the source).
      If a real task T has jobnr. 1 as its only predecessor, write
      [{"task_id": "J1", "type": "FS"}], NOT an empty list.
      Only omit dependencies on the dummy sink (last jobnr.).

5. modes: Locate 'REQUESTS/DURATIONS'. Tasks have multiple modes.
   - mode_id format: "M1", "M2", "M3", etc.
   - For each mode: map duration and resource consumption for both R and N resources.
   - requirements: ONLY include resources where quantity > 0. Omit zero-quantity resources.

6. Output format for each task:
   {
     "id": "J2",
     "dependencies": [{"task_id": "J3", "type": "FS"}],
     "extensions": {"rcpsp": {"modes": [
       {"mode_id": "M1", "duration": 5, "requirements": [{"resource_id": "R1", "quantity": 2}]},
       {"mode_id": "M2", "duration": 3, "requirements": [{"resource_id": "R2", "quantity": 4}]}
     ]}}
   }
"""

PROMPT_RCP = """
You are a data extraction expert. Your task is to parse a raw Patterson (.rcp) file and convert it into a strict JSON structure based on the provided Pydantic schema.

FILE STRUCTURE (skip empty lines when counting):
- Non-empty line 1: [task_count] [resource_count]  <- metadata only, skip
- Non-empty line 2: [cap_R1] [cap_R2] ...           <- resource capacities
- Non-empty lines 3+: one line per task in order
  Format: [duration] [R1_qty] [R2_qty] ... [successor_count] [succ1] [succ2] ...

MAPPING RULES:

1. domain: Strictly "rcpsp".

2. project.name: Generate a name like "Patterson Project".

3. resources: Read capacities from non-empty line 2.
   - Create resources "R1", "R2", etc. in order.
   - Set type strictly to "renewable".

4. tasks — CRITICAL RULES:
   a) EXCLUDE DUMMY TASKS AS TASKS: Task 1 (first task line) and the last task are dummy
      tasks with duration=0. DO NOT output them as task objects.
      BUT: keep their numbers for dependency tracking.
   b) Task id format: Use "T{N}" — task 1 = "T1", task 2 = "T2", etc.
      Dummy task 1 is referenced as "T1" in dependencies.
   c) DEPENDENCY INVERSION — most critical step:
      Build a successor list for ALL tasks (including dummies) first.
      For each real task T: collect all tasks X where T appears in X's successor list.
      Add {"task_id": "T{X}", "type": "FS"} to T's dependencies.
      IMPORTANT: If dummy task 1 lists T as a successor, T's dependencies
      MUST include {"task_id": "T1", "type": "FS"}. Do NOT use an empty list.
      Only omit dependencies where X is the dummy sink (last task number).

5. modes: For each real task, create one mode with mode_id: "M1".
   - duration: integer from position 0 of the task line.
   - requirements: resource quantities from positions 1..K. ONLY include where quantity > 0.

6. Output format example (task 2 whose only predecessor is dummy T1):
   {
     "id": "T2",
     "dependencies": [{"task_id": "T1", "type": "FS"}],
     "extensions": {"rcpsp": {"modes": [{"mode_id": "M1", "duration": 5, "requirements": [
       {"resource_id": "R1", "quantity": 10}
     ]}]}}
   }
"""

PROMPT_MSRCP = """
You are a data extraction expert. Your task is to parse a raw MSRCPSP (.msrcp) file and convert it into a strict JSON structure based on the provided Pydantic schema.

FILE STRUCTURE — READ VERY CAREFULLY:

After "* Project Module *\\": 
  - Line 1: [task_count] [worker_count] [skill_count] [level_count]
  - Lines 2..N (before the task lines): these are METADATA lines (e.g., planning horizon values).
    They are single integers on their own lines. SKIP THEM.
  - Task lines come AFTER the metadata. You must read EXACTLY task_count task entries.
    Each task entry is a line with format: [duration] [succ_count] [succ1] [succ2] ...
    OR just: [duration] [succ_count]  (if succ_count = 0)

HOW TO IDENTIFY TASK LINES vs METADATA LINES:
  - Count task_count from the header. Tasks occupy the LAST task_count non-empty lines
    before the next section marker ("* Workforce Module *").
  - Metadata lines appear BEFORE the task lines.
  - Example for task_count=3:
      84          ← metadata (skip)
      85          ← metadata (skip)
      0   1   2   ← task 1: duration=0, 1 successor (task 2)  — DUMMY START
      6   0       ← task 2: duration=6, 0 successors          — REAL TASK
      0   0       ← task 3: duration=0, 0 successors          — DUMMY FINISH

MAPPING RULES:

1. domain: Strictly "rcpsp".

2. project.name: Generate from context (e.g., "MSRCPSP Project").

3. resources: Locate "* Workforce Module with Skill Levels *\\". This is an NxM matrix (N workers, M skills).
   - Create one resource per skill: "SKILL_1", "SKILL_2", etc.
   - Capacity for each skill = count of workers with a non-zero value in that skill's column.
   - Set type strictly to "renewable".

4. tasks — CRITICAL RULES:
   a) EXCLUDE DUMMY TASKS AS TASKS: Task 1 (dummy start, duration=0) and the last task
      (dummy finish, duration=0). DO NOT output them as task objects.
      BUT keep their numbers for dependency tracking.
   b) Task id format: Use "T{N}" — task 1 = "T1", task 2 = "T2", etc.
      Dummy task 1 is referenced as "T1" in dependencies.
   c) DEPENDENCY INVERSION: Build successor lists for ALL tasks (including dummies) first.
      For each real task T: find all tasks X where T appears in X's successor list.
      Add {"task_id": "T{X}", "type": "FS"} to T's dependencies.
      IMPORTANT: If dummy task 1 lists T as a successor, include {"task_id": "T1", "type": "FS"}.
      Only omit dependencies where X is the dummy sink (last task).

5. modes: Each real task has one mode (mode_id: "M1").
   Duration = first integer of the task line (NOT from the metadata lines above tasks).

6. requirements: Locate "* Skill Requirements Module *\\".
   Lines correspond to ALL tasks in order (including dummies), one line per task.
   - Values are tab/space-separated: [qty_SKILL_1] [qty_SKILL_2] ...
   - Map skill values > 0 as: {"resource_id": "SKILL_N", "quantity": value}.
   - CRITICAL: Do NOT include skills with 0 quantity.

7. min_skill_levels: Locate "* Skill Level Requirements Module *\\". Entries separated by "-1".
   Each entry corresponds to a task in order (including dummies).
   - Entry "-1" alone means no requirements for that task.
   - Otherwise format: [skill_index] [min_level] (until "-1").
   - In the mode's extensions add: {"min_skill_levels": {"SKILL_N": level}} for requirements > 0.

8. Output format example (matching the example file above):
   {
     "id": "T2",
     "dependencies": [{"task_id": "T1", "type": "FS"}],
     "extensions": {"rcpsp": {"modes": [{
       "mode_id": "M1", "duration": 6,
       "requirements": [{"resource_id": "SKILL_2", "quantity": 2}],
       "extensions": {"min_skill_levels": {"SKILL_2": 1}}
     }]}}
   }
"""

SYSTEM_PROMPTS = {
    ".sm": PROMPT_SM,
    ".mm": PROMPT_MM,
    ".rcp": PROMPT_RCP,
    ".msrcp": PROMPT_MSRCP
}

def extract_schedule_data(text: str, ext: str) -> SchedulingProblem:
    system_prompt = SYSTEM_PROMPTS.get(ext.lower())
    if not system_prompt:
        raise ValueError(f'Расширение {ext} не поддерживается')

    return client.chat.completions.create(
        model=model,
        response_model=SchedulingProblem,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.2
    )


def debug_compare(gt_data: dict, pred_data: dict):
    """
    Печатает подробную диагностику расхождений между GT и предсказанием.
    Вызывается один раз для первого файла/прогона.
    """
    print("\n" + "="*60)
    print("DEBUG: Сравнение структур GT vs Prediction")
    print("="*60)

    gt_tasks = gt_data.get("tasks", [])
    pred_tasks = pred_data.get("tasks", [])
    print(f"  Кол-во задач в GT:   {len(gt_tasks)}")
    print(f"  Кол-во задач в Pred: {len(pred_tasks)}")

    # --- task_id типы ---
    if gt_tasks:
        gt_id = gt_tasks[0].get("task_id")
        print(f"\n  GT task_id пример:   {repr(gt_id)} (тип: {type(gt_id).__name__})")
    if pred_tasks:
        pred_id = pred_tasks[0].get("task_id")
        print(f"  Pred task_id пример: {repr(pred_id)} (тип: {type(pred_id).__name__})")

    # --- Структура первой задачи целиком ---
    if gt_tasks:
        print("\n  GT tasks[0] (первые 600 символов):")
        print("  " + json.dumps(gt_tasks[0], indent=2, ensure_ascii=False)[:600])
    if pred_tasks:
        print("\n  Pred tasks[0] (первые 600 символов):")
        print("  " + json.dumps(pred_tasks[0], indent=2, ensure_ascii=False)[:600])

    # --- Проверка пути к duration ---
    def extract_duration(task):
        rcpsp = task.get("extensions", {}).get("rcpsp", {})
        modes = rcpsp.get("modes", [])
        if isinstance(modes, list) and modes:
            return modes[0].get("duration"), "list"
        elif isinstance(modes, dict):
            first = next(iter(modes.values()), {})
            return first.get("duration"), "dict"
        return None, "not found"

    if gt_tasks:
        dur, fmt = extract_duration(gt_tasks[0])
        print(f"\n  GT duration (task[0]):   {repr(dur)} (modes формат: {fmt})")
    if pred_tasks:
        dur, fmt = extract_duration(pred_tasks[0])
        print(f"  Pred duration (task[0]): {repr(dur)} (modes формат: {fmt})")

    # --- Проверка структуры dependencies ---
    def extract_deps(task):
        deps = task.get("dependencies", [])
        if not deps:
            return [], "пусто"
        sample = deps[0]
        if isinstance(sample, dict):
            return deps, f"dict, ключи: {list(sample.keys())}"
        return deps, f"scalar ({type(sample).__name__})"

    if gt_tasks:
        deps, fmt = extract_deps(gt_tasks[0])
        print(f"\n  GT dependencies (task[0]):   {deps[:3]} ({fmt})")
    if pred_tasks:
        deps, fmt = extract_deps(pred_tasks[0])
        print(f"  Pred dependencies (task[0]): {deps[:3]} ({fmt})")

    print("="*60 + "\n")


def get_task_id(task: dict) -> str:
    """
    Извлекает идентификатор задачи.
    GT использует ключ "id" (например "J2"), Pred может использовать "id" или "task_id".
    Возвращает строку.
    """
    return str(task.get("id") or task.get("task_id") or "")


def get_duration(task: dict) -> str | None:
    """
    Извлекает duration из extensions.rcpsp.modes.
    modes может быть списком или словарём.
    """
    rcpsp = task.get("extensions", {}).get("rcpsp", {})
    modes = rcpsp.get("modes", [])
    if isinstance(modes, list) and modes:
        dur = modes[0].get("duration")
    elif isinstance(modes, dict) and modes:
        dur = next(iter(modes.values()), {}).get("duration")
    else:
        return None
    return str(dur) if dur is not None else None


def get_dep_ids(task: dict) -> set:
    """
    Извлекает множество id предшественников из dependencies.
    Поддерживает форматы:
      - [{"task_id": "J1", ...}]   ← GT формат
      - [{"predecessor_id": "3"}]  ← альтернативный формат
      - ["J1", "J2"]               ← скалярный формат
    """
    deps = task.get("dependencies", [])
    result = set()
    for d in deps:
        if isinstance(d, dict):
            pid = d.get("task_id") or d.get("predecessor_id") or d.get("id")
            if pid is not None:
                result.add(str(pid))
        else:
            result.add(str(d))
    return result


def normalize_dep_ids(dep_ids: set, id_map: dict) -> set:
    """
    Приводит id предшественников к единому формату через id_map.
    id_map: {pred_raw_id -> gt_raw_id} — строится при выравнивании задач.
    Если маппинга нет — оставляем как есть (сравнение не сломается, просто не совпадёт).
    """
    return {id_map.get(d, d) for d in dep_ids}


def align_tasks(gt_tasks: list, pred_tasks: list) -> list[tuple[dict, dict]]:
    """
    Выравнивает GT и Pred задачи для попарного сравнения.

    Стратегия:
    1. Пробуем сопоставить по id напрямую (например, "J2" == "J2").
    2. Если прямого совпадения нет — сопоставляем по позиции.
       GT хранит только "реальные" задачи (без dummy старта/финиша),
       Pred может включать dummy задачи с duration=0 в начале и конце.
       Поэтому из Pred сначала отфильтровываем задачи с duration=0
       (типичные dummy), затем выравниваем по порядку.

    Возвращает список пар (gt_task, pred_task).
    pred_task может быть пустым словарём, если совпадение не найдено.
    """
    # Сначала пробуем сопоставление по id
    pred_by_id = {get_task_id(t): t for t in pred_tasks}
    pairs_by_id = []
    unmatched_gt = []

    for gt_task in gt_tasks:
        gt_id = get_task_id(gt_task)
        if gt_id in pred_by_id:
            pairs_by_id.append((gt_task, pred_by_id[gt_id]))
        else:
            unmatched_gt.append(gt_task)

    # Если все GT задачи нашли пару по id — возвращаем
    if not unmatched_gt:
        return pairs_by_id

    # Иначе — позиционное выравнивание для оставшихся
    # Фильтруем pred: убираем задачи у которых duration == 0 (dummy start/finish)
    pred_real = [t for t in pred_tasks if get_duration(t) not in ("0", "None", None)
                 or get_dep_ids(t) or t.get("dependencies")]

    # Если после фильтрации стало меньше чем GT — берём всех pred
    if len(pred_real) < len(unmatched_gt):
        pred_real = pred_tasks

    # Убираем уже сматченные pred задачи
    matched_pred_ids = {get_task_id(t) for _, t in pairs_by_id}
    pred_unmatched = [t for t in pred_real if get_task_id(t) not in matched_pred_ids]

    positional_pairs = []
    for i, gt_task in enumerate(unmatched_gt):
        pred_task = pred_unmatched[i] if i < len(pred_unmatched) else {}
        positional_pairs.append((gt_task, pred_task))

    return pairs_by_id + positional_pairs


def build_id_map(pairs: list[tuple[dict, dict]]) -> dict:
    """
    Строит маппинг pred_id -> gt_id для нормализации зависимостей.
    Нужен когда GT использует "T2", а Pred использует "J2" для той же задачи,
    или когда dummy-старт называется "J1" в Pred и "T1" в GT.
    """
    id_map = {}
    for gt_task, pred_task in pairs:
        if pred_task:
            pred_id = get_task_id(pred_task)
            gt_id = get_task_id(gt_task)
            if pred_id and gt_id and pred_id != gt_id:
                id_map[pred_id] = gt_id

    # Выводим маппинг для dummy-старта по аналогии с первой реальной задачей.
    # Пример: если "J2" -> "T2", то dummy "J1" -> "T1"
    for pred_id, gt_id in list(id_map.items()):
        pred_prefix = ''.join(c for c in pred_id if not c.isdigit())
        gt_prefix = ''.join(c for c in gt_id if not c.isdigit())
        pred_num = ''.join(c for c in pred_id if c.isdigit())
        gt_num = ''.join(c for c in gt_id if c.isdigit())
        # Если номера совпадают — значит только префикс разный, строим маппинг для dummy
        if pred_num == gt_num and pred_prefix != gt_prefix:
            dummy_pred = f"{pred_prefix}1"
            dummy_gt = f"{gt_prefix}1"
            if dummy_pred not in id_map:
                id_map[dummy_pred] = dummy_gt
            break

    return id_map


def evaluate_run(gt_data: dict, pred_data: dict) -> dict:
    """
    Сравнивает предсказание с эталоном.
    Возвращает словарь с метриками для одного прогона.
    """
    metrics = {
        "duration_accuracy": 0.0,
        "dependencies_f1": 0.0
    }

    gt_tasks_list = gt_data.get("tasks", [])
    pred_tasks_list = pred_data.get("tasks", [])

    if not gt_tasks_list:
        return metrics

    # Выравниваем задачи GT и Pred для попарного сравнения
    pairs = align_tasks(gt_tasks_list, pred_tasks_list)

    # Строим маппинг id: pred_id -> gt_id (для нормализации зависимостей)
    id_map = build_id_map(pairs)

    correct_durations = 0
    tp_deps, fp_deps, fn_deps = 0, 0, 0

    for gt_task, pred_task in pairs:
        # --- Duration accuracy ---
        gt_dur = get_duration(gt_task)
        pred_dur = get_duration(pred_task) if pred_task else None

        if gt_dur is not None and gt_dur == pred_dur:
            correct_durations += 1

        # --- Dependencies F1 ---
        gt_deps = get_dep_ids(gt_task)

        raw_pred_deps = get_dep_ids(pred_task) if pred_task else set()
        # Нормализуем pred id к формату GT (например "2" -> "J2")
        pred_deps = normalize_dep_ids(raw_pred_deps, id_map)

        tp_deps += len(gt_deps & pred_deps)
        fp_deps += len(pred_deps - gt_deps)
        fn_deps += len(gt_deps - pred_deps)

    total = len(pairs)
    metrics["duration_accuracy"] = correct_durations / total if total > 0 else 0.0

    precision = tp_deps / (tp_deps + fp_deps) if (tp_deps + fp_deps) > 0 else 0.0
    recall = tp_deps / (tp_deps + fn_deps) if (tp_deps + fn_deps) > 0 else 0.0
    metrics["dependencies_f1"] = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return metrics

def run_experiments(source_dir: str, gt_dir: str, output_base: str, n_runs: int = 5):
    source = Path(source_dir)
    gt_path = Path(gt_dir)
    output = Path(output_base)

    files = [f for f in source.glob("*.*") if f.suffix.lower() in SYSTEM_PROMPTS]
    print(f"Найдено файлов для экспериментов: {len(files)}\n")

    # Флаг: debug_compare нужно вызвать только один раз
    debug_done = False

    for file in files:
        task_name = file.stem
        ext = file.suffix.lower()
        
        # Папка для результатов конкретной задачи
        task_out_dir = output / task_name
        task_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Проверяем, есть ли Ground Truth для этой задачи
        gt_file = gt_path / f"{task_name}.json"
        gt_data = None
        if gt_file.exists():
            with open(gt_file, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
        else:
            print(f"ВНИМАНИЕ: Нет эталона (GT) для {task_name}. Метрики не будут посчитаны.")

        print(f"--- Старт экспериментов: {task_name} ({n_runs} прогонов) ---")
        
        run_metrics = {"duration_accuracy": [], "dependencies_f1": []}
        
        content = file.read_text(encoding="utf-8")
        
        for run in range(1, n_runs + 1):
            run_file_path = task_out_dir / f"run_{run}.json"
            
            try:
                # Генерация (или загрузка, если уже сгенерировано ранее)
                if run_file_path.exists():
                    print(f"Прогон {run}: Загружен из кэша")
                    with open(run_file_path, "r", encoding="utf-8") as f:
                        pred_dict = json.load(f)
                else:
                    print(f"Прогон {run}: Генерация LLM...")
                    result = extract_schedule_data(content, ext)
                    pred_dict = json.loads(result.model_dump_json(exclude_none=True))
                    
                    with open(run_file_path, "w", encoding="utf-8") as f:
                        json.dump(pred_dict, f, indent=2, ensure_ascii=False)
                
                # --- DEBUG: один раз показываем структуры GT и Pred ---
                if gt_data and not debug_done:
                    debug_compare(gt_data, pred_dict)
                    debug_done = True

                # Оценка
                if gt_data:
                    metrics = evaluate_run(gt_data, pred_dict)
                    run_metrics["duration_accuracy"].append(metrics["duration_accuracy"])
                    run_metrics["dependencies_f1"].append(metrics["dependencies_f1"])
                    
            except Exception as e:
                print(f"Ошибка в прогоне {run}: {e}")

        # Агрегация и вывод результатов
        if gt_data and run_metrics["duration_accuracy"]:
            acc_mean = statistics.mean(run_metrics["duration_accuracy"])
            acc_std = statistics.stdev(run_metrics["duration_accuracy"]) if len(run_metrics["duration_accuracy"]) > 1 else 0.0
            
            f1_mean = statistics.mean(run_metrics["dependencies_f1"])
            f1_std = statistics.stdev(run_metrics["dependencies_f1"]) if len(run_metrics["dependencies_f1"]) > 1 else 0.0
            
            print(f"Результаты {task_name}:")
            print(f"Accuracy (Длительность): {acc_mean:.3f} ± {acc_std:.3f}")
            print(f"F1-Score (Связи): {f1_mean:.3f} ± {f1_std:.3f}\n")
            
            # Сохраняем сводку в файл
            with open(task_out_dir / "metrics_summary.txt", "w", encoding="utf-8") as f:
                f.write(f"Duration Accuracy: {acc_mean:.3f} ± {acc_std:.3f}\n")
                f.write(f"Dependencies F1:   {f1_mean:.3f} ± {f1_std:.3f}\n")

if __name__ == "__main__":
    SOURCE_DIR = "data/benchmark/1_raw_data"
    GT_DIR = "data/benchmark/2_ground_truth"
    OUTPUT_BASE = "data/benchmark/3_model_output"

    run_experiments(SOURCE_DIR, GT_DIR, OUTPUT_BASE, n_runs=5)