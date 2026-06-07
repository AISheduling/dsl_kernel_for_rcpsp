"""
evaluate_parsers.py — тестирование детерминированных парсеров форматов RCPSP.

Аналог run_experiments из LLM-пайплайна, но вместо LLM вызываются
parse_sm / parse_mm / parse_rcp / parse_msrcp.

Метрики (по аналогии с evaluate_run):
  • duration_accuracy — доля задач с правильно извлечённой длительностью
  • dependencies_f1 — F1 по множеству предшественников (precision × recall)
  • resources_f1 — F1 по множеству ресурсов {id, capacity}
  • requirements_f1 — F1 по требованиям ресурсов внутри задач/режимов

Запуск:
    python evaluate_parsers.py
    python evaluate_parsers.py --source data/raw --gt data/gt --output data/results --runs 1
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from src.dsl_schema import SchedulingProblem

# Импорт парсеров
from src.parsers import parse_sm, parse_mm, parse_rcp, parse_msrcp

# Реестр парсеров по расширению
PARSERS: dict[str, callable] = {
    ".sm": parse_sm,
    ".mm": parse_mm,
    ".rcp": parse_rcp,
    ".msrcp": parse_msrcp,
}

# Утилиты извлечения полей (аналог get_task_id / get_duration / get_dep_ids)

def get_task_id(task: dict) -> str:
    return str(task.get("id") or task.get("task_id") or "")

def get_duration(task: dict) -> str | None:
    rcpsp = task.get("extensions", {}).get("rcpsp", {})
    modes = rcpsp.get("modes", [])
    if isinstance(modes, list) and modes:
        dur = modes[0].get("duration")
    elif isinstance(modes, dict) and modes:
        dur = next(iter(modes.values()), {}).get("duration")
    else:
        return None
    return str(dur) if dur is not None else None

def get_dep_ids(task: dict) -> set[str]:
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


def normalize_dep_ids(dep_ids: set[str], id_map: dict[str, str]) -> set[str]:
    return {id_map.get(d, d) for d in dep_ids}


def get_resource_signature(res: dict) -> str:
    """Строка-ключ ресурса: 'R1:9' (id:capacity)."""
    return f"{res.get('id', '')}:{res.get('capacity', '')}"


def get_requirements(task: dict) -> set[str]:
    """
    Множество строк вида 'R1:3' из всех режимов задачи.
    Для multi-mode берём объединение по всем режимам.
    """
    rcpsp = task.get("extensions", {}).get("rcpsp", {})
    modes = rcpsp.get("modes", [])
    if isinstance(modes, dict):
        modes = list(modes.values())
    result = set()
    for mode in (modes or []):
        for req in mode.get("requirements", []):
            result.add(f"{req.get('resource_id', '')}:{req.get('quantity', '')}")
    return result


# Выравнивание задач GT ↔ Pred

def align_tasks(gt_tasks: list[dict], pred_tasks: list[dict]) -> list[tuple[dict, dict]]:
    """
    Сопоставляет задачи GT и Pred:
    1. По id напрямую.
    2. По позиции (после фильтрации dummy-задач с duration=0 из Pred).
    """
    pred_by_id = {get_task_id(t): t for t in pred_tasks}
    pairs: list[tuple[dict, dict]] = []
    unmatched_gt: list[dict] = []

    for gt_task in gt_tasks:
        gt_id = get_task_id(gt_task)
        if gt_id in pred_by_id:
            pairs.append((gt_task, pred_by_id[gt_id]))
        else:
            unmatched_gt.append(gt_task)

    if not unmatched_gt:
        return pairs

    # Позиционное выравнивание для несовпавших
    matched_pred_ids = {get_task_id(t) for _, t in pairs}
    pred_real = [
        t for t in pred_tasks
        if get_task_id(t) not in matched_pred_ids
        and (get_duration(t) not in ("0", None) or get_dep_ids(t))
    ]
    if len(pred_real) < len(unmatched_gt):
        pred_real = [t for t in pred_tasks if get_task_id(t) not in matched_pred_ids]

    for i, gt_task in enumerate(unmatched_gt):
        pred_task = pred_real[i] if i < len(pred_real) else {}
        pairs.append((gt_task, pred_task))

    return pairs


def build_id_map(pairs: list[tuple[dict, dict]]) -> dict[str, str]:
    """
    Строит маппинг pred_id → gt_id для нормализации зависимостей
    (например "T2" → "J2" если парсер использует другой префикс).
    """
    id_map: dict[str, str] = {}
    for gt_task, pred_task in pairs:
        if not pred_task:
            continue
        pred_id = get_task_id(pred_task)
        gt_id = get_task_id(gt_task)
        if pred_id and gt_id and pred_id != gt_id:
            id_map[pred_id] = gt_id

    # Распространяем маппинг на dummy-старт по аналогии с первой реальной задачей
    for pred_id, gt_id in list(id_map.items()):
        pred_prefix = "".join(c for c in pred_id if not c.isdigit())
        gt_prefix = "".join(c for c in gt_id if not c.isdigit())
        pred_num = "".join(c for c in pred_id if c.isdigit())
        gt_num = "".join(c for c in gt_id if c.isdigit())
        if pred_num == gt_num and pred_prefix != gt_prefix:
            dummy_pred = f"{pred_prefix}1"
            dummy_gt = f"{gt_prefix}1"
            if dummy_pred not in id_map:
                id_map[dummy_pred] = dummy_gt
            break

    return id_map


# Отладочный вывод

def debug_compare(gt_data: dict, pred_data: dict) -> None:
    print("DEBUG: Сравнение структур GT vs Parser Output")

    gt_tasks = gt_data.get("tasks", [])
    pred_tasks = pred_data.get("tasks", [])
    print(f"  Задач в GT: {len(gt_tasks)}")
    print(f"  Задач в Pred: {len(pred_tasks)}")

    if gt_tasks:
        print(f"\n  GT tasks[0]:\n  " + json.dumps(gt_tasks[0], indent=2, ensure_ascii=False)[:500])
    if pred_tasks:
        print(f"\n  Pred tasks[0]:\n  " + json.dumps(pred_tasks[0], indent=2, ensure_ascii=False)[:500])

    gt_res = gt_data.get("resources", [])
    pred_res = pred_data.get("resources", [])
    print(f"\n  Ресурсов в GT:   {len(gt_res)}")
    print(f"  Ресурсов в Pred: {len(pred_res)}")


# Вычисление метрик для одного прогона

def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def evaluate_run(gt_data: dict, pred_data: dict) -> dict:
    metrics = {
        "duration_accuracy": 0.0,
        "dependencies_f1": 0.0,
        "resources_f1": 0.0,
        "requirements_f1": 0.0,
    }

    # Resources F1
    gt_res_set  = {get_resource_signature(r) for r in gt_data.get("resources", [])}
    pred_res_set = {get_resource_signature(r) for r in pred_data.get("resources", [])}

    print(f"  GT res:   {sorted(gt_res_set)}")
    print(f"  Pred res: {sorted(pred_res_set)}")
    
    tp_r = len(gt_res_set & pred_res_set)
    fp_r = len(pred_res_set - gt_res_set)
    fn_r = len(gt_res_set - pred_res_set)
    metrics["resources_f1"] = _f1(tp_r, fp_r, fn_r)

    # Tasks
    gt_tasks_list = gt_data.get("tasks", [])
    pred_tasks_list = pred_data.get("tasks", [])

    if not gt_tasks_list:
        return metrics

    pairs = align_tasks(gt_tasks_list, pred_tasks_list)
    id_map = build_id_map(pairs)

    correct_durations = 0
    tp_deps, fp_deps, fn_deps = 0, 0, 0
    tp_reqs, fp_reqs, fn_reqs = 0, 0, 0

    for gt_task, pred_task in pairs:
        # Duration accuracy
        gt_dur = get_duration(gt_task)
        pred_dur = get_duration(pred_task) if pred_task else None
        if gt_dur is not None and gt_dur == pred_dur:
            correct_durations += 1

        # Dependencies F1
        gt_deps = get_dep_ids(gt_task)
        pred_deps = normalize_dep_ids(get_dep_ids(pred_task) if pred_task else set(), id_map)
        tp_deps += len(gt_deps & pred_deps)
        fp_deps += len(pred_deps - gt_deps)
        fn_deps += len(gt_deps - pred_deps)

        # Requirements F1 (по ресурсным требованиям задачи)
        gt_reqs = get_requirements(gt_task)
        pred_reqs = get_requirements(pred_task) if pred_task else set()
        tp_reqs += len(gt_reqs & pred_reqs)
        fp_reqs += len(pred_reqs - gt_reqs)
        fn_reqs += len(gt_reqs - pred_reqs)

    total = len(pairs)
    metrics["duration_accuracy"] = correct_durations / total if total > 0 else 0.0
    metrics["dependencies_f1"] = _f1(tp_deps, fp_deps, fn_deps)
    metrics["requirements_f1"] = _f1(tp_reqs, fp_reqs, fn_reqs)

    return metrics


# Основной цикл экспериментов

def run_experiments(source_dir: str, gt_dir: str, output_base: str, n_runs: int = 1) -> None:
    """
    Прогоняет парсеры по всем файлам из source_dir, сравнивает с GT из gt_dir.

    n_runs > 1 имеет смысл если парсер недетерминирован. Для детерминированных
    парсеров достаточно n_runs=1, но повторные прогоны просто дочитают кэш.
    """
    source  = Path(source_dir)
    gt_path = Path(gt_dir)
    output  = Path(output_base)

    files = [f for f in source.glob("*.*") if f.suffix.lower() in PARSERS]
    print(f"Найдено файлов для экспериментов: {len(files)}\n")

    debug_done = False
    all_metrics: dict[str, dict] = {}  # task_name → средние метрики

    for file in sorted(files):
        task_name = file.stem
        ext = file.suffix.lower()

        task_out_dir = output / task_name
        task_out_dir.mkdir(parents=True, exist_ok=True)

        # Ground Truth
        gt_file = gt_path / f"{task_name}.json"
        gt_data = None
        if gt_file.exists():
            with open(gt_file, encoding="utf-8") as f:
                gt_data = json.load(f)
        else:
            print(f"ВНИМАНИЕ: Нет GT для {task_name}. Метрики не будут посчитаны.")

        print(f"--- Старт: {task_name} ({n_runs} прогон(ов)) ---")

        run_metrics: dict[str, list[float]] = {
            "duration_accuracy": [],
            "dependencies_f1": [],
            "resources_f1": [],
            "requirements_f1": [],
        }

        for run in range(1, n_runs + 1):
            run_file = task_out_dir / f"run_{run}.json"

            try:
                # Детерминированный парсер: кэшируем результат первого прогона
                if run_file.exists():
                    print(f"  Прогон {run}: загружен из кэша")
                    with open(run_file, encoding="utf-8") as f:
                        pred_dict = json.load(f)
                else:
                    print(f"  Прогон {run}: запуск парсера...")
                    parser = PARSERS[ext]
                    problem: SchedulingProblem = parser(str(file))

                    # Валидация через Pydantic (бросит ошибку если схема нарушена)
                    pred_dict = json.loads(
                        problem.model_dump_json(exclude_none=True)
                    )
                    with open(run_file, "w", encoding="utf-8") as f:
                        json.dump(pred_dict, f, indent=2, ensure_ascii=False)
                    print(f"  Прогон {run}: сохранён → {run_file}")

                # Debug-вывод один раз за весь запуск
                if gt_data and not debug_done:
                    debug_compare(gt_data, pred_dict)
                    debug_done = True

                # Метрики
                if gt_data:
                    m = evaluate_run(gt_data, pred_dict)
                    for key, val in m.items():
                        run_metrics[key].append(val)

            except Exception as exc:
                print(f"  ОШИБКА в прогоне {run}: {exc}")

        # Агрегация
        if gt_data and run_metrics["duration_accuracy"]:
            def fmt(vals: list[float]) -> str:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                return f"{mean:.3f} ± {std:.3f}"

            summary_lines = [
                f"Duration Accuracy : {fmt(run_metrics['duration_accuracy'])}",
                f"Dependencies F1: {fmt(run_metrics['dependencies_f1'])}",
                f"Resources F1: {fmt(run_metrics['resources_f1'])}",
                f"Requirements F1: {fmt(run_metrics['requirements_f1'])}",
            ]
            print(f"Результаты {task_name}:")
            for line in summary_lines:
                print(f"  {line}")
            print()

            summary_path = task_out_dir / "metrics_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines) + "\n")

            all_metrics[task_name] = {
                k: statistics.mean(v) for k, v in run_metrics.items()
            }

    # Итоговая сводка по всем файлам
    if all_metrics:
        print("ИТОГО по всем файлам:")
        for metric in ("duration_accuracy", "dependencies_f1", "resources_f1", "requirements_f1"):
            vals = [m[metric] for m in all_metrics.values()]
            print(f"  {metric:25s}: {statistics.mean(vals):.3f}")

        global_path = output / "global_metrics.json"
        with open(global_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"Полная сводка сохранена → {global_path}\n")


# CLI

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Тестирование детерминированных RCPSP-парсеров."
    )
    parser.add_argument(
        "--source", default="data/benchmark/1_raw_data",
        help="Папка с исходными файлами (.sm/.mm/.rcp/.msrcp)"
    )
    parser.add_argument(
        "--gt", default="data/benchmark/2_ground_truth",
        help="Папка с Ground Truth файлами (.json)"
    )
    parser.add_argument(
        "--output", default="data/benchmark/3_parser_output",
        help="Папка для сохранения результатов парсеров и метрик"
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Количество прогонов (для детерминированных парсеров = 1)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiments(
        source_dir=args.source,
        gt_dir=args.gt,
        output_base=args.output,
        n_runs=args.runs,
    )