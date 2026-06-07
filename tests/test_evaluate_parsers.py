"""
tests/test_evaluate_parsers.py — юнит-тесты для evaluate_parsers.py.

Покрываются:
  - _f1: базовые случаи (идеальный, нулевой, частичный)
  - get_duration: single-mode, multi-mode, отсутствие modes
  - get_dep_ids: dict-зависимости, плоские id, пустой список
  - get_requirements: multi-mode объединение, нулевые quantity
  - get_resource_signature: формат строки
  - align_tasks: совпадение по id, позиционный фолбэк, смешанный случай
  - build_id_map: разные префиксы, dummy-расширение
  - evaluate_run: идеальное совпадение, пустой pred, частичные ошибки
"""

import pytest
from src.evaluate_parsers import (
    _f1, 
    get_duration,
    get_dep_ids,
    get_requirements,
    get_resource_signature,
    align_tasks,
    build_id_map,
    evaluate_run
)


# Helpers
def make_task(
    task_id: str,
    duration: int | None = None,
    deps: list[dict] | None = None,
    requirements: list[dict] | None = None,
    modes: list[dict] | None = None
) -> dict:
    """Строит минимальный словарь задачи в формате L0."""
    if modes is None and duration is not None:
        reqs = requirements or []
        modes = [{"mode_id": "M1", "duration": duration, "requirements": reqs}]
    return {
        "id": task_id,
        "dependencies": deps or [],
        "extensions": {"rcpsp": {"modes": modes or []}}
    }


def make_resource(res_id: str, capacity: int) -> dict:
    return {"id": res_id, "capacity": capacity}


def make_gt(tasks: list[dict], resources: list[dict] | None = None) -> dict:
    return {"tasks": tasks, "resources": resources or []}


# _f1
class TestF1:
    def test_perfect(self):
        assert _f1(tp=5, fp=0, fn=0) == pytest.approx(1.0)

    def test_zero_tp(self):
        assert _f1(tp=0, fp=3, fn=3) == pytest.approx(0.0)

    def test_all_zero(self):
        assert _f1(tp=0, fp=0, fn=0) == pytest.approx(0.0)

    def test_no_false_positives(self):
        # precision=1, recall=0.5 -> F1 = 0.667
        result = _f1(tp=2, fp=0, fn=2)
        assert result == pytest.approx(2 / 3, rel=1e-4)

    def test_no_false_negatives(self):
        # precision=0.5, recall=1 -> F1 = 0.667
        result = _f1(tp=2, fp=2, fn=0)
        assert result == pytest.approx(2 / 3, rel=1e-4)

    def test_partial(self):
        # precision=2/3, recall=2/4=0.5 -> F1 = 0.571
        result = _f1(tp=2, fp=1, fn=2)
        assert result == pytest.approx(2 * (2/3) * 0.5 / (2/3 + 0.5), rel=1e-4)


# get_duration
class TestGetDuration:
    def test_single_mode(self):
        task = make_task("T1", duration=5)
        assert get_duration(task) == "5"

    def test_multi_mode_returns_first(self):
        task = {
            "id": "T1",
            "dependencies": [],
            "extensions": {"rcpsp": {"modes": [
                {"mode_id": "M1", "duration": 3, "requirements": []},
                {"mode_id": "M2", "duration": 7, "requirements": []}
            ]}}
        }
        assert get_duration(task) == "3"

    def test_no_modes(self):
        task = {"id": "T1", "dependencies": [], "extensions": {"rcpsp": {}}}
        assert get_duration(task) is None

    def test_empty_modes_list(self):
        task = {"id": "T1", "dependencies": [], "extensions": {"rcpsp": {"modes": []}}}
        assert get_duration(task) is None

    def test_no_extensions(self):
        task = {"id": "T1", "dependencies": []}
        assert get_duration(task) is None

    def test_zero_duration(self):
        task = make_task("START", duration=0)
        assert get_duration(task) == "0"

# get_dep_ids
class TestGetDepIds:
    def test_dict_deps(self):
        task = make_task("T2", deps=[
            {"task_id": "START", "type": "FS"},
            {"task_id": "T3", "type": "FS"}
        ])
        assert get_dep_ids(task) == {"START", "T3"}

    def test_flat_string_deps(self):
        task = {"id": "T2", "dependencies": ["T1", "T3"]}
        assert get_dep_ids(task) == {"T1", "T3"}

    def test_empty_deps(self):
        task = make_task("T1", deps=[])
        assert get_dep_ids(task) == set()

    def test_missing_deps_field(self):
        task = {"id": "T1"}
        assert get_dep_ids(task) == set()

    def test_dict_with_predecessor_id_key(self):
        task = {"id": "T2", "dependencies": [{"predecessor_id": "T1", "type": "FS"}]}
        assert get_dep_ids(task) == {"T1"}


# get_requirements
class TestGetRequirements:
    def test_single_mode(self):
        task = make_task("T1", duration=3, requirements=[
            {"resource_id": "R1", "quantity": 2},
            {"resource_id": "R2", "quantity": 0} # должен быть включён
        ])
        result = get_requirements(task)
        assert "R1:2" in result
        assert "R2:0" in result

    def test_multi_mode_union(self):
        task = {
            "id": "T1",
            "dependencies": [],
            "extensions": {"rcpsp": {"modes": [
                {"mode_id": "M1", "duration": 3, "requirements": [{"resource_id": "R1", "quantity": 2}]},
                {"mode_id": "M2", "duration": 5, "requirements": [{"resource_id": "R2", "quantity": 1}]}
            ]}}
        }
        result = get_requirements(task)
        assert result == {"R1:2", "R2:1"}

    def test_no_requirements(self):
        task = make_task("START", duration=0, requirements=[])
        assert get_requirements(task) == set()

    def test_modes_as_dict(self):
        # modes могут прийти как словарь
        task = {
            "id": "T1",
            "dependencies": [],
            "extensions": {"rcpsp": {"modes": {
                "M1": {"mode_id": "M1", "duration": 2, "requirements": [{"resource_id": "R1", "quantity": 3}]}
            }}}
        }
        result = get_requirements(task)
        assert "R1:3" in result

# get_resource_signature
class TestGetResourceSignature:
    def test_basic(self):
        res = make_resource("R1", 9)
        assert get_resource_signature(res) == "R1:9"

    def test_zero_capacity(self):
        res = make_resource("R1", 0)
        assert get_resource_signature(res) == "R1:0"

    def test_non_renewable(self):
        res = make_resource("N1", 100)
        assert get_resource_signature(res) == "N1:100"

# align_tasks
class TestAlignTasks:
    def test_exact_id_match(self):
        gt = [make_task("T2", 3), make_task("T3", 5)]
        pred = [make_task("T3", 5), make_task("T2", 4)]
        pairs = align_tasks(gt, pred)
        # T2 -> T2, T3 -> T3 (независимо от порядка в pred)
        by_gt_id = {get_id(g): get_id(p) for g, p in pairs}
        assert by_gt_id["T2"] == "T2"
        assert by_gt_id["T3"] == "T3"

    def test_positional_fallback(self):
        gt = [make_task("T2", 3), make_task("T3", 5)]
        pred = [make_task("J2", 3), make_task("J3", 5)]
        pairs = align_tasks(gt, pred)
        assert len(pairs) == 2
        # позиционно: GT[0] ↔ Pred[0], GT[1] ↔ Pred[1]
        assert pairs[0][0]["id"] == "T2"
        assert pairs[0][1]["id"] == "J2"

    def test_dummy_filtered_in_positional(self):
        # pred содержит dummy (duration=0), он должен быть отфильтрован при позиционном выравнивании
        gt = [make_task("T2", 3)]
        pred = [make_task("DUMMY", 0), make_task("J2", 3)]
        pairs = align_tasks(gt, pred)
        assert len(pairs) == 1
        assert pairs[0][1]["id"] == "J2"

    def test_mixed_some_id_match(self):
        gt = [make_task("T2", 3), make_task("T3", 5)]
        pred = [make_task("T2", 3), make_task("J3", 5)]
        pairs = align_tasks(gt, pred)
        # T2 совпадает по id, T3 выравнивается позиционно на J3
        by_gt_id = {get_id(g): get_id(p) for g, p in pairs}
        assert by_gt_id["T2"] == "T2"
        assert by_gt_id["T3"] == "J3"

def get_id(task: dict) -> str:
    return task.get("id", "")

# build_id_map
class TestBuildIdMap:
    def test_different_prefix(self):
        gt = [make_task("T2", 3), make_task("T3", 5)]
        pred = [make_task("J2", 3), make_task("J3", 5)]
        pairs = list(zip(gt, pred))
        id_map = build_id_map(pairs)
        assert id_map.get("J2") == "T2"
        assert id_map.get("J3") == "T3"

    def test_same_ids_no_map(self):
        gt = [make_task("T2", 3)]
        pred = [make_task("T2", 3)]
        pairs = list(zip(gt, pred))
        id_map = build_id_map(pairs)
        assert "T2" not in id_map  # если id одинаковые то mapping не нужен

    def test_empty_pred(self):
        gt = [make_task("T2", 3)]
        pairs = [(gt[0], {})]
        id_map = build_id_map(pairs)
        assert id_map == {}

# интеграционные тесты
class TestEvaluateRun:
    def _perfect_pred(self) -> tuple[dict, dict]:
        """GT и pred полностью совпадают."""
        tasks = [
            make_task("START", 0),
            make_task("T2", 3, deps=[{"task_id": "START", "type": "FS"}],
                      requirements=[{"resource_id": "R1", "quantity": 2}]),
            make_task("END", 0, deps=[{"task_id": "T2", "type": "FS"}])
        ]
        resources = [make_resource("R1", 9), make_resource("R2", 4)]
        gt = make_gt(tasks, resources)
        import copy
        pred = copy.deepcopy(gt)
        return gt, pred

    def test_perfect_scores(self):
        gt, pred = self._perfect_pred()
        metrics = evaluate_run(gt, pred)
        assert metrics["duration_accuracy"] == pytest.approx(1.0)
        assert metrics["dependencies_f1"] == pytest.approx(1.0)
        assert metrics["resources_f1"] == pytest.approx(1.0)
        assert metrics["requirements_f1"] == pytest.approx(1.0)

    def test_empty_pred_tasks(self):
        tasks = [make_task("T2", 3)]
        resources = [make_resource("R1", 9)]
        gt = make_gt(tasks, resources)
        pred = make_gt([], [])
        metrics = evaluate_run(gt, pred)
        assert metrics["resources_f1"] == pytest.approx(0.0)

    def test_wrong_duration(self):
        tasks_gt = [make_task("T2", 5)]
        tasks_pred = [make_task("T2", 9)] # неправильная длительность
        gt = make_gt(tasks_gt, [])
        pred = make_gt(tasks_pred, [])
        metrics = evaluate_run(gt, pred)
        assert metrics["duration_accuracy"] == pytest.approx(0.0)

    def test_missing_dependency(self):
        tasks_gt = [
            make_task("START", 0),
            make_task("T2", 3, deps=[{"task_id": "START", "type": "FS"}]),
        ]
        tasks_pred = [
            make_task("START", 0),
            make_task("T2", 3, deps=[]) # зависимость потеряна
        ]
        gt = make_gt(tasks_gt)
        pred = make_gt(tasks_pred)
        metrics = evaluate_run(gt, pred)
        assert metrics["dependencies_f1"] < 1.0

    def test_wrong_resource_capacity(self):
        gt = make_gt([], [make_resource("R1", 9)])
        pred = make_gt([], [make_resource("R1", 1)]) # неправильный capacity
        metrics = evaluate_run(gt, pred)
        assert metrics["resources_f1"] == pytest.approx(0.0)

    def test_correct_resource(self):
        gt = make_gt([], [make_resource("R1", 9)])
        pred = make_gt([], [make_resource("R1", 9)])
        metrics = evaluate_run(gt, pred)
        assert metrics["resources_f1"] == pytest.approx(1.0)

    def test_empty_requirements(self):
        tasks_gt = [make_task("T2", 3, requirements=[{"resource_id": "R1", "quantity": 2}])]
        tasks_pred = [make_task("T2", 3, requirements=[])] # требования не извлечены
        gt = make_gt(tasks_gt)
        pred = make_gt(tasks_pred)
        metrics = evaluate_run(gt, pred)
        assert metrics["requirements_f1"] == pytest.approx(0.0)

    def test_extra_requirement_in_pred(self):
        tasks_gt = [make_task("T2", 3, requirements=[{"resource_id": "R1", "quantity": 2}])]
        tasks_pred = [make_task("T2", 3, requirements=[
            {"resource_id": "R1", "quantity": 2},
            {"resource_id": "R2", "quantity": 1} # лишнее
        ])]
        gt = make_gt(tasks_gt)
        pred = make_gt(tasks_pred)
        metrics = evaluate_run(gt, pred)
        # TP=1, FP=1, FN=0 -> precision=0.5, recall=1 -> F1=0.667
        assert metrics["requirements_f1"] == pytest.approx(2/3, rel=1e-4)

    def test_metrics_keys_present(self):
        gt = make_gt([make_task("T2", 3)], [make_resource("R1", 9)])
        pred = make_gt([make_task("T2", 3)], [make_resource("R1", 9)])
        metrics = evaluate_run(gt, pred)
        assert set(metrics.keys()) == {
            "duration_accuracy",
            "dependencies_f1",
            "resources_f1",
            "requirements_f1"
        }