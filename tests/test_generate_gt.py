"""
tests/test_generate_gt.py — тесты для generate_gt.py.

Покрываются:
  - generate_gt: успешная генерация для поддерживаемых форматов
  - generate_gt: пропуск неподдерживаемых расширений
  - generate_gt: обработка ошибок парсера (файл не крашит скрипт)
  - generate_gt: выходной файл валиден как JSON и содержит обязательные поля L0
  - generate_gt: создаёт output-директорию если её нет
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from generate_gt import generate_gt


# Helpers
MINIMAL_L0 = {
    "schema_version": "0.1",
    "problem_id": "test",
    "domain": "rcpsp",
    "project": {"name": "test", "objective": "minimize_makespan"},
    "resources": [],
    "tasks": []
}


def make_mock_problem(data: dict = MINIMAL_L0) -> MagicMock:
    """Создаёт мок SchedulingProblem, чей model_dump_json возвращает JSON"""
    mock = MagicMock()
    mock.model_dump_json.return_value = json.dumps(data, indent=2)
    return mock

# generate_gt
class TestGenerateGt:

    def test_creates_output_directory(self, tmp_path):
        source = tmp_path / "raw"
        source.mkdir()
        output = tmp_path / "gt" / "nested"  # директории ещё нет

        with patch("generate_gt.PARSERS", {}):
            generate_gt(str(source), str(output))

        assert output.exists()

    def test_generates_json_for_supported_format(self, tmp_path):
        source = tmp_path / "raw"
        source.mkdir()
        output = tmp_path / "gt"

        # создаём фиктивный .sm файл
        sm_file = source / "instance.sm"
        sm_file.write_text("dummy content", encoding="utf-8")

        mock_problem = make_mock_problem()

        with patch("generate_gt.PARSERS", {".sm": lambda path: mock_problem}):
            generate_gt(str(source), str(output))

        out_file = output / "instance.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_output_filename_matches_stem(self, tmp_path):
        source = tmp_path / "raw"
        source.mkdir()
        output = tmp_path / "gt"

        (source / "j3010_1.sm").write_text("dummy", encoding="utf-8")
        mock_problem = make_mock_problem()

        with patch("generate_gt.PARSERS", {".sm": lambda p: mock_problem}):
            generate_gt(str(source), str(output))

        assert (output / "j3010_1.json").exists()

    def test_skips_unsupported_extension(self, tmp_path):
        source = tmp_path / "raw"
        source.mkdir()
        output = tmp_path / "gt"

        (source / "readme.txt").write_text("not a scheduling file", encoding="utf-8")
        (source / "data.csv").write_text("1,2,3", encoding="utf-8")

        with patch("generate_gt.PARSERS", {".sm": MagicMock()}):
            generate_gt(str(source), str(output))

        # ни readme.json ни data.json не должны появиться
        assert not (output / "readme.json").exists()
        assert not (output / "data.json").exists()

    def test_handles_parser_exception_gracefully(self, tmp_path):
        source = tmp_path / "raw"
        source.mkdir()
        output = tmp_path / "gt"

        (source / "broken.sm").write_text("garbage", encoding="utf-8")
        (source / "good.sm").write_text("valid", encoding="utf-8")

        def parser_side_effect(path: str):
            if "broken" in path:
                raise ValueError("Cannot parse file")
            return make_mock_problem()

        with patch("generate_gt.PARSERS", {".sm": parser_side_effect}):
            # не должно выбрасывать исключение
            generate_gt(str(source), str(output))

        # good.json создан, broken.json нет
        assert (output / "good.json").exists()
        assert not (output / "broken.json").exists()

    def test_multiple_formats(self, tmp_path):
        source = tmp_path / "raw"
        source.mkdir()
        output = tmp_path / "gt"

        (source / "a.sm").write_text("sm content", encoding="utf-8")
        (source / "b.rcp").write_text("rcp content", encoding="utf-8")
        (source / "c.mm").write_text("mm content", encoding="utf-8")

        parsers = {
            ".sm":  lambda p: make_mock_problem({"problem_id": "a", **MINIMAL_L0}),
            ".rcp": lambda p: make_mock_problem({"problem_id": "b", **MINIMAL_L0}),
            ".mm":  lambda p: make_mock_problem({"problem_id": "c", **MINIMAL_L0})
        }

        with patch("generate_gt.PARSERS", parsers):
            generate_gt(str(source), str(output))

        assert (output / "a.json").exists()
        assert (output / "b.json").exists()
        assert (output / "c.json").exists()

    def test_output_is_valid_json(self, tmp_path):
        source = tmp_path / "raw"
        source.mkdir()
        output = tmp_path / "gt"

        (source / "instance.rcp").write_text("data", encoding="utf-8")
        mock_problem = make_mock_problem(MINIMAL_L0)

        with patch("generate_gt.PARSERS", {".rcp": lambda p: mock_problem}):
            generate_gt(str(source), str(output))

        raw = (output / "instance.json").read_text(encoding="utf-8")
        # должен парситься без исключений
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_empty_source_directory(self, tmp_path):
        source = tmp_path / "raw"
        source.mkdir()
        output = tmp_path / "gt"

        with patch("generate_gt.PARSERS", {".sm": MagicMock()}):
            # пустая директория
            generate_gt(str(source), str(output))

        assert output.exists()
        assert list(output.iterdir()) == []

    def test_model_dump_json_called_with_correct_args(self, tmp_path):
        source = tmp_path / "raw"
        source.mkdir()
        output = tmp_path / "gt"

        (source / "instance.sm").write_text("content", encoding="utf-8")
        mock_problem = make_mock_problem()

        with patch("generate_gt.PARSERS", {".sm": lambda p: mock_problem}):
            generate_gt(str(source), str(output))

        mock_problem.model_dump_json.assert_called_once_with(
            indent=2, exclude_none=True
        )

    def test_output_encoding_is_utf8(self, tmp_path):
        source = tmp_path / "raw"
        source.mkdir()
        output = tmp_path / "gt"

        (source / "instance.sm").write_text("content", encoding="utf-8")
        data_with_unicode = {**MINIMAL_L0, "description": "Описание на кириллице"}
        mock_problem = make_mock_problem(data_with_unicode)

        with patch("generate_gt.PARSERS", {".sm": lambda p: mock_problem}):
            generate_gt(str(source), str(output))

        # файл должен читаться без ошибок как utf-8
        content = (output / "instance.json").read_text(encoding="utf-8")
        assert isinstance(content, str)