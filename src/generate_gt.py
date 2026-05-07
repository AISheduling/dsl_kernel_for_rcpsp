"""
generate_gt.py — генерация Ground Truth файлов из сырых данных.

Запуск:
    python generate_gt.py
    python generate_gt.py --source data/raw --output data/gt
"""

import argparse
import json
from pathlib import Path

from parsers import parse_sm, parse_mm, parse_rcp, parse_msrcp

PARSERS = {
    ".sm": parse_sm,
    ".mm": parse_mm,
    ".rcp": parse_rcp,
    ".msrcp": parse_msrcp,
}


def generate_gt(source_dir: str, output_dir: str) -> None:
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    files = [f for f in source.glob("*.*") if f.suffix.lower() in PARSERS]
    print(f"Найдено файлов: {len(files)}")

    ok, failed = 0, 0

    for file in sorted(files):
        out_file = output / f"{file.stem}.json"
        try:
            parser = PARSERS[file.suffix.lower()]
            problem = parser(str(file))
            out_file.write_text(
                problem.model_dump_json(indent=2, exclude_none=True),
                encoding="utf-8"
            )
            print(f"  OK  {file.name} → {out_file.name}")
            ok += 1
        except Exception as e:
            print(f"  ERR {file.name}: {e}")
            failed += 1

    print(f"\nГотово: {ok} успешно, {failed} ошибок.")
    print(f"GT сохранён в: {output.resolve()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Генерация Ground Truth из RCPSP-файлов.")
    p.add_argument("--source", default="data/benchmark/1_raw_data")
    p.add_argument("--output", default="data/benchmark/2_ground_truth")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_gt(args.source, args.output)