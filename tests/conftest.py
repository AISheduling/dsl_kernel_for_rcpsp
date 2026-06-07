import sys
from pathlib import Path

# Добавляем src/ в sys.path чтобы тесты могли импортировать модули проекта
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))