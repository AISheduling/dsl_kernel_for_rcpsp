import os
import json
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

# ПРОМПТЫ ДЛЯ РАЗНЫХ ФОРМАТОВ

PROMPT_SM = """
Ты — эксперт по анализу данных. Твоя задача — прочитать сырой текстовый файл в формате PSPLIB (.sm) 
и преобразовать его в строгую JSON-структуру согласно предоставленной Pydantic-схеме.

ПРАВИЛА МАППИНГА:
1. domain: строго "rcpsp".
2. project.name: возьми из строки 'projects :' или придумай на основе 'file with basedata'.
3. resources: Найди блок 'RESOURCEAVAILABILITIES'. Ресурсы (R1, R2...) запиши с их capacity. Укажи type: "renewable".
4. tasks: Найди блок 'PRECEDENCE RELATIONS'. 'jobnr.' - это task_id. Поле 'successors' показывает преемников. Инвертируй логику: укажи для каждой задачи её предшественников (DEPENDENCIES). type всегда "FS".
5. modes и durations: Найди блок 'REQUESTS/DURATIONS'. Запиши длительность и потребление ресурсов (где > 0) в TaskExt -> rcpsp -> modes (с mode_id='1').
"""

PROMPT_MM = """
Ты — эксперт по анализу данных. Твоя задача — прочитать сырой текстовый файл в формате MRCPSP (.mm) 
и преобразовать его в строгую JSON-структуру согласно предоставленной Pydantic-схеме.

ПРАВИЛА МАППИНГА:
1. domain: строго "rcpsp".
2. project.name: возьми из строки 'projects :' или 'file with basedata'.
3. resources: Найди блок 'RESOURCEAVAILABILITIES'. Ресурсы могут быть возобновляемыми (R) и невозобновляемыми (N). Для R укажи type: "renewable", для N — "nonrenewable".
4. tasks: Найди блок 'PRECEDENCE RELATIONS'. 'jobnr.' - это task_id. Инвертируй 'successors' в предшественников (DEPENDENCIES). type всегда "FS".
5. modes: Найди блок 'REQUESTS/DURATIONS'. У задач может быть несколько режимов (mode). Для каждого mode_id запиши длительность и потребление ресурсов (R и N, где значение > 0).
"""

PROMPT_RCP = """
Ты — эксперт по анализу данных. Твоя задача — прочитать сырой текстовый файл в формате Patterson (.rcp) 
и преобразовать его в строгую JSON-структуру согласно предоставленной Pydantic-схеме.

ПРАВИЛА МАППИНГА:
1. domain: строго "rcpsp".
2. project.name: сгенерируй на основе данных, так как в файле имени нет.
3. resources: Вторая строка содержит capacity для ресурсов. Создай ресурсы R1, R2 и т.д., type: "renewable".
4. tasks и durations: Начиная с 3-й строки идут описания задач по порядку (1-я строка данных = task_id '1').
   Формат строки: [длительность] [потребление R1] [потребление R2] ... [кол-во преемников] [преемник 1] [преемник 2] ...
   ИНВЕРТИРУЙ ПРЕЕМНИКОВ В ПРЕДШЕСТВЕННИКОВ (DEPENDENCIES)! Если задача 1 имеет преемника 2, то у задачи 2 должен быть предшественник 1.
5. modes: Длительность и потребление ресурсов (> 0) сохрани в TaskExt -> rcpsp -> modes (mode_id='1').
"""

PROMPT_MSRCP = """
Ты — эксперт по анализу данных. Твоя задача — прочитать сырой текстовый файл в формате MSRCPSP (.msrcp) 
и преобразовать его в строгую JSON-структуру согласно предоставленной Pydantic-схеме.

ПРАВИЛА МАППИНГА:
1. domain: строго "rcpsp".
2. project.name: сгенерируй на основе имени файла, так как в файле имени нет.
3. resources: Блок '* Workforce Module with Skill Levels *' содержит матрицу NxM,
   где N — количество работников, M — количество навыков. Каждый навык — отдельный ресурс (SKILL_1, SKILL_2 и т.д.).
   capacity каждого навыка = количество работников, у которых он ненулевой. type: "renewable".
4. tasks: Первая строка после '* Project Module *' содержит: [кол-во задач] [кол-во работников] [кол-во навыков] [кол-во уровней].
   Далее идут строки задач в формате: [длительность] [кол-во преемников] [преемник 1] [преемник 2] ...
   ИНВЕРТИРУЙ ПРЕЕМНИКОВ В ПРЕДШЕСТВЕННИКОВ (DEPENDENCIES)! type всегда "FS".
5. modes: У каждой задачи один режим (mode_id='1'). Длительность — первое поле строки задачи.
6. requirements: Блок '* Skill Requirements Module *' содержит по одной строке на задачу.
   Каждое значение > 0 — это количество единиц соответствующего навыка (resource_id: SKILL_N, quantity: значение).
7. min_skill_levels: Блок '* Skill Level Requirements Module *' содержит минимальные уровни квалификации.
   Задачи разделены маркером '-1'. Для каждой задачи запиши в extensions режима поле
   'min_skill_levels': {SKILL_N: уровень} только для навыков с требованием > 0.
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

    response = client.chat.completions.create(
        model="iairlab/qwen2.5-72b",
        response_model=SchedulingProblem,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.0,
    )
    return response

def run_extraction(source_dir: str, output_base: str):
    source = Path(source_dir)
    output = Path(output_base)

    files = [f for f in source.glob("*.*") if f.suffix.lower() in SYSTEM_PROMPTS]
    print(f"Найдено файлов для обработки: {len(files)}")

    for file in files:
        task_name = file.stem
        ext = file.suffix.lower()
        task_file = task_name + '.json'
        output_file_path = output / task_file

        if output_file_path.exists():
            print(f'Пропуск {file.name}: {task_file} уже существует')
            continue
        
        print(f"--- Обработка {task_name} ---")
        try:
            content = file.read_text(encoding="utf-8")
            result = extract_schedule_data(content, ext)

            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2, exclude_none=True))
            
            print(f"Готово: {task_name}")

        except Exception as e:
            print(f"Ошибка в задаче {task_name}: {e}")


if __name__ == "__main__":
    SOURCE_DIR = "data/benchmark/1_raw_data"
    OUTPUT_BASE = "data/benchmark/3_model_output"

    run_extraction(SOURCE_DIR, OUTPUT_BASE)