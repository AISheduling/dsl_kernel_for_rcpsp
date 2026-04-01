from pydantic import ValidationError
from src.dsl_schema import SchedulingProblem

def validate_dsl_file(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        json_data = f.read()
    
    try:
        problem = SchedulingProblem.model_validate_json(json_data)
        return problem
        
    except ValidationError as e:
        return e

def load_and_validate_dsl(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        json_data = f.read()
        return SchedulingProblem.model_validate_json(json_data)