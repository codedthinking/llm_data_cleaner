from typing import Dict, Any, Type, List, Optional
import pandas as pd
from pydantic import BaseModel

def load_yaml_instructions(yaml_path:str = None) -> Dict[str, Type[BaseModel]]:
    """
    Load models from YAML file.
    """
    import yaml
    def parse_type(props, model_defs=None, model_cache=None):

        if model_defs is None:
            model_defs = {}
        if model_cache is None:
            model_cache = {}

        typ = props["type"]
        optional = props.get("optional", False)
        if typ == 'str':
            typ = str

        if typ == 'float':
            typ = float

        if typ == 'list':
            items = props["items"]
            if isinstance(items, dict):
                typ = List[parse_type(items, model_defs, model_cache)]
            else:
                typ = List[parse_type({"type": items}, model_defs, model_cache)]

        if optional:
            typ = Optional[typ]

        return typ

    with open(yaml_path, "r") as f:
        schema = yaml.safe_load(f)
    models = {}
    for name, model_def in schema.items():
        fields = model_def["fields"]
        annotations = {}
        defaults = {}
        for field, props in fields.items():
            t = parse_type(props)
            print(t)
            if props.get("optional", False):
                defaults[field] = None
                annotations[field] = (t, None)
            else:
                annotations[field] = (t, ...)
        models[name] = create_model(name, **annotations, __base__=BaseModel)
    return models

def batch_dataframe(df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
    """
    Split a DataFrame into batches.

    Args:
        df: Input DataFrame
        batch_size: Size of each batch

    Returns:
        List of DataFrame batches
    """
    if len(df) <= batch_size:
        return [df]
    
    return [df[i:i + batch_size] for i in range(0, len(df), batch_size)]


def validate_instructions(instructions: dict) -> List[str]:
    """
    Validate the cleaning instructions.

    Args:
        instructions: Dictionary of cleaning instructions

    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    for column, instruction in instructions.items():
        if not isinstance(column, str):
            errors.append(f"Column name must be a string, got {type(column)}")
        
        if not isinstance(instruction, dict):
            errors.append(f"Instruction for column '{column}' must be a dictionary")
            continue
        
        if "prompt" not in instruction:
            errors.append(f"Missing 'prompt' in instructions for column '{column}'")
        
        if not isinstance(instruction.get("prompt", ""), str):
            errors.append(f"Prompt for column '{column}' must be a string")
        
        if "schema" in instruction and not isinstance(instruction["schema"], dict):
            errors.append(f"Schema for column '{column}' must be a dictionary")
    
    return errors


class Example(BaseModel):
    """
    Example class for structured output testing.
    """
    year: List[int] = None
    university: List[str] = None
    job_title: str = None
