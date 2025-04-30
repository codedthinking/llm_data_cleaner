from typing import Dict, Any, Type, List, Optional
import pandas as pd
from pydantic import BaseModel, RootModel
import json

def jsonize(data: Any) -> str:
    """
    Convert data to JSON string. Data may be a literal, a list of literals, a BaseModel, or a list of BaseModels. 
    Dictionaries are not permitted.
    Use proper JSON parsing, do not manually convert.

    Args:
        data: Data to be converted

    Returns:
        JSON string representation of the data
    """
    if isinstance(data, list):
        # Check if all elements are BaseModel instances
        if all(isinstance(item, BaseModel) for item in data):
            return json.dumps([item.dict() for item in data], ensure_ascii=False)
        else:
            return json.dumps(data, ensure_ascii=False)
    elif isinstance(data, BaseModel):
        return data.json()
    else:
        return data

# schema for instructions as a pydantic model
# instructions is a dictionary, with keys as column names and values as dictionaries of "prompt" (a string) amd "schema" (a pydantic model)
class InstructionField(BaseModel):
    prompt: str
    schema_class: Type[BaseModel]  # Pydantic model class for the schema

    def __getitem__(self, item):
        if item == "prompt":
            return self.prompt
        elif item == "schema":
            return self.schema_class
        else:
            raise KeyError(f"Invalid key: {item}")

class InstructionSchema(RootModel):
    root: Dict[str, InstructionField]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]
    
    def items(self):
        return self.root.items()
