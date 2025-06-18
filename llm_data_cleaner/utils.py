from typing import Dict, Any, Type, List, Optional
import pandas as pd
from pydantic import BaseModel, RootModel
import json

def jsonize(data: Any) -> Any:
    """Normalize ``data`` for storage.

    - ``BaseModel`` instances and lists of ``BaseModel`` objects are converted to
      JSON strings.
    - Lists of primitive values are also JSON encoded so their structure is
      preserved.
    - All other values, including dictionaries and integers, are returned
      unchanged.

    Args:
        data: The value to normalise.

    Returns:
        The JSON string if conversion occurs, otherwise the original value.
    """
    if isinstance(data, list):
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
