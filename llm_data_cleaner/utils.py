from typing import Dict, Any, Type, List, Optional
import pandas as pd
from pydantic import BaseModel, RootModel

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
