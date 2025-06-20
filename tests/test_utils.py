import json
import pytest
from pydantic import BaseModel
from llm_data_cleaner.utils import jsonize

class Dummy(BaseModel):
    x: int

def test_jsonize_base_model():
    d = Dummy(x=1)
    assert jsonize(d) == d.json()

def test_jsonize_list_of_models():
    d1 = Dummy(x=1)
    d2 = Dummy(x=2)
    assert jsonize([d1, d2]) == json.dumps([{"x": 1}, {"x": 2}], ensure_ascii=False)

def test_jsonize_literal():
    assert jsonize(42) == 42
    assert jsonize("abc") == "abc"

def test_jsonize_dict_passthrough():
    assert jsonize({"a": 1}) == {"a": 1}
