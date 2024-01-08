import os
import json
import logging
from typing import Any, Dict, Type, Union,  Optional

import sys
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

_model_factory: Dict[str, Type[Any]] = {}

def register_model(name: str, model_type: Any):
    global _model_factory
    if name in _model_factory and _model_factory[name] != model_type:
        raise Exception(f"try register model {name} with type {_model_factory[name]} and {model_type}, confict!")
    _model_factory[name] = model_type