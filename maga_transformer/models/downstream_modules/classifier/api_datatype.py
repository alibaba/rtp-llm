import json
import asyncio
from typing import List, Tuple
from maga_transformer.config.base_model_config import PyDanticModelBase

class ClassifierRequest(PyDanticModelBase):
    input: List[Tuple[str, str]]
    model: str = ""

class ClassifierResponse(PyDanticModelBase):
    object: str = 'list'
    score: List[List[float]]