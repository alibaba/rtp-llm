import json
import asyncio
from typing import List, Tuple, Union
from maga_transformer.config.base_model_config import PyDanticModelBase

class ClassifierRequest(PyDanticModelBase):
    # support merge request pair or raw request
    input: List[Union[Tuple[str, str], str]]
    model: str = ""

class ClassifierResponse(PyDanticModelBase):
    object: str = 'list'
    score: List[List[float]]