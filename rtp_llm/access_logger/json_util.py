import numpy
from typing import Any

import json

def response_encoder(element: Any) -> Any:
    if isinstance(element, numpy.ndarray):
        return element.tolist()
    elif isinstance(element, BaseException):
        return str(element)
    elif isinstance(element, bytes):
        return f"bytes[{len(element)}]"
    return element.__dict__

def dump_json(obj: Any) -> str:
    return json.dumps(obj, default=response_encoder, ensure_ascii=False)
