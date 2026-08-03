import json
from enum import Enum
from typing import Any

import numpy


def response_encoder(element: Any) -> Any:
    if isinstance(element, numpy.ndarray):
        return element.tolist()
    elif isinstance(element, BaseException):
        return str(element)
    elif isinstance(element, bytes):
        return f"bytes[{len(element)}]"
    elif isinstance(element, Enum):
        return element.value  # Or use str(element) for the enum name and value
    elif hasattr(element, "__json__"):
        return element.__json__()
    elif hasattr(element, "__dict__"):
        return element.__dict__
    else:
        return str(element)


def dump_json(obj: Any) -> str:
    return json.dumps(obj, default=response_encoder, ensure_ascii=False)
