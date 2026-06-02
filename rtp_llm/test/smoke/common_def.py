import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel

WORLD_SIZE = "WORLD_SIZE"

_CANDIDATES = [
    "internal_source/rtp_llm/test/smoke",
    "rtp_llm/test/smoke",
]
REL_PATH = next((p for p in _CANDIDATES if os.path.isdir(p)),
                os.path.dirname(os.path.realpath(__file__)))

ABS_PATH = os.path.dirname(os.path.realpath(__file__))


class QueryStatus(Enum):
    OK = "ok"
    COMPARE_FAILED = "compare_failed"
    VISIT_FAILED = "visit_failed"
    VALID_FAILED = "VALID_FAILED"
    OTHERS = "others"


class SmokeException(Exception):
    def __init__(self, error_status: QueryStatus, message: str):
        self.error_status = error_status
        self.message = message
        super().__init__(self.message)


class Tracer(BaseModel):
    query: Optional[BaseModel] = None
    expect_result: Optional[BaseModel] = None
    actual_result: Optional[BaseModel] = None
