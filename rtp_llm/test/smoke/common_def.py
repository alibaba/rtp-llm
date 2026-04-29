import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from smoke.rel_path_config import compute_smoke_rel_path

WORLD_SIZE = "WORLD_SIZE"

# OSS and internal smoke share comparer code but have separate data trees.
# When BOTH internal_source/rtp_llm/test/smoke/ and rtp_llm/test/smoke/ exist,
# the previous "first hit wins" rule misrouted OSS task_info paths to the
# internal data dir. Entry scripts (test_smoke_oss.py / test_smoke_internal.py)
# set SMOKE_REL_PATH_PREFER before importing this module to bias the choice.
#
# Resolution logic lives in ``rel_path_config.py`` (stdlib-only) for verify scripts.
_ABS_PKG = os.path.dirname(os.path.realpath(__file__))
REL_PATH = compute_smoke_rel_path(_ABS_PKG)

ABS_PATH = _ABS_PKG


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
