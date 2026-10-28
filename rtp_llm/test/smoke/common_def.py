import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from rtp_llm.test.smoke.rel_path_config import DataRoot, compute_smoke_rel_path

WORLD_SIZE = "WORLD_SIZE"

# OSS and internal smoke share comparer code but have separate data trees.
# When BOTH internal_source/rtp_llm/test/smoke/ and rtp_llm/test/smoke/ exist,
# the previous "first hit wins" rule misrouted OSS task_info paths to the
# internal data dir. Test entry files (test_smoke_oss.py / test_smoke_internal.py)
# call ``set_default_data_root("oss"|"internal")`` BEFORE importing comparers
# (which capture ``REL_PATH`` at import time) to bias the choice.
#
# Resolution logic lives in ``rel_path_config.py`` (stdlib-only) for verify scripts.
_ABS_PKG = os.path.dirname(os.path.realpath(__file__))
REL_PATH = compute_smoke_rel_path(_ABS_PKG)
ABS_PATH = _ABS_PKG


def set_default_data_root(prefer: DataRoot) -> str:
    """Recompute and reassign module-level REL_PATH for the given data root.

    Must be called by test entry files BEFORE any comparer module is imported
    (comparers do ``from smoke.common_def import REL_PATH`` which captures the
    value at import time). Per-test data_root plumbing is handled by the runner
    in ``smoke_framework`` (B1+); this setter remains for entry-file defaults.
    """
    global REL_PATH
    REL_PATH = compute_smoke_rel_path(_ABS_PKG, prefer=prefer)
    return REL_PATH


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
