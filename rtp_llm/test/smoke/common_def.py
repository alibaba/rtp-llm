import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

WORLD_SIZE = "WORLD_SIZE"

# OSS and internal smoke share comparer code but have separate data trees.
# When BOTH internal_source/rtp_llm/test/smoke/ and rtp_llm/test/smoke/ exist,
# the previous "first hit wins" rule misrouted OSS task_info paths to the
# internal data dir. Entry scripts (test_smoke_oss.py / test_smoke_internal.py)
# set SMOKE_REL_PATH_PREFER before importing this module to bias the choice.
#
# This module always loads from the OSS tree (.../github-opensource/rtp_llm/test/smoke).
# Relative paths like internal_source/... only work when cwd is github-opensource/ and
# that symlink or directory exists. We always include _ABS_PKG (this file's directory)
# first so OSS golden data resolves without symlinks. For internal pytest runs,
# SMOKE_REL_PATH_PREFER=internal + monorepo layout resolves sibling internal_source/.../smoke.
_ABS_PKG = os.path.dirname(os.path.realpath(__file__))
_PREFER = os.environ.get("SMOKE_REL_PATH_PREFER", "")


def _monorepo_internal_smoke_dir() -> Optional[str]:
    """If this file lives under .../github-opensource/rtp_llm/test/smoke, map to sibling internal tree."""
    p = Path(_ABS_PKG).resolve()
    if len(p.parents) < 4:
        return None
    # parents: [0]=test, [1]=rtp_llm, [2]=github-opensource, [3]=monorepo root
    internal = p.parents[3] / "internal_source" / "rtp_llm" / "test" / "smoke"
    if internal.is_dir():
        return str(internal)
    return None


if _PREFER == "internal":
    _resolved = _monorepo_internal_smoke_dir()
    if _resolved is not None:
        REL_PATH = _resolved
    else:
        _CANDIDATES = [
            "internal_source/rtp_llm/test/smoke",
            "rtp_llm/test/smoke",
            _ABS_PKG,
        ]
        REL_PATH = next((p for p in _CANDIDATES if os.path.isdir(p)), _ABS_PKG)
elif _PREFER == "oss":
    _CANDIDATES = [
        _ABS_PKG,
        "rtp_llm/test/smoke",
        "internal_source/rtp_llm/test/smoke",
    ]
    REL_PATH = next((p for p in _CANDIDATES if os.path.isdir(p)), _ABS_PKG)
else:
    _CANDIDATES = [
        _ABS_PKG,
        "internal_source/rtp_llm/test/smoke",
        "rtp_llm/test/smoke",
    ]
    REL_PATH = next((p for p in _CANDIDATES if os.path.isdir(p)), _ABS_PKG)

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
