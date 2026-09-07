"""Shared balance scenario components. Resource and timing semantics live here."""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ..context import CaseContext, CaseDef, rid_base
from ..grade import GradeReport
from ..harness import TTL_DRAIN_TIMEOUT_S, AssertUtils
from .requests import drain_fired as _drain_fired
from .requests import fire_request as _fire_request
from .requests import wait_engine_pending as _poll_engine_pending

STREAM_TIMEOUT_S = 15.0


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"]


def _decode_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "decode"]


# -- shared fire-and-forget helpers (S4 hotspot pattern, cross-case) --


# ===========================================================================
# Balance cases (result-property graded — rework of
# scheduling_smoke.py S1-S12; rid_base family "scheduling" -> "balance"
# in the category reorg)
# ===========================================================================
