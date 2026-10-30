"""Process-wide prefill implementation selection; configure before starting ranks.

0 selects the 70f1ecde04 compatibility path; 1 retains current optimizations.
All ranks must use the same value. Changing it after import is unsupported.
"""

import logging
import os

_value = os.environ.get("RTP_LLM_CP_PREFILL_PIPELINE_OPT", "1")
if _value not in ("0", "1"):
    raise ValueError("RTP_LLM_CP_PREFILL_PIPELINE_OPT must be 0 or 1")
PREFILL_OPTIMIZATIONS_ENABLED = _value == "1"
logging.info(
    "Prefill optimization mode: %s",
    "optimized" if PREFILL_OPTIMIZATIONS_ENABLED else "70f1ecde04 compatibility",
)


# These options already existed in 70f1; compatibility must preserve their
# configured values and historical defaults. Only newer pipeline options and
# the explicitly disabled KV overlap paths are overridden by the master switch.
_LEGACY_OPTIONS = frozenset(
    {
        "RTP_LLM_CP_KV_INDEX_OPT",
        "RTP_CP_PREPARED_SCATTER",
        "RTP_CP_CONTIGUOUS_ROPE_KPE",
        "RTP_CP_REMAP_HISTORY_TOPK",
        "RTP_LLM_CP_INDEXER_REGISTERED",
        "RTP_LLM_CP_KV_REGISTERED",
    }
)


def prefill_optimization_setting(name: str, default: str = "0") -> str:
    """Disable new pipeline paths without changing historical option semantics."""
    if PREFILL_OPTIMIZATIONS_ENABLED or name in _LEGACY_OPTIONS:
        return os.environ.get(name, default)
    return "0"
