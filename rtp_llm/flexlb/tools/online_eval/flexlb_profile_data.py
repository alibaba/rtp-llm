"""Profile data for FLEXLB_CONFIG rendering.

This module owns workload policy defaults and the stress-na130 document.
The DSv4 test calibration lives in data/performance/dsv4_l20_mock_calibration.json;
flexlb_cfg.py owns schema validation and rendering.
"""

import json
import math
from pathlib import Path

_CALIBRATION_PATH = Path(__file__).resolve().parent / "data/performance/dsv4_l20_mock_calibration.json"


def load_mock_calibration(path=_CALIBRATION_PATH):
    """Read the auditable test calibration; no measured values live in code."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (data.get("schema_version") != 1 or data.get("id") != "dsv4_l20_legacy_mock"
            or data.get("status") != "legacy_unverified"
            or not isinstance(data.get("model"), str) or not data["model"].strip()
            or not isinstance(data.get("hardware"), str) or not data["hardware"].strip()
            or not isinstance(data.get("source"), str) or not data["source"].strip()
            or not isinstance(data.get("prefill_expression"), str)
            or not data["prefill_expression"].strip()):
        raise ValueError(f"invalid mock calibration: {path}")
    decode = data.get("decode")
    if (not isinstance(decode, dict)
            or set(decode) != {"step_base_ms", "step_per_running_ms", "tokens_per_step"}
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or value < 0 for value in decode.values())
            or decode["tokens_per_step"] == 0):
        raise ValueError(f"invalid mock decode calibration: {path}")
    return data


DSV4_PREFILL_EXPRESSION = load_mock_calibration()["prefill_expression"]

# ===========================================================================
# Profiles
# ===========================================================================
#
# Functional profile axes and capabilities. The stress profile is render-only.

PROFILES = (
    "batch-window",
    "single-nonbatch",
    "single-batch",
    "window-nonbatch",
)

# decision x dispatcher axes per profile (scheduler is QUEUE, ordering FIFO).
PROFILE_SPECS = {
    "batch-window": {"decision": "fixed_window", "dispatcher": "batch"},
    "single-nonbatch": {"decision": "single", "dispatcher": "non_batch"},
    "single-batch": {"decision": "single", "dispatcher": "batch"},
    "window-nonbatch": {"decision": "fixed_window", "dispatcher": "non_batch"},
}

# Semantic capabilities per profile, used by CaseDef.requires filtering.
PROFILE_CAPS = {
    "batch-window": {
        "queue",
        "fifo",
        "fixed_window",
        "batch_dispatch",
        "enqueue_batch",
        "fetch_response",
    },
    "single-nonbatch": {
        "queue",
        "fifo",
        "single",
        "non_batch_dispatch",
        "frontend_send",
        "generate_stream",
    },
    "single-batch": {
        "queue",
        "fifo",
        "single",
        "batch_dispatch",
        "enqueue_batch",
        "fetch_response",
    },
    "window-nonbatch": {
        "queue",
        "fifo",
        "fixed_window",
        "non_batch_dispatch",
        "frontend_send",
        "generate_stream",
    },
}

# The stress-line baseline profile is render-only: not part of PROFILES.
STRESS_PROFILE = "stress-na130"


STRESS_DECISION_RETYPE_DEFAULTS = {
    "maxRequests": 32,
    "maxCollectionWaitMs": 10,
    "maxPredictedExecutionMs": 550,
}

FUNCTIONAL_PROFILE_QUEUE_TIMEOUT_MS = 60_000

# Functional-test workload defaults, not Java schema defaults.
FUNCTIONAL_DEFAULTS = {
    "ordering": "fifo",
    "decision": "fixed_window",
    "dispatcher": "batch",
    "default_priority": None,
    "preemption": None,
    "max_requests": 32,
    "max_collection_wait_ms": 10,
    "max_predicted_execution_ms": 550,
    "queue_timeout_ms": None,
    "max_inflight_per_prefill_worker": 2,
    "prefill_expression": DSV4_PREFILL_EXPRESSION,
    "request_timeout_ms": 60_000,
    "decision_lifetime": 2.0,
    "status_rpc_ms": 1_000,
    "status_stale_after_ms": None,
    "cleanup_interval_ms": 3_000,
    "decode_max_engine_requests": 132,
    "decode_max_kv_usage_percent": 90,
}

# Profile-layer defaults for the functional generator: the axes plus the
# queue deadline (tight enough that queue-timeout gate cases observe
# expiry without waiting for the Java default of 1h).
FUNCTIONAL_PROFILE_KWARGS = {
    profile: {
        "ordering": "fifo",
        "decision": spec["decision"],
        "dispatcher": spec["dispatcher"],
        "queue_timeout_ms": FUNCTIONAL_PROFILE_QUEUE_TIMEOUT_MS,
    }
    for profile, spec in PROFILE_SPECS.items()
}


# ===========================================================================
# stress-na130 FLEXLB_CONFIG data
# ===========================================================================

STRESS_BASE: dict = {
    "schemaVersion": 3,
    "scheduler": {
        "type": "QUEUE",
        "ordering": {
            "type": "PRIORITY",
            "defaultPriority": 50,
            "preemption": {
                "allowedVictimStages": ["PREFILL_QUEUED", "DECODE_RESERVED"]
            },
        },
        "queueTimeoutMs": 60000,
        "decision": {
            "type": "FIXED_WINDOW",
            "maxRequests": 32,
            "maxCollectionWaitMs": 400,
            "maxPredictedExecutionMs": 550,
        },
    },
    "dispatcher": {"type": "BATCH", "maxInflightPerPrefillWorker": 2},
    "router": {
        "roles": {
            "prefill": {
                "executionTimeEstimator": {
                    "type": "FORMULA",
                    "expression": DSV4_PREFILL_EXPRESSION,
                },
                "cacheAffinity": {"maxExtraTtftMs": 20, "minPrefixHitPercent": 20},
            },
            "decode": {
                "availability": {"maxKvUsagePercent": 95, "maxEngineRequests": 384}
            },
        }
    },
    "workerRegistry": {
        "health": {
            "statusPollIntervalMs": 20,
            "statusRpcTimeoutMs": 5000,
            "statusStaleAfterMs": 10000,
            "cleanupIntervalMs": 3000,
        },
        "cacheStatus": {
            "targetDiffSize": 30,
            "minRefreshIntervalMs": 50,
            "maxRefreshIntervalMs": 3000,
            "fullSnapshotDebugMode": False,
        },
    },
    "observability": {
        "cacheHit": {
            "recentKeyWindow": {
                "writeEnabled": True,
                "durationMs": 1800000,
                "maxKeyOccurrences": 80000000,
            },
            "metricsEnabled": True,
            "requestTraceLogEnabled": False,
        }
    },
    "requestLifecycle": {
        "request": {"timeoutMs": 300000},
        "decision": {"lifetime": 2.0},
    },
}

# All renderable profiles, including workload defaults, share axis metadata.
REGISTERED_PROFILE_SPECS = {
    **PROFILE_SPECS,
    STRESS_PROFILE: {
        "decision": STRESS_BASE["scheduler"]["decision"]["type"].lower(),
        "dispatcher": STRESS_BASE["dispatcher"]["type"].lower(),
    },
}
