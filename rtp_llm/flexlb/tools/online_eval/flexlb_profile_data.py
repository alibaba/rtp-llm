"""Profile data for FLEXLB_CONFIG rendering.

This module owns workload-specific defaults and the stress-na130 document.
The DSv4 fit below is the sole runtime definition; both profile families
refer to it. flexlb_cfg.py owns schema validation and rendering.
"""

# Single authoritative runtime definition of the DSv4 prefill fit.
DSV4_PREFILL_EXPRESSION = (
    "max(196, -68.612174288157 + 0.993068319341 * (max(0, 287.3980926717 + 2.30134977837751 *"
    " batchSize + 0.158123254797307 * sum(hitCacheTokens / 1024.) + 0.575522710053703 *"
    " sum(computeTokens / 1024.) + 0.0517623430739831 * sum(computeTokens / 1024. * computeTokens /"
    " 1024.) + 0.0395308136993267 * sum(hitCacheTokens / 1024. * computeTokens / 1024.) +"
    " 0.0104363634681015 * sum(hitCacheTokens / 1024. * hitCacheTokens / 1024.) + 0.575522710053703 *"
    " max(sum(computeTokens / 1024.) - 16, 0) + 2.82077211814514 * max(sum(computeTokens / 1024.) -"
    " 32, 0) - 0.0254671429192862 * max(sum(computeTokens / 1024.) - 64, 0) + 2.15779213792494 *"
    " max(sum(computeTokens / 1024.) - 96, 0) + 0.247806025472364 * max(sum(hitCacheTokens / 1024.) -"
    " 32, 0) - 0.444522654549492 * max(sum(hitCacheTokens / 1024.) - 64, 0) - 0.427317020061895 *"
    " max(sum(hitCacheTokens / 1024.) - 128, 0) + 0.347029077528455 * max(sum(hitCacheTokens / 1024.)"
    " - 256, 0) - 0.298742307762735 * max(sum(hitCacheTokens / 1024.) - 384, 0) + 2.30134977837751 *"
    " max(batchSize - 8, 0) - 3.54884859699154 * max(batchSize - 16, 0) - 11.3438560779984 *"
    " max(batchSize - 24, 0) + 0.879751992138183 * sum(max(computeTokens / 1024. - 2, 0)) +"
    " 0.636364578079591 * sum(max(computeTokens / 1024. - 4, 0)) - 0.0513345988517118 *"
    " sum(max(computeTokens / 1024. - 8, 0)) - 0.332584389129357 * sum(max(hitCacheTokens / 1024. -"
    " 2, 0)) + 0.305819761192588 * sum(max(hitCacheTokens / 1024. - 4, 0)) - 0.287610979974721 *"
    " sum(max(hitCacheTokens / 1024. - 8, 0)) + 0.191310200712013 * sum(max(hitCacheTokens / 1024. -"
    " 12, 0)) + 0.0130251644478961 * max(batchSize - 8, 0) * sum(hitCacheTokens / 1024.) +"
    " 0.00981382840761646 * max(batchSize - 16, 0) * sum(hitCacheTokens / 1024.) - 0.0299132587297009"
    " * max(batchSize - 24, 0) * sum(hitCacheTokens / 1024.) + 0.0447455122487382 * max(batchSize -"
    " 8, 0) * sum(computeTokens / 1024.) + 0.0104635312001851 * max(batchSize - 16, 0) *"
    " sum(computeTokens / 1024.) + 0.0542737877321807 * max(batchSize - 24, 0) * sum(computeTokens /"
    " 1024.))))"
)

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
