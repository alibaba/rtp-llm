"""flexlb_cfg — the shared schema validator and renderer for FLEXLB_CONFIG.

Profile-specific values live in flexlb_profile_data; this module renders
every FLEXLB_CONFIG document this repo produces:

  * ``render_env(profile, overrides)`` — the master-process env string;
  * ``render_process_config(profile, overrides, jvm_heap)`` — the
    zone_process_setting envelope (FLEXLB_CONFIG + FLEXLB_JVM_HEAP_SIZE
    envs) consumed by the Java mock engine's ``--master-config`` and by
    run_stress.py.  The env string inside the envelope is derived
    from the same render call, so the file and the env cannot drift
    apart (single render, two projections).

Profiles:

  * four functional case-test profiles (``PROFILES``) — the schema-v3
    decision x dispatcher axes (scheduler QUEUE + FIFO ordering), values
    unchanged from the former harness.flexlb_config_for_profile;
  * ``stress-na130`` — the stress workload rendered using schema-v3.
    Removed capacity, selector and acknowledgement settings are not emitted.

Override semantics (``ConfigOverride``):

  * layering is strictly one-way: base template < profile < override;
  * ``None`` (the default) means "do not touch" — the profile value
    survives;
  * ``OMIT`` (sentinel) forces the key OUT of the document (use for
    queue_timeout_ms when the legacy wrapper meant "not emitted — keep
    the Java default", while the functional profiles carry 60000);
  * an override may only change fields the profile's document already
    has (unknown-for-profile fields raise ValueError — e.g.
    unknown legacy capacity knobs);
  * functional profile identity (decision / dispatcher) cannot be overridden;
    ordering remains a configurable policy. The render-only stress profile
    allows schema retyping and drops keys exclusive to the previous type.

The Java side of the contract: ConfigService (STRICT_MAPPER,
FAIL_ON_UNKNOWN_PROPERTIES) + FlexlbConfigValidator.validateQueue — the
Python mirror below fails fast on the same shapes so a malformed config
dies in Python instead of aborting master startup.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from typing import Mapping, Optional, Union

from flexlb_profile_data import (
    DSV4_PREFILL_EXPRESSION,
    FUNCTIONAL_DEFAULTS,
    GENERATOR_DEFAULTS,
    STRESS_BASE,
    PROFILES,
    PROFILE_SPECS,
    PROFILE_CAPS,
    STRESS_PROFILE,
    FUNCTIONAL_PROFILE_KWARGS,
    STRESS_DECISION_RETYPE_DEFAULTS,
)

# scheduler.ordering.preemption.allowedVictimStages enum values
# (flexlb-common VictimStage.java; see PreemptionPolicy).
VICTIM_STAGES = ("PREFILL_QUEUED", "DECODE_RESERVED", "DECODE_ENGINE_OWNED")

_RENDER_PROFILES = PROFILES + (STRESS_PROFILE,)

class ProfileIdentityError(ValueError):
    """A functional profile's decision or dispatcher was changed."""


def validate_profile_identity(profile: str, overrides: Mapping[str, object]) -> None:
    """Validate the closed functional profile axes for all config entry points."""
    for axis, expected in PROFILE_SPECS[profile].items():
        value = overrides.get(axis)
        if value is not None and value != expected:
            raise ProfileIdentityError(f"{axis} is a profile identity field for {profile}")


def profile_dispatches_batch(profile: str) -> bool:
    """True when *profile*'s dispatcher axis is BATCH (master sends via
    EnqueueBatch; clients consume FetchResponse)."""
    if profile == STRESS_PROFILE:
        return STRESS_BASE["dispatcher"]["type"] == "BATCH"
    return PROFILE_SPECS[profile]["dispatcher"] == "batch"


# ===========================================================================
# Override spec
# ===========================================================================


class _Omit:
    """Sentinel: force the config key OUT of the rendered document.

    Distinct from ``None`` (leave the profile value in place).  Meaningful
    for keys whose absence is itself a knob — e.g. queue_timeout_ms, where
    the functional profiles emit 60000 but the legacy fault-family wrappers
    omitted the key entirely (Java default = 1h).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "OMIT"


OMIT = _Omit()

# Fields where OMIT is a legal value (absence is a legal document state).
_OMITTABLE = frozenset({"queue_timeout_ms", "preemption"})


def _validate_affinity(extra_ms, percent):
    if extra_ms is not None and (type(extra_ms) is not int or not 0 <= extra_ms <= 9223372036854775807):
        raise ValueError("cache_affinity_max_extra_ttft_ms must be a nonnegative Java long")
    if percent is not None and (type(percent) not in (int, float) or not math.isfinite(percent) or not 0 <= percent <= 100):
        raise ValueError("cache_affinity_min_prefix_hit_percent must be finite and in [0, 100]")


@dataclass(frozen=True)
class ConfigOverride:
    """Explicit schema-v3 overrides; removed schema-v3 keys are rejected."""

    ordering: Optional[str] = None
    decision: Optional[str] = None
    dispatcher: Optional[str] = None
    default_priority: Optional[int] = None
    preemption: Optional[dict] = None
    max_requests: Optional[int] = None
    max_collection_wait_ms: Optional[int] = None
    max_predicted_execution_ms: Optional[int] = None
    queue_timeout_ms: Union[int, _Omit, None] = None
    max_inflight_per_prefill_worker: Optional[int] = None
    request_timeout_ms: Optional[int] = None
    decision_lifetime: Optional[float] = None
    status_rpc_ms: Optional[int] = None
    status_stale_after_ms: Optional[int] = None
    cleanup_interval_ms: Optional[int] = None
    decode_max_engine_requests: Optional[int] = None
    decode_max_kv_usage_percent: Optional[int] = None
    cache_affinity_max_extra_ttft_ms: Optional[int] = None
    cache_affinity_min_prefix_hit_percent: Optional[float] = None
    prefill_expression: Optional[str] = None
    strip_preemption: bool = False

    def __post_init__(self):
        if self.decision_lifetime is not None and (
            type(self.decision_lifetime) not in (int, float)
            or not math.isfinite(self.decision_lifetime)
            or self.decision_lifetime < 1
        ):
            raise ValueError("decision_lifetime must be finite and >= 1")
        _validate_affinity(self.cache_affinity_max_extra_ttft_ms,
                           self.cache_affinity_min_prefix_hit_percent)

    def omit_map(self) -> dict:
        return {f.name: getattr(self, f.name) is OMIT for f in fields(self)}


# ===========================================================================
# Preemption / ordering blocks (strict schema-v3; Python mirror of
# FlexlbConfigValidator.validateQueue)
# ===========================================================================


@dataclass(frozen=True)
class PreemptionPolicy:
    allowed_victim_stages: tuple[str, ...]
    timeout_ms: Optional[int] = None

    def __post_init__(self):
        if (type(self.allowed_victim_stages) is not tuple
                or not self.allowed_victim_stages
                or any(stage not in VICTIM_STAGES for stage in self.allowed_victim_stages)):
            raise ValueError(
                f"allowed_victim_stages must be a non-empty subset of {VICTIM_STAGES}"
            )
        if self.timeout_ms is not None:
            if "DECODE_ENGINE_OWNED" not in self.allowed_victim_stages:
                raise ValueError("timeout_ms requires DECODE_ENGINE_OWNED")
            if type(self.timeout_ms) is not int or self.timeout_ms <= 0:
                raise ValueError("preemption.timeout_ms must be a positive integer")

    @classmethod
    def from_input(cls, value: dict) -> PreemptionPolicy:
        unknown = set(value) - {"allowed_victim_stages", "timeout_ms"}
        if unknown:
            raise ValueError(f"unknown preemption keys: {sorted(unknown)}")
        return cls(tuple(value.get("allowed_victim_stages") or ()), value.get("timeout_ms"))

    @classmethod
    def from_json(cls, value: dict) -> PreemptionPolicy:
        unknown = set(value) - {"allowedVictimStages", "timeoutMs"}
        if unknown:
            raise ValueError(f"unknown preemption keys: {sorted(unknown)}")
        return cls(tuple(value.get("allowedVictimStages") or ()), value.get("timeoutMs"))

    def to_json(self) -> dict:
        result = {"allowedVictimStages": list(self.allowed_victim_stages)}
        if self.timeout_ms is not None:
            result["timeoutMs"] = self.timeout_ms
        return result


@dataclass(frozen=True)
class FifoOrdering:
    def to_json(self) -> dict:
        return {"type": "FIFO"}


@dataclass(frozen=True)
class PriorityOrdering:
    default_priority: Optional[int] = None
    preemption: Optional[PreemptionPolicy] = None

    def __post_init__(self):
        if self.default_priority is not None and (
            type(self.default_priority) is not int or not 1 <= self.default_priority <= 100
        ):
            raise ValueError(
                f"default_priority must be in [1, 100], got {self.default_priority}"
            )
        if self.preemption is not None and not isinstance(self.preemption, PreemptionPolicy):
            raise TypeError("preemption must be a PreemptionPolicy")

    def to_json(self) -> dict:
        result = {"type": "PRIORITY"}
        if self.default_priority is not None:
            result["defaultPriority"] = self.default_priority
        if self.preemption is not None:
            result["preemption"] = self.preemption.to_json()
        return result


@dataclass(frozen=True)
class SingleDecision:
    def to_json(self) -> dict:
        return {"type": "SINGLE"}


@dataclass(frozen=True)
class FixedWindowDecision:
    max_requests: int
    max_collection_wait_ms: int
    max_predicted_execution_ms: int

    def __post_init__(self):
        for name, value, minimum in (
            ("max_requests", self.max_requests, 1),
            ("max_collection_wait_ms", self.max_collection_wait_ms, 0),
            ("max_predicted_execution_ms", self.max_predicted_execution_ms, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

    def to_json(self) -> dict:
        return {
            "type": "FIXED_WINDOW",
            "maxRequests": self.max_requests,
            "maxCollectionWaitMs": self.max_collection_wait_ms,
            "maxPredictedExecutionMs": self.max_predicted_execution_ms,
        }


@dataclass(frozen=True)
class DispatcherPolicy:
    kind: str
    max_inflight_per_prefill_worker: int

    def __post_init__(self):
        if self.kind not in ("batch", "non_batch"):
            raise ValueError("dispatcher must be batch or non_batch")
        if (type(self.max_inflight_per_prefill_worker) is not int
                or not 1 <= self.max_inflight_per_prefill_worker <= 2_147_483_647):
            raise ValueError("max_inflight_per_prefill_worker must be a positive Java integer")

    def to_json(self) -> dict:
        return {
            "type": self.kind.upper(),
            "maxInflightPerPrefillWorker": self.max_inflight_per_prefill_worker,
        }


def _ordering_policy(ordering, default_priority, preemption):
    """Parse Python arguments into an ordering with no invalid field mix."""
    kind = ordering.lower() if isinstance(ordering, str) else ordering
    if kind not in ("fifo", "priority"):
        raise ValueError(f"ordering must be 'fifo' or 'priority', got {kind!r}")
    if kind == "fifo":
        if default_priority is not None or preemption is not None:
            raise ValueError(
                "default_priority/preemption apply only to ordering='priority' "
                "(the strict FLEXLB_CONFIG parser rejects them under FIFO)"
            )
        return FifoOrdering()
    return PriorityOrdering(default_priority,
                            PreemptionPolicy.from_input(preemption) if preemption is not None else None)


# ===========================================================================
# build_flexlb_config — the functional-template generator
# ===========================================================================
#
# Unified strict schema-v3 generator for the four functional profiles
# (formerly harness.build_flexlb_config, extended with the
# decode_max_engine_requests knob so the JSON-splice call sites could
# migrate onto generator parameters).  The router always gets the FORMULA
# estimator with the profile data fit injected explicitly.


def build_flexlb_config(
    *,
    ordering: str = GENERATOR_DEFAULTS["ordering"],
    decision: str = GENERATOR_DEFAULTS["decision"],
    dispatcher: str = GENERATOR_DEFAULTS["dispatcher"],
    default_priority: Optional[int] = GENERATOR_DEFAULTS["default_priority"],
    preemption: Optional[dict] = GENERATOR_DEFAULTS["preemption"],
    max_requests: int = GENERATOR_DEFAULTS["max_requests"],
    max_collection_wait_ms: int = GENERATOR_DEFAULTS["max_collection_wait_ms"],
    max_predicted_execution_ms: int = GENERATOR_DEFAULTS["max_predicted_execution_ms"],
    queue_timeout_ms: Optional[int] = GENERATOR_DEFAULTS["queue_timeout_ms"],
    # Functional-test workload values come from flexlb_profile_data.
    max_inflight_per_prefill_worker: int = GENERATOR_DEFAULTS["max_inflight_per_prefill_worker"],
    prefill_expression: str = GENERATOR_DEFAULTS["prefill_expression"],
    cache_affinity_max_extra_ttft_ms: int = 20,
    cache_affinity_min_prefix_hit_percent: float = 20,
    request_timeout_ms: int = GENERATOR_DEFAULTS["request_timeout_ms"],
    decision_lifetime: float = GENERATOR_DEFAULTS["decision_lifetime"],
    status_rpc_ms: int = GENERATOR_DEFAULTS["status_rpc_ms"],
    status_stale_after_ms: Optional[int] = GENERATOR_DEFAULTS["status_stale_after_ms"],
    cleanup_interval_ms: int = GENERATOR_DEFAULTS["cleanup_interval_ms"],
    decode_max_engine_requests: int = GENERATOR_DEFAULTS["decode_max_engine_requests"],
    decode_max_kv_usage_percent: int = GENERATOR_DEFAULTS["decode_max_kv_usage_percent"],
) -> str:
    """Generate schema-v3 JSON from scheduling policy and workload budgets."""
    _validate_affinity(cache_affinity_max_extra_ttft_ms, cache_affinity_min_prefix_hit_percent)
    if decision not in ("single", "fixed_window") or dispatcher not in (
        "batch", "non_batch"
    ):
        raise ValueError("unsupported decision or dispatcher")
    for name, value in (
        ("max_inflight_per_prefill_worker", max_inflight_per_prefill_worker),
        ("queue_timeout_ms", queue_timeout_ms),
        ("request_timeout_ms", request_timeout_ms),
    ):
        if value is None and name == "queue_timeout_ms":
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if max_inflight_per_prefill_worker > 2_147_483_647:
        raise ValueError(
            "max_inflight_per_prefill_worker exceeds the Java integer limit"
        )
    if (
        isinstance(decision_lifetime, bool)
        or not math.isfinite(decision_lifetime)
        or decision_lifetime < 1
    ):
        raise ValueError("decision_lifetime must be finite and at least 1")
    decision_policy = (
        FixedWindowDecision(max_requests, max_collection_wait_ms, max_predicted_execution_ms)
        if decision == "fixed_window" else SingleDecision()
    )
    scheduler_cfg: dict = {
        "type": "QUEUE",
        "ordering": _ordering_policy(ordering, default_priority, preemption).to_json(),
        "decision": decision_policy.to_json(),
    }
    if queue_timeout_ms is not None:
        scheduler_cfg["queueTimeoutMs"] = queue_timeout_ms
    return json.dumps(
        {
            "schemaVersion": 3,
            "scheduler": scheduler_cfg,
            "dispatcher": DispatcherPolicy(dispatcher, max_inflight_per_prefill_worker).to_json(),
            "requestLifecycle": {
                "request": {"timeoutMs": request_timeout_ms},
                "decision": {"lifetime": decision_lifetime},
            },
            "router": {
                "roles": {
                    "prefill": {
                        "executionTimeEstimator": {
                            "type": "FORMULA",
                            "expression": prefill_expression,
                        },
                        "cacheAffinity": {
                            "maxExtraTtftMs": cache_affinity_max_extra_ttft_ms,
                            "minPrefixHitPercent": cache_affinity_min_prefix_hit_percent,
                        },
                    },
                    "decode": {
                        "availability": {
                            "maxEngineRequests": decode_max_engine_requests,
                            "maxKvUsagePercent": decode_max_kv_usage_percent,
                        }
                    },
                }
            },
            "workerRegistry": {
                "health": {
                    "statusPollIntervalMs": 20,
                    "statusRpcTimeoutMs": status_rpc_ms,
                    "statusStaleAfterMs": (
                        max(10_000, status_rpc_ms * 2)
                        if status_stale_after_ms is None
                        else status_stale_after_ms
                    ),
                    "cleanupIntervalMs": cleanup_interval_ms,
                }
            },
        },
        separators=(",", ":"),
    )


# override field -> document path for the stress base (edit-in-place).
_STRESS_DOC_PATHS = {
    "cache_affinity_max_extra_ttft_ms": ("router", "roles", "prefill", "cacheAffinity", "maxExtraTtftMs"),
    "cache_affinity_min_prefix_hit_percent": ("router", "roles", "prefill", "cacheAffinity", "minPrefixHitPercent"),
    "prefill_expression": ("router", "roles", "prefill", "executionTimeEstimator", "expression"),
    "queue_timeout_ms": ("scheduler", "queueTimeoutMs"),
    "status_rpc_ms": ("workerRegistry", "health", "statusRpcTimeoutMs"),
    "decode_max_engine_requests": (
        "router",
        "roles",
        "decode",
        "availability",
        "maxEngineRequests",
    ),
    "decode_max_kv_usage_percent": (
        "router",
        "roles",
        "decode",
        "availability",
        "maxKvUsagePercent",
    ),
    "request_timeout_ms": ("requestLifecycle", "request", "timeoutMs"),
    "decision_lifetime": ("requestLifecycle", "decision", "lifetime"),
    "status_stale_after_ms": ("workerRegistry", "health", "statusStaleAfterMs"),
    "cleanup_interval_ms": ("workerRegistry", "health", "cleanupIntervalMs"),
}


def _apply_omits(doc: dict, overrides: ConfigOverride) -> None:
    for name, is_omit in overrides.omit_map().items():
        if not is_omit:
            continue
        if name not in _OMITTABLE:
            raise ValueError(f"ConfigOverride.{name}: OMIT is not a legal value")
        if name == "preemption":
            doc["scheduler"]["ordering"].pop("preemption", None)
        else:
            doc["scheduler"].pop("queueTimeoutMs", None)


def _edit_doc(doc: dict, path: tuple, value, field_name: str) -> None:
    """Set *value* at *path*; the key must already exist (SSOT rule:
    overrides may only change fields the profile document has)."""
    node = doc
    for key in path[:-1]:
        node = node[key]
    leaf = path[-1]
    if leaf not in node:
        raise ValueError(
            f"ConfigOverride.{field_name}: the profile document has no "
            f"{'.'.join(path)} key — overrides may only change existing fields"
        )
    node[leaf] = value


def _retype_ordering(doc: dict, overrides: ConfigOverride) -> None:
    """ordering / default_priority / preemption handling for the stress base."""
    scheduler = doc["scheduler"]
    current = scheduler["ordering"]
    current_type = str(current.get("type", "")).lower()
    if overrides.ordering is not None:
        new_type = overrides.ordering.lower()
        if new_type not in ("fifo", "priority"):
            raise ValueError(
                f"ordering must be 'fifo' or 'priority', got {overrides.ordering!r}"
            )
        if new_type == "fifo" and (
            overrides.default_priority is not None or overrides.preemption is not None
        ):
            raise ValueError(
                "default_priority/preemption apply only to ordering='priority' "
                "(the strict FLEXLB_CONFIG parser rejects them under FIFO)"
            )
        policy = FifoOrdering() if new_type == "fifo" else PriorityOrdering(
            current.get("defaultPriority") if current_type == "priority" else None,
            PreemptionPolicy.from_json(current["preemption"])
            if current_type == "priority" and "preemption" in current else None,
        )
    elif current_type == "priority":
        policy = PriorityOrdering(
            current.get("defaultPriority"),
            PreemptionPolicy.from_json(current["preemption"])
            if "preemption" in current else None,
        )
    else:
        policy = FifoOrdering()
    if isinstance(policy, FifoOrdering):
        scheduler["ordering"] = policy.to_json()
        return
    priority = (policy.default_priority if overrides.default_priority is None
                else overrides.default_priority)
    preemption = policy.preemption
    if overrides.preemption is not None and overrides.preemption is not OMIT:
        preemption = PreemptionPolicy.from_input(overrides.preemption)
    if overrides.strip_preemption:
        preemption = None
    scheduler["ordering"] = PriorityOrdering(
        priority, preemption,
    ).to_json()


def _retype_decision(doc: dict, overrides: ConfigOverride) -> None:
    decision = doc["scheduler"]["decision"]
    if overrides.decision is None and all(
        getattr(overrides, name) is None or getattr(overrides, name) is OMIT
        for name in ("max_requests", "max_collection_wait_ms", "max_predicted_execution_ms")
    ):
        return
    new_type = overrides.decision.lower() if overrides.decision is not None else decision["type"].lower()
    if new_type not in ("single", "fixed_window"):
        raise ValueError(
            "decision must be 'fixed_window' or 'single', got "
            f"{overrides.decision!r}"
        )
    if new_type == "single":
        doc["scheduler"]["decision"] = SingleDecision().to_json()
        for field, key in (("max_requests", "maxRequests"),
                           ("max_collection_wait_ms", "maxCollectionWaitMs"),
                           ("max_predicted_execution_ms", "maxPredictedExecutionMs")):
            value = getattr(overrides, field)
            if value is not None and value is not OMIT:
                _edit_doc(doc, ("scheduler", "decision", key), value, field)
        return
    values = {}
    for field, key in (("max_requests", "maxRequests"),
                       ("max_collection_wait_ms", "maxCollectionWaitMs"),
                       ("max_predicted_execution_ms", "maxPredictedExecutionMs")):
        value = getattr(overrides, field)
        values[key] = (
            decision.get(key, STRESS_DECISION_RETYPE_DEFAULTS[key])
            if value is None or value is OMIT else value
        )
    doc["scheduler"]["decision"] = FixedWindowDecision(
        values["maxRequests"], values["maxCollectionWaitMs"],
        values["maxPredictedExecutionMs"],
    ).to_json()


def _retype_dispatcher(doc: dict, overrides: ConfigOverride) -> None:
    if overrides.dispatcher is None and overrides.max_inflight_per_prefill_worker is None:
        return
    cap = overrides.max_inflight_per_prefill_worker
    if cap is None or cap is OMIT:
        cap = doc["dispatcher"]["maxInflightPerPrefillWorker"]
    doc["dispatcher"] = DispatcherPolicy(
        overrides.dispatcher or doc["dispatcher"]["type"].lower(), cap,
    ).to_json()


def _render_stress(overrides: Optional[ConfigOverride]) -> str:
    doc = json.loads(json.dumps(STRESS_BASE, separators=(",", ":")))
    if overrides is None:
        return json.dumps(doc, separators=(",", ":"))
    _apply_omits(doc, overrides)
    _retype_dispatcher(doc, overrides)
    _retype_decision(doc, overrides)
    _retype_ordering(doc, overrides)
    for name, path in _STRESS_DOC_PATHS.items():
        value = getattr(overrides, name)
        if value is None or value is OMIT:
            continue
        _edit_doc(doc, path, value, name)
    return json.dumps(doc, separators=(",", ":"))


# ===========================================================================
# Public render API
# ===========================================================================


def render_env(profile: str, overrides: Optional[ConfigOverride] = None) -> str:
    """FLEXLB_CONFIG env string for *profile* layered with *overrides*.

    Layering is one-way: base < profile < override; ``None`` override
    fields keep the profile value, ``OMIT`` removes the key, and every
    non-None override must target a field the profile document already
    carries (unknown-for-profile fields raise ValueError).
    """
    if profile not in _RENDER_PROFILES:
        raise ValueError(
            f"unknown profile {profile!r}; valid: {list(_RENDER_PROFILES)}"
        )
    if profile == STRESS_PROFILE:
        return _render_stress(overrides)
    kwargs = {**FUNCTIONAL_DEFAULTS, **FUNCTIONAL_PROFILE_KWARGS[profile]}
    if overrides is not None:
        validate_profile_identity(profile, {
            axis: getattr(overrides, axis) for axis in PROFILE_SPECS[profile]
        })
        for f in fields(ConfigOverride):
            value = getattr(overrides, f.name)
            if value is None:
                continue
            if value is OMIT:
                if f.name not in _OMITTABLE:
                    raise ValueError(
                        f"ConfigOverride.{f.name}: OMIT is not a legal value "
                        "for this field (the key is always present in the "
                        "base document)"
                    )
                # functional generator: OMIT == the generator's None
                # (key simply not emitted)
                if f.name == "queue_timeout_ms":
                    kwargs["queue_timeout_ms"] = None
                elif f.name == "preemption":
                    kwargs["preemption"] = None
                continue
            if f.name == "strip_preemption":
                # functional bases carry no preemption block unless the
                # override itself set one — drop it here
                if overrides.strip_preemption:
                    kwargs["preemption"] = None
                continue
            if f.name == "decode_max_engine_requests":
                kwargs["decode_max_engine_requests"] = value
                continue
            kwargs[f.name] = value
    return build_flexlb_config(**kwargs)


def render_process_config(
    profile: str,
    overrides: Optional[ConfigOverride] = None,
    jvm_heap: str = "32g",
    raw_config: Optional[str] = None,
) -> str:
    """zone_process_setting envelope (the --master-config document).

    Structure and field order mirror the retired
    ``data/config/master_fixed_window.json``: zone_name / global /
    resource_plan (mem SCALAR 57344) / process_info{args, envs}.  The
    envs list carries [FLEXLB_CONFIG, render_env(profile, overrides)]
    and [FLEXLB_JVM_HEAP_SIZE, jvm_heap] — the config string comes from
    the SAME render as render_env, so file and env cannot diverge.

    *raw_config* bypasses the generator entirely (the EnvSpec
    negative-test channel — the atpm strict-reject variants inject
    deliberately-illegal documents).  The raw string is projected into
    the envelope verbatim; harness._master_env injects the SAME string
    into the master env, preserving the single-source/two-projections
    invariant on the bypass path too.
    """
    env_str = raw_config if raw_config is not None else render_env(profile, overrides)
    envelope = {
        "zone_name": "master",
        "zone_process_setting": {
            "global": {},
            "resource_plan": {
                "resources": [
                    {
                        "slot_resources": [
                            {"name": "mem", "type": "SCALAR", "amount": 57344}
                        ]
                    }
                ],
                "meta_tag_list": [],
            },
            "process_info": {
                "args": [],
                "envs": [
                    ["FLEXLB_CONFIG", env_str],
                    ["FLEXLB_JVM_HEAP_SIZE", jvm_heap],
                ],
            },
        },
    }
    # Trailing newline mirrors the retired master_fixed_window.json byte
    # layout (POSIX text file, golden-diff parity).
    return json.dumps(envelope, indent=2) + "\n"


# ===========================================================================
# CLI override parsing (run_stress.py FLEXLB_CONFIG_OVERRIDE)
# ===========================================================================

_INT_FIELDS = frozenset(
    {
        "cache_affinity_max_extra_ttft_ms",
        "cleanup_interval_ms",
        "status_rpc_ms",
        "max_requests",
        "default_priority",
        "queue_timeout_ms",
        "decode_max_engine_requests",
        "decode_max_kv_usage_percent",
        "max_predicted_execution_ms",
        "request_timeout_ms",
        "max_collection_wait_ms",
        "status_stale_after_ms",
        "max_inflight_per_prefill_worker",
    }
)
_FLOAT_FIELDS = frozenset({"decision_lifetime", "cache_affinity_min_prefix_hit_percent"})
_STR_FIELDS = frozenset({"ordering", "decision", "dispatcher", "prefill_expression"})
_BOOL_FIELDS = frozenset({"strip_preemption"})


def parse_overrides(spec: Optional[str]) -> Optional[ConfigOverride]:
    """Parse a ``"k=v,k=v"`` override string (FLEXLB_CONFIG_OVERRIDE).

    Bare ``k`` (no ``=``) is a boolean flag (``k=1``).  Unknown keys raise
    ValueError — the SSOT vocabulary is closed.  Returns None for an
    empty/blank input.
    """
    if spec is None or not spec.strip():
        return None
    known = _INT_FIELDS | _FLOAT_FIELDS | _STR_FIELDS | _BOOL_FIELDS
    kwargs: dict = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, _, raw = item.partition("=")
            key = key.strip()
            raw = raw.strip()
        else:
            key, raw = item, "1"
        if key not in known:
            raise ValueError(
                f"FLEXLB_CONFIG_OVERRIDE: unknown key {key!r}; valid keys: "
                f"{sorted(known)}"
            )
        if key in _INT_FIELDS:
            try:
                kwargs[key] = int(raw)
            except ValueError:
                raise ValueError(
                    f"FLEXLB_CONFIG_OVERRIDE: {key} expects an integer, got {raw!r}"
                ) from None
        elif key in _FLOAT_FIELDS:
            kwargs[key] = float(raw)
        elif key in _BOOL_FIELDS:
            kwargs[key] = raw.strip().lower() in ("1", "true", "yes", "on")
        else:
            kwargs[key] = raw
    if not kwargs:
        return None
    return ConfigOverride(**kwargs)
