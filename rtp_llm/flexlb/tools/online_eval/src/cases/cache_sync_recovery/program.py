"""Shared pressure background; only the intervention and its checks vary."""

from case_api.input_contract import InputContract
from case_api.numeric_parameters import COUNT, POSITIVE_COUNT, NONNEGATIVE
from case_api.registry import CaseDefinition
from adapters.java_flow_spec import JavaFlowSpec
from cases.cache_sync_recovery.actions import HANDLERS

INPUTS = InputContract.from_fields({
    "procedure": dict.fromkeys(("setup_timeout_s", "warmup_timeout_s", "cache_only_s",
        "traffic_warmup_s", "measurement_s", "withdraw_s", "post_s", "seed_interval_s", "baseline_interval_s"), NONNEGATIVE)
        | dict.fromkeys(("unique_keys", "delta_keys", "churn_interval_ms", "expired_workers", "baseline_steps"), POSITIVE_COUNT)
        | {"shared_keys": COUNT},
    "observation": {"sample_s": NONNEGATIVE, "max_gap_s": NONNEGATIVE, "converged_samples": POSITIVE_COUNT},
    "checks": dict.fromkeys(("max_status_age_ms", "min_cache_responses_per_s",
        "min_added_keys_per_s", "max_added_keys_per_s"), NONNEGATIVE),
    "traffic": JavaFlowSpec.input_fields(),
})


def build(case, intervention):
    data = case.bind(INPUTS)
    p, o, c = data.procedure, data.observation, data.checks
    if any(v <= 0 for k, v in p.items() if k not in {"shared_keys", "seed_interval_s"}):
        raise ValueError("pressure windows and key parameters must be positive")
    if case.env.n_prefill < 3 or case.env.discovery != "discovery_file" or not 0 < p["expired_workers"] < case.env.n_prefill:
        raise ValueError("partial withdrawal requires owned discovery and a proper Prefill subset")
    if p["unique_keys"] + 2*p["shared_keys"] > 1_000_000 or p["delta_keys"] > p["unique_keys"]:
        raise ValueError("invalid fixture key budget")
    if not 0 < o["sample_s"] <= o["max_gap_s"] <= c["max_status_age_ms"]/1000:
        raise ValueError("sampling gap must not exceed health stale threshold")
    if not 0 < c["min_added_keys_per_s"] < c["max_added_keys_per_s"]:
        raise ValueError("invalid actual mutation envelope")
    expected = case.env.n_prefill*p["delta_keys"]*1000/p["churn_interval_ms"]
    if not c["min_added_keys_per_s"] <= expected <= c["max_added_keys_per_s"]:
        raise ValueError("mutation envelope excludes the configured input")
    flow = JavaFlowSpec.read(data.traffic)
    seconds = p["traffic_warmup_s"] + p["measurement_s"] + p["withdraw_s"] + p["post_s"]
    if flow.configuration.playback.duration_s < seconds:
        raise ValueError("traffic must cover intervention and the full post window")
    case.step("setup", "setup", timeout_s=p["setup_timeout_s"])
    budget = p["warmup_timeout_s"] + p["cache_only_s"] + seconds + case.env.n_prefill*p["seed_interval_s"] + p["baseline_steps"]*p["baseline_interval_s"] + 360
    case.step("cycle", "cache_recovery_cycle", timeout_s=budget,
        params={"procedure": p, "observation": o, "checks": c, "traffic": data.traffic,
                "intervention": intervention})
    case.step("cleanup", "teardown")


def default(case):
    build(case, False)


def recovery(case):
    build(case, True)


CASE = CaseDefinition(builders={"default": default, "recovery": recovery},
                      input_contract=INPUTS, actions=tuple(HANDLERS))
