"""One functional recovery cycle with a bounded synthetic ring cache."""

from case_api.builder import output
from case_api.input_contract import InputContract
from case_api.numeric_parameters import COUNT, POSITIVE_COUNT, NONNEGATIVE
from case_api.registry import CaseDefinition
from cases.cache_sync_recovery.actions import HANDLERS

INPUTS = InputContract.from_fields({
    "procedure": {
        "setup_timeout_s": NONNEGATIVE, "warmup_timeout_s": NONNEGATIVE,
        "stable_s": NONNEGATIVE, "churn_s": NONNEGATIVE, "withdraw_s": NONNEGATIVE,
        "recovery_timeout_s": NONNEGATIVE, "unique_keys": POSITIVE_COUNT,
        "seed_interval_s": NONNEGATIVE,
        "shared_keys": COUNT, "delta_keys": POSITIVE_COUNT,
        "change_every_responses": POSITIVE_COUNT, "expired_workers": POSITIVE_COUNT,
    },
    "observation": {"sample_s": NONNEGATIVE, "max_gap_s": NONNEGATIVE,
                    "converged_samples": POSITIVE_COUNT},
    "checks": {"max_status_age_ms": NONNEGATIVE, "min_cache_responses_per_s": NONNEGATIVE},
})


def default(case):
    data = case.bind(INPUTS)
    for section in (data.procedure, data.observation, data.checks):
        for key, value in section.items():
            if key not in {"shared_keys", "seed_interval_s"} and value <= 0:
                raise ValueError("recovery parameters must be positive: " + key)
    if case.env.discovery != "discovery_file":
        raise ValueError("cache recovery requires owned file discovery")
    if case.env.n_prefill < 3 or not 0 < data.procedure["expired_workers"] < case.env.n_prefill:
        raise ValueError("cache recovery requires a proper subset of at least three Prefill Workers")
    if (data.procedure["unique_keys"] + 2 * data.procedure["shared_keys"] > 1_000_000
            or data.procedure["delta_keys"] > data.procedure["unique_keys"]):
        raise ValueError("invalid fixture key/diff budget")
    if not data.observation["sample_s"] <= data.observation["max_gap_s"] < data.checks["max_status_age_ms"] / 1000:
        raise ValueError("sampling gap must be below the health stale threshold")
    for profile in case.env.profiles:
        health = case.env.resolved_config(profile)["workerRegistry"]["health"]
        if data.checks["max_status_age_ms"] > health["statusStaleAfterMs"]:
            raise ValueError("health gate cannot be looser than configured Worker expiry")
    case.step("setup", "setup", timeout_s=data.procedure["setup_timeout_s"])
    budget = (data.procedure["warmup_timeout_s"] + data.procedure["stable_s"]
              + data.procedure["churn_s"] + data.procedure["withdraw_s"]
              + data.procedure["recovery_timeout_s"]
              + (case.env.n_prefill - 1) * data.procedure["seed_interval_s"] + 120)
    case.step("cycle", "cache_recovery_cycle", timeout_s=budget,
              params={"procedure": data.procedure, "observation": data.observation,
                      "checks": data.checks})
    case.step("submit", "request", params={"count": 1, "input_len": 16, "output_len": 2})
    case.step("terminal", "wait", params={"requests": output("submit", "requests")})
    case.step("request_completed", "check", params={
        "actual": output("terminal", "completed"), "op": "eq", "expected": True})
    case.step("request_no_errors", "check", params={
        "actual": output("terminal", "error_count"), "op": "eq", "expected": 0})
    case.step("cleanup", "teardown")


CASE = CaseDefinition(builders={"default": default}, input_contract=INPUTS, actions=tuple(HANDLERS))
