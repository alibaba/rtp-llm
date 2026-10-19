"""Adjudicate frozen cache pressure evidence without acquisition or presentation."""

from analysis.checks import CheckResult, evaluate, invalid_check
from analysis.gates import gate_checks

VALIDITY_CHECKS = frozenset({"actual_sync_intensity", "response_shape", "clock_input_intensity",
                             "sampling_complete", "client_complete"})
CHECK_IDS = frozenset({"healthy_survivors", "victims_withdrawn", "victims_rebuilt", "cache_converged"}) | VALIDITY_CHECKS


def analyze(evidence):
    affected = len(evidence.get("affected_workers", []))
    checks = [evaluate("healthy_survivors", affected, "eq", 0)]
    errors = list(evidence.get("errors", []))
    if evidence.get("runtime_incomplete") or errors:
        checks.append(invalid_check("sampling_complete", detail="; ".join(errors) or "observation interrupted"))
    else:
        p = evidence["parameters"]
        limits, procedure, observation = (p[k] for k in ("checks", "procedure", "observation"))
        last = evidence["frames"][-1]["workers"]
        keys = procedure["unique_keys"] + 2*procedure["shared_keys"]
        progress = evidence["cache_progress"]
        from cases.cache_sync_recovery.health import converged
        ready = (converged(last, set(evidence["worker_addresses"]), keys,
                           progress["expected_versions"], limits["max_status_age_ms"])
                 and progress["converged"]
                 and evidence["convergence_samples"] >= observation["converged_samples"])
        measured = evidence["intensity"]["measurement"]
        checks += [evaluate("cache_converged", ready, "eq", True),
                   evaluate("actual_sync_intensity", measured["responses"], "ge", limits["min_cache_responses_per_s"]),
                   evaluate("response_shape", measured["totals"]["entries"] == measured["totals"]["responses"]*keys
                            and measured["totals"]["full_responses"] == measured["totals"]["responses"], "eq", True),
                   evaluate("clock_input_intensity", limits["min_added_keys_per_s"] <= measured["added_keys"]
                            <= limits["max_added_keys_per_s"], "eq", True),
                   evaluate("sampling_complete", evidence["max_sample_gap_s"], "le", observation["max_gap_s"]),
                   evaluate("client_complete", evidence["client"]["complete"], "eq", True)]
        if evidence["intervention"]:
            for identity, field in (("victims_withdrawn", "expired_workers"), ("victims_rebuilt", "rebuilt_workers")):
                checks.append(evaluate(identity, len(evidence[field]), "eq", len(evidence["victims"])))
        else:
            checks += [CheckResult(id=k, status="SKIP", detail="steady has no withdrawal event")
                       for k in ("victims_withdrawn", "victims_rebuilt")]
    # Insufficient offered pressure/coverage is invalid evidence, not a product failure.
    checks = [invalid_check(c.id, detail="workload evidence requirement not satisfied", actual=c.actual,
                            expected=c.expected) if c.id in VALIDITY_CHECKS and c.status == "FAIL" else c
              for c in checks]
    present = {c.id for c in checks}
    checks += [CheckResult(id=key, status="SKIP", detail="observation interrupted") for key in sorted(CHECK_IDS-present)]
    verdict, rows = gate_checks(checks, [])
    validity = "INVALID" if any(c.status == "ERROR" for c in checks) else "VALID"
    return dict(verdict=verdict, validity=validity, checks=rows, errors=errors,
                functional_checks=[c for c in rows if c["id"] not in VALIDITY_CHECKS],
                validity_checks=[c for c in rows if c["id"] in VALIDITY_CHECKS])
