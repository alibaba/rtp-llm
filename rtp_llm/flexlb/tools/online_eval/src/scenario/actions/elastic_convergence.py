"""Pure, generation-aware interpretation of crossfire request and mutation records."""

import math


def convergence_bound(config, margin_s, cap_s):
    health = config["workerRegistry"]["health"]
    bound = (
        health["statusStaleAfterMs"] + health["cleanupIntervalMs"]
    ) / 1000 + margin_s
    if not math.isfinite(bound) or bound > cap_s:
        raise ValueError(
            f"configuration drift: derived convergence bound {bound}s exceeds cap {cap_s}s"
        )
    return bound


def dead_interval(operations, address, issued_s):
    """Only exact advertised addresses match; a later successful add revives them."""
    if not address:
        return None
    events = sorted(
        (
            op
            for op in operations
            if op.get("ok")
            and op.get("address") == address
            and op["ended_s"] <= issued_s
        ),
        key=lambda op: op["ended_s"],
    )
    if events and events[-1]["operation"] == "remove":
        return events[-1]["ended_s"]
    return None


def evaluate_convergence(
    rows,
    operations,
    mutation_end_s,
    bound_s,
    *,
    batch,
    min_post_samples,
    nonbatch_failfast_s,
    batch_failfast_s,
):
    removals = [op for op in operations if op.get("ok") and op["operation"] == "remove"]
    missing = [op for op in removals if not op.get("address")]
    # Include graceful removes that returned after the mutation issuance window.
    quiet_at = max([mutation_end_s] + [op["ended_s"] for op in removals])
    cutoff = quiet_at + bound_s
    post = [r for r in rows if r["issued_s"] >= cutoff]
    post_routed = [r for r in post if r.get("prefill_addr")]
    dead_hits, slow, observed = [], [], []
    unknown_batch_failures = []
    for r in rows:
        removed_at = dead_interval(operations, r.get("prefill_addr"), r["issued_s"])
        if removed_at is not None:
            dead_hits.append(
                dict(
                    request_id=r["wire_request_id"],
                    issued_s=r["issued_s"],
                    removed_at_s=removed_at,
                    address=r["prefill_addr"],
                )
            )
        schedule = r["schedule"]
        # Failed BATCH Schedule has no returned endpoint. Evaluate ALL such
        # failures conservatively, without claiming dead-engine attribution.
        batch_failure = batch and schedule["status"] not in ("OK", None)
        if batch_failure:
            unknown_batch_failures.append(r["wire_request_id"])
        if removed_at is None and not batch_failure:
            continue
        observed.append(r["wire_request_id"])
        started = schedule.get("started_s")
        terminal = schedule.get("ended_s") if batch else r.get("transport_terminal_s")
        limit = batch_failfast_s if batch else nonbatch_failfast_s
        failed = (
            schedule["status"] not in ("OK", None)
            if batch
            else r["stream"]["status"] == "UNAVAILABLE"
        )
        # A deadline/hang is never an explicit fail-fast result, even if a
        # shortened external timeout makes its latency fit the band.
        if (
            not failed
            or started is None
            or terminal is None
            or terminal - started > limit
            or (batch and schedule["status"] in ("DEADLINE_EXCEEDED", "CANCELLED"))
        ):
            slow.append(r["wire_request_id"])
    late = [
        hit for hit in dead_hits if hit["issued_s"] >= hit["removed_at_s"] + bound_s
    ]
    return dict(
        bound_s=bound_s,
        quiet_at_s=quiet_at,
        cutoff_s=cutoff,
        successful_removals=len(removals),
        missing_address_operations=missing,
        post_samples=len(post),
        post_routed_samples=len(post_routed),
        coverage_ok=bool(removals)
        and not missing
        and len(post_routed) >= min_post_samples,
        dead_hits=dead_hits,
        late_dead_hits=late,
        failfast_samples=len(observed),
        failfast_violations=slow,
        unattributed_batch_failures=unknown_batch_failures,
    )
