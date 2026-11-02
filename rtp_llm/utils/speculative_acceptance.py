"""Aggregate cumulative owner-root snapshots without summing stream/TP copies."""

from __future__ import annotations

import time


COUNTERS = ("proposal_rounds", "proposal_candidates", "verify_rounds", "verified_candidates",
            "accepted_draft_tokens", "generated_tokens", "excluded_forced_rounds",
            "accept_length_0", "accept_length_1", "accept_length_2", "accept_length_3")


def checked_counters(metrics):
    if not metrics:
        return {key: 0 for key in COUNTERS}
    if metrics.get("version") != 1:
        raise ValueError("unknown acceptance accounting contract")
    result = {key: metrics.get(key, 0) for key in COUNTERS}
    if any(type(value) is not int or value < 0 for value in result.values()):
        raise ValueError("acceptance counters must be nonnegative integers")
    rounds = result["verify_rounds"]
    accepted = result["accepted_draft_tokens"]
    valid = result["verified_candidates"]
    histogram = [result[f"accept_length_{n}"] for n in range(4)]
    if (sum(histogram) != rounds or sum(n * count for n, count in enumerate(histogram)) != accepted
            or accepted > valid or valid > result["proposal_candidates"]
            or accepted > result["generated_tokens"] or rounds != result["proposal_rounds"]):
        raise ValueError("inconsistent acceptance accounting")
    return result


def summary(counters):
    result = dict(counters)
    valid, rounds = result["verified_candidates"], result["verify_rounds"]
    result["acceptance_rate"] = result["accepted_draft_tokens"] / valid if valid else None
    result["mean_accept_length"] = result["accepted_draft_tokens"] / rounds if rounds else None
    result["accept_length_fractions"] = {
        str(n): result[f"accept_length_{n}"] / rounds if rounds else None for n in range(4)
    }
    return result


class AcceptanceMonitor:
    """The caller polls snapshots from existing output messages every five seconds.

    This object owns no thread, timer, GPU operation, service, or log loop.
    Request generation speed includes target tokens; A/V excludes bonus tokens.
    Cycle wall time is explicitly amortized from a fully observed request's
    first-token to final-response wall interval, not summed kernel durations.
    """

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = self.last_window = clock()
        self.requests = {}
        self.window_counters = {key: 0 for key in COUNTERS}

    def observe(self, request_id, owner, metrics, output_tokens, *, first_token=False, finished=False):
        if not isinstance(request_id, str) or not request_id or not isinstance(owner, str) or not owner:
            raise ValueError("request identity and authoritative Decode owner are required")
        if type(output_tokens) is not int or output_tokens < 0:
            raise ValueError("invalid actual generated token count")
        now = self.clock()
        current = checked_counters(metrics)
        previous = self.requests.get(request_id)
        if previous:
            if previous["owner"] != owner:
                raise ValueError("request appeared on two owners; reconcile routing instead of double counting")
            if any(current[key] < previous["counters"][key] for key in COUNTERS):
                raise ValueError("cumulative counters regressed or service restarted")
            if output_tokens < previous["actual_generated_tokens"]:
                raise ValueError("generated token count regressed")
            initial = previous["first_token_time"]
            ended = previous["finished_at"]
        else:
            initial = now if first_token and output_tokens == 1 and current["verify_rounds"] == 0 else None
            ended = None
        self.requests[request_id] = dict(owner=owner, counters=current, actual_generated_tokens=output_tokens,
                                         first_token_time=initial, finished_at=ended or (now if finished else None),
                                         last_observed_at=now)

    def report(self, *, reset_window=False):
        now = self.clock()
        total = {key: 0 for key in COUNTERS}
        by_owner = {}
        by_request = {}
        for request_id, request in self.requests.items():
            counters = request["counters"]
            owner = by_owner.setdefault(request["owner"], {key: 0 for key in COUNTERS})
            for key in COUNTERS:
                total[key] += counters[key]
                owner[key] += counters[key]
            row = summary(counters)
            row["actual_generated_tokens"] = request["actual_generated_tokens"]
            row["owner"] = request["owner"]
            first, end = request["first_token_time"], request["finished_at"]
            span = end - first if end is not None and first is not None else None
            row["decode_wall_seconds"] = span
            row["amortized_complete_cycle_wall_seconds"] = (
                span / counters["verify_rounds"] if span is not None and counters["verify_rounds"] else None)
            row["generation_tokens_per_second"] = (
                (request["actual_generated_tokens"] - 1) / span if span is not None and span > 0 else None)
            by_request[request_id] = row
        window = {key: total[key] - self.window_counters[key] for key in COUNTERS}
        value = dict(overall=summary(total), by_owner={key: summary(row) for key, row in by_owner.items()},
                     by_request=by_request, window=summary(window), window_seconds=now - self.last_window,
                     elapsed_seconds=now - self.started, actual_generated_tokens=sum(
                         row["actual_generated_tokens"] for row in self.requests.values()))
        if reset_window:
            self.last_window, self.window_counters = now, total
        return value
