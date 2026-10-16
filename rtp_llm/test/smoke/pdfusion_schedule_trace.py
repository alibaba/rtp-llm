"""Validate explicit engine step IDs. Never silently realign ranks by log ordinal."""

import argparse
import collections
import json
import re
from pathlib import Path


def align(lines, run_id, ranks=4, allow_drain_tail=False):
    events = {}
    order = collections.defaultdict(list)
    active = {}
    for line in lines:
        if "MODEL_EXECUTION " in line:
            model = dict(
                re.findall(r"(\w+)=([^\s]+)", line.split("MODEL_EXECUTION ", 1)[1])
            )
            rank = int(model["rank"])
            if rank in active:
                events[(rank, active[rank], "begin")]["model_calls"].append(
                    dict(
                        mode=model["mode"],
                        batch=int(model["batch"]),
                        graph_bs=int(model["graph_bs"]),
                    )
                )
            continue
        if "SCHEDULE_TRACE " not in line:
            continue
        row = dict(re.findall(r"(\w+)=([^\s]+)", line.split("SCHEDULE_TRACE ", 1)[1]))
        if row.get("run") != run_id:
            raise ValueError("trace contains another run identity")
        if row.get("v") != "1" or row.get("event") not in ("begin", "end"):
            raise ValueError("unknown trace schema/event")
        for key, value in list(row.items()):
            if re.fullmatch(r"-?\d+", value):
                row[key] = int(value)
        rank, step, event = row["rank"], row["model_step_id"], row["event"]
        if event == "end" and row["ok"] != 1:
            raise ValueError("failed execution, including in drain tail")
        if rank not in range(ranks) or step < 1:
            raise ValueError("invalid rank/step")
        key = (rank, step, event)
        if key in events:
            raise ValueError(f"duplicate event {key}")
        if event == "begin":
            row["model_calls"] = []
            active[rank] = step
        else:
            active.pop(rank, None)
        events[key] = row
        order[rank].append((step, event))
    if set(order) != set(range(ranks)):
        raise ValueError("missing rank")
    counts = []
    for rank in range(ranks):
        begins = [step for step, event in order[rank] if event == "begin"]
        if begins != list(range(1, max(begins, default=0) + 1)):
            raise ValueError(f"rank {rank}: missing or reordered step IDs")
        expected = [(step, event) for step in begins for event in ("begin", "end")]
        if order[rank] not in (expected, expected[:-1]):
            raise ValueError(f"rank {rank}: missing/reordered begin/end")
        counts.append(len(begins))
    complete = []
    tail = []
    for step in range(1, max(counts) + 1):
        pairs = [
            (events.get((rank, step, "begin")), events.get((rank, step, "end")))
            for rank in range(ranks)
        ]
        if not all(begin and end for begin, end in pairs):
            if not allow_drain_tail or step != max(counts):
                raise ValueError(f"step {step}: incomplete cross-rank execution")
            tail.append(step)
            continue
        rows = []
        for begin, end in pairs:
            if begin["control_epoch"] != end["control_epoch"]:
                raise ValueError("epoch mismatch between begin/end")
            scope, plan = begin["epoch_scope"], begin["global_plan"]
            if scope == "local_model_order":
                if plan != "local":
                    raise ValueError("unexpected local plan")
            elif scope == "global_control":
                if plan not in ("prefill", "decode"):
                    raise ValueError("invalid global plan")
                if (plan == "prefill" and begin["decode"]) or (
                    plan == "decode" and begin["prefill"]
                ):
                    raise ValueError("real batch violates global phase")
                for field in (
                    "ready_prefill",
                    "ready_decode",
                    "oldest_ready_us",
                    "decode_unserved_us",
                    "prepare_us",
                    "control_us",
                    "commit_us",
                ):
                    if begin.get(field, -1) < 0:
                        raise ValueError("missing or invalid global preparation")
                if (
                    begin["prefill"] > begin["ready_prefill"]
                    or begin["decode"] > begin["ready_decode"]
                ):
                    raise ValueError("execution exceeds reservation")
                real = begin["prefill"] + begin["decode"]
                bit = 1 << begin["rank"]
                if (
                    not 0 <= begin["ready_mask"] <= (1 << ranks) - 1
                    or not 1 <= begin["commit_mask"] <= (1 << ranks) - 1
                ):
                    raise ValueError("invalid global masks")
                if bool(begin["ready_mask"] & bit) != bool(begin["ready_prefill"]):
                    raise ValueError("ready mask disagrees with snapshot")
                if bool(begin["commit_mask"] & bit) != bool(real):
                    raise ValueError("commit mask disagrees with execution")
                if begin["fake"] != int(real == 0) or (
                    not real and end["output_tokens"] != 0
                ):
                    raise ValueError("fake participation or output accounting mismatch")
            else:
                raise ValueError("unsupported coordinator mode")
            if (
                end["ok"] != 1
                or not begin["start_us"] <= begin["execute_start_us"] <= end["end_us"]
            ):
                raise ValueError("failed execution or invalid clock order")
            if begin["snapshot_valid"] != 1:
                raise ValueError("missing scheduler observation")
            if (
                begin["committed_prefill"] != begin["prefill"]
                or begin["committed_decode"] != begin["decode"]
            ):
                raise ValueError("scheduler and engine batch differ")
            rows.append(dict(begin, **end))
        if len({row["control_epoch"] for row in rows}) != 1:
            raise ValueError("cross-rank epoch mismatch")
        if len({row["epoch_scope"] for row in rows}) != 1:
            raise ValueError("mixed local/global coordination")
        if complete and rows[0]["control_epoch"] <= complete[-1][0]["control_epoch"]:
            raise ValueError("nonmonotonic control epoch")
        if rows[0]["epoch_scope"] == "global_control":
            for field in ("global_plan", "ready_mask", "commit_mask"):
                if len({row[field] for row in rows}) != 1:
                    raise ValueError("divergent global plan/mask")
        # Same counter values alone are not proof of global synchronization. A
        # common host-executor interval is an additional audit for this single-host EP run.
        overlap = min(row["end_us"] for row in rows) - max(
            row["execute_start_us"] for row in rows
        )
        if overlap < 0:
            raise ValueError(f"step {step}: no overlapping execution interval")
        complete.append(rows)
    return complete, dict(rank_begin_counts=counts, excluded_drain_tail=tail)


def validate_admission_pacing(rows, share, previous):
    fields = (
        "debt_before_scaled",
        "debt_after_scaled",
        "charged_plan",
        "admit_before",
        "decode_debt_us",
    )
    for key in fields:
        if any(key not in row for row in rows) or len({row[key] for row in rows}) != 1:
            raise ValueError("cross-rank admission budget mismatch")
    before, after, charged, admit, debt_us = (rows[0][key] for key in fields)
    credit_fields = ("admit_credit_before_scaled", "admit_credit_after_scaled")
    has_credit = any(key in row for row in rows for key in credit_fields)
    before_credit = after_credit = 0
    if has_credit:
        for key in credit_fields:
            if (
                any(key not in row for row in rows)
                or len({row[key] for row in rows}) != 1
            ):
                raise ValueError("cross-rank admission lookahead mismatch")
        before_credit, after_credit = (rows[0][key] for key in credit_fields)
        if min(before_credit, after_credit) < 0:
            raise ValueError("invalid admission lookahead credit")
    costs = [row.get("last_model_us", -1) for row in rows]
    if any(not 0 <= cost <= 600000000 for cost in costs) or charged not in (0, 1, 2):
        raise ValueError("invalid completed model cost")
    cost = max(costs)
    expected_admit = 0 if charged == 1 else int(before <= before_credit)
    if admit != expected_admit or any(
        row.get("prefill_admit_allowed") != int(admit or not row["ready_decode"])
        for row in rows
    ):
        raise ValueError("admission gate disagrees with Decode budget")
    expected = before
    if charged == 1:
        expected += cost * share
    elif charged == 2:
        credit = cost * (100 - share)
        expected = max(expected - credit, -credit)
    if not any(row["ready_decode"] for row in rows):
        expected = 0
    expected_credit = (
        cost * (100 - share)
        if charged == 2 and any(row["ready_decode"] for row in rows)
        else 0
    )
    if has_credit and after_credit != expected_credit:
        raise ValueError("admission lookahead exceeds completed Decode quantum")
    if after != expected or debt_us != max(0, after) // (100 - share):
        raise ValueError("admission budget cost accounting mismatch")
    epoch = rows[0]["control_epoch"]
    gap = previous is not None and epoch != previous[0] + 1
    if previous is not None and has_credit != previous[4]:
        raise ValueError("admission lookahead schema changed during run")
    if previous is not None and not gap:
        if before != previous[1] or charged != previous[2]:
            raise ValueError("admission budget changed between consecutive epochs")
        expected_before_credit = 0 if charged == 1 else previous[3]
        if before_credit != expected_before_credit:
            raise ValueError("admission lookahead changed between consecutive epochs")
    # IDLE/cancelled zero-work control epochs have no model begin/end. Their
    # explicit epoch gaps are counted; each visible transition is still checked.
    real = any(row.get("prefill", 0) + row["decode"] for row in rows)
    completed = {"prefill": 1, "decode": 2}[rows[0]["global_plan"]] if real else 0
    return (epoch, after, completed, after_credit, has_credit), int(gap)


def validate_cadence(steps, ratio):
    """Replay either policy from reservations and completed real Decode batches."""
    history = collections.deque()
    config = None
    since = None
    phases = {"prefill": 0, "decode": 0}
    reasons = collections.Counter()
    budget_previous, budget_gaps = None, 0
    draining_seen_mask = 0
    for rows in steps:
        configs = {
            (
                r.get("cadence_adaptive", 0),
                r.get("decode_window", 64),
                r.get("low_water_pct", 50),
                r.get("prefill_max_wait_ms", 1000),
                r.get("min_decode_share_pct", 0),
            )
            for r in rows
        }
        if len(configs) != 1:
            raise ValueError("cross-rank cadence configuration mismatch")
        current = configs.pop()
        if config is not None and config != current:
            raise ValueError("cadence configuration changed during run")
        config = current
        adaptive, window, pct, wait_ms, share = config
        if (
            adaptive not in (0, 1)
            or not 1 <= window <= 1000000
            or not 1 <= pct <= 100
            or not 0 <= wait_ms <= 600000
            or not 0 <= share <= 99
        ):
            raise ValueError("invalid cadence configuration")
        prefill = any(r["ready_prefill"] for r in rows)
        decode = any(r["ready_decode"] for r in rows)
        if adaptive:
            if any("local_draining" in r or "drain_mask" in r for r in rows):
                if any(
                    r.get("local_draining") not in (0, 1) or "drain_mask" not in r
                    for r in rows
                ):
                    raise ValueError("invalid or missing drain evidence")
                drain_mask = sum(1 << r["rank"] for r in rows if r["local_draining"])
                if any(r["drain_mask"] != drain_mask for r in rows):
                    raise ValueError("cross-rank drain mask mismatch")
            else:
                drain_mask = 0
            if drain_mask & draining_seen_mask != draining_seen_mask:
                raise ValueError("drain signal regressed")
            draining_seen_mask |= drain_mask
            if share:
                budget_previous, gap = validate_admission_pacing(
                    rows, share, budget_previous
                )
                budget_gaps += gap
            peaks = [
                max((sample[i] for sample in history), default=r["ready_decode"])
                for i, r in enumerate(rows)
            ]
            low_mask = sum(
                1 << r["rank"]
                for i, r in enumerate(rows)
                if not r["ready_decode"] or r["ready_decode"] * 100 <= peaks[i] * pct
            )
            if all(r["ready_prefill"] for r in rows):
                expected, reason = "prefill", "all_ready"
            elif prefill and low_mask:
                expected, reason = "prefill", "decode_low"
            elif prefill and drain_mask:
                expected, reason = "prefill", "drain"
            elif (
                prefill
                and wait_ms > 0
                and max(r["oldest_ready_us"] for r in rows if r["ready_prefill"])
                >= wait_ms * 1000
            ):
                expected, reason = "prefill", "prefill_timeout"
            else:
                expected = "decode" if decode else "idle"
                reason = (
                    ("accumulate" if prefill else "decode_only") if decode else "idle"
                )
            for i, r in enumerate(rows):
                if (
                    r.get("decode_peak") != peaks[i]
                    or r.get("low_decode_mask") != low_mask
                    or r.get("cadence_reason") != reason
                ):
                    raise ValueError("adaptive cadence evidence disagrees with replay")
        else:
            if since is None:
                since = int(ratio)
            expected = (
                "prefill"
                if prefill and (not decode or since >= int(ratio))
                else "decode"
            )
            reason = "ratio"
        plan = rows[0]["global_plan"]
        if any(
            r["epoch_scope"] != "global_control" or r["global_plan"] != expected
            for r in rows
        ):
            raise ValueError("cadence trace violates scheduling policy")
        phases[plan] += 1
        reasons[reason] += 1
        if not adaptive:
            since = 0 if plan == "prefill" else min(since + 1, int(ratio))
        if adaptive and plan == "decode" and any(r["decode"] for r in rows):
            history.append(tuple(r["decode"] for r in rows))
            while len(history) > window:
                history.popleft()
    return dict(
        phases=phases,
        reasons=dict(reasons),
        policy="adaptive" if config and config[0] else "ratio",
        decode_window=config[1] if config else 64,
        low_water_pct=config[2] if config else 50,
        prefill_max_wait_ms=config[3] if config else 1000,
        min_decode_share_pct=config[4] if config else 0,
        budget_continuity_gaps=budget_gaps,
    )


def summarize(steps, lo_us=0, hi_us=float("inf")):
    groups = collections.defaultdict(list)
    for rows in steps:
        if (
            min(row["end_us"] for row in rows) < lo_us
            or max(row["end_us"] for row in rows) >= hi_us
        ):
            continue
        count = sum(row["prefill"] > 0 for row in rows)
        groups["prefill_present" if count else "pure_decode"].append((rows, count))
    result = {}
    for kind, entries in groups.items():
        durations = [
            max(r["end_us"] - r["execute_start_us"] for r in rows) / 1e6
            for rows, _ in entries
        ]
        result[kind] = dict(
            steps=len(entries),
            max_rank_host_executor_sum_s=sum(durations),
            mean_host_executor_ms=1000 * sum(durations) / len(durations),
            prefill_rank_hist=dict(collections.Counter(count for _, count in entries)),
            output_tokens=sum(r["output_tokens"] for rows, _ in entries for r in rows),
            decode=sum(r["decode"] for rows, _ in entries for r in rows),
            fake=sum(r["fake"] for rows, _ in entries for r in rows),
            observed_graph_hist=dict(
                collections.Counter(
                    call["graph_bs"]
                    for rows, _ in entries
                    for r in rows
                    for call in r["model_calls"]
                    if call["mode"] == "graph"
                )
            ),
            model_calls_unobserved=sum(
                not r["model_calls"] for rows, _ in entries for r in rows
            ),
            real_model_calls_unobserved=sum(
                not r["model_calls"] and (r["prefill"] + r["decode"] > 0)
                for rows, _ in entries
                for r in rows
            ),
            rejected_batch=sum(r["reject_batch"] for rows, _ in entries for r in rows),
            rejected_tokens=sum(
                r["reject_tokens"] for rows, _ in entries for r in rows
            ),
            rejected_kv=sum(r["reject_kv"] for rows, _ in entries for r in rows),
            intent_prefill_fallback=sum(
                r["epoch_scope"] == "local_model_order"
                and r["intent"] == "prefill"
                and r["prefill"] == 0
                for rows, _ in entries
                for r in rows
            ),
            global_empty_ranks=sum(
                r["epoch_scope"] == "global_control" and r["prefill"] + r["decode"] == 0
                for rows, _ in entries
                for r in rows
            ),
            global_steps=sum(
                rows[0]["epoch_scope"] == "global_control" for rows, _ in entries
            ),
            max_rank_control_sum_s=sum(
                max(r.get("control_us", 0) for r in rows) for rows, _ in entries
            )
            / 1e6,
            oldest_ready_max_us=max(
                (r.get("oldest_ready_us", 0) for rows, _ in entries for r in rows),
                default=0,
            ),
            decode_unserved_max_us=max(
                (r.get("decode_unserved_us", 0) for rows, _ in entries for r in rows),
                default=0,
            ),
            admission_throttled_snapshots=sum(
                r.get("prefill_admit_allowed", 1) == 0
                for rows, _ in entries
                for r in rows
            ),
            decode_debt_max_us=max(
                (r.get("decode_debt_us", 0) for rows, _ in entries for r in rows),
                default=0,
            ),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ranks", type=int, choices=(2, 4), default=4)
    parser.add_argument("--steady-summary", type=Path)
    parser.add_argument("--allow-drain-tail", action="store_true")
    parser.add_argument(
        "--cadence-ratio",
        type=int,
        help="Replay global cadence policy (ratio ignored by adaptive mode)",
    )
    args = parser.parse_args()
    with args.log.open(errors="strict") as stream:
        steps, metadata = align(
            stream,
            args.run_id,
            ranks=args.ranks,
            allow_drain_tail=args.allow_drain_tail,
        )
    result = dict(
        metadata,
        run_id=args.run_id,
        completed_steps=len(steps),
        all=summarize(steps),
        note="Explicit model/control IDs, epoch scope, global plan/masks when present and overlap checked. "
        "Host executor time is not GPU kernel or peer-wait time. Graph bins come only from MODEL_EXECUTION records inside explicit begin/end brackets; missing records stay unobserved.",
    )
    if args.steady_summary:
        result["phases"] = {
            name: summarize(
                steps,
                (s["started_at"] + s["measurement"]["start_s"]) * 1e6,
                (s["started_at"] + s["measurement"]["end_s"]) * 1e6,
            )
            for name, s in json.loads(args.steady_summary.read_text()).items()
        }
    if args.cadence_ratio is not None:
        result["cadence"] = validate_cadence(steps, args.cadence_ratio)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
