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
            if (
                begin["epoch_scope"] != "local_model_order"
                or begin["global_plan"] != "local"
            ):
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
        # Same counter values alone are not proof of global synchronization. A
        # common host-executor interval is an additional audit for this single-host EP run.
        overlap = min(row["end_us"] for row in rows) - max(
            row["execute_start_us"] for row in rows
        )
        if overlap < 0:
            raise ValueError(f"step {step}: no overlapping execution interval")
        complete.append(rows)
    return complete, dict(rank_begin_counts=counts, excluded_drain_tail=tail)


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
                r["intent"] == "prefill" and r["prefill"] == 0
                for rows, _ in entries
                for r in rows
            ),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steady-summary", type=Path)
    parser.add_argument("--allow-drain-tail", action="store_true")
    args = parser.parse_args()
    with args.log.open(errors="strict") as stream:
        steps, metadata = align(
            stream, args.run_id, allow_drain_tail=args.allow_drain_tail
        )
    result = dict(
        metadata,
        run_id=args.run_id,
        completed_steps=len(steps),
        all=summarize(steps),
        note="Engine-local explicit IDs and overlap checked; no synchronized control channel. "
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
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
