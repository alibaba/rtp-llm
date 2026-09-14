"""Fail closed when a scheduled smoke scenario was not actually exercised.

These events describe engine plans/rounds, not per-request numerical accuracy.
The request runner independently validates answers and owner routing.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def ordered_subsequence(values, expected):
    position = 0
    for value in values:
        if value == expected[position]:
            position += 1
            if position == len(expected):
                return True
    return False


def _verify(events: list[dict], role: str, replay_seen: bool = False) -> dict:
    checks = {}
    if role == "prefill":
        rounds = [e for e in events if e.get("kind") == "chunk"]
        checks["chunk_rounds_present"] = bool(rounds)
        checks["tp8_padding_valid"] = bool(rounds) and all(
            e["tp"] == 8
            and e["physical_tokens"] % 8 == 0
            and e["physical_tokens"] - e["logical_tokens"] == (-e["logical_tokens"]) % 8
            and e["physical_requests"] - e["logical_requests"]
            == int(e["physical_tokens"] > e["logical_tokens"])
            for e in rounds
        )
        for padding in (1, 7):
            checks[f"actual_padding_{padding}"] = any(
                e["physical_tokens"] - e["logical_tokens"] == padding for e in rounds
            )
    else:
        plans = [e for e in events if e.get("kind") == "ktp"]
        # Logs may be mirrored into service.log and main_<rank>.log. Deduplicate
        # by rank/step, checking contradictory copies rather than ignoring them.
        unique = {}
        consistent_copies = True
        for e in plans:
            key = (e["rank"], e["step"])
            if key in unique and unique[key] != e:
                consistent_copies = False
            unique[key] = e
        checks["consistent_log_copies"] = consistent_copies
        checks["all_ranks_reported"] = {r for r, _ in unique} == set(range(8))
        steps = {}
        for (rank, step), event in unique.items():
            steps.setdefault(step, {})[rank] = event
        checks["all_ranks_agree_each_step"] = bool(steps) and all(
            set(ranks) == set(range(8))
            and len(
                {
                    json.dumps(
                        {k: v for k, v in e.items() if k != "rank"}, sort_keys=True
                    )
                    for e in ranks.values()
                }
            )
            == 1
            for ranks in steps.values()
        )
        canonical = [e for (rank, _), e in sorted(unique.items()) if rank == 0]
        checks["valid_plan_shapes"] = bool(canonical) and all(
            len(e["valid"]) == 8
            and min(e["valid"]) >= 0
            and e["physical"] >= max(e["valid"])
            and (not e["graph"] or e["physical"] == e["bucket"] in (1, 2, 4, 8))
            for e in canonical
        )
        checks["all_owners_active"] = all(
            any(e["valid"][r] > 0 for e in canonical) for r in range(8)
        )
        checks["rank0_idle_rank7_active"] = any(
            e["valid"][:7] == [0] * 7 and e["valid"][7] > 0 for e in canonical
        )
        graph = [e for e in canonical if e["graph"] and e["mode"] == "TARGET_VERIFY"]
        for bucket in (1, 2, 4, 8):
            checks[f"target_verify_bucket_{bucket}"] = any(
                e["bucket"] == bucket for e in graph
            )
        checks["bucket8_slot_reuse_7_5_6"] = ordered_subsequence(
            [e["valid"][7] for e in graph if e["bucket"] == 8], [7, 5, 6]
        )
        checks["graph_replay_observed"] = replay_seen
    return {
        "role": role,
        "passed": all(checks.values()),
        "checks": checks,
        "event_count": len(events),
    }


def verify(events: list[dict], role: str, replay_seen: bool = False) -> dict:
    try:
        return _verify(events, role, replay_seen)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return {
            "role": role,
            "passed": False,
            "checks": {"valid_event_schema": False},
            "error": str(exc),
            "event_count": len(events),
        }


def collect(root: pathlib.Path):
    events, replay_seen = [], False
    # Explicit service/rank paths avoid recursively ingesting saved evidence.
    paths = (
        list(root.glob("*.log"))
        + list(root.glob("runtime/logs/*/main_*.log"))
        + list(root.glob("runtime/work/*/logs/engine.log"))
    )
    for path in sorted(set(paths)):
        with path.open(errors="replace") as source:
            for line in source:
                replay_seen |= "[K3_PROJECTION_KTP_GRAPH_REPLAY]" in line
                if "[K3_SMOKE_EVENT] " in line:
                    event = json.loads(line.split("[K3_SMOKE_EVENT] ", 1)[1])
                    events.append(event)
    return events, replay_seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        events, replay = collect(args.root)
        report = verify(events, args.role, replay)
    except (ValueError, OSError) as exc:
        report = {
            "role": args.role,
            "passed": False,
            "checks": {"readable_events": False},
            "error": str(exc),
        }
    (args.root / "smoke-runtime-coverage.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(
            "smoke runtime coverage incomplete: "
            + ", ".join(k for k, v in report["checks"].items() if not v)
        )


if __name__ == "__main__":
    main()
