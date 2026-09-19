"""Fail closed when a scheduled smoke scenario was not actually exercised.

These events describe engine plans/rounds, not per-request numerical accuracy.
The request runner independently validates answers and owner routing.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re


# Decode Page-RR (DCP) is KTP1, so it emits none of the Projection-KTP plan
# events. Its runtime paths are proved by engine markers instead: every TP rank
# builds the A2A communicator, the local cache geometry is validated once, and
# the C++ graph runner captures the scheduled buckets per rank.
_MLA_DCP = re.compile(r"\[MLA_DCP\] backend=(\w+) tp=(\d+) rank=(\d+)")
_PAGE_RR_TARGET = re.compile(
    r"\[K3_PAGE_RR_TARGET\] role=(\w+) TP=(\d+) B=(\d+) V=(\d+)"
)
_GRAPH_CAPTURE = re.compile(r"captured batch size (\d+):")
_RANK_PREFIX = re.compile(r"\[RANK (\d+)\]")
_MAIN_LOG = re.compile(r"main_(\d+)\.log")
GRAPH_BUCKETS = (1, 2, 4, 8)


def physical_graph_buckets(tp: int, proposal_tokens: int = 3) -> tuple[int, ...]:
    """Mirror request alignment in CudaGraphRunner::getDecodeBatchSizesToCapture."""
    if tp < 1 or proposal_tokens < 1:
        raise ValueError("TP and proposal token count must be positive")
    buckets = set()
    for width in (1, proposal_tokens + 1):
        alignment = tp // math.gcd(tp, width)
        buckets.update(
            ((size + alignment - 1) // alignment) * alignment
            for size in GRAPH_BUCKETS
        )
    return tuple(sorted(buckets))


def adjacent_unique(values):
    return [
        value
        for index, value in enumerate(values)
        if index == 0 or value != values[index - 1]
    ]


def _dcp_checks(markers: dict, replay_seen: bool, events: list[dict], proposal_tokens: int = 3, expected_tp: int | None = None, dp_size: int = 1, expected_block_size: int | None = None, source_tp: int | None = None) -> dict:
    """Prove the Decode Page-RR path ran on every rank of a KTP1 topology."""
    backends = markers.get("dcp_backends", set())
    sizes = {tp for tp, _ in backends}
    ranks = {rank for _, rank in backends}
    tp = min(sizes) if sizes else None
    world = (expected_tp or tp or 0) * dp_size
    checks = {
        "dcp_backend_a2a_single_size": len(sizes) == 1 and (expected_tp is None or sizes == {expected_tp}),
        "dcp_communicator_all_ranks": tp is not None and ranks == set(range(tp)),
    }
    if dp_size > 1:
        checks["dcp_communicator_all_workers"] = markers.get("dcp_workers", set()) == {
            (rank, rank // (expected_tp or tp), rank % (expected_tp or tp)) for rank in range(world)
        }
    targets = markers.get("page_rr_targets", set())
    decode_targets = {
        (tp, pages, checkpoints)
        for role_, tp, pages, checkpoints in targets
        if role_ == "Decode"
    }
    checks["page_rr_target_decode_geometry"] = (
        all(role_ == "Decode" for role_, _, _, _ in targets)
        and len(decode_targets) == 1
        and all(
            pages > 0 and checkpoints > 0 and {tp} == sizes
            and (expected_block_size is None or (pages == expected_block_size and checkpoints == max(tp, source_tp or tp) * pages))
            for tp, pages, checkpoints in decode_targets
        )
    )
    captures = markers.get("graph_captures", set())
    expected_buckets = physical_graph_buckets(tp or 8, proposal_tokens)
    checks["graph_capture_buckets"] = all(
        any(bucket == expected for _, bucket in captures) for expected in expected_buckets
    )
    checks["graph_capture_all_ranks"] = tp is not None and all(
        {rank for rank, bucket in captures if bucket == expected and rank is not None}
        == set(range(world))
        for expected in expected_buckets
    )
    # A KTP>1 topology must not silently satisfy a DCP round.
    checks["projection_ktp_inactive"] = not replay_seen and not any(
        event.get("kind") == "ktp" for event in events
    )
    observations = {
        "dcp_tp": tp,
        "dcp_ranks": sorted(ranks),
        "page_rr_targets": sorted(
            f"{role_}:TP{tp}:B{pages}:V{checkpoints}"
            for role_, tp, pages, checkpoints in targets
        ),
        "graph_capture_ranks_per_bucket": {
            str(expected): sorted(
                rank for rank, bucket in captures if bucket == expected and rank is not None
            )
            for expected in expected_buckets
        },
        # The TP1 fan-in marker is diagnostic; DCP destinations store sharded pages.
        "pd_page_rr_fan_in_lines": markers.get("pd_page_rr_fan_in", 0),
    }
    return {"checks": checks, "observations": observations}


def _verify(
    events: list[dict],
    role: str,
    replay_seen: bool = False,
    markers: dict | None = None,
    decode_page_rr: bool = False,
    proposal_tokens: int = 3,
    expected_tp: int | None = None,
    dp_size: int = 1,
    expected_block_size: int | None = None,
    source_tp: int | None = None,
    prefill_page_rr: bool = False,
) -> dict:
    checks = {}
    observations = {}
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
        if prefill_page_rr:
            prefix_plans = [
                event
                for event in events
                if event.get("kind") == "mla_prefix"
                and event.get("backend") == "page_rr"
                and event.get("route") == "hybrid"
                and event.get("tp") == 8
            ]

            def valid_multi_launch(event: dict) -> bool:
                launches = event["launch_tokens"]
                capacity = event["capacity_tokens"]
                alignment = event["alignment_tokens"]
                return (
                    isinstance(launches, list)
                    and len(launches) >= 2
                    and isinstance(capacity, int)
                    and capacity > 0
                    and isinstance(alignment, int)
                    and alignment > 0
                    and all(
                        isinstance(tokens, int)
                        and 0 < tokens <= capacity
                        and tokens % alignment == 0
                        for tokens in launches
                    )
                    and sum(launches) == event["prefix_tokens"]
                )

            valid_prefix_plans = [
                event for event in prefix_plans if valid_multi_launch(event)
            ]
            checks["page_rr_prefix_multi_launch"] = bool(valid_prefix_plans)
            observations["page_rr_max_launch_count"] = max(
                (len(event["launch_tokens"]) for event in valid_prefix_plans),
                default=0,
            )
    elif decode_page_rr:
        report = _dcp_checks(markers or {}, replay_seen, events, proposal_tokens, expected_tp, dp_size, expected_block_size, source_tp)
        checks.update(report["checks"])
        observations.update(report["observations"])
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
        # The request runner synchronously validates the 7 -> 5 -> 6 owner-7
        # waves. Their HTTP barrier cannot guarantee that every request is
        # resident in one Decode step, so gate on the exercised Graph path and
        # retain instantaneous occupancy only as diagnostic evidence.
        owner7_bucket8 = [
            e
            for e in graph
            if e["bucket"] == 8
            and e["valid"][:7] == [0] * 7
            and e["valid"][7] > 0
        ]
        checks["bucket8_owner7_target_verify"] = bool(owner7_bucket8)
        observations["bucket8_owner7_valid_transitions"] = adjacent_unique(
            [e["valid"][7] for e in owner7_bucket8]
        )
        checks["graph_replay_observed"] = replay_seen
    return {
        "role": role,
        "passed": all(checks.values()),
        "checks": checks,
        "observations": observations,
        "event_count": len(events),
    }


def verify(
    events: list[dict],
    role: str,
    replay_seen: bool = False,
    markers: dict | None = None,
    decode_page_rr: bool = False,
    proposal_tokens: int = 3,
    expected_tp: int | None = None,
    dp_size: int = 1,
    expected_block_size: int | None = None,
    source_tp: int | None = None,
    prefill_page_rr: bool = False,
) -> dict:
    try:
        return _verify(
            events,
            role,
            replay_seen,
            markers,
            decode_page_rr,
            proposal_tokens,
            expected_tp,
            dp_size,
            expected_block_size,
            source_tp,
            prefill_page_rr,
        )
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
    markers = {
        "dcp_backends": set(),
        "dcp_workers": set(),
        "page_rr_targets": set(),
        "graph_captures": set(),
        "pd_page_rr_fan_in": 0,
    }
    # Explicit service/rank paths avoid recursively ingesting saved evidence.
    paths = (
        list(root.glob("*.log"))
        + list(root.glob("runtime/logs/*/main_*.log"))
        + list(root.glob("runtime/work/*/logs/engine.log"))
    )
    for path in sorted(set(paths)):
        # Rank logs carry their rank in the filename; engine logs prefix it.
        log_rank = _MAIN_LOG.fullmatch(path.name)
        log_rank = int(log_rank.group(1)) if log_rank else None
        with path.open(errors="replace") as source:
            for line in source:
                replay_seen |= "[K3_PROJECTION_KTP_GRAPH_REPLAY]" in line
                if "[K3_SMOKE_EVENT] " in line:
                    event = json.loads(line.split("[K3_SMOKE_EVENT] ", 1)[1])
                    events.append(event)
                backend = _MLA_DCP.search(line)
                if backend and backend.group(1) == "a2a":
                    markers["dcp_backends"].add(
                        (int(backend.group(2)), int(backend.group(3)))
                    )
                worker = re.search(r"\[MLA_DCP\].* world_rank=(\d+) dp_rank=(\d+)", line)
                if backend and worker and backend.group(1) == "a2a":
                    markers["dcp_workers"].add((int(worker.group(1)), int(worker.group(2)), int(backend.group(3))))
                target = _PAGE_RR_TARGET.search(line)
                if target:
                    markers["page_rr_targets"].add(
                        (
                            target.group(1),
                            int(target.group(2)),
                            int(target.group(3)),
                            int(target.group(4)),
                        )
                    )
                capture = _GRAPH_CAPTURE.search(line)
                if capture:
                    rank = _RANK_PREFIX.search(line)
                    markers["graph_captures"].add(
                        (
                            int(rank.group(1)) if rank else log_rank,
                            int(capture.group(1)),
                        )
                    )
                markers["pd_page_rr_fan_in"] += "[K3_PD_PAGE_RR_FAN_IN]" in line
    return events, replay_seen, markers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--decode-page-rr",
        choices=("0", "1"),
        default="0",
        help="Decode runs a Page-RR (DCP) KTP1 topology instead of Projection-KTP",
    )
    parser.add_argument("--proposal-tokens", type=int, default=3)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--source-tp-size", type=int)
    parser.add_argument(
        "--prefill-page-rr",
        choices=("0", "1"),
        default="0",
        help="Require a real TP8 Page-RR multi-launch prefix plan on Prefill",
    )
    args = parser.parse_args()
    try:
        events, replay, markers = collect(args.root)
        report = verify(
            events,
            args.role,
            replay,
            markers,
            args.decode_page_rr == "1",
            args.proposal_tokens,
            args.tp_size,
            args.dp_size,
            args.block_size,
            args.source_tp_size,
            args.prefill_page_rr == "1",
        )
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
