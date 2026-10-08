"""Audit replay ownership and actual mixed verification without tensor dumps."""

import re

from request_smoke import require


def check_replay_events(events):
    previous = {}
    reuses = 0
    commits = 0
    seen = set()
    for event in sorted(events, key=lambda item: item["time_ns"]):
        if event.get("event") != "linear_replay_commit":
            continue
        identity = (
            event["time_ns"],
            event["slot"],
            event["generation"],
            event["verify_epoch"],
            event["request_id"],
        )
        if identity in seen:
            continue
        seen.add(identity)
        commits += 1
        slot, generation, epoch = (
            event["slot"],
            event["generation"],
            event["verify_epoch"],
        )
        request = event["request_id"]
        require(
            slot >= 0 and generation > 0 and epoch > 0 and event["produced_steps"] == 4,
            "invalid replay lease or verify width",
        )
        if slot in previous:
            old_generation, old_epoch, old_request = previous[slot]
            require(
                generation >= old_generation, "replay slot generation went backwards"
            )
            if generation == old_generation:
                require(
                    request == old_request and epoch > old_epoch,
                    "replay lease crossed requests or stale epoch",
                )
            else:
                reuses += 1
        previous[slot] = generation, epoch, request
    require(commits > 0 and reuses > 0, "replay commit/slot reuse was not exercised")
    return {"commits": commits, "slot_reuses": reuses}


def check_replay_owners(decode, owner_count):
    require(
        owner_count > 0
        and len(decode) % owner_count == 0
        and sorted(decode) == list(range(len(decode))),
        "invalid Decode TP/DP layout",
    )
    width = len(decode) // owner_count
    require(width > 0, "missing Decode ranks")
    # Stream leases are committed by each DP group's TP rank zero.
    # Model execution on the TP followers is covered by the all-rank path audit.
    return {
        str(rank): check_replay_events(decode[rank])
        for rank in range(0, len(decode), width)
    }


def audit_request_runtime(result, decode, decode_engine_log):
    records = result.get("protocol_requests", [])
    require(
        records and all(row["passed"] for row in records),
        "missing or failed protocol smoke",
    )
    replay = check_replay_owners(
        decode, max(1, len(result.get("decode_role_addrs", [])))
    )
    mixed = {}
    for prefix in ("grammar_mixed_", "logprobs_mixed_"):
        group = [row for row in records if row["name"].startswith(prefix)]
        require(len(group) == 3, f"missing mixed requests: {prefix}")
        start, end = min(row["start_ns"] for row in group), max(
            row["end_ns"] for row in group
        )
        observed = any(
            event.get("event") == "mtp_target_verify_forward"
            and event.get("stream_count", 0) >= 2
            and start <= event["time_ns"] <= end
            for events in decode.values()
            for event in events
        )
        require(
            observed,
            f"concurrent requests did not enter mixed target verification: {prefix}",
        )
        mixed[prefix] = True
    require(decode_engine_log is not None, "pool audit requires the Decode engine log")
    text = decode_engine_log.read_text(errors="replace")
    pools = [
        (tag, int(concurrency), int(blocks))
        for tag, concurrency, blocks in re.findall(
            r"Decode LINEAR pool tag=(\S+) concurrency=(\d+) blocks=(\d+)", text
        )
    ]
    require(
        pools
        and all(blocks == 2 * (concurrency + 4) for _, concurrency, blocks in pools),
        "missing or incorrect automatic Decode LINEAR capacity",
    )
    return {
        "replay": replay,
        "mixed_verify": mixed,
        "linear_pools": pools,
        "protocol_request_count": len(records),
        "images_added": False,
    }
