"""Receive native P/D evidence; no transport, model runner or device access."""

import argparse
import json
from pathlib import Path


def receive(prefill, decode):
    for item, role in ((prefill, "prefill"), (decode, "decode")):
        if item.get("native_role_ok") is not True or item.get("role") != role or item.get("sharded") is not True:
            raise AssertionError("N6 requires successful explicit sharded native roles")
        if item["transport"] != "native RDMA CacheStore":
            raise AssertionError("N6 requires native RDMA without fallback")
    if prefill["snapshot"] != decode["snapshot"] or prefill["cache_protocol"] != decode["cache_protocol"]:
        raise AssertionError("source/destination snapshot or cache profile mismatch")
    source = [item for item in prefill["state_fingerprints"] if item["prefix_lengths"] == [0]]
    if len(source) != 8 or {item["rank"] for item in source} != set(range(8)):
        raise AssertionError("one cold CP8 Prefill call on every source rank is required")
    destinations = [item for item in decode["state_fingerprints"] if item["prefix_lengths"] == [257]]
    if len(destinations) != 1:
        raise AssertionError("one real D business owner must receive prefix257 before continuation")
    destination = destinations[0]
    source_bindings = [item for item in prefill["cache_placement"] if item["call"] == source[0]["call"]]
    destination_bindings = next(item for item in decode["cache_placement"]
                                if (item["rank"], item["call"]) == (destination["rank"], destination["call"]))
    target_pools = {(p["layer"], p["tag"], p["request"]): p for p in destination_bindings["pools"]}
    for binding in source_bindings:
        if len(binding["pools"]) != len(target_pools):
            raise AssertionError("source/destination physical pool bindings differ")
        for pool in binding["pools"]:
            target = target_pools[pool["layer"], pool["tag"], pool["request"]]
            if not pool["sharded"] or target["sharded"] or pool["cp_rank"] != binding["rank"]:
                raise AssertionError("physical pool source/destination ownership mismatch")
            if pool["physical_span"] != target["physical_span"] or pool["kernel_span"] != target["kernel_span"]:
                raise AssertionError("source/destination per-tag geometry differs")
            if pool["tag"] not in ("global_kv_1", "global_kv_2", "indexer_kv"):
                if pool["local_block_stride_bytes"] * 8 != target["local_block_stride_bytes"]:
                    raise AssertionError("native fixed-pool slices do not assemble the destination full stride")
                if pool["full_entries"] != target["entries"]:
                    raise AssertionError("native aligned ring entry counts differ")
    collectives = prefill["collectives"]
    traces = [item for item in collectives if item["call"] == source[0]["call"]]
    if len(traces) != 8 or any(item["pools"] != traces[0]["pools"] or item["current_rows"] != traces[0]["current_rows"] for item in traces):
        raise AssertionError("CP ranks issued inconsistent native pool collective order")
    canonical = destination["state"]
    if not canonical:
        raise AssertionError("missing destination native live content")
    ownership = []
    for key, target in canonical.items():
        positions = target["positions"]
        if not positions or target["bytes"] % len(positions):
            raise AssertionError("invalid canonical logical byte domain")
        row_bytes = target["bytes"] // len(positions)
        intervals = {position: [] for position in positions}
        owner_bytes = 0
        for rank in source:
            if set(rank["state"]) != set(canonical):
                raise AssertionError("source/destination pool domains differ")
            state = rank["state"][key]
            # These bytes are logical gathers, not a claim that resident P
            # allocations are replicated. The owner proof below is mandatory.
            if any(state[field] != target[field] for field in ("positions", "sha256", "bytes", "nonzero_bytes")):
                raise AssertionError(f"native canonical handoff mismatch: {key}")
            owner = state["source_owner"]
            if owner["cp_size"] != 8 or owner["cp_rank"] != rank["rank"] or owner["matches_canonical"] is not True:
                raise AssertionError("invalid physical source-owner evidence")
            count = 0
            for fragment in owner["fragments"]:
                pos, begin, end = (fragment[field] for field in ("position", "begin", "end"))
                if pos not in intervals or not 0 <= begin < end <= row_bytes:
                    raise AssertionError("owner fragment lies outside the logical domain")
                intervals[pos].append((begin, end, rank["rank"]))
                count += end - begin
            if count != owner["bytes"]:
                raise AssertionError("owner byte accounting differs from actual fragments")
            owner_bytes += count
        for fragments in intervals.values():
            end = 0
            for begin, stop, rank in sorted(fragments):
                if begin != end:
                    raise AssertionError("CP owners have a gap or duplicate logical bytes")
                end = stop
            if end != row_bytes:
                raise AssertionError("CP source owners do not cover destination content")
        if owner_bytes != target["bytes"]:
            raise AssertionError("sum of unique source-owned bytes differs from destination")
        ownership.append({"pool": key, "logical_rows": len(positions), "unique_owned_bytes": owner_bytes,
                          "destination_sha256": target["sha256"], "exact_owner_cover": True})
    if decode["evidence"]["prefill_executed"] is not False:
        raise AssertionError("D must continue without Prefill")
    return {"schema": "v41-ppu-n6-native-handoff-v1", "SMOKE": "PASS_NATIVE_SHARDED_RDMA",
            "destination_owner": destination["rank"], "ownership": ownership,
            "source_layout": prefill["cache_protocol"]["placement"],
            "source_pool_gathers_per_rank": len(traces[0]["pools"]),
            "source_current_row_gathers_per_rank": len(traces[0]["current_rows"]),
            "physical_bindings": prefill["cache_placement"],
            "capacity_scope": "one short request; no all-owner/long-capacity matrix claim",
            "FULL_VALIDATION": "DEFERRED_TO_N10"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-summary", type=Path, required=True)
    parser.add_argument("--decode-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = receive(json.loads(args.prefill_summary.read_text()), json.loads(args.decode_summary.read_text()))
    with args.output.open("x") as writer:
        json.dump(result, writer, indent=2)
        writer.write("\n")


if __name__ == "__main__":
    main()
