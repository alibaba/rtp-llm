"""Offline native-run assertions; synthetic parser tests are not device evidence.

The byte oracle below deliberately imports no NativeCacheBridge, CacheVersion,
cache codec or publisher. It interprets recorded native tables and raw bytes.
Only task-local captures from an actual native role may satisfy the role gate.
"""

import gzip
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.tools.v41.native_smoke import (
    SSETokenChain,
    make_query,
    output_tokens,
    run_request,
    validate_finished,
)

FULL = {"global_kv_1", "global_kv_2", "indexer_kv"}
WIDTH = {
    "global_kv_1": 288,
    "global_kv_2": 288,
    "indexer_kv": 68,
    "swa_kv": 528,
    "csa_state": 1024,
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def physical_entry(pool, row, position):
    """Independent absolute-token -> native physical page/element mapping."""
    table = pool["tables"][row]
    logical_block = position // pool["span"]
    require(0 <= logical_block < len(table), "position exceeds recorded native table")
    block = int(table[logical_block])
    require(0 <= block < pool["shape"][0], "unallocated native block")
    offset = (
        (position % pool["span"]) // pool["ratio"]
        if pool["tag"] in FULL
        else position % pool["entries"]
    )
    require(offset < pool["entries"], "native element exceeds physical capacity")
    if pool["tag"] == "indexer_kv":
        columns = list(range(offset * 64, offset * 64 + 64)) + list(
            range(
                pool["entries"] * 64 + offset * 4, pool["entries"] * 64 + offset * 4 + 4
            )
        )
    else:
        width = WIDTH[pool["tag"]]
        columns = list(range(offset * width, (offset + 1) * width))
    size = 4 if pool["tag"] == "csa_state" else 1
    byte_columns = [column * size + lane for column in columns for lane in range(size)]
    require(
        max(byte_columns) < pool["shape"][1] * size, "native byte exceeds page stride"
    )
    return pool["blocks"].index(block), byte_columns


def apply_payloads(pools, raw, payloads, written_bytes=None, written_entries=None):
    expected = {key: value.clone() for key, value in raw.items()}
    assigned = {}
    for item in payloads:
        key = tuple(item["key"])
        require(key in pools, f"missing native pool {key}")
        positions = item["positions"].tolist()
        if not positions:
            require(item["packed"].numel() == 0, "empty payload has bytes")
            continue
        packed = (
            item["packed"].contiguous().view(torch.uint8).reshape(len(positions), -1)
        )
        for index, position in enumerate(positions):
            page, columns = physical_entry(pools[key], item["row"], int(position))
            require(len(columns) == packed.shape[1], "candidate byte width mismatch")
            destination = (key, pools[key]["blocks"][page], columns[0])
            if destination in assigned and assigned[destination][0] != item["row"]:
                require(
                    torch.equal(assigned[destination][1], packed[index]),
                    "cross-request native payload conflict",
                )
            if destination not in assigned and written_bytes is not None:
                written_bytes[key] += len(columns)
            assigned[destination] = (item["row"], packed[index])
            if written_entries is not None:
                written_entries.add(destination)
            expected[key][page, columns] = packed[index]
    return expected


def assert_imports(pools, raw, payloads):
    for item in payloads:
        key = tuple(item["key"])
        positions = item["positions"].tolist()
        if not positions:
            continue
        packed = (
            item["packed"].contiguous().view(torch.uint8).reshape(len(positions), -1)
        )
        for index, position in enumerate(positions):
            page, columns = physical_entry(pools[key], item["row"], int(position))
            require(
                torch.equal(raw[key][page, columns], packed[index]),
                f"native import mismatch {key}",
            )


def assert_noop_rows(capture, pool):
    """Independently validate no-op ownership and report actual byte coverage."""
    if "noop_rows" not in pool:
        return False, []  # Legacy captures keep every existing requirement.
    require(pool["tag"] == "csa_state", "no-op metadata on a non-CSA pool")
    require(
        pool["dtype"] == "torch.float32"
        and pool["ratio"] == 2
        and pool["entries"] >= 2,
        "invalid no-op CSA geometry",
    )
    require(
        capture["kind"] in ("prefill", "verify", "proposal", "commit"),
        "invalid no-op kind",
    )
    prefixes, lengths = capture["prefixes"], capture["lengths"]
    require(
        prefixes and len(prefixes) == len(lengths) == len(pool["tables"]),
        "no-op request/table rows differ",
    )
    explicit = capture.get("observation_positions")
    require(
        explicit is None or len(explicit) == len(prefixes),
        "no-op explicit position rows differ",
    )
    if capture["kind"] in ("prefill", "verify", "proposal"):
        require(explicit is None, "unexpected explicit target observation positions")
    expected_rows = []
    for row, (base, length) in enumerate(zip(prefixes, lengths)):
        require(base >= 0 and length >= 0, "invalid no-op request bounds")
        end = base + length
        if explicit is not None:
            affected = explicit[row]
        elif capture["kind"] == "prefill":
            affected = [end - 1] if end % 2 else []
        else:
            affected = [n - 1 for n in range(base + 1, end + 1) if n % 2]
        imported = [base - 1] if base % 2 else []
        for position in list(affected) + imported:
            span, table = pool["span"], pool["tables"][row]
            require(
                span > 0 and 0 <= position < len(table) * span,
                "CSA semantic position exceeds table",
            )
            require(
                0 < table[position // span] < pool["shape"][0],
                "CSA affected/imported position is unallocated or padding",
            )
        if not affected and not imported:
            expected_rows.append(row)
    declared = pool["noop_rows"]
    require(
        isinstance(declared, list)
        and all(type(entry["row"]) is int for entry in declared)
        and [entry["row"] for entry in declared] == expected_rows,
        "no-op declaration discards real affected/imported positions",
    )
    require(
        pool["before"].dtype == pool["after"].dtype == torch.uint8
        and tuple(pool["before"].shape)
        == tuple(pool["after"].shape)
        == (len(pool["blocks"]), pool["shape"][1] * 4),
        "no-op raw byte coverage shape differs",
    )
    reports, owned_blocks = [], set()
    for entry in declared:
        row = entry["row"]
        end = prefixes[row] + lengths[row]
        span, table = pool["span"], pool["tables"][row]
        require(span > 0 and end >= 0, "invalid no-op request geometry")
        logical_end = (end + span - 1) // span
        require(logical_end <= len(table), "no-op request table is incomplete")
        witnesses = []
        for logical, block in enumerate(table[:logical_end]):
            require(
                block in (-1, 0) or 0 < block < pool["shape"][0],
                "invalid no-op owned table entry",
            )
            if block > 0:
                witnesses.append(min((logical + 1) * span, end) - 1)
        coverage = (
            "UNTOUCHED_OWNED_PAGES" if witnesses else "NO_OWNED_PAGES_NO_BYTE_COVERAGE"
        )
        require(
            entry["end"] == end
            and entry["witness_positions"] == witnesses
            and entry["coverage"] == coverage,
            "incomplete or false no-op witness coverage",
        )
        for payload in capture.get("expected", []) + capture.get("imports", []):
            if tuple(payload["key"]) == tuple(pool["key"]) and payload["row"] == row:
                require(
                    payload["positions"].numel() == payload["packed"].numel() == 0,
                    "no-op observation invented a semantic payload",
                )
        blocks = {table[position // span] for position in witnesses}
        owned_blocks.update(blocks)
        for block in blocks:
            require(block in pool["blocks"], "no-op owned page was not captured")
            index = pool["blocks"].index(block)
            require(
                torch.equal(pool["before"][index], pool["after"][index]),
                "no-op owned page bytes changed",
            )
        reports.append(
            dict(
                key=list(pool["key"]),
                row=row,
                coverage=coverage,
                witness_blocks=sorted(blocks),
                compared_bytes=len(blocks) * pool["shape"][1] * 4,
            )
        )
    for row, position in pool["neighbor_positions"]:
        if row in expected_rows:
            end = prefixes[row] + lengths[row]
            require(0 <= position < end, "no-op neighbor is future/padding")
            require(
                pool["tables"][row][position // pool["span"]] in owned_blocks,
                "no-op neighbor is not request-owned",
            )
    all_noop = expected_rows == list(range(len(prefixes)))
    if all_noop:
        require(
            pool["blocks"] == sorted(owned_blocks),
            "no-op capture includes unrelated/future physical pages",
        )
    return all_noop, reports


def assert_capture(capture):
    require(capture.get("schema_version") == 1, "unsupported native capture schema")
    require(capture.get("complete") is True, "native capture is incomplete")
    if capture["kind"] == "verify":
        require(
            len(capture["prefixes"])
            == len(capture["lengths"])
            == len(capture["accepted_input_rows"]),
            "native transaction row metadata differs",
        )
        require(
            all(
                0 <= count <= width
                for count, width in zip(
                    capture["accepted_input_rows"], capture["lengths"]
                )
            ),
            "accepted-input count exceeds query",
        )
        if capture["outcome"] == "accept":
            require(
                capture["candidate_ends"]
                == [
                    list(range(prefix + 1, prefix + width + 1))
                    for prefix, width in zip(capture["prefixes"], capture["lengths"])
                ],
                "candidate versions do not match causal stages",
            )
            require(
                capture["selected_versions"]
                == [
                    dict(row=row, end=prefix + count, accepted_input_rows=count)
                    for row, (prefix, count) in enumerate(
                        zip(capture["prefixes"], capture["accepted_input_rows"])
                    )
                    if count
                ],
                "selected version does not match accepted input end",
            )
    pools = {tuple(pool["key"]): pool for pool in capture["pools"]}
    require(
        set(pools) == {tuple(key) for key in capture["pool_keys"]},
        "omitted native pool",
    )
    expected_layers = 40 if capture["model"] == "target" else 3
    require(
        {key[0] for key in pools if key[1] == "swa_kv"} == set(range(expected_layers)),
        "capture must contain every real target/draft SWA layer",
    )
    require(
        len({pool["data_ptr"] for key, pool in pools.items() if key[1] == "swa_kv"})
        == expected_layers,
        "native SWA layers alias the same allocation",
    )
    before = {tuple(p["key"]): p["before"] for p in capture["pools"]}
    noop_coverage = []
    for pool in capture["pools"]:
        require(pool["before"].dtype == torch.uint8, "raw page capture is not bytes")
        all_noop, coverage = assert_noop_rows(capture, pool)
        noop_coverage.extend(coverage)
        require(
            pool["tables"] and (pool["blocks"] or all_noop),
            "missing physical page/table observation",
        )
        require(pool["data_ptr"] > 0, "capture has no native allocation identity")
        if pool["tag"] in FULL:
            require(
                pool.get("full_neighbor_scan_complete") is True,
                "FULL neighbor scan is incomplete",
            )
        else:
            require(
                pool["neighbor_positions"] or all_noop,
                "capture declares no boundary probes",
            )
        require(
            pool["before"].shape == pool["after"].shape, "native capture shape changed"
        )
        if capture["kind"] in ("verify", "proposal"):
            require(
                torch.equal(pool["before"], pool["verified"]),
                "query mutated committed native bytes",
            )
    assert_imports(pools, before, capture.get("imports", []))
    if capture["outcome"] in ("abort", "rollback", "proposal"):
        require(not capture["expected"], "unpublished round has accepted payloads")
    written_bytes = defaultdict(int)
    written_entries = set()
    expected = apply_payloads(
        pools, before, capture["expected"], written_bytes, written_entries
    )
    for pool in capture["pools"]:
        key = tuple(pool["key"])
        require(
            torch.equal(pool["after"], expected[key]),
            f"native accepted bytes or untouched page/plane/padding changed: {key}",
        )
        if pool["tag"] in FULL:
            for row, position in pool["neighbor_positions"]:
                page, columns = physical_entry(pool, row, position)
                require(
                    (key, pool["blocks"][page], columns[0]) not in written_entries,
                    "FULL neighbor aliases an actually written physical entry",
                )
    require(
        any(pool["before"].numel() for pool in capture["pools"]),
        "capture has no physical byte coverage",
    )
    return {
        "kind": capture["kind"],
        "outcome": capture["outcome"],
        "native_pools": len(pools),
        "native_blocks": sum(len(p["blocks"]) for p in capture["pools"]),
        "untouched_byte_comparisons": sum(
            before[key].numel() - written_bytes[key] for key in pools
        ),
        "noop_coverage": noop_coverage,
        "noop_owned_byte_comparisons": sum(c["compared_bytes"] for c in noop_coverage),
    }


def read_observations(directory, snapshot):
    headers, events, captures = [], [], []
    paths = sorted(Path(directory).glob("observer-*.jsonl"))
    require(paths, "no real native observer output")
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        require(rows and rows[0]["event"] == "start", "missing observer identity")
        header = dict(rows[0])
        require(
            "ZW-M890P" in header["hardware"] and header["device"].startswith("cuda"),
            "observer is not an actual expected PPU native run",
        )
        for key in ("snapshot_id", "source_sha256", "public_commit", "internal_commit"):
            require(
                header["source"][key] == snapshot[key],
                f"observer source mismatch {key}",
            )
            require(
                header["runtime"][key] == snapshot[key],
                f"observer runtime mismatch {key}",
            )
        require(rows[-1]["event"] == "complete", "observer did not close normally")
        require(rows[-1]["failed"] is False, "native diagnostic failed")
        require(
            [r["sequence"] for r in rows] == list(range(len(rows))),
            "observer event loss",
        )
        require(
            not any(r["event"] in ("error", "limit") for r in rows),
            "diagnostic error/limit: missing evidence",
        )
        require(rows[-1]["open_captures"] == 0, "unfinished native capture")
        require(
            rows[-1]["capture_count"] == sum(r["event"] == "capture" for r in rows),
            "capture loss",
        )
        selected_ids = [
            r["capture_id"] for r in rows if r["event"] == "begin" and r["selected"]
        ]
        captured_ids = [r["capture_id"] for r in rows if r["event"] == "capture"]
        require(
            len(captured_ids) == len(set(captured_ids))
            and set(selected_ids) == set(captured_ids),
            "selected native capture missing/duplicated",
        )
        votes = [r for r in rows if r["event"] == "vote"]
        require(rows[-1]["vote_count"] == len(votes), "native vote record loss")
        last_index, previous = -1, None
        for vote in votes:
            gap = vote["vote_index"] - last_index - 1
            require(
                gap >= 0 and gap == vote["omitted_idle_votes"],
                "native vote ordering/loss",
            )
            require(
                not gap
                or previous is not None
                and previous["global_rows"] == 0
                and not previous["global_stop"],
                "omitted a non-idle native vote",
            )
            last_index, previous = vote["vote_index"], vote
        require(
            rows[-1]["total_existing_votes"] >= last_index + 1,
            "invalid terminal vote index",
        )
        require(
            rows[-1]["total_existing_votes"] == last_index + 1
            or previous is not None
            and previous["global_rows"] == 0
            and not previous["global_stop"],
            "lost terminal non-idle native vote",
        )
        for row in rows:
            row["identity"] = header
            if row["event"] == "capture":
                target = path.parent / row["file"]
                require(
                    target.parent == path.parent, "capture path escaped task directory"
                )
                require(
                    hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"],
                    "native capture bytes changed",
                )
                capture = torch.load(target, map_location="cpu", weights_only=False)
                require(
                    capture["capture_id"] == row["capture_id"], "capture event mismatch"
                )
                capture["byte_proof"] = assert_capture(capture)
                capture["identity"] = header
                captures.append(capture)
        headers.append(header)
        events.extend(rows)
    return headers, events, captures


def assert_work_votes(events):
    result = {}
    for model in ("target", "draft"):
        owners, rounds = set(), defaultdict(dict)
        for row in events:
            h = row["identity"]
            if h["model"] != model or row["event"] != "vote" or row.get("warmup"):
                continue
            owner = h["dp_rank"]
            owners.add(owner)
            index = row["vote_index"]
            require(owner not in rounds[index], "duplicate native owner vote")
            rounds[index][owner] = row
        require(owners == set(range(8)), f"missing {model} DP owners")
        one_busy, skew, all_empty, busy_seen = [], [], [], False
        for index, group in sorted(rounds.items()):
            require(set(group) == set(range(8)), f"unaligned/lost {model} vote {index}")
            total = sum(row["local_rows"] for row in group.values())
            require(
                all(row["global_rows"] == total for row in group.values()),
                "invalid native reduced work sum",
            )
            require(
                len({row["global_stop"] for row in group.values()}) == 1,
                "unaligned native stop vote",
            )
            require(
                all(
                    row["global_stop"] == any(v["local_stop"] for v in group.values())
                    for row in group.values()
                ),
                "invalid native reduced stop vote",
            )
            active = [row["local_rows"] for row in group.values() if row["local_rows"]]
            if total:
                busy_seen = True
            if len(active) == 1:
                one_busy.append(index)
            if len(active) > 1 and len(set(active)) > 1:
                skew.append(index)
            if (
                total == 0
                and busy_seen
                and not any(row["global_stop"] for row in group.values())
            ):
                all_empty.append(index)
        require(
            one_busy and skew and all_empty,
            f"{model} missing actual one-busy/skew/post-work-all-empty votes",
        )
        result[model] = dict(
            one_busy=one_busy, skew=skew, post_work_all_empty=all_empty
        )
    return result


def read_model_inputs(directory):
    records, groups = [], defaultdict(list)
    for path in sorted(Path(directory).glob("*.pt")):
        match = re.search(r"_r(\d+)_s\d+_p\d+_", path.name)
        require(match is not None, "model dump lacks rank/PID identity")
        chunk = torch.load(path, map_location="cpu", weights_only=False)
        group = path.name[: match.end()]
        for row in (
            chunk["records"]
            if chunk["record_type"] == "model_inputs_chunk"
            else [chunk]
        ):
            require(
                row["record_type"] == "gpt_model_inputs_snapshot",
                "model input gap/unknown record",
            )
            require(row["dropped_before"] == 0, "model input logger dropped records")
            row["native_owner"] = int(match.group(1))
            records.append(row)
            groups[group].append(row["record_sequence"])
    require(records, "no actual model input dumps")
    for sequence in groups.values():
        sequence.sort()
        require(
            sequence == list(range(sequence[0], sequence[0] + len(sequence))),
            "model input sequence gap",
        )
    return records


def assert_first_anchors(records, events, cases, *, terminal_prefetch_rows=0):
    finished = defaultdict(list)
    computed = defaultdict(list)
    for row in events:
        h = row["identity"]
        if (
            row["event"] == "compute"
            and h["model"] == "target"
            and row["kind"] == "verify"
        ):
            for rid, position, token in zip(
                row["request_ids"], row["positions"], row["tokens"]
            ):
                computed[(h["dp_rank"], row["capture_id"], rid)].append(
                    (row["stage"], position, token)
                )
    for row in events:
        h = row["identity"]
        if (
            row["event"] != "finish"
            or h["model"] != "target"
            or row["kind"] != "verify"
        ):
            continue
        require(
            row["request_ids"] is not None,
            "target round lacks scheduler request identities",
        )
        for rid, prefix, accepted in zip(
            row["request_ids"], row["prefixes"], row["accepted_input_rows"]
        ):
            key = (h["dp_rank"], rid, prefix)
            require(
                not accepted or not any(count for count, _ in finished[key]),
                "duplicate committed absolute prefix",
            )
            finished[key].append((accepted, row["capture_id"]))
    observed_owners, evidence = set(), []
    for case in cases:
        trace, outputs = case["trace_id"], case["output_ids"][0]
        prompt = case["aux_info"]["input_len"]
        chain = []
        consumed_rounds = defaultdict(int)
        for record in sorted(records, key=lambda r: r["record_sequence"]):
            if (
                record["execution_stage"] != "target_verify"
                or record["warmup"]
                or record["is_fake_stream"]
                or trace not in record["trace_ids"]
            ):
                continue
            ids = record["request_id"].reshape(-1).tolist()
            lengths = record["input_lengths"].reshape(-1).tolist()
            prefixes = record["prefix_lengths"].reshape(-1).tolist()
            tokens = record["combo_tokens"].reshape(-1).tolist()
            position_tensor = record.get("combo_position_ids")
            positions = (
                position_tensor.reshape(-1).tolist()
                if isinstance(position_tensor, torch.Tensor) and position_tensor.numel()
                else None
            )
            require(
                len(ids) == len(lengths) == len(prefixes) == len(record["trace_ids"]),
                "target request-major identity mismatch",
            )
            cursor = 0
            for row, length in enumerate(lengths):
                if record["trace_ids"][row] == trace:
                    owner, prefix = record["native_owner"], prefixes[row]
                    key = (owner, ids[row], prefix)
                    ordinal = consumed_rounds[key]
                    require(
                        ordinal < len(finished[key]),
                        "target input has no completed native transaction",
                    )
                    count, capture_id = finished[key][ordinal]
                    consumed_rounds[key] += 1
                    require(
                        0 <= count <= length, "accepted input rows exceed target query"
                    )
                    actual = sorted(computed[(owner, capture_id, ids[row])])
                    require(
                        len(actual) == length
                        and [r[0] for r in actual] == list(range(length)),
                        "native causal compute stages missing/repeated",
                    )
                    actual_positions = [r[1] for r in actual]
                    require(
                        actual_positions == list(range(prefix, prefix + length)),
                        "actual target compute positions changed",
                    )
                    require(
                        [r[2] for r in actual] == tokens[cursor : cursor + length],
                        "actual target compute tokens differ from the model-input dump",
                    )
                    if positions is not None:
                        require(
                            positions[cursor : cursor + length] == actual_positions,
                            "scheduler/actual target absolute positions differ",
                        )
                    offset = prefix - prompt
                    compare = min(count, max(0, len(outputs) - offset))
                    require(
                        offset >= 0
                        and tokens[cursor : cursor + compare]
                        == outputs[offset : offset + compare],
                        "accepted target inputs differ from client token prefix",
                    )
                    chain.append(
                        (
                            prefix,
                            count,
                            owner,
                            ids[row],
                            tokens[cursor],
                            record["record_sequence"],
                        )
                    )
                cursor += length
        chain.sort(key=lambda row: row[-1])
        require(
            chain and chain[0][0] == prompt, f"missing native first anchor for {trace}"
        )
        require(chain[0][4] == outputs[0], "P first token differs from D first anchor")
        require(
            sum(row[0] == prompt and row[1] > 0 for row in chain) == 1,
            "first anchor committed more than once or never committed",
        )
        require(
            len({(row[2], row[3]) for row in chain}) == 1,
            "request moved native owner/identity",
        )
        owner, request_id = chain[0][2:4]
        require(
            len(chain)
            == sum(
                len(counts)
                for (rank, rid, _), counts in finished.items()
                if rank == owner and rid == request_id
            ),
            "model-input dump omitted a native business transaction",
        )
        terminal_prefetch = []
        end = prompt + len(outputs) - 1
        if (
            terminal_prefetch_rows
            and not case.get("intentionally_disconnected")
            and len(outputs) == case["query"]["generate_config"]["max_new_tokens"]
        ):
            # The C++ stream can complete while an async input is in flight.
            # The scheduler and compute evidence above still covers that input.
            while chain[-1][1] == 0 and end <= chain[-1][0] <= end + terminal_prefetch_rows:
                terminal_prefetch.append(chain.pop()[0])
        for previous, following in zip(chain, chain[1:]):
            require(
                following[0] == previous[0] + previous[1],
                "target accepted prefix skipped/repeated",
            )
        committed = [
            position
            for prefix, count, *_ in chain
            for position in range(prefix, prefix + count)
        ]
        require(
            len(committed) == len(set(committed)),
            "accepted input position committed twice",
        )
        if not case.get("intentionally_disconnected"):
            require(
                chain[-1][0] + chain[-1][1] == end,
                "final committed input prefix is off by one",
            )
        observed_owners.add(chain[0][2])
        evidence.append(
            dict(
                trace_id=trace,
                owner=chain[0][2],
                request_id=chain[0][3],
                rounds=len(chain) + len(terminal_prefetch),
            )
        )
        if terminal_prefetch:
            evidence[-1]["terminal_zero_publication_prefixes"] = terminal_prefetch[::-1]
    require(observed_owners == set(range(8)), "first-anchor proof misses D owners")
    return evidence


def access_records(directory):
    result = []
    paths = {
        path
        for pattern in ("access_r*_s*.log*", "access.log*")
        for path in Path(directory).rglob(pattern)
        if path.is_file()
    }
    for path in sorted(paths):
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    start = line.find("{")
                    if start < 0:
                        continue
                    try:
                        row = json.loads(line[start:])
                    except ValueError:
                        continue
                    if "request" in row and "response" in row:
                        result.append(row)
        except FileNotFoundError:
            # An active frontend can rotate between glob and open. Its next
            # terminal-access poll reads the renamed file/current path.
            continue
    return result


def assert_cancellation(case, recovery, access, native_text):
    require(case.get("intentionally_disconnected") is True, "no actual HTTP disconnect")
    require(case["output_ids"][0], "cancel happened before any native output")
    matched = [
        r
        for r in access
        if r["request"]["request_json"].get("generate_config", {}).get("trace_id")
        == case["trace_id"]
    ]
    require(
        len(matched) == 1 and matched[0]["response"].get("exception") is not None,
        "missing actual cancelled frontend access record",
    )
    identity = str(matched[0]["id"])
    pattern = (
        r"request \[(?:[^\]\n]*_request_id_)?"
        + re.escape(identity)
        + r"\] (?:cancelled by user|write outputs pb failed)"
    )
    require(
        re.search(pattern, native_text),
        "disconnect lacks matching native RPC cancellation",
    )
    require(
        recovery["started_ns"] > case["disconnected_ns"],
        "recovery was not issued after disconnect",
    )
    require(
        recovery.get("http_pd_generation_ok") is True,
        "post-cancel native recovery did not complete",
    )
    return dict(
        frontend_request_id=matched[0]["id"],
        trace_id=case["trace_id"],
        recovery_trace_id=recovery["trace_id"],
    )


def handoff_initial_rows(capture, decode_events):
    """Resolve actual scheduler rounds before selecting individual import rows."""
    frame = capture
    if capture["kind"] == "proposal":
        # Proposal allocation/table identities can change when a batch shrinks.
        # Only its immediate actual target call supplies stable request rounds.
        following = [
            event
            for event in decode_events
            if event["event"] == "begin"
            and event["identity"]["pid"] == capture["identity"]["pid"]
            and event["identity"]["dp_rank"] == capture["identity"]["dp_rank"]
            and event["monotonic_ns"] >= capture["began_ns"]
            and event["capture_id"] != capture["capture_id"]
        ]
        require(following, "proposal lacks a matching actual target identity frame")
        frame = min(following, key=lambda event: event["monotonic_ns"])
        require(
            frame["kind"] == "verify" and frame["identity"]["model"] == "target",
            "proposal is not immediately followed by actual target verify",
        )
        require(
            frame["prefixes"] == capture["prefixes"],
            "proposal and immediate target request-major prefixes differ",
        )
        require(
            capture["request_ids"] is None
            or capture["request_ids"] == frame["request_ids"],
            "proposal scheduler IDs differ from immediate target verify",
        )
    ids, rounds = frame["request_ids"], frame["rounds"]
    require(
        ids is not None
        and len(ids) == len(rounds) == len(capture["prefixes"])
        and all(type(identity) is int and identity >= 0 for identity in ids)
        and len(set(ids)) == len(ids)
        and all(type(index) is int and index >= 0 for index in rounds),
        "D import lacks unambiguous scheduler identity/round rows",
    )
    return [
        (row, identity, capture["prefixes"][row])
        for row, (identity, index) in enumerate(zip(ids, rounds))
        if index == 0
    ]


def assert_native_handoff(prefill_captures, decode_captures, decode_events):
    """Match actual P payload bytes to actual D initial native import bytes."""
    published = {}
    p_owners = defaultdict(set)
    for capture in prefill_captures:
        if capture["kind"] not in ("prefill", "commit"):
            continue
        require(
            capture["request_ids"] is not None,
            "P publication lacks native request identities",
        )
        model = capture["model"]
        for pool in capture["pools"]:
            for row, request_id in enumerate(capture["request_ids"]):
                end = capture["prefixes"][row] + capture["lengths"][row]
                if pool["tag"] in FULL:
                    positions = [
                        index * pool["ratio"]
                        for index in range(
                            max(0, end // pool["ratio"] - 2), end // pool["ratio"]
                        )
                    ]
                elif pool["tag"] == "swa_kv":
                    positions = range(max(0, end - capture["window"]), end)
                else:
                    positions = [end - 1] if end % 2 else []
                for position in positions:
                    page, columns = physical_entry(pool, row, position)
                    packed = pool["after"][page, columns]
                    key = (model, request_id, end, tuple(pool["key"]), position)
                    if key in published:
                        require(
                            torch.equal(published[key], packed),
                            "P CP owners published different native payload bytes",
                        )
                    published[key] = packed
                p_owners[(model, request_id, end)].add(capture["identity"]["cp_rank"])
    matched, owners = defaultdict(int), set()
    owners_by_model = defaultdict(set)
    for capture in decode_captures:
        if capture["kind"] not in ("verify", "proposal"):
            continue
        initial_rows = handoff_initial_rows(capture, decode_events)
        for pool in capture["pools"]:
            for row, request_id, prefix in initial_rows:
                if pool["tag"] in FULL:
                    positions = [
                        i * pool["ratio"]
                        for i in range(
                            max(0, prefix // pool["ratio"] - 2), prefix // pool["ratio"]
                        )
                    ]
                elif pool["tag"] == "swa_kv":
                    positions = range(max(0, prefix - capture["window"]), prefix)
                else:
                    positions = [prefix - 1] if prefix % 2 else []
                for position in positions:
                    key = (
                        capture["model"],
                        request_id,
                        prefix,
                        tuple(pool["key"]),
                        position,
                    )
                    require(
                        key in published,
                        f"missing actual P publication bytes for D import {key}",
                    )
                    page, columns = physical_entry(pool, row, position)
                    require(
                        torch.equal(pool["before"][page, columns], published[key]),
                        "native P/D cache payload mismatch",
                    )
                    matched[capture["model"]] += 1
                    owners.add(capture["identity"]["dp_rank"])
                    owners_by_model[capture["model"]].add(
                        capture["identity"]["dp_rank"]
                    )
    require(
        all(matched[model] > 0 for model in ("target", "draft")),
        "missing actual target/draft P/D handoff",
    )
    require(owners == set(range(8)), "native handoff byte proof misses D owners")
    require(
        all(owners_by_model[model] == set(range(8)) for model in ("target", "draft")),
        "native handoff byte proof misses D owners by model",
    )
    expected_cp = prefill_captures[0]["identity"]["cp_size"]
    require(
        all(len(ranks) == expected_cp for ranks in p_owners.values()),
        "P publication capture misses CP owners",
    )
    if prefill_captures[0]["identity"]["dp_size"] == 8:
        require(
            {c["identity"]["dp_rank"] for c in prefill_captures} == set(range(8)),
            "native P publication misses DP owners",
        )
    return dict(
        matched_native_rows=dict(matched),
        decode_owners=sorted(owners),
        decode_owners_by_model={
            model: sorted(owners_by_model[model]) for model in ("target", "draft")
        },
    )


class EvidenceParserTest(unittest.TestCase):
    @staticmethod
    def _handoff_fixture(*, mixed=False):
        """Independent byte fixtures; these are not native/device evidence."""
        prefill, decode, events = [], [], []
        for owner in range(8):
            request_id = 100 + owner
            prefixes = [1027, 129] if mixed and owner == 0 else [129]
            rounds = [2, 0] if len(prefixes) == 2 else [0]
            ids = [999, request_id] if len(prefixes) == 2 else [request_id]
            for model, kind in (("draft", "proposal"), ("target", "verify")):
                data = torch.full(
                    (1, 84480),
                    10 + owner + (80 if model == "draft" else 0),
                    dtype=torch.uint8,
                )
                pool = dict(
                    key=(0, "swa_kv"),
                    tag="swa_kv",
                    span=256,
                    entries=132,
                    ratio=1,
                    shape=[16, 84480],
                    tables=[[12]],
                    blocks=[12],
                    after=data,
                )
                for cp in range(8):
                    prefill.append(
                        dict(
                            model=model,
                            kind="commit" if model == "draft" else "prefill",
                            request_ids=[request_id],
                            prefixes=[0],
                            lengths=[129],
                            window=128,
                            identity=dict(cp_rank=cp, cp_size=8, dp_rank=0, dp_size=1),
                            pools=[deepcopy(pool)],
                        )
                    )
                imported = deepcopy(pool)
                imported["before"] = data.clone()
                if len(prefixes) == 2:
                    imported.update(
                        tables=[[-1, -1, -1, 10, 11], [12, 0, 0, 0, 0]],
                        blocks=[10, 11, 12],
                        before=torch.cat((torch.zeros_like(data).repeat(2, 1), data)),
                    )
                capture = dict(
                    model=model,
                    kind=kind,
                    request_ids=None if kind == "proposal" else ids,
                    prefixes=list(prefixes),
                    lengths=[3 if kind == "proposal" else 4] * len(ids),
                    rounds=list(rounds),
                    window=128,
                    pools=[imported],
                    capture_id=f"unit-{kind}-{owner}",
                    began_ns=owner * 1000 + (100 if kind == "proposal" else 110),
                    identity=dict(
                        model=model, pid=1000 + owner, dp_rank=owner, cp_rank=0
                    ),
                )
                decode.append(capture)
                events.append(
                    dict(
                        event="begin",
                        kind=kind,
                        capture_id=capture["capture_id"],
                        monotonic_ns=capture["began_ns"] + 1,
                        identity=deepcopy(capture["identity"]),
                        request_ids=list(ids),
                        prefixes=list(prefixes),
                        rounds=list(rounds),
                    )
                )
        return prefill, decode, events

    def test_handoff_mixed_first_rows_are_checked_for_both_models(self):
        prefill, decode, events = self._handoff_fixture(mixed=True)
        proof = assert_native_handoff(prefill, decode, events)
        self.assertEqual(proof["decode_owners"], list(range(8)))
        self.assertEqual(
            proof["decode_owners_by_model"],
            dict(target=list(range(8)), draft=list(range(8))),
        )
        self.assertEqual(proof["matched_native_rows"], dict(target=1024, draft=1024))
        for model in ("target", "draft"):
            with self.subTest(model=model):
                broken = deepcopy(decode)
                capture = next(c for c in broken if c["model"] == model)
                # Row1 is the actual first request, even though row0 is round2.
                capture["pools"][0]["before"][2, 528] ^= 1
                with self.assertRaisesRegex(AssertionError, "payload mismatch"):
                    assert_native_handoff(prefill, broken, events)

    def test_handoff_batch_compaction_round_reset_is_not_a_new_import(self):
        prefill, decode, events = self._handoff_fixture(mixed=True)
        for kind in ("proposal", "verify"):
            later = deepcopy(next(c for c in decode if c["kind"] == kind))
            later.update(
                capture_id="later-" + kind,
                request_ids=None if kind == "proposal" else [100],
                prefixes=[131],
                lengths=[3 if kind == "proposal" else 4],
                rounds=[0 if kind == "proposal" else 2],
                began_ns=200 if kind == "proposal" else 210,
            )
            later["pools"][0].update(tables=[[12]], blocks=[12])
            decode.append(later)
            events.append(
                dict(
                    event="begin",
                    kind=kind,
                    capture_id=later["capture_id"],
                    monotonic_ns=later["began_ns"] + 1,
                    identity=deepcopy(later["identity"]),
                    request_ids=[100],
                    prefixes=[131],
                    rounds=[2],
                )
            )
        # P published129, never131. D rounds129/130 have already committed.
        proof = assert_native_handoff(prefill, decode, events)
        self.assertEqual(proof["matched_native_rows"], dict(target=1024, draft=1024))

    def test_handoff_first_scheduler_round_ignores_proposal_allocation_counter(self):
        prefill, decode, events = self._handoff_fixture()
        proposal = next(c for c in decode if c["kind"] == "proposal")
        proposal["rounds"] = [3]
        self.assertEqual(
            assert_native_handoff(prefill, decode, events)["matched_native_rows"][
                "draft"
            ],
            1024,
        )
        proposal["pools"][0]["before"][0, 528] ^= 1
        with self.assertRaisesRegex(AssertionError, "payload mismatch"):
            assert_native_handoff(prefill, decode, events)

    def test_handoff_missing_first_publication_never_skips_the_row(self):
        for model in ("target", "draft"):
            with self.subTest(model=model):
                prefill, decode, events = self._handoff_fixture(mixed=True)
                prefill = [
                    c
                    for c in prefill
                    if not (c["model"] == model and c["request_ids"] == [100])
                ]
                with self.assertRaisesRegex(
                    AssertionError, "missing actual P publication"
                ):
                    assert_native_handoff(prefill, decode, events)

    def test_handoff_wrong_identity_or_immediate_association_fails(self):
        for defect, reason in (
            ("provided_id", "scheduler IDs differ"),
            ("unknown_id", "missing actual P publication"),
            ("missing_ids", "unambiguous scheduler"),
            ("missing_rounds", "unambiguous scheduler"),
            ("wrong_owner", "lacks a matching actual target"),
            ("wrong_pid", "lacks a matching actual target"),
            ("wrong_prefix_then_matching", "prefixes differ"),
            ("intervening_proposal", "not immediately followed"),
            ("wrong_target_model", "not immediately followed"),
        ):
            with self.subTest(defect=defect):
                prefill, decode, events = self._handoff_fixture()
                proposal = next(c for c in decode if c["kind"] == "proposal")
                following = next(e for e in events if e["kind"] == "verify")
                if defect == "provided_id":
                    proposal["request_ids"] = [999]
                elif defect == "unknown_id":
                    following["request_ids"] = [999]
                elif defect == "missing_ids":
                    following["request_ids"] = None
                elif defect == "missing_rounds":
                    following["rounds"] = []
                elif defect == "wrong_owner":
                    following["identity"]["dp_rank"] = 1
                elif defect == "wrong_pid":
                    following["identity"]["pid"] = 9999
                elif defect == "wrong_prefix_then_matching":
                    matching = deepcopy(following)
                    matching.update(capture_id="later-matching", monotonic_ns=112)
                    events.append(matching)
                    following["prefixes"] = [130]
                elif defect == "intervening_proposal":
                    extra = deepcopy(events[0])
                    extra.update(capture_id="intervening", monotonic_ns=105)
                    events.append(extra)
                else:
                    following["identity"]["model"] = "draft"
                with self.assertRaisesRegex(AssertionError, reason):
                    assert_native_handoff(prefill, decode, events)

    def test_handoff_native_owner_gates_remain_mandatory(self):
        for defect, reason in (
            ("D_owner", "misses D owners"),
            ("draft_D_owner", "misses D owners by model"),
            ("P_CP_owner", "misses CP owners"),
            ("draft_model", "missing actual target/draft"),
        ):
            with self.subTest(defect=defect):
                prefill, decode, events = self._handoff_fixture()
                if defect == "D_owner":
                    decode = [c for c in decode if c["identity"]["dp_rank"] != 7]
                elif defect == "draft_D_owner":
                    decode = [
                        c
                        for c in decode
                        if not (c["model"] == "draft" and c["identity"]["dp_rank"] == 7)
                    ]
                elif defect == "P_CP_owner":
                    prefill = [c for c in prefill if c["identity"]["cp_rank"] != 7]
                else:
                    decode = [c for c in decode if c["model"] != "draft"]
                with self.assertRaisesRegex(AssertionError, reason):
                    assert_native_handoff(prefill, decode, events)

    @staticmethod
    def _sse_event(tokens, total, step, *, finished=False, input_len=129):
        return dict(
            output_ids=[tokens],
            finished=finished,
            aux_info=dict(
                output_len=total,
                step_output_len=step,
                input_len=input_len,
                pd_sep=True,
                cum_log_probs=[],
            ),
        )

    def test_sse_native_two_incremental_packets_preserve_ids_and_disconnect(self):
        # Native contract request46 emitted these exact IDs/counts despite
        # return_incremental=False. Unrelated auxiliary fields are omitted.
        packets = [self._sse_event([1], 1, 1), self._sse_event([90], 2, 1)]
        with self.assertRaisesRegex(RuntimeError, "output_len"):
            output_tokens(packets[1], expected_input_tokens=129)
        lines = [
            ("data: " + json.dumps(packet) + "\r\n").encode() for packet in packets
        ]
        wire = [lines[0], b"\r\n", lines[1], b"\r\n", b"data: [DONE]\r\n"]
        response, connection = MagicMock(), MagicMock()
        response.status = 200
        response.__iter__.return_value = iter(wire)
        connection.getresponse.return_value = response
        query = make_query("hello", 256, stream=True)
        query["generate_config"]["ignore_eos"] = True
        with tempfile.TemporaryDirectory() as directory, patch(
            "rtp_llm.tools.v41.native_smoke.http.client.HTTPConnection",
            return_value=connection,
        ), patch("builtins.print"):
            record = run_request(
                "http://127.0.0.1:8080",
                query,
                directory,
                request=46,
                expected_input_tokens=129,
                cancel_after_tokens=2,
            )
            root = Path(directory)
            events = json.loads((root / "request-46.events.json").read_text())
            self.assertEqual([event["response"] for event in events], packets)
            self.assertEqual(events[-1]["cumulative_output_ids"], [[1, 90]])
            self.assertEqual(events[-1]["output_ids_mode"], "incremental")
            self.assertEqual(
                (root / "request-46.sse.txt").read_bytes(), b"".join(wire[:3])
            )
        self.assertEqual(record["output_ids"], [[1, 90]])
        self.assertTrue(record["intentionally_disconnected"])
        self.assertTrue(record["cancellation_requires_native_terminal_evidence"])
        self.assertFalse(events[-1]["response"]["finished"])
        response.close.assert_called_once()
        connection.close.assert_called_once()

    def test_sse_cumulative_and_multi_token_deltas_form_one_exact_chain(self):
        chain = SSETokenChain(expected_input_tokens=129, max_new_tokens=6)
        self.assertEqual(chain.append(self._sse_event([7], 1, 1)), [7])
        self.assertEqual(chain.append(self._sse_event([7, 8, 8], 3, 2)), [7, 8, 8])
        self.assertEqual(chain.last_mode, "cumulative")
        self.assertEqual(chain.append(self._sse_event([9, 10], 5, 2)), [7, 8, 8, 9, 10])
        self.assertEqual(chain.last_mode, "incremental")
        final = self._sse_event([7, 8, 8, 9, 10, 10], 6, 1, finished=True)
        self.assertEqual(chain.append(final), [7, 8, 8, 9, 10, 10])
        self.assertEqual(final["output_ids"], [[7, 8, 8, 9, 10, 10]])
        with self.assertRaisesRegex(RuntimeError, "followed a finished"):
            chain.append(self._sse_event([11], 7, 1))

    def test_sse_malformed_counts_ids_and_metadata_never_advance_chain(self):
        invalid = []
        for field in ("output_len", "step_output_len"):
            for value in (None, True, 2.0, "2", -1, 0):
                event = self._sse_event([90], 2, 1)
                if value is None:
                    del event["aux_info"][field]
                else:
                    event["aux_info"][field] = value
                invalid.append((f"{field}={value!r}", event))
        invalid.extend(
            [
                ("duplicate", self._sse_event([1], 1, 1)),
                ("regression", self._sse_event([], 0, 1)),
                ("gap", self._sse_event([90], 3, 1)),
                ("wrong step", self._sse_event([90], 2, 2)),
                ("short delta", self._sse_event([90], 3, 2)),
                ("excess IDs", self._sse_event([1, 90, 91, 92], 3, 2)),
                ("changed prefix", self._sse_event([2, 90], 2, 1)),
                ("negative ID", self._sse_event([-1], 2, 1)),
                ("boolean ID", self._sse_event([True], 2, 1)),
                ("string ID", self._sse_event(["90"], 2, 1)),
                ("changed input", self._sse_event([90], 2, 1, input_len=130)),
                ("boolean input", self._sse_event([90], 2, 1, input_len=True)),
                ("missing input", self._sse_event([90], 2, 1, input_len=None)),
                ("missing finished", dict(output_ids=[[90]], aux_info={})),
                ("nonboolean finished", self._sse_event([90], 2, 1, finished=0)),
                ("wrong shape", dict(output_ids=[90], aux_info={}, finished=False)),
                ("not an object", []),
                ("native error", dict(error_code=1)),
            ]
        )
        for field, value in (("pd_sep", False), ("cum_log_probs", [float("nan")])):
            event = self._sse_event([90], 2, 1)
            event["aux_info"][field] = value
            invalid.append((field, event))
        for name, event in invalid:
            with self.subTest(name=name):
                # Even without an expected length, the observed input count
                # must remain stable. A rejected packet cannot advance state.
                chain = SSETokenChain(max_new_tokens=3)
                self.assertEqual(chain.append(self._sse_event([1], 1, 1)), [1])
                with self.assertRaises(RuntimeError):
                    chain.append(event)
                self.assertEqual(chain.append(self._sse_event([90], 2, 1)), [1, 90])
        chain = SSETokenChain(expected_input_tokens=129, max_new_tokens=1)
        with self.assertRaisesRegex(RuntimeError, "boundary case"):
            chain.append(self._sse_event([1], 1, 1, input_len=128))
        chain.append(self._sse_event([1], 1, 1))
        with self.assertRaisesRegex(RuntimeError, "output cap"):
            chain.append(self._sse_event([90], 2, 1))

    def test_sse_failed_or_finished_packet_is_saved_without_cancellation_success(self):
        for bad in (
            self._sse_event([90], 3, 1),
            self._sse_event([90], 2, 1, finished=True),
        ):
            with self.subTest(packet=bad):
                packets = [self._sse_event([1], 1, 1), bad]
                lines = [
                    ("data: " + json.dumps(packet) + "\n").encode()
                    for packet in packets
                ]
                response, connection = MagicMock(), MagicMock()
                response.status = 200
                response.__iter__.return_value = iter(lines)
                connection.getresponse.return_value = response
                with tempfile.TemporaryDirectory() as directory, patch(
                    "rtp_llm.tools.v41.native_smoke.http.client.HTTPConnection",
                    return_value=connection,
                ):
                    with self.assertRaises(RuntimeError):
                        run_request(
                            "http://127.0.0.1:8080",
                            make_query("hello", 256, stream=True),
                            directory,
                            expected_input_tokens=129,
                            cancel_after_tokens=2,
                        )
                    root = Path(directory)
                    events = json.loads((root / "request-0.events.json").read_text())
                    self.assertEqual([event["response"] for event in events], packets)
                    self.assertEqual(
                        (root / "request-0.sse.txt").read_bytes(), b"".join(lines)
                    )
                    self.assertFalse((root / "request-0.summary.json").exists())
                response.close.assert_called_once()
                connection.close.assert_called_once()

    def test_log_backup_count_uses_existing_config_cli_and_environment_binding(self):
        from rtp_llm.config.py_config_modules import PyEnvConfigs
        from rtp_llm.server.server_args.profile_debug_logging_group_args import (
            init_profile_debug_logging_group_args,
        )
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        for args, env, expected in (
            ([], {}, 16),
            ([], {"LOG_FILE_BACKUP_COUNT": "4096"}, 4096),
            (None, {"LOG_FILE_BACKUP_COUNT": "4096"}, 4096),
            (
                ["--log_file_backup_count", "128"],
                {"LOG_FILE_BACKUP_COUNT": "4096"},
                128,
            ),
        ):
            with self.subTest(args=args, env=env), patch.dict(
                os.environ, env, clear=True
            ), patch.object(sys, "argv", ["server"]):
                config = PyEnvConfigs()
                parser = EnvArgumentParser()
                parser.set_root_config(config)
                init_profile_debug_logging_group_args(
                    parser, config.profiling_debug_logging_config
                )
                parser.parse_args(args)
                self.assertEqual(
                    config.profiling_debug_logging_config.log_file_backup_count,
                    expected,
                )

    def test_only_contracts_retains_full_finite_model_input_plan(self):
        from rtp_llm.models_py.modules.dsv41.ppu.test.dspark_workload import (
            configure_contract_logging,
        )

        initial = {
            "LOG_FILE_BACKUP_COUNT": "23",
            "ENABLE_MODEL_INPUTS_LOG": "0",
            "RTP_LLM_MODEL_INPUTS_LOG_MAX_RECORD_BYTES": "65536",
            "RTP_LLM_MODEL_INPUTS_LOG_MAX_QUEUE_BYTES": "131072",
        }
        env, output = dict(initial), Path("unit-output")
        self.assertIsNone(
            configure_contract_logging(env, "smoke", output, [129, 257, 1025], 8, 3)
        )
        self.assertEqual(env, initial)
        for gamma, token_cap, record_bound in (
            (1, 1132, 3797),
            (3, 1172, 3917),
            (5, 1212, 4037),
        ):
            with self.subTest(gamma=gamma):
                env = dict(initial)
                retention = configure_contract_logging(
                    env, "contracts", output, [129, 257, 1025], 8, gamma
                )
                self.assertEqual(retention["cases"], 80)
                self.assertEqual(retention["output_token_cap"], token_cap)
                self.assertEqual(retention["model_input_record_bound"], record_bound)
                self.assertEqual(retention["log_file_backup_count"], 4096)
                self.assertLess(record_bound, int(env["LOG_FILE_BACKUP_COUNT"]))
                self.assertEqual(env["ENABLE_MODEL_INPUTS_LOG"], "1")
                self.assertEqual(env["DISABLE_ACCESS_LOG"], "0")
                self.assertEqual(
                    int(env["RTP_LLM_MODEL_INPUTS_LOG_MAX_RECORD_BYTES"]),
                    retention["model_input_record_byte_limit"],
                )
                self.assertEqual(
                    retention["model_input_record_byte_limit"], 128 * 1024**2
                )
                self.assertEqual(
                    retention["model_input_chunk_byte_limit"],
                    retention["model_input_record_byte_limit"],
                )
                self.assertEqual(
                    int(env["RTP_LLM_MODEL_INPUTS_LOG_MAX_QUEUE_BYTES"]),
                    retention["model_input_queue_byte_limit"],
                )
                self.assertEqual(
                    retention["model_input_queue_byte_limit"], 512 * 1024**2
                )
                self.assertEqual(
                    env["RTP_V41_DSPARK_OBSERVE_DIR"],
                    str(output / "native-observations"),
                )
        retention = configure_contract_logging(
            env, "contracts", output, [8193, 32769, 33793], 32, 3
        )
        self.assertEqual(retention["log_file_backup_count"], 16384)
        self.assertGreaterEqual(
            retention["log_file_backup_count"], retention["model_input_record_bound"]
        )

    def test_contract_logger_capacity_keeps_complete_32k_cp8_features(self):
        from rtp_llm.models_py.modules.dsv41.ppu.test.dspark_workload import (
            configure_contract_logging,
        )

        env = {}
        retention = configure_contract_logging(
            env, "contracts", Path("unit-output"), [129, 257, 32768], 8, 5
        )
        # Actual observed width/dtype, source-derived CP8 zigzag row count.
        feature_bytes = 4096 * 15360 * 2
        record_budget = feature_bytes + 8 * 32768 + 48 * 128 + 65536
        self.assertGreater(feature_bytes, 64 * 1024**2)
        self.assertLess(record_budget, retention["model_input_record_byte_limit"])
        self.assertLess(4 * record_budget, retention["model_input_queue_byte_limit"])
        self.assertEqual(
            int(env["RTP_LLM_MODEL_INPUTS_LOG_MAX_RECORD_BYTES"]), 128 * 1024**2
        )
        self.assertEqual(
            int(env["RTP_LLM_MODEL_INPUTS_LOG_MAX_QUEUE_BYTES"]), 512 * 1024**2
        )
        self.assertEqual(retention["log_file_backup_count"], 4096)

    def test_real_access_filenames_and_rotations_are_read_from_disk(self):
        from rtp_llm.access_logger.log_utils import get_process_log_filename

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "logs"
            directory.mkdir()
            names = [
                get_process_log_filename("access.log", 0, 0),
                get_process_log_filename("access.log", 5, 2) + ".1",
                get_process_log_filename("access.log", 7, 1) + ".2.gz",
                "access.log.1",
            ]
            ids = [3624731568649859078, 2, 3, 4]
            for name, identity in zip(names, ids):
                text = (
                    "prefix "
                    + json.dumps(
                        dict(
                            id=identity,
                            request=dict(request_json={}),
                            response=dict(exception="CancelledError"),
                        )
                    )
                    + "\ninvalid\n{}\n"
                )
                path = directory / name
                if path.suffix == ".gz":
                    with gzip.open(path, "wt", encoding="utf-8") as stream:
                        stream.write(text)
                else:
                    path.write_text(text)
            (directory / "query_access_r0_s0.log").write_text(
                json.dumps(dict(id=99, request={}, response={}))
            )
            rows = access_records(Path(root))
            self.assertEqual({row["id"] for row in rows}, set(ids))
            self.assertEqual(len(rows), len(ids))

    @staticmethod
    def _observer_pool_fixture(tag, tables, *, entries=6, span=256, blocks=64):
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_observation import (
            NativeDSparkObserver,
        )

        # Observed span, CSA entries and owned block IDs; a smaller real CPU
        # pool keeps fixtures bounded. No model/bridge implementation is used.
        tensor = torch.zeros(
            (blocks, entries * WIDTH[tag]),
            dtype=torch.float32 if tag == "csa_state" else torch.uint8,
            device="cpu",
        )
        for block in (46, 47):
            if block < blocks:
                tensor[block].fill_(block)
        pool = SimpleNamespace(
            framework_tag="csa_kv" if tag == "global_kv_2" else tag,
            tensor=tensor,
            table=torch.tensor(tables, dtype=torch.int32, device="cpu"),
            entries=entries,
            span=span,
        )
        bridge = SimpleNamespace(
            pools={(2, tag): pool},
            logical=SimpleNamespace(
                geometry=SimpleNamespace(window=128, ratio=lambda _: 2)
            ),
        )
        observer = NativeDSparkObserver.__new__(NativeDSparkObserver)
        observer.block_limit = 32
        return observer, bridge, pool

    @staticmethod
    def _observer_capture_fixture(spec, before, after, prefixes, lengths):
        """Independent raw-byte oracle fixture, never native-run evidence."""
        selected = {k: deepcopy(v) for k, v in spec.items() if k != "native_pool"}
        selected.update(before=before, after=after)
        pools = [selected]
        for layer in range(40):
            raw = torch.full((1, 132 * 528), layer, dtype=torch.uint8, device="cpu")
            pools.append(
                dict(
                    key=(layer, "swa_kv"),
                    tag="swa_kv",
                    entries=132,
                    span=256,
                    ratio=1,
                    tables=[[1] for _ in prefixes],
                    blocks=[1],
                    shape=[64, 132 * 528],
                    data_ptr=raw.data_ptr(),
                    neighbor_positions=[[0, 0]],
                    before=raw,
                    after=raw.clone(),
                )
            )
        return dict(
            schema_version=1,
            complete=True,
            model="target",
            kind="prefill",
            prefixes=prefixes,
            lengths=lengths,
            observation_positions=None,
            outcome="accept",
            accepted_input_rows=list(lengths),
            pools=pools,
            pool_keys=[p["key"] for p in pools],
            imports=[],
            expected=[],
        )

    def test_noop_csa_sparse_even_cold_8k_32k_keeps_owned_bytes(self):
        for end in (8192, 32768):
            with self.subTest(end=end):
                table = [-1] * (end // 256 - 2) + [46, 47, 0]
                observer, bridge, pool = self._observer_pool_fixture(
                    "csa_state", [table]
                )
                spec = observer._specs(bridge, [0], [end], "prefill", None)[0]
                self.assertEqual(spec["blocks"], [46, 47])
                self.assertEqual(spec["neighbor_positions"], [[0, end - 258]])
                self.assertEqual(
                    spec["noop_rows"],
                    [
                        dict(
                            row=0,
                            end=end,
                            witness_positions=[end - 257, end - 1],
                            coverage="UNTOUCHED_OWNED_PAGES",
                        )
                    ],
                )
                before = observer._snapshot([spec])[0]
                self.assertTrue(
                    torch.equal(before[0], pool.tensor[46].view(torch.uint8))
                )
                self.assertTrue(
                    torch.equal(before[1], pool.tensor[47].view(torch.uint8))
                )
                self.assertEqual(
                    observer._cache_payloads(bridge, None, [spec], [end]), []
                )
                self.assertEqual(
                    observer._cache_payloads(bridge, None, [spec], [0]), []
                )
                capture = self._observer_capture_fixture(
                    spec, before, before.clone(), [0], [end]
                )
                result = assert_capture(capture)
                self.assertEqual(result["noop_owned_byte_comparisons"], 2 * 6144 * 4)
                self.assertEqual(result["noop_coverage"][0]["witness_blocks"], [46, 47])

    def test_noop_csa_every_owned_page_mutation_is_rejected(self):
        for block in (46, 47):
            with self.subTest(block=block):
                observer, bridge, pool = self._observer_pool_fixture(
                    "csa_state", [[-1] * 30 + [46, 47, 0]]
                )
                spec = observer._specs(bridge, [0], [8192], "prefill", None)[0]
                before = observer._snapshot([spec])[0]
                pool.tensor[block, -1] += 1
                after = observer._snapshot([spec])[0]
                capture = self._observer_capture_fixture(
                    spec, before, after, [0], [8192]
                )
                with self.assertRaisesRegex(
                    AssertionError, "no-op owned page bytes changed"
                ):
                    assert_capture(capture)

    def test_noop_csa_no_owned_pages_reports_no_byte_coverage(self):
        for absent in (-1, 0):
            with self.subTest(absent=absent):
                observer, bridge, _ = self._observer_pool_fixture(
                    "csa_state", [[absent] * 32 + [0]]
                )
                spec = observer._specs(bridge, [0], [8192], "prefill", None)[0]
                self.assertEqual(spec["blocks"], [])
                self.assertEqual(spec["neighbor_positions"], [])
                self.assertEqual(spec["noop_rows"][0]["witness_positions"], [])
                self.assertEqual(
                    spec["noop_rows"][0]["coverage"], "NO_OWNED_PAGES_NO_BYTE_COVERAGE"
                )
                before = observer._snapshot([spec])[0]
                self.assertEqual(tuple(before.shape), (0, 6144 * 4))
                capture = self._observer_capture_fixture(
                    spec, before, before.clone(), [0], [8192]
                )
                result = assert_capture(capture)
                self.assertEqual(result["noop_owned_byte_comparisons"], 0)
                self.assertEqual(
                    result["noop_coverage"][0]["coverage"],
                    "NO_OWNED_PAGES_NO_BYTE_COVERAGE",
                )
                self.assertGreater(
                    result["untouched_byte_comparisons"], 0
                )  # Other fixture pages.
                empty_frame = deepcopy(capture)
                for other in empty_frame["pools"][1:]:
                    other["before"] = other["before"][:0]
                    other["after"] = other["after"][:0]
                with self.assertRaisesRegex(
                    AssertionError, "no physical byte coverage"
                ):
                    assert_capture(empty_frame)

    def test_noop_csa_padding_neighbor_is_not_request_owned(self):
        observer, bridge, _ = self._observer_pool_fixture(
            "csa_state", [[0, 46, 0]], span=1
        )
        spec = observer._specs(bridge, [0], [2], "prefill", None)[0]
        self.assertEqual(spec["blocks"], [46])
        self.assertEqual(spec["neighbor_positions"], [])
        self.assertEqual(spec["noop_rows"][0]["witness_positions"], [1])
        before = observer._snapshot([spec])[0]
        capture = self._observer_capture_fixture(spec, before, before.clone(), [0], [2])
        self.assertEqual(
            assert_capture(capture)["noop_owned_byte_comparisons"], 6144 * 4
        )

    def test_noop_csa_true_affected_holes_still_fail(self):
        for kind, width in (("prefill", 1), ("verify", 2)):
            for missing in (-1, 0):
                with self.subTest(kind=kind, missing=missing):
                    observer, bridge, _ = self._observer_pool_fixture(
                        "csa_state", [[missing] * 34]
                    )
                    with self.assertRaisesRegex(
                        ValueError, "selected native row is unallocated"
                    ):
                        observer._specs(bridge, [8192], [width], kind, None)

    def test_noop_csa_true_imported_hole_still_fails(self):
        for missing in (-1, 0):
            with self.subTest(missing=missing):
                observer, bridge, _ = self._observer_pool_fixture(
                    "csa_state", [[missing] * 34]
                )
                with self.assertRaisesRegex(
                    ValueError, "selected native row is unallocated"
                ):
                    observer._specs(bridge, [8193], [1], "prefill", None)

    def test_noop_csa_odd_boundary_and_import_keep_semantic_payloads(self):
        for base in (8192, 8193):
            with self.subTest(base=base):
                observer, bridge, pool = self._observer_pool_fixture(
                    "csa_state", [[-1] * 31 + [46, 47, 0]]
                )
                spec = observer._specs(bridge, [base], [1], "prefill", None)[0]
                self.assertEqual(spec["noop_rows"], [])
                self.assertEqual(spec["blocks"], [46, 47])
                before = observer._snapshot([spec])[0]
                packed = torch.full(
                    (1, 1024),
                    17 if base == 8192 else 47,
                    dtype=torch.float32,
                    device="cpu",
                )
                cache = SimpleNamespace(
                    slots_at_positions=MagicMock(
                        return_value=torch.tensor([0], device="cpu")
                    ),
                    _gather=MagicMock(return_value=packed),
                )
                payloads = observer._cache_payloads(bridge, cache, [spec], [8193])
                self.assertEqual(payloads[0]["positions"].tolist(), [8192])
                if base == 8192:
                    pool.tensor[47, 2048:3072].copy_(
                        packed[0]
                    )  # Observed entries6:8192%6 ==2.
                else:
                    self.assertEqual(
                        observer._cache_payloads(bridge, cache, [spec], [8194]), []
                    )
                after = observer._snapshot([spec])[0]
                capture = self._observer_capture_fixture(
                    spec, before, after, [base], [1]
                )
                capture["expected" if base == 8192 else "imports"] = payloads
                self.assertEqual(assert_capture(capture)["noop_coverage"], [])
                forged = deepcopy(capture)
                forged["pools"][0]["noop_rows"] = [
                    dict(
                        row=0,
                        end=base + 1,
                        witness_positions=[],
                        coverage="NO_OWNED_PAGES_NO_BYTE_COVERAGE",
                    )
                ]
                with self.assertRaisesRegex(AssertionError, "real affected/imported"):
                    assert_capture(forged)

    def test_noop_csa_forged_witness_or_payload_is_rejected(self):
        observer, bridge, _ = self._observer_pool_fixture(
            "csa_state", [[-1] * 30 + [46, 47, 0]]
        )
        spec = observer._specs(bridge, [0], [8192], "prefill", None)[0]
        before = observer._snapshot([spec])[0]
        original = self._observer_capture_fixture(
            spec, before, before.clone(), [0], [8192]
        )
        for error in (
            "omitted_page",
            "semantic_payload",
            "future_page",
            "false_coverage",
        ):
            with self.subTest(error=error):
                capture = deepcopy(original)
                pool = capture["pools"][0]
                if error == "omitted_page":
                    pool["noop_rows"][0]["witness_positions"].pop(0)
                elif error == "semantic_payload":
                    capture["expected"] = [
                        dict(
                            key=(2, "csa_state"),
                            row=0,
                            positions=torch.tensor([8192], device="cpu"),
                            packed=torch.zeros(
                                (1, 1024), dtype=torch.float32, device="cpu"
                            ),
                        )
                    ]
                elif error == "future_page":
                    pool["blocks"].insert(0, 0)
                    for field in ("before", "after"):
                        pool[field] = torch.cat(
                            (
                                torch.zeros(
                                    (1, 6144 * 4), dtype=torch.uint8, device="cpu"
                                ),
                                pool[field],
                            )
                        )
                else:
                    pool["noop_rows"][0]["coverage"] = "NO_OWNED_PAGES_NO_BYTE_COVERAGE"
                with self.assertRaises(AssertionError):
                    assert_capture(capture)

    def test_noop_csa_corrupt_table_truncation_and_limit_still_fail(self):
        cases = (
            ([-1] * 30 + [64, 47, 0], 8192, "invalid physical block"),
            ([-1] * 31, 8192, "exceeds table"),
            (list(range(1, 34)) + [0], 33 * 256, "physical block limit"),
        )
        for table, end, error in cases:
            with self.subTest(error=error):
                observer, bridge, _ = self._observer_pool_fixture("csa_state", [table])
                with self.assertRaisesRegex(ValueError, error):
                    observer._specs(bridge, [0], [end], "prefill", None)

    def test_noop_csa_mixed_rows_keep_other_rows_semantic(self):
        observer, bridge, pool = self._observer_pool_fixture(
            "csa_state", [[-1] * 30 + [46, 47, 0], [30, 31] + [-1] * 31]
        )
        spec = observer._specs(bridge, [0, 0], [8192, 257], "prefill", None)[0]
        self.assertEqual([r["row"] for r in spec["noop_rows"]], [0])
        self.assertEqual(spec["blocks"], [30, 31, 46, 47])
        before = observer._snapshot([spec])[0]
        packed = torch.full((1, 1024), 19, dtype=torch.float32, device="cpu")
        pool.tensor[31, 4096:5120].copy_(packed[0])  # 256%6 ==4.
        after = observer._snapshot([spec])[0]
        capture = self._observer_capture_fixture(
            spec, before, after, [0, 0], [8192, 257]
        )
        capture["expected"] = [
            dict(
                key=(2, "csa_state"),
                row=1,
                positions=torch.tensor([256], device="cpu"),
                packed=packed,
            )
        ]
        self.assertEqual(
            assert_capture(capture)["noop_coverage"][0]["witness_blocks"], [46, 47]
        )

    def test_noop_fix_preserves_full_indexer_swa_selection(self):
        for tag, entries in (
            ("global_kv_2", 128),
            ("indexer_kv", 256),
            ("swa_kv", 132),
        ):
            with self.subTest(tag=tag):
                table = list(range(73, 106)) if tag in FULL else [-1] * 30 + [46, 47, 0]
                observer, bridge, _ = self._observer_pool_fixture(
                    tag, [table], entries=entries, blocks=128
                )
                spec = observer._specs(bridge, [0], [8192], "prefill", None)[0]
                self.assertNotIn("noop_rows", spec)
                self.assertEqual(spec["blocks"], [104, 105] if tag in FULL else [0, 47])
                self.assertEqual(
                    spec["neighbor_positions"],
                    [[0, 8192]] if tag in FULL else [[0, 8063], [0, 8192]],
                )

    def test_noop_fix_does_not_drop_full_indexer_swa_required_holes(self):
        for tag, entries in (
            ("global_kv_2", 128),
            ("indexer_kv", 256),
            ("swa_kv", 132),
        ):
            with self.subTest(tag=tag):
                table = list(range(73, 106)) if tag in FULL else [-1] * 30 + [46, 47, 0]
                table[31] = -1
                observer, bridge, _ = self._observer_pool_fixture(
                    tag, [table], entries=entries, blocks=128
                )
                with self.assertRaisesRegex(
                    ValueError, "selected native row is unallocated"
                ):
                    observer._specs(bridge, [0], [8192], "prefill", None)

    def test_native_queue_timeout_cli_default_override_and_invalid(self):
        from rtp_llm.tools.v41 import launch

        for extra, expected in (
            ([], "1800000"),
            (["--prefill-wait-timeout-ms", "5400000"], "5400000"),
        ):
            with self.subTest(expected=expected):
                argv = [
                    "launch",
                    "prefill",
                    "--checkpoint-path",
                    "/unused-checkpoint",
                    "--print-command",
                    *extra,
                ]
                with patch.dict(os.environ, {}, clear=True), patch.object(
                    sys, "argv", argv
                ), patch.object(sys, "stdout", io.StringIO()) as output, patch.object(
                    launch, "collect_report"
                ) as report, patch.object(
                    launch.os, "execvpe"
                ) as execute:
                    self.assertEqual(launch.main(), 0)
                    self.assertIn(
                        "PREFILL_MAX_WAIT_TIMEOUT_MS=" + expected, output.getvalue()
                    )
                    report.assert_not_called()
                    execute.assert_not_called()
        for value in ("0", "-1", "nan", "inf", "86400001"):
            with self.subTest(value=value):
                argv = [
                    "launch",
                    "prefill",
                    "--checkpoint-path",
                    "/unused-checkpoint",
                    "--prefill-wait-timeout-ms",
                    value,
                    "--print-command",
                ]
                with patch.object(sys, "argv", argv), patch.object(
                    sys, "stderr", io.StringIO()
                ), patch.object(launch, "collect_report") as report, patch.object(
                    launch.os, "execvpe"
                ) as execute:
                    with self.assertRaises(SystemExit) as rejected:
                        launch.main()
                    self.assertEqual(rejected.exception.code, 2)
                    report.assert_not_called()
                    execute.assert_not_called()

    def test_contract_request_deadline_bounds_query_and_http_timeout(self):
        from rtp_llm.models_py.modules.dsv41.ppu.test import dspark_workload as role

        for requested, remaining, expected in (
            (3600, 10000, 3600),
            (7200, 10000, 7200),
            (7200, 10, 10),
        ):
            with self.subTest(
                requested=requested, remaining=remaining
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text(json.dumps(dict(eos_token_id=1)))
                with patch.object(
                    role.time, "monotonic", return_value=100
                ), patch.object(
                    role,
                    "run_request",
                    side_effect=RuntimeError("intercepted HTTP only"),
                ) as request:
                    with self.assertRaisesRegex(RuntimeError, "intercepted HTTP only"):
                        role.exercise_contract_cases(
                            root / "native-unit",
                            "127.0.0.1:8090",
                            [129],
                            8,
                            3,
                            root,
                            False,
                            SimpleNamespace(),
                            100 + remaining,
                            request_timeout_seconds=requested,
                        )
                query = request.call_args.args[1]
                self.assertEqual(
                    query["generate_config"]["timeout_ms"], expected * 1000
                )
                self.assertEqual(request.call_args.kwargs["timeout"], expected)
                self.assertEqual(query["generate_config"]["max_new_tokens"], 8)
                self.assertEqual(query["generate_config"]["min_new_tokens"], 8)
                self.assertEqual(query["generate_config"]["random_seed"], 42)

    def test_contract_request_expired_deadline_never_dispatches(self):
        from rtp_llm.models_py.modules.dsv41.ppu.test import dspark_workload as role

        for remaining in (0, -1, 0.0005):
            with self.subTest(
                remaining=remaining
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text(json.dumps(dict(eos_token_id=1)))
                with patch.object(
                    role.time, "monotonic", return_value=100
                ), patch.object(role, "run_request") as request:
                    with self.assertRaises(TimeoutError):
                        role.exercise_contract_cases(
                            root / "native-unit",
                            "127.0.0.1:8090",
                            [129],
                            8,
                            3,
                            root,
                            False,
                            SimpleNamespace(),
                            100 + remaining,
                            request_timeout_seconds=7200,
                        )
                    request.assert_not_called()

    def test_full_pool_neighbors_cross_compressed_pages_without_write_alias(self):
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_observation import (
            full_pool_neighbor_positions,
        )

        for tag, entries, width in (("global_kv_2", 4, 288), ("indexer_kv", 8, 68)):
            with self.subTest(tag=tag):
                probes = full_pool_neighbor_positions(
                    [[2, 4]], 8, entries, 2, 8, [7], [1], "verify", [[2, 4, 6]]
                )
                self.assertEqual(probes, [[0, 4], [0, 8]])
                pool = dict(
                    tag=tag,
                    entries=entries,
                    span=8,
                    ratio=2,
                    tables=[[2, 4]],
                    blocks=[2, 4],
                    shape=[8, entries * width + 16],
                )
                coordinates = [
                    physical_entry(pool, row, position) for row, position in probes
                ]
                scale = 64 if tag == "indexer_kv" else 288
                self.assertEqual(
                    [
                        (pool["blocks"][page], columns[0] // scale)
                        for page, columns in coordinates
                    ],
                    [(2, 2), (4, 0)],
                )
        self.assertEqual(
            full_pool_neighbor_positions(
                [[2, -1]], 8, 4, 2, 8, [0], [8], "prefill", [[4, 6]]
            ),
            [],
        )
        self.assertEqual(
            full_pool_neighbor_positions(
                [[2, 2]], 8, 4, 2, 8, [0], [8], "prefill", [[4, 6]]
            ),
            [],
        )
        # Row0's next page is allocated, but its entry0 is written by row1.
        self.assertEqual(
            full_pool_neighbor_positions(
                [[2, 4], [4, 5]], 8, 4, 2, 8, [7, 1], [1, 1], "verify", [[2, 4, 6], [0]]
            ),
            [[0, 4], [1, 2]],
        )

    def test_only_committed_inputs_consume_positions_and_bonus_waits(self):
        records, events, cases = [], [], []
        for owner in range(8):
            trace, rid = f"unit-{owner}", 500 + owner
            cases.append(
                dict(
                    trace_id=trace,
                    output_ids=[[11, 12, 13, 14]],
                    aux_info=dict(input_len=129),
                )
            )
            for sequence, prefix, tokens, count in (
                (0, 129, [11, 12, 999, 888], 0),
                (1, 129, [11, 12, 999, 888], 2),
                (2, 131, [13, 777], 1),
            ):
                records.append(
                    dict(
                        record_sequence=sequence,
                        execution_stage="target_verify",
                        warmup=False,
                        is_fake_stream=False,
                        trace_ids=[trace],
                        request_id=torch.tensor([rid]),
                        input_lengths=torch.tensor([len(tokens)]),
                        prefix_lengths=torch.tensor([prefix]),
                        combo_tokens=torch.tensor(tokens),
                        combo_position_ids=torch.arange(prefix, prefix + len(tokens)),
                        native_owner=owner,
                    )
                )
                events.append(
                    dict(
                        event="finish",
                        kind="verify",
                        identity=dict(model="target", dp_rank=owner),
                        request_ids=[rid],
                        prefixes=[prefix],
                        accepted_input_rows=[count],
                        capture_id=f"unit-{owner}-{sequence}",
                    )
                )
                for stage, token in enumerate(tokens):
                    events.append(
                        dict(
                            event="compute",
                            kind="verify",
                            identity=dict(model="target", dp_rank=owner),
                            capture_id=f"unit-{owner}-{sequence}",
                            stage=stage,
                            request_ids=[rid],
                            positions=[prefix + stage],
                            tokens=[token],
                        )
                    )
            fake = dict(records[-1], is_fake_stream=True, native_owner=99)
            records.append(fake)
        proof = assert_first_anchors(records, events, cases)
        self.assertEqual({row["owner"] for row in proof}, set(range(8)))
        self.assertTrue(all(row["rounds"] == 3 for row in proof))

    def test_votes_require_actual_owner_sums_skew_and_empty_transition(self):
        events = []
        for model in ("target", "draft"):
            for index, counts in enumerate(([4] + [0] * 7, [8, 4] + [0] * 6, [0] * 8)):
                for owner, count in enumerate(counts):
                    events.append(
                        dict(
                            event="vote",
                            identity=dict(model=model, dp_rank=owner),
                            vote_index=index,
                            local_rows=count,
                            global_rows=sum(counts),
                            global_stop=False,
                            local_stop=False,
                            warmup=False,
                        )
                    )
        self.assertEqual(set(assert_work_votes(events)), {"target", "draft"})
        with self.assertRaisesRegex(AssertionError, "unaligned/lost"):
            assert_work_votes(events[:-1])
        events[0]["global_rows"] = 100
        with self.assertRaisesRegex(AssertionError, "reduced work sum"):
            assert_work_votes(events)

    def test_http_disconnect_without_native_cancel_does_not_pass(self):
        cancel = dict(
            intentionally_disconnected=True,
            output_ids=[[11, 12]],
            trace_id="unit-cancel",
            disconnected_ns=10,
        )
        recovery = dict(
            started_ns=11, http_pd_generation_ok=True, trace_id="unit-recovery"
        )
        access = [
            dict(
                id=500,
                request=dict(
                    request_json=dict(generate_config=dict(trace_id="unit-cancel"))
                ),
                response=dict(exception="CancelledError"),
            )
        ]
        with self.assertRaisesRegex(AssertionError, "native RPC cancellation"):
            assert_cancellation(cancel, recovery, access, "")
        self.assertEqual(
            assert_cancellation(
                cancel,
                recovery,
                access,
                "request [client_request_id_500] cancelled by user",
            )["frontend_request_id"],
            500,
        )

    def test_response_eos_padding_cannot_prove_eos(self):
        query = make_query("hello", 16, min_new_tokens=0)
        result = {
            "finished": True,
            "output_ids": [[7, 8, 1]],
            "aux_info": {"output_len": 2, "pd_sep": True},
        }
        self.assertEqual(output_tokens(result), [7, 8])
        with self.assertRaisesRegex(RuntimeError, "EOS"):
            validate_finished(result, query, eos_token_ids=[1])
        result["aux_info"]["output_len"] = 3
        self.assertEqual(validate_finished(result, query, eos_token_ids=[1]), [7, 8, 1])

    def test_independent_indexer_scale_plane_and_csa_byte_mapping(self):
        pool = dict(
            tag="indexer_kv",
            span=8,
            entries=8,
            ratio=2,
            tables=[[4, 2]],
            blocks=[2, 4],
            shape=[6, 560],
        )
        page, columns = physical_entry(pool, 0, 10)
        self.assertEqual(page, 0)
        self.assertEqual(columns, list(range(64, 128)) + list(range(516, 520)))
        pool.update(tag="csa_state", entries=4, shape=[6, 4112])
        page, columns = physical_entry(pool, 0, 9)
        self.assertEqual(page, 0)
        self.assertEqual(columns[0], 4096)
        self.assertEqual(columns[-1], 8191)

    def test_missing_observation_never_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, "no real native"):
                read_observations(directory, {})

    def test_native_padding_corruption_is_rejected_by_independent_oracle(self):
        pools, payloads = [], []
        for layer in range(3):
            raw = torch.full((1, 1072), 173, dtype=torch.uint8)
            pool = dict(
                key=(layer, "swa_kv"),
                tag="swa_kv",
                span=8,
                entries=2,
                ratio=1,
                tables=[[1]],
                blocks=[1],
                shape=[3, 1072],
                data_ptr=1000 + layer,
                neighbor_positions=[[0, 0]],
                before=raw,
                after=raw.clone(),
            )
            pools.append(pool)
            payloads.append(
                dict(
                    key=pool["key"],
                    row=0,
                    positions=torch.tensor([1]),
                    packed=torch.full((1, 528), layer + 1, dtype=torch.uint8),
                )
            )
        expected = apply_payloads(
            {tuple(p["key"]): p for p in pools},
            {tuple(p["key"]): p["before"] for p in pools},
            payloads,
        )
        for pool in pools:
            pool["after"] = expected[tuple(pool["key"])]
        capture = dict(
            schema_version=1,
            complete=True,
            model="draft",
            kind="commit",
            outcome="accept",
            pool_keys=[p["key"] for p in pools],
            pools=pools,
            expected=payloads,
        )
        assert_capture(capture)
        pools[1]["after"][0, -1] ^= 1
        with self.assertRaisesRegex(AssertionError, "padding changed"):
            assert_capture(capture)


if __name__ == "__main__":
    unittest.main()
