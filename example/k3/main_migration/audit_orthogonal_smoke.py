#!/usr/bin/env python3
"""Independently check K3 smoke answers and all-rank execution events."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import gcd
from typing import Any

MARKER = "[K3_SMOKE_EVENT] "
PAGED_MARKER = "[K3_MLA_PAGED_INPUT] "


def read_paged_contract(line: str) -> dict[str, Any]:
    timestamp = re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]", line)
    if timestamp is None:
        raise ValueError("paged input contract lacks a log timestamp")
    # The K3 launchers' host and container logs use China Standard Time.
    stamp = datetime.fromisoformat(timestamp.group(1)).replace(
        tzinfo=timezone(timedelta(hours=8))
    )
    values = dict(
        field.split("=", 1) for field in line.split(PAGED_MARKER, 1)[1].split()
    )
    for key in ("graph", "logical_batch", "physical_batch", "q", "physical_tokens"):
        values[key] = int(values[key])
    values.update(kind="mla_paged_input_contract", time_ns=int(stamp.timestamp() * 1e9))
    return values


def eager_paged_contract(
    events: list[dict[str, Any]],
    *,
    end_ns: int,
    phase: str,
    logical_batch: int,
    tp: int,
) -> bool:
    width = 1 if phase == "proposal_or_decode" else 4
    alignment = tp // gcd(tp, width)
    physical = (logical_batch + alignment - 1) // alignment * alignment
    dtype = "torch.float8_e4m3fn" if phase == "target_verify" else "torch.bfloat16"
    # Contracts are logged once per unique shape/workspace. An earlier
    # preparation is valid evidence; a later contract cannot certify this stage.
    return any(
        event.get("kind") == "mla_paged_input_contract"
        and event["time_ns"] <= end_ns
        and event.get("phase") == phase
        and event.get("graph") == 0
        and event.get("logical_batch") == logical_batch
        and event.get("physical_batch") == physical
        and event.get("q") == width
        and event.get("physical_tokens") == physical * width
        and event.get("operand_dtype") == dtype
        and event.get("projection_dtype") == "bf16"
        and event.get("backend") == "tokenspeed_page_rr"
        for event in events
    )


def cancel_host_load_evidence(
    item: dict[str, Any], prefill: dict[int, list[dict[str, Any]]]
) -> bool:
    started = item.get("load_started")
    cancelled = item.get("cancelled")
    done = item.get("load_done")
    if not isinstance(started, dict) or not isinstance(cancelled, dict):
        return False
    if (
        item.get("cancel_status") != 1
        or item.get("fetch", {}).get("code") != "StatusCode.RESOURCE_EXHAUSTED"
        or "preempted" not in item.get("fetch", {}).get("details", "")
        or started.get("time_ns", -1) > cancelled.get("time_ns", -1)
        or (
            isinstance(done, dict)
            and cancelled.get("time_ns", -1) >= done.get("time_ns", -1)
        )
    ):
        return False
    request_id = item.get("request_id")
    return all(
        any(
            event.get("event") == event_name
            and event.get("request_id") == request_id
            and event.get("time_ns") == recorded.get("time_ns")
            for rank in prefill.values()
            for event in rank
        )
        for event_name, recorded in (
            ("host_cache_load_started", started),
            ("prefill_priority_cancel_accepted", cancelled),
        )
    )


def read_events(
    directory: pathlib.Path, ranks: int = 8, engine_log: pathlib.Path | None = None
) -> dict[int, list[dict[str, Any]]]:
    result = {}
    for rank in range(ranks):
        path = directory / f"main_{rank}.log"
        if not path.is_file():
            raise ValueError(f"missing rank log: {path}")
        events = []
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if PAGED_MARKER in line:
                    events.append(read_paged_contract(line))
                    continue
                if MARKER not in line:
                    continue
                try:
                    item = json.loads(line.split(MARKER, 1)[1])
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"malformed smoke event in {path}: {line[:300]}"
                    ) from exc
                # Existing MLA planning diagnostics use the same marker but
                # have no timestamp. Execution is checked using the separate
                # timed mla_prefix_executed event below.
                if item.get("kind") == "mla_prefix" and "time_ns" not in item:
                    continue
                if not isinstance(item, dict) or not isinstance(
                    item.get("time_ns"), int
                ):
                    raise ValueError(f"smoke event lacks time_ns in {path}: {item!r}")
                events.append(item)
        result[rank] = events
    if engine_log is not None:
        if not engine_log.is_file():
            raise ValueError(f"missing C++ engine log: {engine_log}")
        with engine_log.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if MARKER not in line:
                    continue
                match = re.search(r"\[RANK (\d+)\]", line.split(MARKER, 1)[0])
                if match is None or int(match.group(1)) not in result:
                    raise ValueError(
                        f"smoke event lacks valid rank in {engine_log}: {line[:300]}"
                    )
                try:
                    item = json.loads(line.split(MARKER, 1)[1])
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"malformed smoke event in {engine_log}: {line[:300]}"
                    ) from exc
                if not isinstance(item, dict) or not isinstance(
                    item.get("time_ns"), int
                ):
                    raise ValueError(
                        f"smoke event lacks time_ns in {engine_log}: {item!r}"
                    )
                result[int(match.group(1))].append(item)
        for events in result.values():
            events.sort(key=lambda event: event["time_ns"])
    return result


def within(
    events: list[dict[str, Any]], stage: dict[str, Any], *, offset_ns: int = 0
) -> list[dict[str, Any]]:
    start = int(stage["start_time_ns"]) + offset_ns
    end = int(stage["end_time_ns"]) + offset_ns
    return [event for event in events if start <= event["time_ns"] <= end]


def frontend_ids(
    events: dict[int, list[dict[str, Any]]], stages: list[dict[str, Any]]
) -> dict[str, int]:
    mapping = {}
    windows = [
        (int(stage["start_time_ns"]), int(stage["end_time_ns"]))
        for stage in stages
        if "start_time_ns" in stage and "end_time_ns" in stage
    ]
    for event in events[0]:
        if event.get("event") != "frontend_request":
            continue
        timestamp = event.get("time_ns")
        if not isinstance(timestamp, int) or not any(
            start <= timestamp <= end for start, end in windows
        ):
            continue
        name = event.get("case")
        request_id = event.get("request_id")
        if not isinstance(name, str) or not isinstance(request_id, int):
            continue
        if name in mapping and mapping[name] != request_id:
            raise ValueError(f"case {name} maps to multiple internal request IDs")
        mapping[name] = request_id
    for stage in stages:
        if stage.get("transport") != "pd_group_rpc":
            continue
        group_ids = stage.get("request_ids")
        if not isinstance(group_ids, dict) or set(group_ids) != set(
            stage.get("case_names", [])
        ):
            raise ValueError(
                "PD group stage lacks a complete case-to-request ID mapping"
            )
        if any(not isinstance(value, int) for value in group_ids.values()) or (
            len(set(group_ids.values())) != len(group_ids)
        ):
            raise ValueError("PD group stage has invalid or duplicate request IDs")
        for name, request_id in group_ids.items():
            if name in mapping and mapping[name] != request_id:
                raise ValueError(f"PD group case {name} conflicts with frontend event")
            mapping[name] = request_id
    return mapping


def independent_answer_check(
    row: dict[str, Any], *, diagnostic: bool = False
) -> list[str]:
    errors = []
    name = row.get("name", "<unnamed>")
    if row.get("phase") != "formal":
        return errors
    if row.get("pd_sep") is not True:
        errors.append(f"{name}: PD absent")
    if row.get("finish_reason") == "content_filter" or (
        row.get("finish_reason") == "length" and not diagnostic
    ):
        errors.append(f"{name}: incomplete answer")
    content = row.get("content")
    if diagnostic and (not isinstance(content, str) or not content.strip()):
        content = row.get("reasoning_content")
    if (
        not isinstance(content, str)
        or not content.strip()
        or (not diagnostic and "\ufffd" in content)
    ):
        errors.append(f"{name}: empty or malformed answer")
    expected = row.get("expected_json")
    if expected is not None:
        try:
            pairs = json.loads(content, object_pairs_hook=lambda values: values)
            if not isinstance(pairs, list) or len(pairs) != len(
                set(k for k, _ in pairs)
            ):
                errors.append(f"{name}: duplicate or malformed JSON fields")
            elif dict(pairs) != expected:
                errors.append(f"{name}: JSON answer mismatch")
        except (TypeError, ValueError, json.JSONDecodeError):
            errors.append(f"{name}: invalid JSON answer")
    elif (
        isinstance(content, str)
        and row.get("expected_regex")
        and not re.search(row["expected_regex"], content, flags=re.IGNORECASE)
    ):
        errors.append(f"{name}: answer failed independent regex check")
    if int(row.get("output_len", 0)) <= 0:
        errors.append(f"{name}: no output tokens")
    if row.get("require_mtp_draft") and row.get("mtp_draft_rounds", 0) <= 0:
        errors.append(f"{name}: no MTP draft round")
    if row.get("require_mtp") and row.get("mtp_accepted_tokens", 0) <= 0:
        errors.append(f"{name}: no MTP acceptance")
    if row.get("allow_long_history") and (
        int(row.get("input_len", 0)) - int(row.get("effective_reuse_len", 0)) > 65536
    ):
        errors.append(f"{name}: current Q exceeds 64K")
    return errors


def answer_from_question(prompt: str) -> str | dict[str, str]:
    square = re.findall(r"只回答数字[：:]\s*(\d+)\s*的平方是多少", prompt)
    if square:
        number = int(square[-1])
        return str(number * number)
    lookup = re.findall(r"检索\s*key=([^\s]+)\s*的\s*value", prompt)
    if lookup:
        records = re.findall(r"^key=([^;\n]+);\s*value=([^\n]+)$", prompt, re.M)
        values = [value for name, value in records if name == lookup[-1]]
        if len(values) != 1:
            raise ValueError(f"lookup {lookup[-1]!r} has {len(values)} records")
        return {"value": values[0]}
    sequence = re.findall(r'\{"value":"000 001 \.\.\. (\d+)"\}', prompt)
    if sequence:
        last = int(sequence[-1])
        return {"value": " ".join(f"{item:03d}" for item in range(last + 1))}
    literal = re.findall(r'\{"value":"([^"\n]*)"\}', prompt)
    if literal:
        return {"value": literal[-1]}
    raise ValueError("question pattern is not covered by independent audit")


def audit_raw_answers(
    request_dir: pathlib.Path, rows: dict[str, dict[str, Any]]
) -> list[str]:
    errors = []
    seen = set()
    for path in sorted(request_dir.glob("*.json")):
        artifact = json.loads(path.read_text())
        if artifact.get("phase") != "formal" or artifact.get("skipped"):
            continue
        name = artifact.get("name")
        if name in seen:
            errors.append(f"{name}: duplicate request artifact")
            continue
        seen.add(name)
        try:
            if artifact.get("transport") == "pd_group_rpc":
                if artifact.get("rpc_status") != "OK" or not isinstance(
                    artifact.get("request_id"), int
                ):
                    raise ValueError("PD group RPC status or request ID missing")
            elif artifact.get("status") != 200:
                raise ValueError(f"HTTP status {artifact.get('status')}")
            wire = json.loads(artifact["response_body"])
            choice = wire["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"finish_reason={choice.get('finish_reason')!r}")
            message = choice["message"]
            answer = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            if not isinstance(answer, str) or not isinstance(reasoning, str):
                raise ValueError("non-text output channel")
            for channel, value in (("content", answer), ("reasoning", reasoning)):
                value.encode("utf-8", errors="strict")
                if "\ufffd" in value or "\x00" in value:
                    raise ValueError(
                        f"{channel} contains replacement or NUL characters"
                    )
                if re.search(r"([^\s])\1{7,}", value):
                    raise ValueError(f"{channel} contains repeated characters")
            prompt = artifact["request"]["messages"][-1]["content"]
            expected = answer_from_question(prompt)
            observed = (
                answer.strip()
                if isinstance(expected, str)
                else json.loads(answer, object_pairs_hook=_unique_keys)
            )
            if observed != expected:
                raise ValueError(
                    f"prompt-derived answer mismatch: {observed!r} != {expected!r}"
                )
            if wire["aux_info"].get("pd_sep") is not True:
                raise ValueError("raw PD evidence absent")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    expected_names = {
        name for name, row in rows.items() if row.get("phase") == "formal"
    }
    if seen != expected_names:
        errors.append(
            f"formal raw artifact mismatch: missing={sorted(expected_names - seen)}, "
            f"extra={sorted(seen - expected_names)}"
        )
    return errors


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def audit(
    result: dict[str, Any],
    prefill: dict[int, list[dict[str, Any]]],
    decode: dict[int, list[dict[str, Any]]],
    *,
    decode_dp: int,
    decode_graph: bool = True,
    decode_clock_offset_ns: int = 0,
    request_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    errors = []
    if result.get("passed") is not True:
        errors.append("smoke runner did not pass")
    rows = {row["name"]: row for row in result.get("cases", [])}
    raw_answer_errors = []
    if result.get("suite") == "main-text-64k-capped":
        if request_dir is None or not request_dir.is_dir():
            raw_answer_errors.append(
                "missing raw request artifacts for independent answer review"
            )
        else:
            raw_answer_errors.extend(audit_raw_answers(request_dir, rows))
    errors.extend(raw_answer_errors)
    diagnostic = result.get("suite") == "orthogonal-flow"
    for row in rows.values():
        errors.extend(independent_answer_check(row, diagnostic=diagnostic))
    stages = result.get("stages", [])
    prefill_ids = frontend_ids(prefill, stages)
    for stage in stages:
        if stage.get("transport") != "pd_group_rpc":
            continue
        if request_dir is None or not request_dir.is_dir():
            errors.append("PD group raw request artifacts are missing")
            continue
        artifacts = {}
        for path in request_dir.glob("*.json"):
            artifact = json.loads(path.read_text())
            if artifact.get("transport") == "pd_group_rpc":
                artifacts[artifact.get("name")] = artifact
        for name, request_id in stage["request_ids"].items():
            artifact = artifacts.get(name)
            if (
                artifact is None
                or artifact.get("request_id") != request_id
                or (artifact.get("rpc_status") != "OK")
            ):
                errors.append(f"{name}: PD group ID or ACK lacks matching raw evidence")
    path_results = {}
    prefill_tp = len(prefill)
    decode_tp = 8 // decode_dp

    for stage in stages:
        name = stage.get("name", "")
        if name.startswith("orthogonal_cache_mixed_") and stage.get("path_passed"):
            ids = {prefill_ids.get(case) for case in stage["case_names"]}
            valid = len(ids) == 4 and None not in ids
            for rank in range(prefill_tp):
                observed = within(prefill[rank], stage)
                valid &= any(
                    event.get("kind") == "target_prefill_forward"
                    and event.get("actual_batch") == 4
                    and set(event.get("request_ids", [])) == ids
                    for event in observed
                )
            path_results["cache_mixed_forward"] = bool(valid)
            if not valid:
                errors.append(
                    "four cache tiers did not share one all-rank target Prefill forward"
                )
            handoff_ok = True
            for case in stage["case_names"]:
                request_id = prefill_ids.get(case)
                owner = rows[case]["decode_owner_rank"]
                for rank in range(owner * decode_tp, (owner + 1) * decode_tp):
                    if not any(
                        event.get("event") == "pd_cache_loaded"
                        and event.get("request_id") == request_id
                        and event.get("dp_rank") == owner
                        for event in decode[rank]
                    ):
                        handoff_ok = False
                        errors.append(f"{case}: missing Decode rank {rank} PD handoff")
            path_results["mixed_pd_handoff"] = handoff_ok

        if name == "decode_dp_cross_owner_cached_64k":
            case = name
            cold = "prefill_64k_single_0_cold"
            request_id = prefill_ids.get(case)
            cold_id = prefill_ids.get(cold)
            valid = bool(
                decode_dp == 2
                and request_id is not None
                and cold_id is not None
                and rows.get(case, {}).get("decode_owner_rank") == 1
                and rows.get(cold, {}).get("decode_owner_rank") == 0
                and rows[case].get("effective_reuse_len", 0) > 0
            )
            for rank in range(8):
                owner, attention_rank = divmod(rank, decode_tp)
                loaded = {
                    event.get("request_id")
                    for event in decode[rank]
                    if event.get("event") == "pd_cache_loaded"
                }
                valid &= (cold_id in loaded) == (owner == 0)
                valid &= (request_id in loaded) == (owner == 1)
                if owner != 1:
                    continue
                transfers = {
                    event.get("source_tp_rank"): event
                    for event in decode[rank]
                    if event.get("event") == "pd_transfer_submitted"
                    and event.get("request_id") == request_id
                }
                valid &= set(transfers) == set(range(8))
                for source_rank, event in transfers.items():
                    valid &= (
                        event.get("source_tp") == 8
                        and event.get("destination_tp") == 4
                        and event.get("dp_rank") == 1
                        and event.get("attn_tp_rank") == attention_rank
                        and event.get("kda_source_partitions") == 1
                        and event.get("kda_destination_partitions") == 2
                        and event.get("kda_destination_partition") == source_rank % 2
                    )
                    valid &= (event.get("kda_blocks", 0) > 0) == (
                        source_rank // 2 == attention_rank
                    )
                    if event.get("mla_pages", 0) > 0:
                        valid &= (
                            event.get("first_mla_page", -1) % 8 == source_rank
                            and event.get("last_mla_page", -1) % 8 == source_rank
                        )
            path_results["decode_dp_cross_owner_handoff"] = bool(valid)
            if not valid:
                errors.append(
                    "Decode DP owner switch lacks isolated MLA PageRR/KDA TP handoff"
                )

        if name.startswith("orthogonal_page_"):
            actual = []
            for rank in range(prefill_tp):
                actual.extend(
                    event
                    for event in within(prefill[rank], stage)
                    if event.get("kind") == "mla_page_rr_prefill"
                    and event.get("rank") == rank
                )
            valid = len(actual) >= prefill_tp and all(
                event.get("shards") == prefill_tp
                and event.get("page_tokens") == 4096
                and event.get("padding_owned_slots") == 0
                and event.get("physical_tokens", 0) >= event.get("logical_tokens", 0)
                for event in actual
            )
            if name == "orthogonal_page_32768_+0":
                valid &= {
                    event["rank"]
                    for event in actual
                    if event.get("owned_token_rows", 0) > 0
                } == set(range(prefill_tp))
            if name in ("orthogonal_page_4096_-1", "orthogonal_page_32768_+1"):
                valid &= any(event.get("padding_rows", 0) > 0 for event in actual)
            path_results[name] = bool(valid)
            if not valid:
                errors.append(
                    f"{name}: incomplete PageRR ownership or padding evidence"
                )

        if name == "orthogonal_kv_final":
            valid = (
                rows.get("orthogonal_kv_final", {}).get("effective_reuse_len", 0)
                >= 589_824
            )
            for rank in range(prefill_tp):
                actual = [
                    event
                    for event in within(prefill[rank], stage)
                    if event.get("kind") == "mla_prefix_executed"
                    and event.get("backend") == "fp8_tokenspeed"
                    and event.get("rank") == rank
                ]
                grouped = defaultdict(set)
                for event in actual:
                    if event.get("length", 0) > 0:
                        grouped[(event.get("layer_id"), event.get("owner"))].add(
                            event.get("start")
                        )
                valid &= any(len(starts) >= 2 for starts in grouped.values())
            path_results["chunk_kv_executed"] = bool(valid)
            if not valid:
                errors.append(
                    "589824-token historic FP8 MLA did not execute multiple chunks on every rank"
                )

        match = re.fullmatch(r"orthogonal_decode_\d+_batch_(\d+)", name)
        if match:
            size = int(match.group(1))
            valid = True
            for owner in range(decode_dp):
                local = sum(
                    rows[case]["decode_owner_rank"] == owner
                    for case in stage["case_names"]
                )
                if local == 0:
                    continue
                for rank in range(owner * decode_tp, (owner + 1) * decode_tp):
                    events = within(
                        decode[rank], stage, offset_ns=decode_clock_offset_ns
                    )
                    root = rank == owner * decode_tp

                    def paired_replay(forward_name: str, role: int) -> bool:
                        forwards = [
                            event
                            for event in events
                            if event.get("event") == forward_name
                        ]
                        for index, forward in enumerate(forwards):
                            if forward.get("input_rows") != local or (
                                root and forward.get("stream_count") != local
                            ):
                                continue
                            next_time = (
                                forwards[index + 1]["time_ns"]
                                if index + 1 < len(forwards)
                                else int(stage["end_time_ns"]) + decode_clock_offset_ns
                            )
                            for event in events:
                                if (
                                    event.get("event") != "cuda_graph_replay"
                                    or event.get("role") != role
                                    or not forward["time_ns"]
                                    <= event["time_ns"]
                                    < next_time
                                ):
                                    continue
                                if role == 2:
                                    rows_used = event.get("real_batch", 0)
                                else:
                                    rows_used = event.get("real_tokens", 0)
                                physical_batch = event.get("real_batch", 0)
                                # A one-request owner can replay bucket 1.
                                # Verify can expose padding as a physical
                                # virtual row (DP1: 63 -> real_batch 64) or
                                # as Graph bucket padding (DP2: 31 -> 32).
                                # Draft may capture its exact MTP token count.
                                needs_padding = (
                                    role == 2
                                    and local > 1
                                    and bool(local & (local - 1))
                                )
                                if (
                                    physical_batch >= local
                                    and rows_used > 0
                                    and event.get("bucket", 0) >= rows_used
                                    and event.get("padding_rows")
                                    == event["bucket"] - rows_used
                                    and (
                                        not needs_padding
                                        or physical_batch > local
                                        or event["padding_rows"] > 0
                                    )
                                ):
                                    return True
                        return False

                    if decode_graph:
                        valid &= paired_replay("mtp_target_verify_forward", 2)
                        valid &= paired_replay("mtp_draft_decode_forward", 4)
                    else:
                        valid &= not any(
                            event.get("event") == "cuda_graph_replay"
                            and event.get("role") in (2, 3, 4)
                            for event in events
                        )
                        for forward_name in (
                            "mtp_target_verify_forward",
                            "mtp_draft_decode_forward",
                        ):
                            valid &= any(
                                event.get("event") == forward_name
                                and event.get("input_rows") == local
                                and (not root or event.get("stream_count") == local)
                                and (
                                    event.get("token_rows") == local * 4
                                    if forward_name == "mtp_target_verify_forward"
                                    else event.get("propose_step") == 3
                                )
                                for event in events
                            )
                        valid &= all(
                            eager_paged_contract(
                                decode[rank],
                                end_ns=int(stage["end_time_ns"])
                                + decode_clock_offset_ns,
                                phase=phase,
                                logical_batch=local,
                                tp=decode_tp,
                            )
                            for phase in (
                                "target_verify",
                                "proposal_or_decode",
                                "mtp_update",
                            )
                        )
                        valid &= not any(
                            event.get("kind") == "mla_prefix_executed"
                            for event in events
                        )
            path_results[name] = bool(valid)
            if not valid:
                evidence = (
                    "paired MTP Graph replay"
                    if decode_graph
                    else "eager paged MTP modeling"
                )
                errors.append(f"{name}: logical request batch or {evidence} missing")

        if name == "orthogonal_cancel_recovery":
            attempts = stage.get("cancel_attempts", [])
            valid = any(cancel_host_load_evidence(item, prefill) for item in attempts)
            path_results["cancel_during_host_load"] = bool(valid)
            if not valid:
                errors.append("Host cache cancellation did not happen during load")

    orthogonal = result.get("profile") == "orthogonal-pd-page-rr" or result.get(
        "suite"
    ) in ("orthogonal-flow", "main-text-64k-capped")
    all_phases = {"cache", "cancel", "page", "chunk", "decode"}
    phases: set[str] = set()
    required = set()
    if orthogonal:
        declared_phases = result.get("orthogonal_phases", ())
        if declared_phases:
            phases = set(declared_phases)
            if len(phases) != len(declared_phases) or not phases <= all_phases:
                errors.append("invalid orthogonal phase selection")
        elif result.get("suite") == "orthogonal-flow":
            # Older diagnostic artifacts predate the recorded phase list.
            names = {stage.get("name", "") for stage in stages}
            phases = {
                phase
                for phase, prefix in (
                    ("cache", "orthogonal_cache_"),
                    ("cancel", "orthogonal_cancel_"),
                    ("page", "orthogonal_page_"),
                    ("chunk", "orthogonal_kv_"),
                    ("decode", "orthogonal_decode_"),
                )
                if any(name.startswith(prefix) for name in names)
            }
        else:
            phases = all_phases
        if not phases:
            errors.append("orthogonal smoke has no executed phase")
        if result.get("suite") == "main-text-64k-capped" and phases != all_phases:
            errors.append("full capped smoke must run every orthogonal phase")
        if "cache" in phases:
            required |= {"cache_mixed_forward", "mixed_pd_handoff"}
        if "cancel" in phases:
            required.add("cancel_during_host_load")
        if "chunk" in phases:
            required.add("chunk_kv_executed")
        if "page" in phases:
            required |= {
                f"orthogonal_page_{boundary}_{delta:+d}"
                for boundary in (4096, 32768)
                for delta in (-1, 0, 1)
            }
        if "decode" in phases:
            decode_sizes = (
                (1, 4, 8, 63, 64)
                if result.get("case_profile") == "compact-64k-v1"
                else (1, 7, 8, 9, 31, 32, 33, 63, 64, 64, 63, 33, 1, 64)
            )
            required |= {
                f"orthogonal_decode_{index:02d}_batch_{size}"
                for index, size in enumerate(decode_sizes)
            }
        if decode_dp == 2 and result.get("suite") == "main-text-64k-capped":
            required.add("decode_dp_cross_owner_handoff")
    for name in sorted(required - path_results.keys()):
        errors.append(f"missing runtime stage: {name}")
    if (
        orthogonal
        and "decode" in phases
        and not any(
            name.endswith("_batch_64") and value for name, value in path_results.items()
        )
    ):
        evidence = "Graph replay" if decode_graph else "eager paged modeling"
        errors.append(f"Decode batch 64 {evidence} was not proven")
    return {
        "passed": not errors,
        "decode_graph_expected": decode_graph,
        "selected_phases": sorted(phases),
        "all_orthogonal_phases_selected": phases == all_phases,
        "answer_passed": not any(
            independent_answer_check(row, diagnostic=diagnostic)
            for row in rows.values()
        )
        and not raw_answer_errors,
        "path_passed": all(path_results.values()) and required <= path_results.keys(),
        "path_results": path_results,
        "errors": errors,
        "case_count": len(rows),
        "skipped_cases": result.get("skipped_cases", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=pathlib.Path, required=True)
    parser.add_argument("--prefill-log-dir", type=pathlib.Path, required=True)
    parser.add_argument("--decode-log-dir", type=pathlib.Path, required=True)
    parser.add_argument("--prefill-engine-log", type=pathlib.Path)
    parser.add_argument("--decode-engine-log", type=pathlib.Path)
    parser.add_argument("--prefill-ranks", type=int, choices=(4, 8), default=8)
    parser.add_argument("--decode-dp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--decode-graph", type=int, choices=(0, 1), default=1)
    parser.add_argument("--decode-clock-offset-ns", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = json.loads(args.result.read_text())
    verdict = audit(
        result,
        read_events(args.prefill_log_dir, args.prefill_ranks, args.prefill_engine_log),
        read_events(args.decode_log_dir, engine_log=args.decode_engine_log),
        decode_dp=args.decode_dp,
        decode_graph=bool(args.decode_graph),
        decode_clock_offset_ns=args.decode_clock_offset_ns,
        request_dir=args.result.parent / "requests",
    )
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": verdict["passed"], "errors": verdict["errors"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
