"""Check K3 smoke responses and case completeness without production probes.

A passing result certifies the recorded functional checks. It does not prove
internal kernel dispatch, graph replay, or cancellation during Host cache DMA.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter
from typing import Any


def failure_details(accuracy_path: pathlib.Path | str) -> str:
    """Keep the first case/stage failure visible in the outer Bazel test log."""
    path = pathlib.Path(accuracy_path)
    details = [f"result={path}"]
    try:
        result = json.loads(path.read_text())
        if result.get("error"):
            details.append(str(result["error"]))
        for key, label in (
            ("failures", "case"),
            ("stages", "stage"),
            ("protocol_requests", "protocol case"),
        ):
            for row in result.get(key, []):
                if row.get("error"):
                    details.append(
                        f"{label}={row.get('name')} owner={row.get('owner')} "
                        f"artifact={row.get('artifact', path)}: {row['error']}"
                    )
                    break
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        details.append(f"result unavailable: {exc}")
    return "; ".join(details)[:4096]


def independent_answer_check(row: dict[str, Any]) -> list[str]:
    errors = []
    name = row.get("name", "<unnamed>")
    if row.get("phase") != "formal":
        return errors
    if row.get("pd_sep") is not True:
        errors.append(f"{name}: PD absent")
    if row.get("finish_reason") in ("content_filter", "length"):
        errors.append(f"{name}: incomplete answer")
    content = row.get("content")
    if not isinstance(content, str) or not content.strip() or "\ufffd" in content:
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


def _cache_errors(row: dict[str, Any], block_size: int) -> list[str]:
    name = row["name"]
    errors = []
    length, reuse = int(row["input_len"]), int(row["effective_reuse_len"])
    if length <= 0 or not 0 <= reuse <= length or reuse % block_size:
        errors.append(f"{name}: invalid cache reuse length or alignment")
    expectation = row.get("expected_reuse")
    valid = {
        "miss": reuse == 0,
        "hit": reuse > 0,
        "partial": 0 < reuse < length,
        "any": True,
    }
    if expectation not in valid or not valid[expectation]:
        errors.append(f"{name}: expected cache {expectation}, observed {reuse}")
    for field, actual in (
        ("expected_input_len", length),
        ("expected_reuse_len", reuse),
    ):
        if row.get(field) is not None and row[field] != actual:
            errors.append(f"{name}: {field} mismatch")
    tiers = row.get("prefill_cache_tiers")
    if tiers is not None:
        total, local, device, memory, disk = (
            int(tiers[key]) for key in ("total", "local", "device", "memory", "disk")
        )
        if min(total, local, device, memory, disk) < 0 or not (
            total == reuse and device + memory + disk == local <= total
        ):
            errors.append(f"{name}: inconsistent Prefill cache tier counters")
    tier = row.get("expected_cache_tier")
    if tier is not None:
        valid_tier = tiers is not None and (
            (tier == "miss" and tiers["total"] == 0)
            or (
                tier in ("device", "memory", "disk")
                and tiers[tier] > 0
                and all(
                    tiers[other] == 0
                    for other in ("device", "memory", "disk")
                    if other != tier
                )
            )
        )
        if not valid_tier:
            errors.append(f"{name}: expected Prefill cache tier {tier}")
    for boundary in row.get("decode_crossings", []):
        if not length <= boundary <= length + int(row["output_len"]) - 2:
            errors.append(f"{name}: Decode did not cross committed boundary {boundary}")
    selected = row.get("selected_decode_role_addr")
    if selected is None or selected not in row.get("observed_decode_role_addrs", []):
        errors.append(f"{name}: requested Decode owner was not returned")
    return errors


def _raw_consistency(
    request_dir: pathlib.Path, rows: dict[str, dict[str, Any]]
) -> list[str]:
    errors = []
    for path in sorted(request_dir.glob("*.json")):
        artifact = json.loads(path.read_text())
        if artifact.get("phase") != "formal" or artifact.get("skipped"):
            continue
        name = artifact["name"]
        if name not in rows:
            continue
        row = rows[name]
        wire = json.loads(artifact["response_body"])
        choice, aux = wire["choices"][0], wire["aux_info"]
        comparisons = {
            "content": choice["message"].get("content") or "",
            "reasoning_content": choice["message"].get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"),
            "input_len": aux["input_len"],
            "output_len": aux["output_len"],
            "iter_count": aux["iter_count"],
            "effective_reuse_len": aux.get(
                "prefill_total_reuse_len", aux.get("reuse_len", 0)
            ),
            "mtp_draft_rounds": aux.get("speculative_draft_rounds", 0),
        }
        for key, value in comparisons.items():
            if row.get(key) != value:
                errors.append(f"{name}: summary {key} differs from raw response")
        if row.get("mtp_accepted_tokens") != aux["output_len"] - aux["iter_count"]:
            errors.append(f"{name}: MTP acceptance summary differs from raw response")
        if row.get("observed_decode_role_addrs") != aux.get("role_addrs", []):
            errors.append(f"{name}: Decode owner summary differs from raw response")
        tiers = row.get("prefill_cache_tiers")
        if tiers is not None:
            for field, key in (
                ("total", "prefill_total_reuse_len"),
                ("local", "prefill_local_reuse_len"),
                ("memory", "prefill_memory_reuse_len"),
                ("disk", "prefill_disk_reuse_len"),
            ):
                if tiers[field] != aux.get(key):
                    errors.append(
                        f"{name}: {field} cache tier differs from raw response"
                    )
    return errors


def _protocol_names() -> set[str]:
    names = {
        "request_no_think_sync",
        "request_no_think_sse",
        "request_sampling_defaults",
        "request_thinking",
        "request_invalid_temperature",
        "request_invalid_top_p",
        "request_invalid_n",
        "request_tool_call",
        "request_tool_sse",
        "request_missing_tool_result",
        "request_tool_roundtrip",
        "request_dash_no_think",
        "request_dash_invalid_temperature",
        "request_after_rejections",
        "grammar_json_sync",
        "grammar_json_sse",
        "grammar_thinking_budget",
        "grammar_mixed_plain",
        "grammar_mixed_json",
        "grammar_mixed_thinking",
        "logprobs_mixed_none",
        "logprobs_mixed_default",
        "logprobs_mixed_original",
        "logprobs_dash_False",
        "logprobs_dash_True",
    }
    names.update(
        f"logprobs_{mode}_{transport}_top{top}"
        for mode in ("default", "original")
        for transport in ("sync", "sse")
        for top in (0, 3)
    )
    names.update(f"logprobs_length_stop_{limit}" for limit in (3, 4, 5))
    names.update(f"tail_reuse_slot_{index}" for index in range(4))
    return names


def _check_protocol(
    records: list[dict[str, Any]], directory: pathlib.Path
) -> list[str]:
    from rtp_llm.test.smoke.kimi_k3.grammar_smoke import check_json_answer
    from rtp_llm.test.smoke.kimi_k3.logprobs_smoke import COPY_TEXT, check_logprobs
    from rtp_llm.test.smoke.kimi_k3.request_smoke import merge_sse, validate_answer

    errors = []
    names = [row["name"] for row in records]
    required = _protocol_names()
    if len(names) != len(set(names)) or set(names) != required:
        errors.append(
            f"protocol matrix mismatch: missing={sorted(required - set(names))}, extra={sorted(set(names) - required)}"
        )
    try:
        if json.loads((directory / "summary.json").read_text()) != records:
            errors.append("protocol summary differs from accuracy result")
    except (OSError, ValueError) as error:
        errors.append(f"missing or invalid protocol summary: {error}")
    for row in records:
        name = row["name"]
        try:
            if row.get("passed") is not True or row.get("error"):
                raise ValueError("protocol request did not pass")
            if not 0 < row["start_ns"] <= row["end_ns"]:
                raise ValueError("invalid request timing")
            if row["transport"] == "dashsc":
                # Tokenization and DashSc probability checks run in SmokeRequests.dash
                # against the real model tokenizer. Retain the frames and token IDs.
                if not row.get("frames"):
                    raise ValueError("missing DashSc frames")
                if name == "request_dash_invalid_temperature":
                    if (
                        row.get("generated_ids")
                        or "temperature" not in json.dumps(row["frames"]).lower()
                    ):
                        raise ValueError("missing DashSc rejection")
                elif not row.get("generated_ids"):
                    raise ValueError("missing DashSc generated token IDs")
                if (
                    name.startswith("logprobs_dash_")
                    and row.get("multi_token_frames", 0) <= 0
                ):
                    raise ValueError("DashSc multi-token frame was not exercised")
                continue
            if row["transport"] != "openai":
                raise ValueError("unknown protocol transport")
            body = row["response"]
            if (
                name.startswith("request_invalid_")
                or name == "request_missing_tool_result"
            ):
                field = (
                    name.removeprefix("request_invalid_")
                    if name.startswith("request_invalid_")
                    else "tool"
                )
                if (
                    not (
                        row["http_status"] >= 400
                        or body.get("error")
                        or body.get("exception")
                    )
                    or field not in json.dumps(body).lower()
                ):
                    raise ValueError("missing expected OpenAI rejection")
                continue
            if row["http_status"] != 200 or body.get("error"):
                raise ValueError("OpenAI request failed")
            if row["request"].get("stream"):
                merged = merge_sse(
                    row["events"],
                    include_usage=bool(
                        (row["request"].get("stream_options") or {}).get(
                            "include_usage"
                        )
                    ),
                )
                if merged != body:
                    raise ValueError("SSE frames differ from recorded response")
            if body.get("aux_info") and body["aux_info"].get("pd_sep") is not True:
                raise ValueError("protocol request bypassed PD")
            if name in ("request_tool_call", "request_tool_sse"):
                message = validate_answer(body, expected=None, finish="tool_calls")
                calls = message["tool_calls"]
                if (
                    len(calls) != 1
                    or not calls[0]["id"]
                    or calls[0]["function"]["name"] != "add"
                    or json.loads(
                        calls[0]["function"]["arguments"],
                        object_pairs_hook=_unique_keys,
                    )
                    != {"x": 2, "y": 3}
                ):
                    raise ValueError("invalid tool call")
            elif name.startswith("grammar_") and name != "grammar_mixed_plain":
                check_json_answer(body)
            elif name.startswith("logprobs_") and name != "logprobs_mixed_none":
                request = row["request"]
                limited = name.startswith("logprobs_length_stop_")
                check_logprobs(
                    body,
                    mode=request.get("logprobs_mode", "default"),
                    top_count=request.get("top_logprobs", 0),
                    expected=None if limited else COPY_TEXT,
                    finish="length" if limited else "stop",
                )
            else:
                validate_answer(
                    body, COPY_TEXT if name == "logprobs_mixed_none" else "5"
                )
                if name == "logprobs_mixed_none" and body["choices"][0].get("logprobs"):
                    raise ValueError("unrequested probabilities")
                if name == "request_sampling_defaults":
                    config = body["debug_info"]["generate_config"]
                    if (
                        config["temperature"] != 0.6
                        or abs(config["top_p"] - 0.95) >= 1e-6
                    ):
                        raise ValueError("K3 sampling defaults were not applied")
        except (AssertionError, KeyError, IndexError, TypeError, ValueError) as error:
            errors.append(f"{name}: {error}")
    return errors


def _stage_errors(result: dict[str, Any], rows: dict[str, dict[str, Any]]) -> list[str]:
    errors = []
    stages = result["stages"]
    stage_names = [stage["name"] for stage in stages]
    if len(stage_names) != len(set(stage_names)):
        errors.append("duplicate stage names")
    assigned = []
    successful_mixed_cache = False
    for stage in stages:
        if stage.get("skipped"):
            continue
        name, cases = stage["name"], stage["case_names"]
        if (
            stage.get("passed") is not True
            or not cases
            or len(cases) != len(set(cases))
        ):
            errors.append(f"{name}: missing, duplicate, or failed stage cases")
        assigned.extend(cases)
        if stage.get("transport") == "pd_group_rpc":
            mapping = stage["request_ids"]
            if (
                set(mapping) != set(cases)
                or len(set(mapping.values())) != len(cases)
                or any(type(value) is not int for value in mapping.values())
            ):
                errors.append(f"{name}: invalid PD group request IDs")
        if match := re.fullmatch(
            r"orthogonal_decode_(?:\d{2}|graph_long_output)_batch_(\d+)", name
        ):
            if len(cases) != int(match[1]):
                errors.append(f"{name}: wrong Decode request batch size")
            owners = len(result["decode_role_addrs"])
            if owners not in (1, 2) or any(
                rows.get(case, {}).get("decode_owner_rank") != index % owners
                for index, case in enumerate(cases)
            ):
                errors.append(
                    f"{name}: Decode requests were not distributed across configured owners"
                )
        if match := re.fullmatch(r"orthogonal_page_(\d+)_([+-]\d+)", name):
            if len(cases) != 1 or rows.get(cases[0], {}).get("input_len") != int(
                match[1]
            ) + int(match[2]):
                errors.append(f"{name}: wrong page-boundary request length")
        if name.startswith("orthogonal_cache_mixed_"):
            checks = stage.get("cache_tier_checks", {})
            kinds = ("device", "memory", "partial", "miss")
            observed = {}
            current_q = 0
            if len(cases) == 4 and all(
                case.endswith("-" + kind) and case in rows
                for case, kind in zip(cases, kinds)
            ):
                for case, kind in zip(cases, kinds):
                    row = rows[case]
                    tiers = row.get("prefill_cache_tiers") or {}
                    reuse, length = row["effective_reuse_len"], row["input_len"]
                    current_q += length - reuse
                    if kind == "miss":
                        observed[kind] = tiers.get("total") == 0
                    elif kind == "partial":
                        observed[kind] = (
                            0 < reuse < length and tiers.get("device", 0) > 0
                        )
                    else:
                        observed[kind] = tiers.get(kind, 0) > 0 and all(
                            tiers.get(other, 0) == 0
                            for other in ("device", "memory", "disk")
                            if other != kind
                        )
            if (
                set(observed) != set(kinds)
                or checks != observed
                or stage.get("current_q_total") != current_q
            ):
                errors.append(
                    f"{name}: mixed cache summary differs from request results"
                )
            # Cache pressure may need up to three probes to establish the four
            # requested tiers. Every probe still needs valid answers, while one
            # complete tier combination establishes the functional coverage.
            successful_mixed_cache |= (
                set(observed) == set(kinds)
                and all(observed.values())
                and current_q <= 65536
                and stage.get("passed") is True
            )
        if name == "orthogonal_kv_final":
            if (
                len(cases) != 1
                or rows.get(cases[0], {}).get("effective_reuse_len", 0) < 589824
            ):
                errors.append(
                    "historical-KV stage did not reuse at least 589824 tokens"
                )
        if name == "orthogonal_cancel_recovery":
            if set(cases) != {
                "cancel-same-prefix-recovery",
                "cancel-independent-recovery",
            }:
                errors.append(
                    "cancel recovery lacks same-prefix or independent recovery request"
                )
            attempts = stage.get("cancel_attempts", [])
            if not any(
                item.get("cancel_status") == 1
                and item.get("fetch", {}).get("code") == "StatusCode.RESOURCE_EXHAUSTED"
                and "preempted" in item.get("fetch", {}).get("details", "")
                for item in attempts
            ):
                errors.append(
                    "cancel recovery lacks accepted Cancel and rejected Fetch"
                )
    if Counter(assigned) != Counter(rows.keys()):
        errors.append("stage case membership is incomplete or duplicated")
    phases = result.get("orthogonal_phases", [])
    all_phases = {"cache", "cancel", "page", "chunk", "decode"}
    if len(phases) != len(set(phases)) or not set(phases) <= all_phases or not phases:
        errors.append("invalid orthogonal phase selection")
    required = set()
    if "cache" in phases and not successful_mixed_cache:
        errors.append("missing successful mixed-cache stage")
    if "cancel" in phases:
        required.add("orthogonal_cancel_recovery")
    if "chunk" in phases:
        required.add("orthogonal_kv_final")
    if "page" in phases:
        required.update(
            f"orthogonal_page_{boundary}_{delta:+d}"
            for boundary in (4096, 32768)
            for delta in (-1, 0, 1)
        )
    if "decode" in phases:
        required.update(
            f"orthogonal_decode_{index:02d}_batch_{size}"
            for index, size in enumerate((1, 4, 8, 63, 64))
        )
    if result["suite"] == "main-text-64k-capped":
        if set(phases) != all_phases:
            errors.append("full capped smoke must execute all five phases")
        required.update(
            (
                "historical_four_squares",
                "rolling_refill",
                "batch_all_miss",
                "batch_all_hit",
                "batch_mixed_hit_miss",
                "batch_mixed_then_all_hit",
                "prefill_64k_single_cold",
                "prefill_64k_batch_cold",
                "prefill_64k_batch_reuse",
                "prefix_A_seed",
                "prefix_B_partial",
                "prefix_A_return",
                "prefix_AB_mixed",
                "orthogonal_decode_graph_long_output_batch_64",
            )
        )
        required.update(
            f"batch_shape_{index}_{size}"
            for index, size in enumerate((1, 2, 3, 7, 8, 9, 3, 1, 9))
        )
        required.update(
            f"cache_block_{boundary}_{mode}"
            for boundary in range(4096, 32769, 4096)
            for mode in ("cold", "repeat")
        )
        required.update(
            f"decode_page_short_{boundary}_{mode}"
            for boundary in (*range(4096, 32769, 4096), 65536)
            for mode in ("cold", "repeat")
        )
        if len(result["decode_role_addrs"]) == 2:
            required.add("decode_dp_cross_owner_cached_64k")
    executed = {
        stage["name"]
        for stage in stages
        if not stage.get("skipped") and stage.get("passed") is True
    }
    errors.extend(
        f"missing functional stage: {name}" for name in sorted(required - executed)
    )
    return errors


def _group_artifact_errors(
    stages: list[dict[str, Any]], request_dir: pathlib.Path
) -> list[str]:
    artifacts = {}
    for path in request_dir.glob("*.json"):
        item = json.loads(path.read_text())
        if item.get("transport") == "pd_group_rpc":
            artifacts[item["name"]] = item
    errors = []
    for stage in stages:
        if stage.get("transport") != "pd_group_rpc":
            continue
        for name, request_id in stage["request_ids"].items():
            item = artifacts.get(name)
            if (
                item is None
                or item.get("request_id") != request_id
                or item.get("rpc_status") != "OK"
            ):
                errors.append(f"{name}: PD group ID or ACK differs from raw artifact")
    return errors


def check_result(accuracy_path: pathlib.Path | str) -> dict[str, Any]:
    """Validate stored requests, responses, matrix membership, and protocol checks."""
    path = pathlib.Path(accuracy_path)
    verdict = {
        "passed": False,
        "scope": "functional responses and request matrix",
        "kernel_paths_verified": False,
        "host_load_timing_verified": False,
        "full_original_suite_passed": False,
        "errors": [],
    }
    errors = verdict["errors"]
    try:
        result = json.loads(path.read_text(), object_pairs_hook=_unique_keys)
        if (
            result.get("passed") is not True
            or result.get("error")
            or result.get("failures")
        ):
            errors.append("smoke runner did not pass")
        if (
            result.get("suite") != "main-text-64k-capped"
            or result.get("case_profile") != "compact-64k-v1"
        ):
            errors.append("unsupported smoke suite or case profile")
        records = result["cases"]
        rows = {row["name"]: row for row in records}
        if (
            not rows
            or len(rows) != len(records)
            or not all(isinstance(name, str) and name for name in rows)
        ):
            errors.append("empty or duplicate case names")
        if result["summary"]["case_count"] != len(records):
            errors.append("case count differs from summary")
        answer_errors = []
        for row in records:
            if row.get("phase") not in ("formal", "preparation"):
                errors.append(f"{row['name']}: invalid request phase")
            answer_errors.extend(independent_answer_check(row))
            errors.extend(_cache_errors(row, int(result["block_size"])))
            owner = row["decode_owner_rank"]
            roles = result["decode_role_addrs"]
            if (
                type(owner) is not int
                or not 0 <= owner < len(roles)
                or row.get("selected_decode_role_addr") != roles[owner]
            ):
                errors.append(
                    f"{row['name']}: Decode owner differs from requested topology"
                )
        if not any(row.get("phase") == "formal" for row in records):
            errors.append("no formal requests executed")
        request_dir = path.parent / "requests"
        if not request_dir.is_dir():
            answer_errors.append("missing raw request artifacts")
        else:
            answer_errors.extend(audit_raw_answers(request_dir, rows))
            errors.extend(_raw_consistency(request_dir, rows))
            errors.extend(_group_artifact_errors(result["stages"], request_dir))
        errors.extend(answer_errors)
        errors.extend(_stage_errors(result, rows))
        protocol = result.get("protocol_requests", [])
        errors.extend(_check_protocol(protocol, path.parent / "protocol"))
        skipped = result.get("skipped_cases", [])
        if result["summary"]["skipped_case_count"] != len(skipped):
            errors.append("skipped case count differs from summary")
        if any(
            row.get("sent") is not False or not row.get("reason") for row in skipped
        ):
            errors.append("skipped cases lack explicit unsent status or reason")
        verdict.update(
            passed=not errors,
            answer_passed=not answer_errors,
            case_count=len(records),
            formal_case_count=sum(row["phase"] == "formal" for row in records),
            preparation_case_count=sum(
                row["phase"] == "preparation" for row in records
            ),
            protocol_request_count=len(protocol),
            skipped_cases=skipped,
            selected_phases=result.get("orthogonal_phases", []),
            elapsed_s=result.get("elapsed_s"),
        )
    except (
        AssertionError,
        KeyError,
        IndexError,
        OSError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as error:
        errors.append(f"invalid result artifacts: {error}")
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    verdict = check_result(args.result)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": verdict["passed"], "errors": verdict["errors"]}))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
