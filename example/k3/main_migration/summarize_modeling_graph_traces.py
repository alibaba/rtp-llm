"""Aggregate complete eight-rank modeling windows without mixing topologies."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import median

from analyze_modeling_graph_trace import analyze, trace_events


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trace_origin_ns(path):
    with Path(path).open() as stream:
        header = stream.read(65536)
    match = re.search(r'"baseTimeNanoseconds"\s*:\s*(\d+)', header)
    if not match:
        raise ValueError(f"{path}: missing trace clock origin")
    return int(match.group(1))


def intervals_overlap(values):
    return max(v["gpu_start_us"] for v in values) < min(v["gpu_end_us"] for v in values)


def align_ranks(rank_rounds, sample_rounds=32):
    """Match the same EP8 forward using overlapping GPU intervals, not row indices.

    Each captured model forward includes the real EP8 collective. Every rank
    must be in the corresponding forward at some common time. A trace can begin
    at a different round on one rank; accepting equal list indices would silently
    pair different requests/iterations. Ambiguous intervals fail instead.
    """
    if set(rank_rounds) != set(range(8)):
        raise ValueError("Each window needs all ranks 0 through 7 exactly once")
    if sample_rounds < 1:
        raise ValueError("sample_rounds must be positive")
    for rank, records in rank_rounds.items():
        if not records:
            raise ValueError(f"rank {rank}: no complete B32 Graph rounds")
        for phase, count in (("proposal", 2), ("verify", 1), ("update", 1)):
            for ordinal in range(count):
                activities = {r[phase][ordinal]["activities"] for r in records}
                if len(activities) != 1:
                    raise ValueError(
                        f"rank {rank}: {phase}/{ordinal} GPU activity counts differ; "
                        "possible truncated or changed Graph"
                    )
    used = {rank: set() for rank in range(8)}
    matched = []
    for reference in rank_rounds[0]:
        batch = {0: reference}
        for rank in range(1, 8):
            candidates = [
                (i, record)
                for i, record in enumerate(rank_rounds[rank])
                if i not in used[rank]
                and intervals_overlap([reference["verify"][0], record["verify"][0]])
            ]
            if len(candidates) > 1:
                raise ValueError(
                    f"rank {rank}: ambiguous Target Verify round alignment"
                )
            if not candidates:
                break
            index, record = candidates[0]
            batch[rank] = (index, record)
        if len(batch) != 8:
            continue
        for rank in range(1, 8):
            index, record = batch[rank]
            batch[rank] = record
            used[rank].add(index)
        calls = [[batch[r]["proposal"][i] for r in range(8)] for i in range(2)]
        calls += [
            [batch[r]["verify"][0] for r in range(8)],
            [batch[r]["update"][0] for r in range(8)],
        ]
        if not all(intervals_overlap(call) for call in calls):
            raise ValueError(
                "Matched Target Verify has non-overlapping MTP forwards across ranks"
            )
        max_spans = [max(call[r]["gpu_span_us"] for r in range(8)) for call in calls]
        matched.append(
            {
                "source_round_by_rank": {str(r): batch[r]["round"] for r in range(8)},
                "proposal_1_gpu_us": max_spans[0],
                "proposal_2_gpu_us": max_spans[1],
                "target_verify_gpu_us": max_spans[2],
                "update_gpu_us": max_spans[3],
                "mtp_modeling_gpu_us": max_spans[0] + max_spans[1] + max_spans[3],
            }
        )
    if len(matched) < sample_rounds:
        raise ValueError(f"Only {len(matched)} eight-rank rounds; need {sample_rounds}")
    # A fixed central selection, independent of latency, avoids selecting the
    # fastest rounds and excludes the profiler boundary symmetrically.
    start = (len(matched) - sample_rounds) // 2
    selected = matched[start : start + sample_rounds]
    metrics = (
        "proposal_1_gpu_us",
        "proposal_2_gpu_us",
        "update_gpu_us",
        "target_verify_gpu_us",
        "mtp_modeling_gpu_us",
    )
    return {
        "all_rank_matched_rounds": len(matched),
        "selected_start_index": start,
        "selected_rounds": selected,
        "window_medians_us": {key: median(r[key] for r in selected) for key in metrics},
    }


def load_window(window, root, sample_rounds):
    entries = window["traces"]
    if len(entries) != 8 or {e["rank"] for e in entries} != set(range(8)):
        raise ValueError("Each window needs exactly eight unique rank traces")
    if window["warmup_completed_batches"] < 10:
        raise ValueError(
            "At least 10 completed same-shape warmup batches required per window"
        )
    client_path = root / window["client_summary_path"]
    if file_sha256(client_path) != window["client_summary_sha256"]:
        raise ValueError("Serving warmup/client record SHA mismatch")
    client = json.loads(client_path.read_text())
    saved = [w for w in client["windows"] if w["window"] == window["window"]]
    if len(saved) != 1 or saved[0]["traces"] != entries:
        raise ValueError(
            "Trace set does not match the completed serving capture record"
        )
    labels = saved[0]["warmup_group_labels"]
    if len(set(labels)) != window["warmup_completed_batches"]:
        raise ValueError("Completed warmup batch count differs from the capture record")
    if (
        client["input_tokens"] != 65536
        or client["actual_batch_required_per_owner"] != 32
    ):
        raise ValueError("Warmup was not the required 64K/B32 workload")
    for label in labels:
        groups = [g for g in client["groups"] if g["label"] == label]
        if len(groups) != 1:
            raise ValueError("Missing or duplicate completed warmup batch")
        requests = groups[0]["requests"]
        expected = {
            (owner, request) for owner in range(client["dp"]) for request in range(32)
        }
        if {(r["owner"], r["request"]) for r in requests} != expected or len(
            requests
        ) != len(expected):
            raise ValueError("Warmup batch omitted requests or Decode owners")
        if any(
            r["http_status"] != 200
            or r["input_tokens"] != 65536
            or r["output_tokens"] != client["output_tokens"]
            for r in requests
        ):
            raise ValueError("Warmup request failed or used another shape")
        ready = groups[0].get("ready_snapshot", {})
        if set(map(int, ready)) != set(range(client["dp"])) or any(
            len(row.get("running_task_info", [])) != 32
            or any(
                task.get("is_waiting", False)
                or int(task.get("input_length", -1)) != 65536
                for task in row.get("running_task_info", [])
            )
            for row in ready.values()
        ):
            raise ValueError("Warmup did not observe actual B32 on every owner")
        queues = groups[0]["queues_after"]["decode"]
        if set(map(int, queues)) != set(range(client["dp"])) or any(
            row.get("running_task_info") for row in queues.values()
        ):
            raise ValueError("Warmup Decode queues were not drained")
    origins = {e["rank"]: trace_origin_ns(root / e["path"]) for e in entries}
    origin = min(origins.values())
    rounds = {}
    evidence = []
    for entry in entries:
        path, rank = root / entry["path"], entry["rank"]
        digest = file_sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"rank {rank}: trace SHA mismatch")
        result = analyze(trace_events(path), require_contract=True)
        if result["rejected_rounds"] or result["graph_launches_missing_gpu_activities"]:
            raise ValueError(
                f"rank {rank}: incomplete/eager Graph evidence; audit the raw trace"
            )
        records = result["complete_rounds"]
        delta_us = (origins[rank] - origin) / 1000
        for record in records:
            for phase in ("proposal", "verify", "update"):
                for call in record[phase]:
                    call["gpu_start_us"] += delta_us
                    call["gpu_end_us"] += delta_us
        rounds[rank] = records
        evidence.append(
            {
                "rank": rank,
                "path": str(path),
                "sha256": digest,
                "complete_rounds": len(records),
                "clock_origin_ns": origins[rank],
            }
        )
    result = align_ranks(rounds, sample_rounds)
    result.update(
        {
            "window": window["window"],
            "traces": evidence,
            "warmup_completed_batches": window["warmup_completed_batches"],
        }
    )
    return result


def compare_runs(runs):
    expected = {(version, 8) for version in ("integration", "feat")}
    actual = {(run["version"], run["tp"]) for run in runs}
    if actual != expected or len(runs) != len(expected):
        raise ValueError(
            "Need integration and fixed feat for DP1/TP8/EP8; DP2/TP4 is correctness-only"
        )
    by_key = {(r["version"], r["tp"]): r for r in runs}
    output = []
    for tp in (8,):
        pair = {v: by_key[v, tp] for v in ("integration", "feat")}
        contracts = [r["contract"] for r in pair.values()]
        if contracts[0] != contracts[1]:
            raise ValueError(f"TP{tp}: measurement configurations differ")
        contract = contracts[0]
        required = {
            "history_tokens": 65536,
            "batch_per_owner": 32,
            "n_step": 3,
            "layers": 93,
            "world_size": 8,
            "tp": tp,
            "dp": 8 // tp,
            "ep": 8,
            "cuda_graph": True,
            "mtp_dtype": "bf16",
            "mla_backend": "tokenspeed_page_rr",
        }
        if any(contract.get(k) != v for k, v in required.items()):
            raise ValueError(f"TP{tp}: unsupported modeling workload contract")
        for key in (
            "input_ids_sha256",
            "target_weight_index_sha256",
            "mtp_weight_index_sha256",
        ):
            if not re.fullmatch(r"[a-f0-9]{64}", contract.get(key, "")):
                raise ValueError(f"TP{tp}: missing {key} provenance")
        for key in ("async", "device_states"):
            if not isinstance(contract.get(key), bool):
                raise ValueError(f"TP{tp}: missing actual {key} state")
        if not contract.get("moe_strategy"):
            raise ValueError(f"TP{tp}: missing actual MoE strategy")
        for version, run in pair.items():
            windows = run["windows"]
            if len(windows) != 3 or {w["window"] for w in windows} != {1, 2, 3}:
                raise ValueError(
                    f"{version}/TP{tp}: exactly three distinct hot windows required"
                )
            if (
                version == "feat"
                and run["commit"] != "edaaf0d8aeea61593729dec43a811e29217c32dc"
            ):
                raise ValueError("feat baseline must be fixed edaaf0d8")
        metrics = {}
        for key in (
            "proposal_1_gpu_us",
            "proposal_2_gpu_us",
            "update_gpu_us",
            "target_verify_gpu_us",
            "mtp_modeling_gpu_us",
        ):
            values = {
                v: median(w["window_medians_us"][key] for w in r["windows"])
                for v, r in pair.items()
            }
            if values["feat"] <= 0:
                raise ValueError("Non-positive baseline GPU time")
            metrics[key] = {
                **values,
                "integration_over_feat": values["integration"] / values["feat"],
            }
        passed = all(
            metrics[k]["integration_over_feat"] <= 1.05
            for k in ("target_verify_gpu_us", "mtp_modeling_gpu_us")
        )
        output.append(
            {
                "tp": tp,
                "dp": 8 // tp,
                "contract": contract,
                "metrics": metrics,
                "performance_pass": passed,
            }
        )
    return {
        "boundary": "max-rank GPU span per Graph forward; MTP = sum of the two Q1 "
        "and one Q4 max-rank spans; median across rounds then three windows",
        "topologies": output,
        "performance_pass": all(r["performance_pass"] for r in output),
        "scope": "DP1/TP8/EP8 modeling performance only; DP2/TP4/EP8 correctness/smoke acceptance is separate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--sample-rounds", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    runs = []
    for run in manifest["runs"]:
        runs.append(
            {
                **run,
                "windows": [
                    load_window(w, args.manifest.parent, args.sample_rounds)
                    for w in run["windows"]
                ],
            }
        )
    result = compare_runs(runs)
    result["runs"] = runs
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}))
    if not result["performance_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
