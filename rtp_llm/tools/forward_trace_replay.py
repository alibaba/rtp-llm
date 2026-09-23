"""Inspect profiler batch metadata and export one shape-replay case.

This never sends requests or runs a model. It exports the exact Q/prefix pairs
for a model-boundary harness, bypassing nondeterministic serving batch assembly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA_VERSION = 1


def export_case(trace: dict, forward_id: int) -> dict:
    metadata = trace.get("rtp_forward_metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("trace has no supported RTP forward metadata")
    if not metadata.get("complete"):
        raise ValueError("trace metadata is incomplete; cannot certify a replay case")
    records = {item["forward_id"]: item for item in metadata["forwards"]}
    if forward_id not in records:
        raise ValueError(f"forward {forward_id} not found")
    record = records[forward_id]
    if not record.get("lengths_complete") or record.get("status") != "ok":
        raise ValueError("forward did not complete with exact lengths")
    if record.get("kind") == "skipped":
        raise ValueError("selected forward did not execute")
    q, prefix, kv = (record[key] for key in ("q_lens", "prefix_lens", "kv_lens"))
    if not (len(q) == len(prefix) == len(kv) == record["logical_sequences"]):
        raise ValueError("length arrays disagree with logical batch size")
    if any(a <= 0 or b < 0 or c != a + b for a, b, c in zip(q, prefix, kv)):
        raise ValueError("invalid Q/prefix/KV convention")
    if sum(q) != record["total_q_tokens"]:
        raise ValueError("total Q disagrees with per-sequence lengths")

    # Chunk records inherit model configuration from their enclosing forward.
    inherited = {}
    seen = {forward_id}
    ancestor = record.get("parent_forward_id", 0)
    while ancestor:
        if ancestor in seen or ancestor not in records:
            raise ValueError("invalid forward parent chain")
        seen.add(ancestor)
        parent = records[ancestor]
        for key, value in parent.items():
            inherited.setdefault(key, value)
        ancestor = parent.get("parent_forward_id", 0)
    config_keys = (
        "model_id", "dtype", "layers", "hidden_size", "attention_heads", "kv_heads", "head_dim",
        "act_qscheme", "kv_cache_dtype", "attention_sparse", "indexer_topk", "mla_ops_type",
        "world_rank", "tp_size", "tp_rank", "dp_size", "dp_rank", "ep_size", "ep_rank", "pp_size",
        "ktp_size", "cp_enabled", "cp_kv_sharded", "kv_block_tokens", "kernel_kv_block_tokens",
        "mtp_propose_steps", "whole_model_chunk_budget", "force_disable_sp_run",
    )
    config = {key: record.get(key, inherited.get(key)) for key in config_keys}
    raw = record.get("recorded_inputs", {})
    rows = raw.get("original_batch_indices", list(range(len(q))))
    if len(rows) != len(q):
        raise ValueError("chunk request mapping has wrong size")
    return {
        "schema_version": SCHEMA_VERSION,
        "replay_scope": "shape_only",
        "source_forward_id": forward_id,
        "phase": record["phase"],
        "logical_sequences": len(q),
        "request_count": record.get("request_count"),
        "total_q_tokens": sum(q),
        "requests": [
            {"batch_row": i, "original_batch_row": row, "q_len": a, "prefix_len": b, "kv_len": c}
            for i, (row, a, b, c) in enumerate(zip(rows, q, prefix, kv))
        ],
        "execution": {key: record.get(key) for key in (
            "physical_requests", "physical_tokens", "cuda_graph", "graph_batch_bucket",
            "graph_sequence_bucket", "chunk_index", "kind", "micro_batch_count",
        )},
        "model_config": config,
        "recorded_inputs": raw,
        "warmup": {"minimum_representative_iterations": 10, "restore_prefix_state_before_measurement": True},
        "requirements": [
            "Use the matching model checkpoint, software revision and hardware.",
            "Prepare each prefix outside timing and preserve batch row order.",
            "Match physical padding, graph bucket, parallel layout and attention backend.",
            "Token/KV values, KDA state and MoE/sparse routing are not captured; this is not numerical replay.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--forward", type=int, dest="forward_id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.trace.open() as handle:
        trace = json.load(handle)
    if args.forward_id is None:
        metadata = trace.get("rtp_forward_metadata", {})
        print(json.dumps({"complete": metadata.get("complete"), "forwards": [
            {key: item.get(key) for key in ("forward_id", "parent_forward_id", "phase",
             "logical_sequences", "total_q_tokens", "physical_tokens", "q_lens", "kv_lens", "lengths_complete")}
            for item in metadata.get("forwards", [])
        ]}, ensure_ascii=False, indent=2))
        return
    try:
        result = export_case(trace, args.forward_id)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
