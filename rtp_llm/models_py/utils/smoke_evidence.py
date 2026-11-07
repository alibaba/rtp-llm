"""Opt-in host metadata evidence; no tensor reads, synchronization or collectives."""
import json
import logging
import os
import time

_ENABLED = os.environ.get("KIMI_K3_SMOKE_EVIDENCE") == "1"


def submitted_prefill_conv_snapshot(world_rank, prefixes, lengths, sources, aliases):
    if not _ENABLED:
        return
    rows = [dict(prefix_tokens=prefix, query_tokens=length, input_block=source)
            for prefix, length, source, alias in zip(prefixes, lengths, sources, aliases) if alias]
    logging.getLogger(__name__).info("[K3_SMOKE_EVENT] %s", json.dumps({
        "event": "linear_prefill_conv_snapshot_submitted", "world_rank": world_rank,
        "rows": rows, "time_ns": time.time_ns(),
    }, separators=(",", ":")))


def completed_dsa_chunks(phase, params, count, rows):
    if not _ENABLED or params.is_warmup or not params.is_prefill:
        return
    logging.getLogger(__name__).info("[K3_SMOKE_EVENT] %s", json.dumps({
        "event": "dsa_kv_chunks_completed", "phase": phase,
        "shard_rank": params.shard_rank, "shard_size": params.shard_size,
        "fp8_indexer": bool(params.indexer_execution.fp8_indexer),
        "query_cp": bool(params.query_cp), "local_query_rows": rows,
        "completed_chunks": count, "time_ns": time.time_ns(),
    }, separators=(",", ":")))
