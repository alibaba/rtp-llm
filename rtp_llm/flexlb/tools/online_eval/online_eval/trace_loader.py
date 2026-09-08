"""Trace/query loading for online FlexLB replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .rt_model import (
    BLOCK_SIZE,
    compute_block_keys,
    synthetic_token_ids,
    to_signed_int64,
)


@dataclass
class ReplayRequest:
    request_id: int
    source_rid: str
    trace_id: str
    ts_ms: int
    input_len: int
    output_len: int
    block_keys: List[int]
    token_ids: Optional[List[int]]
    prod_prefill: str = ""
    prod_decode: str = ""
    prod_ttfb_ms: float = 0.0
    prod_total_ms: float = 0.0


def load_replay_requests(
    path: str,
    *,
    limit: int = 0,
    duration_s: float = 0.0,
    zero_output_policy: str = "skip",
    include_token_ids: bool = True,
    block_size: int = BLOCK_SIZE,
) -> List[ReplayRequest]:
    records: List[ReplayRequest] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            req = parse_record(
                raw,
                zero_output_policy=zero_output_policy,
                include_token_ids=include_token_ids,
                block_size=block_size,
            )
            if req is not None:
                records.append(req)
    records.sort(key=lambda r: r.ts_ms)
    if duration_s and records:
        first_ts = records[0].ts_ms
        end_ts = first_ts + int(duration_s * 1000)
        records = [record for record in records if record.ts_ms <= end_ts]
    if limit:
        records = records[:limit]
    return records


def parse_record(
    raw: dict,
    *,
    zero_output_policy: str,
    include_token_ids: bool,
    block_size: int,
) -> ReplayRequest | None:
    input_len = int(
        raw.get("il", raw.get("input_token_len", raw.get("backend_input_token_len", 0)))
        or 0
    )
    if input_len <= 0:
        return None

    output_len = int(raw.get("ol", raw.get("output_token_len", 0)) or 0)
    if output_len <= 0:
        if zero_output_policy == "skip":
            return None
        if zero_output_policy == "one":
            output_len = 1
        elif zero_output_policy == "default100":
            output_len = 100
        else:
            raise ValueError(f"unknown zero_output_policy: {zero_output_policy}")

    source_rid = str(raw.get("request_id", raw.get("rid", stable_request_id(raw))))
    trace_id = extract_trace_id(raw) or source_rid
    request_id = to_signed_int64(
        raw.get("request_id_int", stable_request_id(source_rid))
    )
    ts_ms = int(
        raw.get("ts", raw.get("request_enter_ts_epoch_ms", raw.get("ts_epoch_ms", 0)))
        or 0
    )

    token_ids = raw.get("input_ids")
    if token_ids is not None:
        token_ids = [int(x) for x in token_ids]
    elif include_token_ids:
        token_ids = synthetic_token_ids(input_len)

    block_keys = raw.get("bh") or raw.get("block_cache_keys")
    if block_keys is not None:
        block_keys = [to_signed_int64(x) for x in block_keys]
    elif token_ids is not None:
        block_keys = compute_block_keys(token_ids, block_size)
    else:
        block_keys = []

    return ReplayRequest(
        request_id=request_id,
        source_rid=source_rid,
        trace_id=trace_id,
        ts_ms=ts_ms,
        input_len=input_len,
        output_len=output_len,
        block_keys=block_keys,
        token_ids=token_ids,
        prod_prefill=str(raw.get("pep", "")),
        prod_decode=str(raw.get("dep", "")),
        prod_ttfb_ms=float(raw.get("ttfb", raw.get("latency_ttfb_ms", 0.0)) or 0.0),
        prod_total_ms=float(raw.get("total", raw.get("latency_total_ms", 0.0)) or 0.0),
    )


def stable_request_id(value) -> int:
    data = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    digest = hashlib.blake2b(data.encode("utf-8"), digest_size=8).digest()
    return (
        int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFF_FFFF_FFFF_FFFF
    )


def extract_trace_id(raw: dict) -> str:
    value = raw.get("trace_id")
    if value:
        return str(value)

    controls = raw.get("request_controls")
    if not isinstance(controls, dict):
        return ""

    params = controls.get("parameters")
    if isinstance(params, dict):
        for key in ("trace_id", "traceparent"):
            value = params.get(key)
            if value:
                return str(value)

    metadata = controls.get("metadata")
    if isinstance(metadata, list):
        for item in metadata:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).lower()
            if key in ("eagleeye-traceid", "trace-id", "x-trace-id"):
                value = item.get("value")
                if value:
                    return str(value)
    return ""
