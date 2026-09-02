#!/usr/bin/env python3
"""l1_grid_runner.py — L1 time-grid runner: plan generation + real-side median extraction.

L1 validates the mock engine's MockPerformanceModel timing formulas against
real-engine measurements on a controlled shape grid. This tool owns the two
offline halves; the replay itself is executed by the EXISTING JavaLoadClient
(no request logic is invented here — the plan subcommand emits a standard
trace JSONL that JavaLoadClient replays verbatim):

  plan      Generate the grid plan (l1_grid_plan.json) and the JavaLoadClient
            compatible trace (l1_grid_trace.jsonl).
  extract   Post-run: join client_events.jsonl (+ optional engine_events.jsonl)
            back onto the plan and emit per-cell medians (l1_real_medians.json).

Grid design (defaults, all overridable):
  prefill axis: input_len ∈ {512,1k,2k,4k,8k,16k,32k} × batch ∈ {1,4,8,16,32}
                × cache_mode ∈ {zero, warm}, repeats=5, output_len=1
                (prefill-only: a 1-token decode tail keeps TTFT ≈ prefill).
  decode axis:  output_len ∈ {8,32,128,512} × batch ∈ {1,4,16}, repeats=5,
                fixed input_len=1024, cache_mode=zero.

Serialization contract (why the trace looks the way it does):
  * Requests inside one measurement round share one ts (fired concurrently) —
    that is how a batch of N becomes one prefill batch / one decode running
    set. Rounds are separated by settle+est gaps so inflight drains to zero
    between rounds (JavaLoadClient paces by ts deltas at REPLAY_SPEED=1).
  * Every row carries explicit "bh" block keys (mock-side cache semantics)
    AND full-length "input_ids" (real-side prefix semantics). Without
    input_ids JavaLoadClient substitutes all-zero token ids, which makes
    every request share one prefix on the real engine — the zero-cache
    cells would silently become warm. First-block randomness is what keeps
    real-engine prefix matching truncated at block 1 for zero cells.
  * Warm cells repeat ONE fixed input_ids/bh pattern (prefill round first)
    so the warm-up request populates the (mock LRU / real prefix) cache and
    every measured request hits the full prefix.
  * ts starts at 1000 (ts=0 disables pacing in JavaLoadClient) and every
    round start = previous round start + max(settle_ms, est) where
    est = il/1000*est_ktok_ms + ol*est_token_ms is a conservative
    single-round duration estimate for ts spacing only.

Usage:
  python3 l1_grid_runner.py plan --out-dir /tmp/l1 \\
      [--settle-ms 5000] [--est-ktok-ms 100] [--est-token-ms 10] \\
      [--prefill-input-lens 512,1024,...] [--prefill-batch-sizes 1,4,...] \\
      [--cache-modes zero,warm] [--prefill-repeats 5] \\
      [--decode-output-lens 8,32,128,512] [--decode-batch-sizes 1,4,16] \\
      [--decode-input-len 1024] [--decode-repeats 5] [--seed 20260902]

  python3 l1_grid_runner.py extract \\
      --plan /tmp/l1/l1_grid_plan.json \\
      --client-events <runDir>/client_events.jsonl \\
      [--engine-events <runDir>/engine_events.jsonl] \\
      --out /tmp/l1/l1_real_medians.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

BLOCK_SIZE = 1024  # JavaLoadClient cache-key block size (BLOCK_SIZE constant).
TRACE_TS_BASE_MS = 1000  # ts=0 disables pacing in JavaLoadClient; keep ts > 0.
RID_PREFIX = "l1"
NEUTRAL_PRIORITY = 50  # neutral QoS level (client default; avoids p0 rejection)

# ---------------------------------------------------------------------------
# Deterministic key/token derivation (seeded, reproducible plans).


def _digest_to_i64(*parts) -> int:
    """Deterministic positive int64 from the given parts (blake2b, 8 bytes)."""
    h = hashlib.blake2b(
        "|".join(str(p) for p in parts).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(h, "big") & 0x7FFF_FFFF_FFFF_FFFF


def _cell_block_keys(
    seed: str, grid_id: str, input_len: int, salt: str = "cell"
) -> list[int]:
    """Fixed per-cell block keys: ceil(input_len / BLOCK_SIZE) entries.

    Used verbatim for warm cells (shared by warm-up + all measured requests)
    and as a shape template for zero cells in the mock reference (the empty
    cache keeps their hit at 0 regardless of key values).
    """
    n_blocks = (input_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    return [_digest_to_i64(seed, grid_id, salt, "blk", str(i)) for i in range(n_blocks)]


def _cell_token_ids(seed: str, grid_id: str, input_len: int, salt: str) -> list[int]:
    """Full-length token ids: first block randomized, remainder zero.

    Same-content cells (same salt) reproduce identical requests (warm
    reuse); distinct salts (per round/request on zero cells) guarantee a
    distinct first block, so real-engine prefix matching truncates at block
    1 (hit=0) even though the tail blocks share content.
    """
    tokens: list[int] = [0] * input_len
    first_block = min(BLOCK_SIZE, input_len)
    for i in range(first_block):
        tokens[i] = _digest_to_i64(seed, grid_id, salt, "tok", str(i)) % 10
    return tokens


def _est_round_ms(
    input_len: int, output_len: int, est_ktok_ms: float, est_token_ms: float
) -> int:
    """Conservative single-round duration estimate (ts spacing only)."""
    return int(input_len / 1000.0 * est_ktok_ms + output_len * est_token_ms)


# ---------------------------------------------------------------------------
# Grid plan + trace generation.


def _prefill_grid_id(input_len: int, batch: int, cache_mode: str) -> str:
    return f"p-il{input_len}-b{batch}-{cache_mode}"


def _decode_grid_id(output_len: int, batch: int) -> str:
    return f"d-ol{output_len}-b{batch}"


def _rid(axis: str, grid_id: str, round_idx: int, seq: int, warmup: bool) -> str:
    """rid encodes (axis-level grid id, round, seq) for extract-time join.

    Round marker: 'w' for the warm-up request (excluded from samples),
    'r<round>' for measured rounds.
    """
    marker = "w" if warmup else f"r{round_idx}"
    return f"{RID_PREFIX}|{axis}|{grid_id}|{marker}|s{seq}"


def parse_rid(rid: str) -> dict | None:
    """Inverse of _rid: {'axis','grid_id','warmup','round','seq'} or None."""
    parts = rid.split("|")
    if len(parts) != 5 or parts[0] != RID_PREFIX:
        return None
    _, axis, grid_id, marker, seq_part = parts
    if not seq_part.startswith("s"):
        return None
    try:
        seq = int(seq_part[1:])
    except ValueError:
        return None
    if marker == "w":
        return {
            "axis": axis,
            "grid_id": grid_id,
            "warmup": True,
            "round": None,
            "seq": seq,
        }
    if marker.startswith("r"):
        try:
            round_idx = int(marker[1:])
        except ValueError:
            return None
        return {
            "axis": axis,
            "grid_id": grid_id,
            "warmup": False,
            "round": round_idx,
            "seq": seq,
        }
    return None


def cmd_plan(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "l1_grid_plan.json"
    trace_path = out_dir / "l1_grid_trace.jsonl"

    prefill_input_lens = [int(x) for x in args.prefill_input_lens.split(",")]
    prefill_batch_sizes = [int(x) for x in args.prefill_batch_sizes.split(",")]
    cache_modes = [m.strip() for m in args.cache_modes.split(",")]
    decode_output_lens = [int(x) for x in args.decode_output_lens.split(",")]
    decode_batch_sizes = [int(x) for x in args.decode_batch_sizes.split(",")]

    for mode in cache_modes:
        if mode not in ("zero", "warm"):
            print(
                f"ERROR: unknown cache mode {mode!r} (expected zero|warm)",
                file=sys.stderr,
            )
            return 2
    for name, values in (
        ("prefill_input_lens", prefill_input_lens),
        ("prefill_batch_sizes", prefill_batch_sizes),
        ("decode_output_lens", decode_output_lens),
        ("decode_batch_sizes", decode_batch_sizes),
    ):
        if not values or any(v <= 0 for v in values):
            print(
                f"ERROR: {name} must be a non-empty list of positive ints",
                file=sys.stderr,
            )
            return 2

    seed = args.seed
    cells: list[dict] = []
    rows: list[dict] = []
    # Cursor in trace-relative ms. Round starts advance by
    # max(settle_ms, est(round shape)) AFTER each round; every row in a
    # round shares the round's ts.
    ts_ms = TRACE_TS_BASE_MS
    request_id_int = 0

    def emit(
        axis: str,
        grid_id: str,
        input_len: int,
        output_len: int,
        round_idx: int,
        seq: int,
        warmup: bool,
        block_keys: list[int],
        token_ids: list[int],
    ) -> None:
        nonlocal request_id_int
        request_id_int += 1
        rows.append(
            {
                "rid": _rid(axis, grid_id, round_idx, seq, warmup),
                "request_id_int": request_id_int,
                "ts": ts_ms,
                "il": input_len,
                "ol": output_len,
                "priority": NEUTRAL_PRIORITY,
                "bh": block_keys,
                "input_ids": token_ids,
            }
        )

    # ── prefill axis ──
    for input_len in prefill_input_lens:
        for batch in prefill_batch_sizes:
            for mode in cache_modes:
                grid_id = _prefill_grid_id(input_len, batch, mode)
                cell_keys = _cell_block_keys(seed, grid_id, input_len)
                # Warm cells: one fixed token pattern shared by warm-up and
                # all measured requests. Zero cells: per-round distinct salt.
                warm_tokens = _cell_token_ids(seed, grid_id, input_len, "warm-body")
                est = _est_round_ms(input_len, 1, args.est_ktok_ms, args.est_token_ms)
                cells.append(
                    {
                        "grid_id": grid_id,
                        "axis": "prefill",
                        "input_len": input_len,
                        "output_len": 1,
                        "batch_size": batch,
                        "cache_mode": mode,
                        "repeats": args.prefill_repeats,
                        "est_round_ms": est,
                        "block_keys": cell_keys,
                    }
                )
                if mode == "warm":
                    # Warm-up request: same body as the measured requests,
                    # own settle gap so the cache write completes before
                    # round 1 fires.
                    emit("p", grid_id, input_len, 1, 0, 0, True, cell_keys, warm_tokens)
                    ts_ms += max(args.settle_ms, est)
                for round_idx in range(1, args.prefill_repeats + 1):
                    for seq in range(batch):
                        if mode == "warm":
                            keys, tokens = cell_keys, warm_tokens
                        else:
                            salt = f"zero-r{round_idx}-s{seq}"
                            keys = [
                                _digest_to_i64(seed, grid_id, salt, "blk", str(i))
                                for i in range(len(cell_keys))
                            ]
                            tokens = _cell_token_ids(seed, grid_id, input_len, salt)
                        emit(
                            "p",
                            grid_id,
                            input_len,
                            1,
                            round_idx,
                            seq,
                            False,
                            keys,
                            tokens,
                        )
                    ts_ms += max(args.settle_ms, est)

    # ── decode axis ──
    for output_len in decode_output_lens:
        for batch in decode_batch_sizes:
            grid_id = _decode_grid_id(output_len, batch)
            input_len = args.decode_input_len
            cell_keys = _cell_block_keys(seed, grid_id, input_len)
            est = _est_round_ms(
                input_len, output_len, args.est_ktok_ms, args.est_token_ms
            )
            cells.append(
                {
                    "grid_id": grid_id,
                    "axis": "decode",
                    "input_len": input_len,
                    "output_len": output_len,
                    "batch_size": batch,
                    "cache_mode": "zero",
                    "repeats": args.decode_repeats,
                    "est_round_ms": est,
                    "block_keys": cell_keys,
                }
            )
            for round_idx in range(1, args.decode_repeats + 1):
                for seq in range(batch):
                    salt = f"zero-r{round_idx}-s{seq}"
                    keys = [
                        _digest_to_i64(seed, grid_id, salt, "blk", str(i))
                        for i in range(len(cell_keys))
                    ]
                    tokens = _cell_token_ids(seed, grid_id, input_len, salt)
                    emit(
                        "d",
                        grid_id,
                        input_len,
                        output_len,
                        round_idx,
                        seq,
                        False,
                        keys,
                        tokens,
                    )
                ts_ms += max(args.settle_ms, est)

    trace_span_ms = (ts_ms - TRACE_TS_BASE_MS) if rows else 0
    plan = {
        "tool": "l1_grid_runner",
        "subcommand": "plan",
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params": {
            "seed": seed,
            "block_size": BLOCK_SIZE,
            "settle_ms": args.settle_ms,
            "est_ktok_ms": args.est_ktok_ms,
            "est_token_ms": args.est_token_ms,
            "prefill_input_lens": prefill_input_lens,
            "prefill_batch_sizes": prefill_batch_sizes,
            "cache_modes": cache_modes,
            "prefill_repeats": args.prefill_repeats,
            "decode_output_lens": decode_output_lens,
            "decode_batch_sizes": decode_batch_sizes,
            "decode_input_len": args.decode_input_len,
            "decode_repeats": args.decode_repeats,
            # Replay hints for whoever executes the real-side run (the trace
            # is a plain JavaLoadClient trace file; nothing else is needed).
            "replay_hints": {
                "client": "JavaLoadClient (TRACE_FILE=l1_grid_trace.jsonl)",
                "replay_speed": 1,
                "note": (
                    "REPLAY_SPEED=1 keeps ts deltas intact; round gaps "
                    "are settle+est by construction. Bump --settle-ms "
                    "and regenerate if the real engine is slower than "
                    "the est model."
                ),
            },
        },
        "cells": cells,
        "trace_summary": {
            "trace_file": trace_path.name,
            "requests": len(rows),
            "trace_span_ms": trace_span_ms,
            "approx_span_min": round(trace_span_ms / 60000.0, 1),
        },
    }

    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    with trace_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    n_prefill_cells = sum(1 for c in cells if c["axis"] == "prefill")
    n_decode_cells = sum(1 for c in cells if c["axis"] == "decode")
    print(f"plan written: {plan_path}")
    print(
        f"trace written: {trace_path} ({len(rows)} requests, "
        f"span ≈ {trace_span_ms / 60000.0:.1f} min)"
    )
    print(
        f"cells: {n_prefill_cells} prefill ({len(prefill_input_lens)} il "
        f"× {len(prefill_batch_sizes)} batch × {len(cache_modes)} mode) + "
        f"{n_decode_cells} decode ({len(decode_output_lens)} ol × "
        f"{len(decode_batch_sizes)} batch)"
    )
    return 0


# ---------------------------------------------------------------------------
# extract: run products → per-cell medians.


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _grid_key(axis: str, grid_id: str) -> str:
    return f"{axis}|{grid_id}"


def cmd_extract(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text())
    plan_by_key = {}
    for cell in plan["cells"]:
        key = _grid_key("p" if cell["axis"] == "prefill" else "d", cell["grid_id"])
        plan_by_key[key] = cell

    client_rows = _load_jsonl(Path(args.client_events))
    # request_id (int, client_events) ↔ request_id_int (trace) is the exact
    # join key; rid carries the grid coordinates.
    client_by_request_id = {}
    for row in client_rows:
        rid_info = parse_rid(str(row.get("rid", "")))
        if rid_info is None:
            continue  # not one of ours (foreign replay rows in the same file)
        client_by_request_id[int(row["request_id"])] = (row, rid_info)

    engine_by_request_id: dict[int, dict] = {}
    if args.engine_events:
        engine_path = Path(args.engine_events)
        if engine_path.exists():
            for row in _load_jsonl(engine_path):
                if row.get("event") in ("prefill_done", "decode_done"):
                    engine_by_request_id[int(row["rid"])] = row
        else:
            print(
                f"WARNING: engine events file not found: {engine_path}", file=sys.stderr
            )

    # per-cell sample buckets
    samples: dict[str, dict] = {}

    def bucket(cell: dict) -> dict:
        key = _grid_key("p" if cell["axis"] == "prefill" else "d", cell["grid_id"])
        if key not in samples:
            samples[key] = {
                "client": [],
                "engine": [],
                "errors": 0,
                "warmups": 0,
            }
        return samples[key]

    for cell in plan["cells"]:
        bucket(cell)

    # Enumerate expected request ids per cell from the plan+trace pairing:
    # the trace file is the source of truth for request_id_int ↔ rid.
    trace_path = Path(args.plan).parent / plan["trace_summary"]["trace_file"]
    if not trace_path.exists():
        # Fall back to scanning client rows only (missing requests counted
        # from plan expectations).
        trace_rows = []
    else:
        trace_rows = _load_jsonl(trace_path)

    expected: dict[str, int] = {}
    for row in trace_rows:
        rid_info = parse_rid(str(row.get("rid", "")))
        if rid_info is None:
            continue
        key = f"{rid_info['axis']}|{rid_info['grid_id']}"
        expected[key] = expected.get(key, 0) + 1

    for row, rid_info in client_by_request_id.values():
        key = f"{rid_info['axis']}|{rid_info['grid_id']}"
        cell = plan_by_key.get(key)
        if cell is None:
            continue
        b = samples[key]
        if rid_info["warmup"]:
            b["warmups"] += 1
            continue
        status = str(row.get("status", ""))
        ttft = row.get("ttft_ms")
        total = row.get("total_ms")
        if (
            status != "success"
            or ttft is None
            or total is None
            or ttft < 0
            or total <= 0
        ):
            b["errors"] += 1
            continue
        if cell["axis"] == "prefill":
            b["client"].append(float(ttft))  # TTFT ≈ prefill + hand-off
        else:
            b["client"].append(float(total) - float(ttft))  # decode tail

    for request_id, erow in engine_by_request_id.items():
        entry = client_by_request_id.get(request_id)
        if entry is None:
            continue
        _, rid_info = entry
        if rid_info["warmup"]:
            continue
        key = f"{rid_info['axis']}|{rid_info['grid_id']}"
        cell = plan_by_key.get(key)
        if cell is None or erow.get("cancelled"):
            continue
        exec_ms = erow.get("exec_ms")
        if exec_ms is None or exec_ms < 0:
            continue
        samples[key]["engine"].append(float(exec_ms))

    out_cells = []
    for cell in plan["cells"]:
        key = _grid_key("p" if cell["axis"] == "prefill" else "d", cell["grid_id"])
        b = samples[key]
        # expected measured requests = trace rows for this cell − warmups
        n_expected = max(expected.get(key, 0) - b["warmups"], 0)
        entry = {
            "grid_id": cell["grid_id"],
            "axis": cell["axis"],
            "input_len": cell["input_len"],
            "output_len": cell["output_len"],
            "batch_size": cell["batch_size"],
            "cache_mode": cell["cache_mode"],
            "repeats": cell["repeats"],
        }
        if cell["axis"] == "prefill":
            entry["metric"] = "client_ttft_ms"
            entry["engine_metric"] = "prefill_done_exec_ms"
        else:
            entry["metric"] = "client_total_minus_ttft_ms"
            entry["engine_metric"] = "decode_done_exec_ms"
        entry["n_expected"] = n_expected
        entry["n_samples"] = len(b["client"])
        entry["n_errors"] = b["errors"]
        entry["n_warmups"] = b["warmups"]
        if b["client"]:
            entry["samples_ms"] = [round(v, 3) for v in sorted(b["client"])]
            entry["median_ms"] = round(statistics.median(b["client"]), 3)
        if b["engine"]:
            entry["engine_samples_ms"] = [round(v, 3) for v in sorted(b["engine"])]
            entry["engine_median_ms"] = round(statistics.median(b["engine"]), 3)
        out_cells.append(entry)

    out = {
        "tool": "l1_grid_runner",
        "subcommand": "extract",
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "plan": str(args.plan),
            "client_events": str(args.client_events),
            "engine_events": str(args.engine_events) if args.engine_events else None,
        },
        "cells": out_cells,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    n_ok = sum(1 for c in out_cells if c.get("median_ms") is not None)
    print(
        f"real medians written: {out_path} "
        f"({n_ok}/{len(out_cells)} cells with client samples)"
    )
    for c in out_cells:
        if c.get("median_ms") is None:
            print(
                f"  WARNING: no client samples for {c['grid_id']} "
                f"(expected {c['n_expected']}, errors {c['n_errors']})",
                file=sys.stderr,
            )
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="l1_grid_runner.py",
        description="L1 time-grid runner: grid/trace planner + real-side "
        "median extractor (replay itself is JavaLoadClient's).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser(
        "plan", help="generate grid plan + JavaLoadClient trace JSONL"
    )
    p_plan.add_argument(
        "--out-dir", required=True, help="output directory for plan + trace"
    )
    p_plan.add_argument(
        "--settle-ms",
        type=int,
        default=5000,
        help="minimum inter-round gap in ms (default 5000)",
    )
    p_plan.add_argument(
        "--est-ktok-ms",
        type=float,
        default=100.0,
        help="prefill est per kTok for ts spacing (default 100)",
    )
    p_plan.add_argument(
        "--est-token-ms",
        type=float,
        default=10.0,
        help="decode est per output token for ts spacing " "(default 10)",
    )
    p_plan.add_argument(
        "--prefill-input-lens", default="512,1024,2048,4096,8192,16384,32768"
    )
    p_plan.add_argument("--prefill-batch-sizes", default="1,4,8,16,32")
    p_plan.add_argument("--cache-modes", default="zero,warm")
    p_plan.add_argument("--prefill-repeats", type=int, default=5)
    p_plan.add_argument("--decode-output-lens", default="8,32,128,512")
    p_plan.add_argument("--decode-batch-sizes", default="1,4,16")
    p_plan.add_argument("--decode-input-len", type=int, default=1024)
    p_plan.add_argument("--decode-repeats", type=int, default=5)
    p_plan.add_argument("--seed", type=int, default=20260902)
    p_plan.set_defaults(func=cmd_plan)

    p_ext = sub.add_parser("extract", help="extract per-cell medians from run products")
    p_ext.add_argument(
        "--plan", required=True, help="l1_grid_plan.json from the plan subcommand"
    )
    p_ext.add_argument(
        "--client-events",
        required=True,
        help="client_events.jsonl written by JavaLoadClient",
    )
    p_ext.add_argument(
        "--engine-events",
        default=None,
        help="optional engine_events.jsonl (engine exec_ms " "caliber when present)",
    )
    p_ext.add_argument("--out", required=True, help="output l1_real_medians.json path")
    p_ext.set_defaults(func=cmd_extract)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
