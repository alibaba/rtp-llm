#!/usr/bin/env python3
"""Live HTTP PD-disaggregated performance, parity, and Kineto benchmark for Qwen 4.5-Flash.

Runs against live Prefill + Decode RTP-LLM servers (`PD_SEPARATION=1`) and verifies:
- Greedy golden token parity across optimization waves (`--golden-out` / `--golden-in`)
- Zero leaked KV cache blocks via `/cache_status`
- Device Cache + Host Memory Cache hits, Barex RDMA (0 TCP fallback), QSA sparse,
  Decode CUDA Graph capture/replay, and DFlash2 speculative acceptance
- Prefill Cold/Warm TTFT & tok/s matrix (`T=1024, 4096` x `B=1, 4, 8, 16`)
- Decode TPOT, tok/s, and mean accepted tokens matrix (`ctx=512, 2048` x `B=1, 4, 8, 16, 32, 64`)
- Sustained concurrent 4P+4D mixed Prefill+Decode saturation sweep (`C=16, 32, 64, 128`)
- Rank 0 Kineto profiling trigger (`/start_profile`) and CUDA kernel breakdown parser
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import glob
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_GOLDEN_PROMPTS: List[Dict[str, str]] = [
    {
        "name": "reasoning_math",
        "prompt": (
            "Solve step by step: A train travels 180 km at 90 km/h and then "
            "210 km at 70 km/h. What is the average speed for the entire trip?"
        ),
    },
    {
        "name": "python_code",
        "prompt": (
            "Write a Python function `merge_intervals(intervals)` that merges all "
            "overlapping [start, end] intervals and returns them sorted by start."
        ),
    },
    {
        "name": "systems_networking",
        "prompt": (
            "Explain concisely why RDMA GPUDirect bypasses the host CPU kernel network "
            "stack during KV cache transfer between prefill and decode workers."
        ),
    },
    {
        "name": "chinese_qa",
        "prompt": "请用三句话简要说明混合注意力与线性注意力结合在大语言模型中的优势。",
    },
    {
        "name": "structured_json",
        "prompt": (
            'Return valid JSON with keys "model", "stages", and "metrics" describing '
            "prefill-decode disaggregated LLM serving."
        ),
    },
    {
        "name": "factual_summary",
        "prompt": (
            "List four key differences between speculative decoding verification and "
            "standard single-token autoregressive decoding."
        ),
    },
]


# ---------------------------------------------------------------------------
# HTTP & Server Helpers
# ---------------------------------------------------------------------------


def _http_json(
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 300.0,
) -> Any:
    """Perform a JSON GET or POST request and parse JSON/text response."""
    data = None
    headers: Dict[str, str] = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def wait_for_health(url: str, label: str, timeout_s: float = 600.0) -> None:
    """Poll `/health` until the server reports ok."""
    deadline = time.monotonic() + timeout_s
    health_url = f"{url.rstrip('/')}/health"
    last_err = ""
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                body = resp.read().decode("utf-8", errors="replace").strip()
                if resp.status == 200 and "ok" in body.lower():
                    return
                last_err = f"status={resp.status} body={body}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(2.0)
    raise TimeoutError(f"{label} ({health_url}) did not become healthy within {timeout_s}s: {last_err}")


def get_cache_status(url: str, timeout_s: float = 15.0) -> Dict[str, Any]:
    """Query `/cache_status` without full key dump."""
    resp = _http_json(
        f"{url.rstrip('/')}/cache_status",
        payload={"need_cache_keys": False},
        timeout_s=timeout_s,
    )
    if not isinstance(resp, dict) or "error" in resp:
        raise RuntimeError(f"Invalid /cache_status from {url}: {resp}")
    return resp


def tokenizer_encode(prefill_url: str, prompt: str, timeout_s: float = 30.0) -> List[int]:
    """Encode a prompt via the running frontend `/tokenizer/encode` route."""
    resp = _http_json(
        f"{prefill_url.rstrip('/')}/tokenizer/encode",
        payload={"prompt": prompt},
        timeout_s=timeout_s,
    )
    if not isinstance(resp, dict) or "token_ids" not in resp:
        raise RuntimeError(f"Unexpected /tokenizer/encode response: {resp}")
    return [int(x) for x in resp["token_ids"]]


_DEFAULT_ROLE_ADDRS: Optional[List[Dict[str, Any]]] = None


def set_decode_role_addrs(decode_url: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Derive PD `role_addrs` (`DECODE`) from `--decode-url` (`grpc_port = http_port + 1`)."""
    global _DEFAULT_ROLE_ADDRS
    if not decode_url:
        _DEFAULT_ROLE_ADDRS = None
        return None
    parsed = urllib.parse.urlparse(decode_url if "://" in decode_url else f"http://{decode_url}")
    if not parsed.hostname:
        raise ValueError(f"Could not extract hostname from decode URL: {decode_url!r}")
    http_port = int(parsed.port or 80)
    _DEFAULT_ROLE_ADDRS = [
        {
            "role": "DECODE",
            "ip": parsed.hostname,
            "http_port": http_port,
            "grpc_port": http_port + 1,
        }
    ]
    return _DEFAULT_ROLE_ADDRS


def post_generate(
    prefill_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    reuse_cache: bool,
    ignore_eos: bool = False,
    random_seed: int = 42,
    role_addrs: Optional[List[Dict[str, Any]]] = None,
    timeout_s: float = 600.0,
) -> Dict[str, Any]:
    """Send a greedy generation request and return decoded result + client wall latency."""
    gen_cfg: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 0.0,
        "random_seed": int(random_seed),
        "reuse_cache": bool(reuse_cache),
        "ignore_eos": bool(ignore_eos),
        "can_use_pd_separation": True,
        "return_input_ids": True,
        "return_output_ids": True,
        "return_logits": False,
        "aux_info": True,
    }
    effective_role_addrs = role_addrs if role_addrs is not None else _DEFAULT_ROLE_ADDRS
    if effective_role_addrs:
        gen_cfg["role_addrs"] = effective_role_addrs
    payload = {
        "prompt": prompt,
        "generate_config": gen_cfg,
    }
    t0 = time.perf_counter()
    resp = _http_json(f"{prefill_url.rstrip('/')}/", payload=payload, timeout_s=timeout_s)
    wall_s = time.perf_counter() - t0
    if not isinstance(resp, dict) or "error" in resp or "error_code" in resp:
        raise RuntimeError(f"Generation failed: {resp}")
    return {"resp": resp, "wall_s": wall_s}


def _flatten_int_list(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            return [int(x) for x in raw[0]]
        return [int(x) for x in raw]
    return []


# ---------------------------------------------------------------------------
# Exact Prompt Token-Length Construction
# ---------------------------------------------------------------------------


_FILLER_WORDS = (
    " alpha",
    " beta",
    " gamma",
    " delta",
    " theta",
    " lambda",
    " sigma",
    " omega",
)


def _format_filler_units(num_units: int, seed_offset: int) -> str:
    return "".join(_FILLER_WORDS[(seed_offset + i) % len(_FILLER_WORDS)] for i in range(num_units))


def _adjust_prompt_to_exact_tokens(
    prefill_url: str,
    header: str,
    target_tokens: int,
    initial_units: int,
    single_token_unit: str = " the",
) -> Tuple[str, int, int]:
    """Adjust filler word count around `initial_units` until exact `target_tokens`."""
    cur_units = max(0, initial_units)
    for _ in range(16):
        cur_len = len(tokenizer_encode(prefill_url, header + _format_filler_units(cur_units, 0)))
        diff = target_tokens - cur_len
        if 0 <= diff <= 12:
            break
        cur_units = max(0, cur_units + diff)
    text = header + _format_filler_units(cur_units, 0)
    cur_len = len(tokenizer_encode(prefill_url, text))
    guard = 0
    while cur_len < target_tokens and guard < 64:
        text += single_token_unit
        cur_len = len(tokenizer_encode(prefill_url, text))
        guard += 1
    while cur_len > target_tokens and text.endswith(single_token_unit) and guard < 128:
        text = text[: -len(single_token_unit)]
        cur_len = len(tokenizer_encode(prefill_url, text))
        guard += 1
    return text, cur_len, cur_units


def build_exact_token_prompt(
    prefill_url: str,
    target_tokens: int,
    prefix_tag: str,
    *,
    single_token_unit: str = " the",
    initial_units: Optional[int] = None,
) -> Tuple[str, int]:
    """Construct a prompt that encodes to exactly `target_tokens` tokens."""
    header = f"BenchTag={prefix_tag} Request:"
    if initial_units is not None:
        text, cur_len, _ = _adjust_prompt_to_exact_tokens(
            prefill_url, header, target_tokens, initial_units, single_token_unit
        )
        return text, cur_len

    probe_len = len(tokenizer_encode(prefill_url, header + _format_filler_units(64, 0)))
    hdr_len = len(tokenizer_encode(prefill_url, header))
    if hdr_len >= target_tokens:
        raise ValueError(f"Header already {hdr_len} >= target_tokens={target_tokens}")
    tpu = max(0.5, (probe_len - hdr_len) / 64.0)
    est = max(0, int((target_tokens - hdr_len - 4) / tpu))
    lo = max(0, est - 32)
    hi = est + 32

    while True:
        hi_len = len(tokenizer_encode(prefill_url, header + _format_filler_units(hi, 0)))
        if hi_len >= target_tokens:
            break
        lo = hi
        hi *= 2
    while lo > 0:
        lo_len = len(tokenizer_encode(prefill_url, header + _format_filler_units(lo, 0)))
        if lo_len <= target_tokens:
            break
        lo = max(0, lo // 2)

    best_units = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        cur_len = len(tokenizer_encode(prefill_url, header + _format_filler_units(mid, 0)))
        if cur_len <= target_tokens:
            best_units = mid
            lo = mid + 1
        else:
            hi = mid - 1

    text, cur_len, _ = _adjust_prompt_to_exact_tokens(
        prefill_url, header, target_tokens, best_units, single_token_unit
    )
    return text, cur_len


def build_prompt_batch(
    prefill_url: str,
    target_tokens: int,
    batch_size: int,
    run_tag: str,
) -> List[Tuple[str, int]]:
    """Build `batch_size` distinct prompts each with exact length `target_tokens`."""
    header0 = f"BenchTag={run_tag}_T{target_tokens}_B{batch_size}_000 Request:"
    probe_len = len(tokenizer_encode(prefill_url, header0 + _format_filler_units(64, 0)))
    hdr_len = len(tokenizer_encode(prefill_url, header0))
    tpu = max(0.5, (probe_len - hdr_len) / 64.0)
    est_units = max(0, int((target_tokens - hdr_len - 4) / tpu))
    text0, len0, base_units = _adjust_prompt_to_exact_tokens(
        prefill_url, header0, target_tokens, est_units
    )
    if batch_size == 1:
        return [(text0, len0)]

    prompts: List[Optional[Tuple[str, int]]] = [(text0, len0)] + [None] * (batch_size - 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, batch_size)) as pool:
        futs = {
            pool.submit(
                build_exact_token_prompt,
                prefill_url,
                target_tokens=target_tokens,
                prefix_tag=f"{run_tag}_T{target_tokens}_B{batch_size}_{idx:03d}",
                initial_units=base_units,
            ): idx
            for idx in range(1, batch_size)
        }
        for fut in concurrent.futures.as_completed(futs):
            prompts[futs[fut]] = fut.result()
    return [p for p in prompts if p is not None]


# ---------------------------------------------------------------------------
# Concurrent Benchmark Runners & Statistics
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(xs[lo])
    frac = idx - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def run_concurrent_batch(
    prefill_url: str,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    reuse_cache: bool,
    ignore_eos: bool,
    timeout_s: float,
) -> Tuple[List[Dict[str, Any]], float]:
    """Execute `len(prompts)` requests concurrently and return results + wall time."""
    if len(prompts) == 1:
        t0 = time.perf_counter()
        res = [
            post_generate(
                prefill_url,
                prompts[0],
                max_new_tokens=max_new_tokens,
                reuse_cache=reuse_cache,
                ignore_eos=ignore_eos,
                timeout_s=timeout_s,
            )
        ]
        return res, time.perf_counter() - t0

    t0 = time.perf_counter()
    results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        fut_to_idx = {
            pool.submit(
                post_generate,
                prefill_url,
                p,
                max_new_tokens=max_new_tokens,
                reuse_cache=reuse_cache,
                ignore_eos=ignore_eos,
                timeout_s=timeout_s,
            ): idx
            for idx, p in enumerate(prompts)
        }
        for fut in concurrent.futures.as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            results[idx] = fut.result()
    wall_s = time.perf_counter() - t0
    return [r for r in results if r is not None], wall_s


def _aux_time_ms(
    aux: Dict[str, Any],
    key_ms: str,
    key_us: str,
    fallback_ms: float = 0.0,
) -> float:
    """Extract a timing field from `aux_info` in milliseconds (supporting `*_ms`/bare or `*_us`)."""
    if aux.get(key_ms) is not None:
        return float(aux[key_ms])
    if aux.get(key_us) is not None:
        return float(aux[key_us]) / 1000.0
    return float(fallback_ms)


def _aux_reuse_len(aux: Dict[str, Any], pd_key: str, plain_key: str) -> int:
    """Read KV reuse length from PD (`prefill_*`) or single-engine (`*reuse_len`) fields."""
    pd_val = int(aux.get(pd_key) or 0)
    if pd_val > 0:
        return pd_val
    return int(aux.get(plain_key) or 0)


def summarize_prefill_batch(
    results: Sequence[Dict[str, Any]],
    wall_s: float,
) -> Dict[str, Any]:
    """Aggregate TTFT, reuse lengths, and prefill throughput."""
    ttft_ms: List[float] = []
    e2e_ms: List[float] = []
    wait_ms: List[float] = []
    input_lens: List[int] = []
    total_reuse = 0
    local_reuse = 0
    mem_reuse = 0
    disk_reuse = 0

    for item in results:
        aux = item["resp"].get("aux_info") or {}
        cost = _aux_time_ms(aux, "cost_time", "cost_time_us")
        ft = _aux_time_ms(aux, "first_token_cost_time", "first_token_cost_time_us", cost)
        ttft_ms.append(ft)
        e2e_ms.append(cost if cost > 0.0 else ft)
        wait_ms.append(_aux_time_ms(aux, "wait_time", "wait_time_us"))
        in_len = int(aux.get("input_len") or len(_flatten_int_list(item["resp"].get("input_ids"))))
        input_lens.append(in_len)
        total_reuse += _aux_reuse_len(aux, "prefill_total_reuse_len", "reuse_len")
        local_reuse += _aux_reuse_len(aux, "prefill_local_reuse_len", "local_reuse_len")
        mem_reuse += _aux_reuse_len(aux, "prefill_memory_reuse_len", "memory_reuse_len")
        disk_reuse += _aux_reuse_len(aux, "prefill_disk_reuse_len", "disk_reuse_len")

    total_input = sum(input_lens)
    return {
        "batch_size": len(results),
        "total_input_tokens": total_input,
        "batch_wall_ms": round(wall_s * 1000.0, 3),
        "ttft_mean_ms": round(statistics.mean(ttft_ms), 3) if ttft_ms else 0.0,
        "ttft_p50_ms": round(_percentile(ttft_ms, 0.50), 3),
        "ttft_p95_ms": round(_percentile(ttft_ms, 0.95), 3),
        "ttft_max_ms": round(max(ttft_ms), 3) if ttft_ms else 0.0,
        "e2e_mean_ms": round(statistics.mean(e2e_ms), 3) if e2e_ms else 0.0,
        "wait_mean_ms": round(statistics.mean(wait_ms), 3) if wait_ms else 0.0,
        "prefill_tok_s": round(total_input / max(wall_s, 1e-9), 2),
        "prefill_total_reuse_len": total_reuse,
        "prefill_local_reuse_len": local_reuse,
        "prefill_memory_reuse_len": mem_reuse,
        "prefill_disk_reuse_len": disk_reuse,
        "cache_hit_ratio": round(total_reuse / max(total_input, 1), 4),
    }


def summarize_decode_batch(
    results: Sequence[Dict[str, Any]],
    wall_s: float,
) -> Dict[str, Any]:
    """Aggregate Decode TPOT, throughput, and speculative acceptance metrics."""
    tpots_ms: List[float] = []
    step_ms: List[float] = []
    ttfts_ms: List[float] = []
    decode_times_ms: List[float] = []
    output_lens: List[int] = []
    iter_counts: List[int] = []
    total_rounds = 0
    total_draft_accepted = 0
    per_pos_sum: List[int] = []

    for item in results:
        aux = item["resp"].get("aux_info") or {}
        ft_ms = _aux_time_ms(aux, "first_token_cost_time", "first_token_cost_time_us")
        cost_ms = _aux_time_ms(aux, "cost_time", "cost_time_us")
        decode_ms = max(0.001, cost_ms - ft_ms)
        out_ids = _flatten_int_list(item["resp"].get("output_ids"))
        out_len = int(aux.get("output_len") or len(out_ids))
        iters = int(aux.get("iter_count") or 0)
        rounds = int(aux.get("speculative_draft_rounds") or 0)
        per_pos = [int(x) for x in (aux.get("speculative_accepted_tokens_per_pos") or [])]

        output_lens.append(out_len)
        iter_counts.append(iters)
        ttfts_ms.append(ft_ms)
        decode_times_ms.append(decode_ms)
        if out_len > 1:
            tpots_ms.append(decode_ms / (out_len - 1))
        if iters > 1:
            step_ms.append(decode_ms / (iters - 1))

        total_rounds += rounds
        total_draft_accepted += sum(per_pos)
        if len(per_pos) > len(per_pos_sum):
            per_pos_sum.extend([0] * (len(per_pos) - len(per_pos_sum)))
        for pos_idx, cnt in enumerate(per_pos):
            per_pos_sum[pos_idx] += cnt

    total_out = sum(output_lens)
    total_decode_toks = sum(max(0, n - 1) for n in output_lens)
    mean_tpot = statistics.mean(tpots_ms) if tpots_ms else 0.0
    max_decode_s = max(decode_times_ms) / 1000.0 if decode_times_ms else wall_s
    batch_decode_tok_s = (
        len(results) * 1000.0 / mean_tpot if mean_tpot > 0 else 0.0
    )
    mean_acc_draft = (
        total_draft_accepted / total_rounds if total_rounds > 0 else 0.0
    )
    mean_tok_per_round = (
        (total_rounds + total_draft_accepted) / total_rounds if total_rounds > 0 else 0.0
    )
    out_per_round = total_out / total_rounds if total_rounds > 0 else 0.0

    return {
        "batch_size": len(results),
        "total_output_tokens": total_out,
        "batch_wall_ms": round(wall_s * 1000.0, 3),
        "ttft_mean_ms": round(statistics.mean(ttfts_ms), 3) if ttfts_ms else 0.0,
        "decode_wall_max_ms": round(max(decode_times_ms), 3) if decode_times_ms else 0.0,
        "tpot_mean_ms": round(mean_tpot, 3),
        "tpot_p50_ms": round(_percentile(tpots_ms, 0.50), 3),
        "tpot_p95_ms": round(_percentile(tpots_ms, 0.95), 3),
        "step_time_mean_ms": round(statistics.mean(step_ms), 3) if step_ms else 0.0,
        "mean_iter_count": round(statistics.mean(iter_counts), 2) if iter_counts else 0.0,
        "decode_tok_s": round(total_decode_toks / max(max_decode_s, 1e-9), 2),
        "batch_decode_tok_s_from_mean_tpot": round(batch_decode_tok_s, 2),
        "e2e_tok_s": round(total_out / max(wall_s, 1e-9), 2),
        "speculative_draft_rounds_total": total_rounds,
        "speculative_accepted_draft_tokens_total": total_draft_accepted,
        "mean_accepted_draft_tokens": round(mean_acc_draft, 3),
        "mean_tokens_per_round": round(mean_tok_per_round, 3),
        "output_tokens_per_round": round(out_per_round, 3),
        "accepted_tokens_per_pos_sum": per_pos_sum,
    }


def _build_saturation_workload(
    prefill_url: str,
    *,
    contexts: Sequence[int],
    warm_slots: int,
    run_id: int,
) -> Dict[str, Any]:
    """Build exact-length prompt pools and prime warm prefixes once before saturation."""
    warm_prompts: List[str] = []
    cold_prompts: List[str] = []
    for ctx in contexts:
        cold_p, _ = build_exact_token_prompt(
            prefill_url,
            ctx,
            prefix_tag=f"sat_cold_{run_id}_T{ctx}",
        )
        cold_prompts.append(cold_p)
        for slot in range(warm_slots):
            warm_p, _ = build_exact_token_prompt(
                prefill_url,
                ctx,
                prefix_tag=f"sat_warm_{run_id}_T{ctx}_s{slot}",
            )
            warm_prompts.append(warm_p)
    return {"warm": warm_prompts, "cold": cold_prompts}


def run_saturation_level(
    prefill_url: str,
    *,
    concurrency: int,
    duration_s: float,
    workload: Dict[str, Any],
    max_new_tokens: int,
    warm_ratio: float,
    timeout_s: float,
) -> Dict[str, Any]:
    """Drive `concurrency` continuous mixed Prefill+Decode workers for `duration_s`."""
    warm_prompts: List[str] = workload["warm"]
    cold_prompts: List[str] = workload["cold"]
    warm_pct = max(0, min(100, int(round(warm_ratio * 100))))

    seq_lock = threading.Lock()
    seq_counter = 0
    results_lock = threading.Lock()
    completed: List[Tuple[Dict[str, Any], bool]] = []
    errors: List[str] = []

    barrier = threading.Barrier(concurrency)
    timing: Dict[str, float] = {}

    def _next_request() -> Tuple[int, str, bool]:
        nonlocal seq_counter
        with seq_lock:
            seq = seq_counter
            seq_counter += 1
        use_warm = bool(warm_prompts) and ((seq * 37) % 100 < warm_pct)
        if use_warm:
            return seq, warm_prompts[seq % len(warm_prompts)], True
        return seq, cold_prompts[seq % len(cold_prompts)], False

    def _worker() -> None:
        barrier.wait()
        with seq_lock:
            if "start_mono" not in timing:
                timing["start_mono"] = time.perf_counter()
                timing["start_epoch"] = time.time()
                timing["deadline_mono"] = timing["start_mono"] + max(1.0, float(duration_s))
        deadline = timing["deadline_mono"]
        while time.perf_counter() < deadline:
            seq, prompt, use_warm = _next_request()
            try:
                res = post_generate(
                    prefill_url,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    reuse_cache=use_warm,
                    ignore_eos=True,
                    random_seed=42 + seq,
                    timeout_s=timeout_s,
                )
                with results_lock:
                    completed.append((res, use_warm))
            except Exception as exc:
                with results_lock:
                    errors.append(f"seq={seq} warm={use_warm}: {exc}")

    threads = [threading.Thread(target=_worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    end_mono = time.perf_counter()
    end_epoch = time.time()
    start_mono = timing["start_mono"]
    start_epoch = timing["start_epoch"]
    wall_s = max(1e-6, end_mono - start_mono)
    steady_start_epoch = start_epoch + min(1.0, duration_s * 0.15)
    steady_end_epoch = max(steady_start_epoch + 0.5, start_epoch + duration_s)

    ttfts_ms: List[float] = []
    tpots_ms: List[float] = []
    e2es_ms: List[float] = []
    total_input = 0
    total_reuse = 0
    total_out = 0
    total_decode = 0
    total_rounds = 0
    total_draft_accepted = 0
    warm_completed = 0

    for item, is_warm in completed:
        if is_warm:
            warm_completed += 1
        aux = item["resp"].get("aux_info") or {}
        ft_ms = _aux_time_ms(aux, "first_token_cost_time", "first_token_cost_time_us")
        cost_ms = _aux_time_ms(aux, "cost_time", "cost_time_us", ft_ms)
        decode_ms = max(0.001, cost_ms - ft_ms)
        in_len = int(aux.get("input_len") or len(_flatten_int_list(item["resp"].get("input_ids"))))
        out_len = int(aux.get("output_len") or len(_flatten_int_list(item["resp"].get("output_ids"))))
        reuse_len = _aux_reuse_len(aux, "prefill_total_reuse_len", "reuse_len")
        rounds = int(aux.get("speculative_draft_rounds") or 0)
        per_pos = [int(x) for x in (aux.get("speculative_accepted_tokens_per_pos") or [])]

        total_input += in_len
        total_reuse += reuse_len
        total_out += out_len
        total_decode += max(0, out_len - 1)
        total_rounds += rounds
        total_draft_accepted += sum(per_pos)

        ttfts_ms.append(ft_ms)
        e2es_ms.append(cost_ms)
        if out_len > 1:
            tpots_ms.append(decode_ms / (out_len - 1))

    mean_tok_per_round = (
        (total_rounds + total_draft_accepted) / total_rounds if total_rounds > 0 else 0.0
    )
    return {
        "concurrency": concurrency,
        "target_duration_s": round(float(duration_s), 2),
        "wall_s": round(wall_s, 3),
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "steady_start_epoch": steady_start_epoch,
        "steady_end_epoch": steady_end_epoch,
        "requests_completed": len(completed),
        "warm_requests": warm_completed,
        "cold_requests": len(completed) - warm_completed,
        "error_count": len(errors),
        "errors_sample": errors[:5],
        "requests_per_s": round(len(completed) / wall_s, 3),
        "total_input_tokens": total_input,
        "total_reuse_tokens": total_reuse,
        "cache_hit_ratio": round(total_reuse / max(total_input, 1), 4),
        "total_output_tokens": total_out,
        "system_prefill_tok_s": round(total_input / wall_s, 2),
        "system_decode_tok_s": round(total_out / wall_s, 2),
        "system_total_tok_s": round((total_input + total_out) / wall_s, 2),
        "ttft_mean_ms": round(statistics.mean(ttfts_ms), 3) if ttfts_ms else 0.0,
        "ttft_p50_ms": round(_percentile(ttfts_ms, 0.50), 3),
        "ttft_p95_ms": round(_percentile(ttfts_ms, 0.95), 3),
        "tpot_mean_ms": round(statistics.mean(tpots_ms), 3) if tpots_ms else 0.0,
        "tpot_p50_ms": round(_percentile(tpots_ms, 0.50), 3),
        "tpot_p95_ms": round(_percentile(tpots_ms, 0.95), 3),
        "e2e_mean_ms": round(statistics.mean(e2es_ms), 3) if e2es_ms else 0.0,
        "e2e_p50_ms": round(_percentile(e2es_ms, 0.50), 3),
        "e2e_p95_ms": round(_percentile(e2es_ms, 0.95), 3),
        "speculative_draft_rounds_total": total_rounds,
        "mean_tokens_per_round": round(mean_tok_per_round, 3),
    }


def run_saturation_sweep(
    prefill_url: str,
    *,
    concurrencies: Sequence[int],
    duration_s: float,
    contexts: Sequence[int],
    max_new_tokens: int,
    warm_ratio: float,
    run_id: int,
    timeout_s: float,
) -> Tuple[List[Dict[str, Any]], Optional[float], Optional[float]]:
    """Run sustained mixed-PD saturation sweeps across concurrency levels."""
    if not concurrencies or not contexts:
        return [], None, None
    workload = _build_saturation_workload(
        prefill_url,
        contexts=contexts,
        warm_slots=4,
        run_id=run_id,
    )
    # Prime warm prefix entries into P Device/Host cache before steady-state measurement.
    if workload["warm"]:
        run_concurrent_batch(
            prefill_url,
            workload["warm"],
            max_new_tokens=max(4, min(16, max_new_tokens)),
            reuse_cache=True,
            ignore_eos=True,
            timeout_s=timeout_s,
        )

    rows: List[Dict[str, Any]] = []
    sat_start = time.time()
    for conc in concurrencies:
        row = run_saturation_level(
            prefill_url,
            concurrency=conc,
            duration_s=duration_s,
            workload=workload,
            max_new_tokens=max_new_tokens,
            warm_ratio=warm_ratio,
            timeout_s=timeout_s,
        )
        rows.append(row)
        print(
            f"[SATURATION C={conc} T={contexts} out={max_new_tokens}] "
            f"reqs={row['requests_completed']} err={row['error_count']} "
            f"sys={row['system_total_tok_s']} tok/s "
            f"(pf={row['system_prefill_tok_s']}, dec={row['system_decode_tok_s']}) | "
            f"TTFT p50/p95={row['ttft_p50_ms']}/{row['ttft_p95_ms']}ms | "
            f"TPOT p50/p95={row['tpot_p50_ms']}/{row['tpot_p95_ms']}ms",
            flush=True,
        )
    sat_end = time.time()
    return rows, sat_start, sat_end


# ---------------------------------------------------------------------------
# Verification Gates (Golden Parity, Cache, RDMA, CUDA Graph, QSA, DFlash2)
# ---------------------------------------------------------------------------


def run_golden_suite(
    prefill_url: str,
    prompts: Sequence[Dict[str, str]],
    *,
    max_new_tokens: int = 64,
    expect_dspark: bool = True,
    golden_in: Optional[str] = None,
    golden_out: Optional[str] = None,
    timeout_s: float = 300.0,
) -> Dict[str, Any]:
    """Run greedy golden prompts, optionally saving or asserting token-ID parity."""
    records: List[Dict[str, Any]] = []
    for item in prompts:
        res = post_generate(
            prefill_url,
            item["prompt"],
            max_new_tokens=max_new_tokens,
            reuse_cache=False,
            ignore_eos=False,
            random_seed=42,
            timeout_s=timeout_s,
        )
        resp = res["resp"]
        aux = resp.get("aux_info") or {}
        out_ids = _flatten_int_list(resp.get("output_ids"))
        in_ids = _flatten_int_list(resp.get("input_ids"))
        if expect_dspark:
            rounds = int(aux.get("speculative_draft_rounds") or 0)
            per_pos = aux.get("speculative_accepted_tokens_per_pos") or []
            if rounds <= 0 or not per_pos:
                raise AssertionError(
                    f"DFlash2 inactive on golden prompt {item['name']!r}: aux_info={aux}"
                )
        records.append(
            {
                "name": item["name"],
                "prompt": item["prompt"],
                "input_ids": in_ids,
                "output_ids": out_ids,
                "response": resp.get("response", ""),
                "aux_info": aux,
            }
        )

    if golden_out:
        out_path = Path(golden_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"records": records}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    parity_report: Dict[str, Any] = {"checked": False, "all_exact_match": None, "details": []}
    if golden_in:
        ref = json.loads(Path(golden_in).read_text(encoding="utf-8"))
        ref_by_name = {r["name"]: r for r in ref.get("records", [])}
        all_ok = True
        for rec in records:
            ref_rec = ref_by_name.get(rec["name"])
            if ref_rec is None:
                all_ok = False
                parity_report["details"].append(
                    {"name": rec["name"], "exact_match": False, "reason": "missing_reference"}
                )
                continue
            ref_ids = ref_rec["output_ids"]
            cur_ids = rec["output_ids"]
            prefix_match = 0
            for a, b in zip(ref_ids, cur_ids):
                if a != b:
                    break
                prefix_match += 1
            exact = cur_ids == ref_ids and rec["input_ids"] == ref_rec["input_ids"]
            if not exact:
                all_ok = False
            parity_report["details"].append(
                {
                    "name": rec["name"],
                    "exact_match": exact,
                    "prefix_match_len": prefix_match,
                    "ref_len": len(ref_ids),
                    "cur_len": len(cur_ids),
                }
            )
        parity_report["checked"] = True
        parity_report["all_exact_match"] = all_ok

    return {"records": records, "parity": parity_report}


def probe_device_and_host_cache(
    prefill_url: str,
    p_status: Dict[str, Any],
    *,
    probe_host_spill: bool = True,
    max_host_spill_tokens: int = 131072,
    timeout_s: float = 300.0,
) -> Dict[str, Any]:
    """Verify Device Cache hit and optionally probe Host Memory Cache eviction hit."""
    nonce = int(time.time() * 1000)
    probe_prompt, probe_len = build_exact_token_prompt(
        prefill_url, 1024, prefix_tag=f"cache_probe_{nonce}"
    )
    post_generate(
        prefill_url,
        probe_prompt,
        max_new_tokens=1,
        reuse_cache=True,
        timeout_s=timeout_s,
    )
    second = post_generate(
        prefill_url,
        probe_prompt,
        max_new_tokens=1,
        reuse_cache=True,
        timeout_s=timeout_s,
    )
    aux2 = second["resp"].get("aux_info") or {}
    device_hit_ok = _aux_reuse_len(aux2, "prefill_total_reuse_len", "reuse_len") > 0

    total_blocks = int(p_status.get("total_kv_cache") or 0)
    block_size = int(p_status.get("block_size") or 64)
    device_token_cap = total_blocks * block_size

    host_probe: Dict[str, Any] = {
        "device_token_capacity": device_token_cap,
        "attempted": False,
        "memory_reuse_len": _aux_reuse_len(aux2, "prefill_memory_reuse_len", "memory_reuse_len"),
        "skipped_reason": "",
    }
    if not device_hit_ok:
        raise AssertionError(f"Warm Device Cache hit check failed: aux_info={aux2}")

    if probe_host_spill and device_token_cap > 0:
        if device_token_cap > max_host_spill_tokens:
            host_probe["skipped_reason"] = (
                f"device_token_capacity={device_token_cap} exceeds cap={max_host_spill_tokens}"
            )
        else:
            host_probe["attempted"] = True
            chunk_tokens = min(8192, max(2048, device_token_cap // 2))
            needed = device_token_cap + block_size * 8
            n_chunks = int(math.ceil(needed / chunk_tokens))
            for i in range(n_chunks):
                filler_p, _ = build_exact_token_prompt(
                    prefill_url, chunk_tokens, prefix_tag=f"spill_{nonce}_{i}"
                )
                post_generate(
                    prefill_url,
                    filler_p,
                    max_new_tokens=1,
                    reuse_cache=True,
                    timeout_s=timeout_s,
                )
            re_hit = post_generate(
                prefill_url,
                probe_prompt,
                max_new_tokens=1,
                reuse_cache=True,
                timeout_s=timeout_s,
            )
            aux_h = re_hit["resp"].get("aux_info") or {}
            host_probe["memory_reuse_len"] = _aux_reuse_len(
                aux_h, "prefill_memory_reuse_len", "memory_reuse_len"
            )
            host_probe["total_reuse_len"] = _aux_reuse_len(
                aux_h, "prefill_total_reuse_len", "reuse_len"
            )

    return {
        "device_hit_ok": device_hit_ok,
        "probe_len": probe_len,
        "warm_aux": {
            "prefill_total_reuse_len": _aux_reuse_len(
                aux2, "prefill_total_reuse_len", "reuse_len"
            ),
            "prefill_local_reuse_len": _aux_reuse_len(
                aux2, "prefill_local_reuse_len", "local_reuse_len"
            ),
            "prefill_memory_reuse_len": _aux_reuse_len(
                aux2, "prefill_memory_reuse_len", "memory_reuse_len"
            ),
        },
        "host_spill_probe": host_probe,
    }


def _read_log_glob(pattern: str) -> str:
    paths: List[str] = []
    for sub in pattern.split(","):
        pat = sub.strip()
        if pat:
            paths.extend(sorted(glob.glob(pat)))
    chunks: List[str] = []
    for p in dict.fromkeys(paths):
        try:
            chunks.append(Path(p).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def verify_logs(
    *,
    prefill_log: Optional[str],
    decode_log: Optional[str],
    expect_qsa: bool = True,
    expect_cuda_graph: bool = True,
    expect_dspark: bool = True,
) -> Dict[str, Any]:
    """Check RDMA active / 0 TCP fallback, QSA, CUDA Graph, MemoryCache, DSpark, and CacheStore 1:1."""
    report: Dict[str, Any] = {"prefill": {}, "decode": {}}
    rdma_markers = (
        "rdma client init success",
        "reg user mr success",
        "rdma connection to",
        "TransferBackend=kBarexRdma",
        "[RKEY_TRACE]",
    )
    tcp_markers = (
        "cache_store_rdma_mode: 0",
        "TransferBackend=kTcp",
        "NoRdmaMemoryUtilImpl",
        "tcp mode, no memory",
    )

    for role, pattern in (("prefill", prefill_log), ("decode", decode_log)):
        if not pattern:
            continue
        text = _read_log_glob(pattern)
        if not text:
            raise FileNotFoundError(f"No readable log files matched {role} pattern: {pattern}")

        rdma_cfg = "cache_store_rdma_mode: 1" in text
        rdma_hits = [m for m in rdma_markers if m in text]
        tcp_hits = [m for m in tcp_markers if m in text]
        qsa_active = "[Qwen4_5 QSA] enabled" in text
        mem_cache_enabled = ("enable_memory_cache: True" in text) or ("enable_memory_cache: 1" in text)
        cuda_graph_enabled = "CUDA graph capture is enabled" in text
        cuda_graph_end = "Capture Decode End" in text
        cg_batches = sorted(
            {
                int(x)
                for x in re.findall(
                    r"(?:\[CudaGraph Memory\] captured batch size|capture success for batch size:)\s*(\d+)",
                    text,
                )
            }
        )
        cg_errors = [
            m
            for m in (
                "initCapture prepare_fmha_impl failed",
                "initCapture forward for output datatype failed",
                "WarmUp forward failed",
            )
            if m in text
        ]
        sp_errors = [
            m
            for m in (
                "speculative target commit prepare failed",
                "speculative target side-state commit failed",
            )
            if m in text
        ]

        role_res = {
            "rdma_mode_1": rdma_cfg,
            "rdma_markers_found": rdma_hits,
            "tcp_fallback_markers_found": tcp_hits,
            "qsa_enabled_logged": qsa_active,
            "memory_cache_logged": mem_cache_enabled,
            "cuda_graph_enabled": cuda_graph_enabled,
            "cuda_graph_capture_end": cuda_graph_end,
            "cuda_graph_captured_batch_sizes": cg_batches,
            "cuda_graph_errors": cg_errors,
            "speculative_errors": sp_errors,
        }
        report[role] = role_res

        if tcp_hits:
            raise AssertionError(f"{role} log contains TCP fallback markers: {tcp_hits}")
        if not (rdma_cfg and rdma_hits):
            raise AssertionError(f"{role} log missing active Barex RDMA evidence: {role_res}")
        if expect_qsa and not qsa_active:
            raise AssertionError(f"{role} log missing '[Qwen4_5 QSA] enabled'")
        if role == "decode" and expect_cuda_graph:
            if not (cuda_graph_enabled and cuda_graph_end and cg_batches) or cg_errors:
                raise AssertionError(f"Decode CUDA Graph verification failed: {role_res}")
        if role == "decode" and expect_dspark and sp_errors:
            raise AssertionError(f"Decode speculative commit errors found: {sp_errors}")

    return report


# ---------------------------------------------------------------------------
# Kineto Chrome Trace Parser & Profiler Trigger
# ---------------------------------------------------------------------------


def _categorize_kernel(name: str) -> str:
    low = name.lower()
    if "nccl" in low or "allreduce" in low or "allgather" in low or "reducescatter" in low:
        return "nccl_comm"
    if "flashinfer" in low or "fmha" in low or "attention" in low or "paged_kv" in low:
        return "attention"
    if (
        "mxfp4" in low
        or "deepgemm" in low
        or "moe" in low
        or "cutlass" in low
        or "cublas" in low
        or "gemm" in low
        or "nvjet" in low
    ):
        return "gemm_moe"
    if "gdn" in low or "gated_delta" in low or "fla" in low or "conv" in low:
        return "gdn_conv"
    if "triton" in low:
        return "triton_other"
    if "norm" in low or "rmsnorm" in low or "rope" in low or "silu" in low or "situ" in low:
        return "elementwise_norm"
    return "other"


def parse_kineto_trace(trace_path: Path, top_k: int = 15) -> Dict[str, Any]:
    """Parse a Chrome/Kineto JSON trace and return top CUDA kernels + category shares."""
    raw = json.loads(trace_path.read_text(encoding="utf-8", errors="replace"))
    events = raw.get("traceEvents", []) if isinstance(raw, dict) else raw
    if not isinstance(events, list):
        raise ValueError(f"Unsupported trace format in {trace_path}")

    kernel_stats: Dict[str, List[float]] = {}
    category_us: Dict[str, float] = {}
    memcpy_us = 0.0
    total_kernel_us = 0.0

    for ev in events:
        if not isinstance(ev, dict):
            continue
        cat = str(ev.get("cat") or "").lower()
        dur = float(ev.get("dur") or 0.0)
        if dur <= 0.0:
            continue
        if cat in ("gpu_memcpy", "gpu_memset"):
            memcpy_us += dur
            continue
        if cat != "kernel":
            continue
        name = str(ev.get("name") or "unknown")
        stats = kernel_stats.setdefault(name, [0.0, 0.0])  # [total_us, count]
        stats[0] += dur
        stats[1] += 1.0
        total_kernel_us += dur
        bucket = _categorize_kernel(name)
        category_us[bucket] = category_us.get(bucket, 0.0) + dur

    sorted_kernels = sorted(kernel_stats.items(), key=lambda kv: kv[1][0], reverse=True)
    top_rows = []
    for name, (tot_us, count) in sorted_kernels[:top_k]:
        top_rows.append(
            {
                "name": name,
                "total_ms": round(tot_us / 1000.0, 3),
                "pct": round(100.0 * tot_us / max(total_kernel_us, 1e-9), 2),
                "calls": int(count),
                "mean_us": round(tot_us / max(count, 1.0), 2),
                "category": _categorize_kernel(name),
            }
        )

    cat_summary = {
        k: {
            "total_ms": round(v / 1000.0, 3),
            "pct": round(100.0 * v / max(total_kernel_us, 1e-9), 2),
        }
        for k, v in sorted(category_us.items(), key=lambda kv: kv[1], reverse=True)
    }
    return {
        "trace_file": str(trace_path),
        "total_cuda_kernel_ms": round(total_kernel_us / 1000.0, 3),
        "gpu_memcpy_memset_ms": round(memcpy_us / 1000.0, 3),
        "unique_kernels": len(kernel_stats),
        "categories": cat_summary,
        "top_kernels": top_rows,
    }


def trigger_kineto_capture(
    prefill_url: str,
    decode_url: Optional[str],
    *,
    trace_prefix: str,
    traces_dir: Optional[str] = None,
    prefill_steps: int = 3,
    decode_steps: int = 6,
    max_new_tokens: int = 96,
    batch_size: int = 1,
    timeout_s: float = 300.0,
) -> Dict[str, Any]:
    """Trigger Rank 0 `/start_profile` on P and D and run a capture request/batch."""
    p_name = f"{trace_prefix}_prefill"
    d_name = f"{trace_prefix}_decode"
    p_prof = _http_json(
        f"{prefill_url.rstrip('/')}/start_profile",
        payload={
            "trace_name": p_name,
            "start_step": 0,
            "num_steps": int(prefill_steps),
            "enable_all_rank": False,
        },
        timeout_s=15.0,
    )
    d_prof = None
    if decode_url:
        d_prof = _http_json(
            f"{decode_url.rstrip('/')}/start_profile",
            payload={
                "trace_name": d_name,
                "start_step": 0,
                "num_steps": int(decode_steps),
                "enable_all_rank": False,
            },
            timeout_s=15.0,
        )

    prompts = [
        p for p, _ in build_prompt_batch(prefill_url, 2048, max(1, int(batch_size)), f"kineto_{int(time.time())}")
    ]
    cap_len = len(tokenizer_encode(prefill_url, prompts[0]))
    batch_res, _ = run_concurrent_batch(
        prefill_url,
        prompts,
        max_new_tokens=max_new_tokens,
        reuse_cache=False,
        ignore_eos=True,
        timeout_s=timeout_s,
    )
    res = batch_res[0]
    p_files: List[Path] = []
    d_files: List[Path] = []
    if traces_dir:
        tdir = Path(traces_dir)
        deadline = time.monotonic() + 30.0
        prev_sizes: Dict[Path, int] = {}
        while True:
            p_files = sorted(tdir.glob(f"{p_name}*wr0*.json"))
            d_files = sorted(tdir.glob(f"{d_name}*wr0*.json"))
            cur = p_files + d_files
            if cur:
                cur_sizes = {f: f.stat().st_size for f in cur if f.exists()}
                stable = bool(cur_sizes) and all(
                    sz > 0 and prev_sizes.get(f) == sz for f, sz in cur_sizes.items()
                )
                if stable:
                    break
                prev_sizes = cur_sizes
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
    else:
        time.sleep(2.0)

    out: Dict[str, Any] = {
        "prefill_start_profile_resp": p_prof,
        "decode_start_profile_resp": d_prof,
        "capture_input_tokens": cap_len,
        "capture_batch_size": max(1, int(batch_size)),
        "capture_aux_info": res["resp"].get("aux_info") or {},
        "prefill_trace": parse_kineto_trace(p_files[-1]) if p_files else None,
        "decode_trace": parse_kineto_trace(d_files[-1]) if d_files else None,
    }
    return out


def parse_dmon_log(
    dmon_path: Path,
    *,
    start_epoch: Optional[float] = None,
    end_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarize `nvidia-smi dmon -s mu -d 1 -o DT` samples (per-GPU and cluster)."""
    per_gpu: Dict[int, Dict[str, List[float]]] = {}
    sample_count = 0
    for raw_line in dmon_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Expected: YYYYMMDD HH:MM:SS gpu fb bar1 ccpm sm mem enc dec jpg ofa
        if len(parts) < 8:
            continue
        try:
            ts = dt.datetime.strptime(f"{parts[0]} {parts[1]}", "%Y%m%d %H:%M:%S").timestamp()
            gpu_idx = int(parts[2])
        except ValueError:
            continue
        if start_epoch is not None and ts < start_epoch - 1.0:
            continue
        if end_epoch is not None and ts > end_epoch + 1.0:
            continue

        def _num(tok: str) -> Optional[float]:
            if tok in ("-", "N/A"):
                return None
            return float(tok)

        fb = _num(parts[3])
        sm = _num(parts[6])
        mem = _num(parts[7])
        bucket = per_gpu.setdefault(gpu_idx, {"fb": [], "sm": [], "mem": []})
        if fb is not None:
            bucket["fb"].append(fb)
        if sm is not None:
            bucket["sm"].append(sm)
        if mem is not None:
            bucket["mem"].append(mem)
        sample_count += 1

    gpu_summary: Dict[str, Any] = {}
    all_sm: List[float] = []
    all_mem: List[float] = []
    all_fb: List[float] = []
    for g in sorted(per_gpu):
        d = per_gpu[g]
        all_sm.extend(d["sm"])
        all_mem.extend(d["mem"])
        all_fb.extend(d["fb"])
        gpu_summary[f"gpu_{g}"] = {
            "samples": len(d["sm"]),
            "sm_mean_pct": round(statistics.mean(d["sm"]), 2) if d["sm"] else 0.0,
            "sm_max_pct": round(max(d["sm"]), 2) if d["sm"] else 0.0,
            "mem_mean_pct": round(statistics.mean(d["mem"]), 2) if d["mem"] else 0.0,
            "mem_max_pct": round(max(d["mem"]), 2) if d["mem"] else 0.0,
            "fb_max_mib": round(max(d["fb"]), 1) if d["fb"] else 0.0,
        }
    return {
        "dmon_file": str(dmon_path),
        "samples": sample_count,
        "all_gpus_sm_mean_pct": round(statistics.mean(all_sm), 2) if all_sm else 0.0,
        "all_gpus_sm_max_pct": round(max(all_sm), 2) if all_sm else 0.0,
        "all_gpus_mem_mean_pct": round(statistics.mean(all_mem), 2) if all_mem else 0.0,
        "all_gpus_mem_max_pct": round(max(all_mem), 2) if all_mem else 0.0,
        "all_gpus_fb_max_mib": round(max(all_fb), 1) if all_fb else 0.0,
        "per_gpu": gpu_summary,
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen 4.5-Flash Live PD Perf & Parity Benchmark")
    parser.add_argument("--parse-trace", default="", help="Standalone mode: parse one Kineto JSON file and exit")
    parser.add_argument("--parse-dmon", default="", help="Standalone mode: parse one dmon -o DT log and exit")
    parser.add_argument("--dmon-start-epoch", type=float, default=None, help="Optional start epoch for --parse-dmon")
    parser.add_argument("--dmon-end-epoch", type=float, default=None, help="Optional end epoch for --parse-dmon")
    parser.add_argument("--prefill-url", default="http://127.0.0.1:31000", help="Prefill frontend URL (entrance)")
    parser.add_argument("--decode-url", default="", help="Decode backend HTTP URL (health, cache_status, start_profile)")
    parser.add_argument("--prefill-log", default="", help="Optional Prefill log file/glob for RDMA/QSA verification")
    parser.add_argument("--decode-log", default="", help="Optional Decode log file/glob for RDMA/QSA/CG verification")
    parser.add_argument("--prefill-dmon", default="", help="Optional Prefill nvidia-smi dmon -o DT log")
    parser.add_argument("--decode-dmon", default="", help="Optional Decode nvidia-smi dmon -o DT log")
    parser.add_argument("--traces-dir", default="", help="Optional trace directory if locally readable")
    parser.add_argument("--out-json", default="", help="Save full JSON report to this path")
    parser.add_argument("--golden-out", default="", help="Save greedy golden records JSON")
    parser.add_argument("--golden-in", default="", help="Verify exact greedy token-ID parity against saved JSON")
    parser.add_argument("--eval-prompts-json", default="", help="Optional JSON list of {name, prompt}")
    parser.add_argument("--golden-max-new-tokens", type=int, default=64)
    parser.add_argument("--expect-dspark", dest="expect_dspark", action="store_true", default=True)
    parser.add_argument("--no-dspark", dest="expect_dspark", action="store_false")
    parser.add_argument("--expect-qsa", dest="expect_qsa", action="store_true", default=True)
    parser.add_argument("--no-qsa", dest="expect_qsa", action="store_false")
    parser.add_argument("--expect-cuda-graph", dest="expect_cuda_graph", action="store_true", default=True)
    parser.add_argument("--no-cuda-graph", dest="expect_cuda_graph", action="store_false")
    parser.add_argument("--probe-host-cache", dest="probe_host_cache", action="store_true", default=True)
    parser.add_argument("--no-host-cache-probe", dest="probe_host_cache", action="store_false")
    parser.add_argument("--prefill-lengths", default="1024,4096")
    parser.add_argument("--prefill-batches", default="1,4,8,16")
    parser.add_argument("--decode-contexts", default="512,2048")
    parser.add_argument("--decode-batches", default="1,4,8,16,32,64")
    parser.add_argument("--decode-max-new-tokens", type=int, default=64)
    parser.add_argument("--saturation-concurrencies", default="16,32,64,128")
    parser.add_argument("--saturation-duration-s", type=float, default=12.0)
    parser.add_argument("--saturation-contexts", default="1024,2048")
    parser.add_argument("--saturation-max-new-tokens", type=int, default=32)
    parser.add_argument("--saturation-warm-ratio", type=float, default=0.5)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--kineto", dest="kineto", action="store_true", default=True)
    parser.add_argument("--no-kineto", dest="kineto", action="store_false")
    parser.add_argument("--trace-prefix", default="qwen45_pd")
    parser.add_argument("--kineto-batch-size", type=int, default=1)
    parser.add_argument("--kineto-only", action="store_true", help="Only capture and parse Kineto trace")
    parser.add_argument("--allow-parity-diff", action="store_true", help="Record golden parity mismatch without aborting")
    parser.add_argument("--skip-bench", action="store_true", help="Only run health, cache probe, golden, and log checks")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()

    if args.parse_trace:
        summary = parse_kineto_trace(Path(args.parse_trace))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    if args.parse_dmon:
        summary = parse_dmon_log(
            Path(args.parse_dmon),
            start_epoch=args.dmon_start_epoch,
            end_epoch=args.dmon_end_epoch,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    set_decode_role_addrs(args.decode_url or None)

    wait_for_health(args.prefill_url, "Prefill", timeout_s=args.timeout_s)
    if args.decode_url:
        wait_for_health(args.decode_url, "Decode", timeout_s=args.timeout_s)

    if args.kineto_only:
        k_only = trigger_kineto_capture(
            args.prefill_url,
            args.decode_url or None,
            trace_prefix=args.trace_prefix,
            traces_dir=args.traces_dir or None,
            batch_size=args.kineto_batch_size,
            timeout_s=args.timeout_s,
        )
        if args.out_json:
            out_p = Path(args.out_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(k_only, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            print(json.dumps(k_only, indent=2, ensure_ascii=False))
        return 0

    p_cache_init = get_cache_status(args.prefill_url)
    d_cache_init = get_cache_status(args.decode_url) if args.decode_url else None

    for i in range(max(0, args.warmup_iters)):
        post_generate(
            args.prefill_url,
            f"Warmup check iteration {i}: summarize parallel tensor execution.",
            max_new_tokens=16,
            reuse_cache=False,
            ignore_eos=False,
            timeout_s=args.timeout_s,
        )

    prompts = DEFAULT_GOLDEN_PROMPTS
    if args.eval_prompts_json:
        raw_p = json.loads(Path(args.eval_prompts_json).read_text(encoding="utf-8"))
        if isinstance(raw_p, dict) and "prompts" in raw_p:
            raw_p = raw_p["prompts"]
        prompts = [
            (
                {"name": f"prompt_{i}", "prompt": x}
                if isinstance(x, str)
                else {"name": str(x.get("name", f"prompt_{i}")), "prompt": str(x["prompt"])}
            )
            for i, x in enumerate(raw_p)
        ]

    golden_res = run_golden_suite(
        args.prefill_url,
        prompts,
        max_new_tokens=args.golden_max_new_tokens,
        expect_dspark=args.expect_dspark,
        golden_in=args.golden_in or None,
        golden_out=args.golden_out or None,
        timeout_s=args.timeout_s,
    )
    if golden_res["parity"]["checked"] and not golden_res["parity"]["all_exact_match"]:
        if args.allow_parity_diff:
            print(f"[PARITY WARN] {golden_res['parity']}", flush=True)
        else:
            raise AssertionError(f"Greedy golden parity FAILED: {golden_res['parity']}")

    time.sleep(1.5)
    d_cache_post_golden = get_cache_status(args.decode_url) if args.decode_url else None

    prefill_matrix: List[Dict[str, Any]] = []
    decode_matrix: List[Dict[str, Any]] = []
    saturation_sweep: List[Dict[str, Any]] = []
    cache_probe: Dict[str, Any] = {}
    kineto_res: Dict[str, Any] = {}
    phase_windows: Dict[str, Any] = {}

    if not args.skip_bench:
        cache_probe = probe_device_and_host_cache(
            args.prefill_url,
            p_cache_init,
            probe_host_spill=args.probe_host_cache,
            timeout_s=args.timeout_s,
        )

        run_id = int(time.time())
        pf_start_epoch = time.time()
        for length in _parse_int_list(args.prefill_lengths):
            for bs in _parse_int_list(args.prefill_batches):
                batch_prompts = [
                    p for p, _ in build_prompt_batch(args.prefill_url, length, bs, f"pf_{run_id}")
                ]
                cold_prompts = [
                    p for p, _ in build_prompt_batch(args.prefill_url, length, bs, f"pfc_{run_id}")
                ]
                # Prime P Device/Host cache (also warms Cold JIT for exact shape),
                # then warm suffix-hit JIT, then measure Warm & Cold Prefill.
                run_concurrent_batch(
                    args.prefill_url,
                    batch_prompts,
                    max_new_tokens=1,
                    reuse_cache=True,
                    ignore_eos=True,
                    timeout_s=args.timeout_s,
                )
                run_concurrent_batch(
                    args.prefill_url,
                    batch_prompts,
                    max_new_tokens=1,
                    reuse_cache=True,
                    ignore_eos=True,
                    timeout_s=args.timeout_s,
                )
                warm_res, warm_wall = run_concurrent_batch(
                    args.prefill_url,
                    batch_prompts,
                    max_new_tokens=1,
                    reuse_cache=True,
                    ignore_eos=True,
                    timeout_s=args.timeout_s,
                )
                warm_summary = summarize_prefill_batch(warm_res, warm_wall)

                cold_res, cold_wall = run_concurrent_batch(
                    args.prefill_url,
                    cold_prompts,
                    max_new_tokens=1,
                    reuse_cache=False,
                    ignore_eos=True,
                    timeout_s=args.timeout_s,
                )
                cold_summary = summarize_prefill_batch(cold_res, cold_wall)
                row = {
                    "target_tokens": length,
                    "batch_size": bs,
                    "cold": cold_summary,
                    "warm": warm_summary,
                }
                prefill_matrix.append(row)
                print(
                    f"[PREFILL T={length} B={bs}] "
                    f"cold_ttft={cold_summary['ttft_mean_ms']}ms ({cold_summary['prefill_tok_s']} tok/s) | "
                    f"warm_ttft={warm_summary['ttft_mean_ms']}ms ({warm_summary['prefill_tok_s']} tok/s, "
                    f"hit={warm_summary['cache_hit_ratio']})",
                    flush=True,
                )
        pf_end_epoch = time.time()

        dec_start_epoch = time.time()
        for ctx in _parse_int_list(args.decode_contexts):
            for bs in _parse_int_list(args.decode_batches):
                batch_prompts = [
                    p for p, _ in build_prompt_batch(args.prefill_url, ctx, bs, f"dec_{run_id}")
                ]
                dec_res, dec_wall = run_concurrent_batch(
                    args.prefill_url,
                    batch_prompts,
                    max_new_tokens=args.decode_max_new_tokens,
                    reuse_cache=False,
                    ignore_eos=True,
                    timeout_s=args.timeout_s,
                )
                if args.expect_dspark:
                    for r in dec_res:
                        aux = r["resp"].get("aux_info") or {}
                        if int(aux.get("speculative_draft_rounds") or 0) <= 0:
                            raise AssertionError(f"DFlash2 inactive in Decode ctx={ctx} B={bs}: {aux}")
                dec_summary = summarize_decode_batch(dec_res, dec_wall)
                row = {"context_tokens": ctx, **dec_summary}
                decode_matrix.append(row)
                print(
                    f"[DECODE ctx={ctx} B={bs}] "
                    f"tpot={dec_summary['tpot_mean_ms']}ms (p95={dec_summary['tpot_p95_ms']}ms) | "
                    f"decode_tok/s={dec_summary['decode_tok_s']} "
                    f"(batch_tpot_tok/s={dec_summary['batch_decode_tok_s_from_mean_tpot']}) | "
                    f"mean_acc_tok/rnd={dec_summary['mean_tokens_per_round']}",
                    flush=True,
                )
        dec_end_epoch = time.time()
        phase_windows = {
            "prefill_start_epoch": pf_start_epoch,
            "prefill_end_epoch": pf_end_epoch,
            "decode_start_epoch": dec_start_epoch,
            "decode_end_epoch": dec_end_epoch,
        }

        if args.kineto:
            kineto_res = trigger_kineto_capture(
                args.prefill_url,
                args.decode_url or None,
                trace_prefix=args.trace_prefix,
                traces_dir=args.traces_dir or None,
                batch_size=args.kineto_batch_size,
                timeout_s=args.timeout_s,
            )

        sat_concs = _parse_int_list(args.saturation_concurrencies)
        sat_ctxs = _parse_int_list(args.saturation_contexts)
        if sat_concs and sat_ctxs:
            saturation_sweep, sat_start, sat_end = run_saturation_sweep(
                args.prefill_url,
                concurrencies=sat_concs,
                duration_s=args.saturation_duration_s,
                contexts=sat_ctxs,
                max_new_tokens=args.saturation_max_new_tokens,
                warm_ratio=args.saturation_warm_ratio,
                run_id=run_id,
                timeout_s=args.timeout_s,
            )
            phase_windows["saturation_start_epoch"] = sat_start
            phase_windows["saturation_end_epoch"] = sat_end

    d_cache_end = None
    d_leaked_blocks = None
    if args.decode_url and d_cache_init is not None:
        drain_deadline = time.monotonic() + 10.0
        while True:
            d_cache_end = get_cache_status(args.decode_url)
            d_leaked_blocks = int(d_cache_init["available_kv_cache"]) - int(
                d_cache_end["available_kv_cache"]
            )
            if d_leaked_blocks == 0 or time.monotonic() >= drain_deadline:
                break
            time.sleep(0.5)
        if d_leaked_blocks != 0:
            raise AssertionError(
                f"Decode KV cache leak detected: init={d_cache_init['available_kv_cache']}, "
                f"end={d_cache_end['available_kv_cache']}, leaked={d_leaked_blocks}"
            )
    p_cache_end = get_cache_status(args.prefill_url)

    log_checks = verify_logs(
        prefill_log=args.prefill_log or None,
        decode_log=args.decode_log or None,
        expect_qsa=args.expect_qsa,
        expect_cuda_graph=args.expect_cuda_graph,
        expect_dspark=args.expect_dspark,
    )

    dmon_summary: Dict[str, Any] = {}
    for role, dmon_file in (("prefill", args.prefill_dmon), ("decode", args.decode_dmon)):
        if dmon_file and Path(dmon_file).is_file():
            role_dmon: Dict[str, Any] = {
                "overall": parse_dmon_log(Path(dmon_file)),
            }
            if phase_windows:
                if "prefill_start_epoch" in phase_windows:
                    role_dmon["during_prefill"] = parse_dmon_log(
                        Path(dmon_file),
                        start_epoch=phase_windows["prefill_start_epoch"],
                        end_epoch=phase_windows["prefill_end_epoch"],
                    )
                if "decode_start_epoch" in phase_windows:
                    role_dmon["during_decode"] = parse_dmon_log(
                        Path(dmon_file),
                        start_epoch=phase_windows["decode_start_epoch"],
                        end_epoch=phase_windows["decode_end_epoch"],
                    )
                if "saturation_start_epoch" in phase_windows:
                    role_dmon["during_saturation"] = parse_dmon_log(
                        Path(dmon_file),
                        start_epoch=phase_windows["saturation_start_epoch"],
                        end_epoch=phase_windows["saturation_end_epoch"],
                    )
                if saturation_sweep:
                    role_dmon["during_saturation_by_c"] = {
                        f"c{row['concurrency']}": parse_dmon_log(
                            Path(dmon_file),
                            start_epoch=row["steady_start_epoch"],
                            end_epoch=row["steady_end_epoch"],
                        )
                        for row in saturation_sweep
                    }
            dmon_summary[role] = role_dmon

    report = {
        "prefill_url": args.prefill_url,
        "decode_url": args.decode_url,
        "phase_windows": phase_windows,
        "cache_status": {
            "prefill_init": p_cache_init,
            "prefill_end": p_cache_end,
            "decode_init": d_cache_init,
            "decode_post_golden": d_cache_post_golden,
            "decode_end": d_cache_end,
            "decode_leaked_blocks": d_leaked_blocks,
        },
        "cache_probe": cache_probe,
        "golden_parity": golden_res["parity"],
        "golden_count": len(golden_res["records"]),
        "prefill_matrix": prefill_matrix,
        "decode_matrix": decode_matrix,
        "saturation_sweep": saturation_sweep,
        "kineto": kineto_res,
        "dmon": dmon_summary,
        "log_checks": log_checks,
    }
    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
