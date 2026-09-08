"""Benchmark-only UT for DSV4 TensorCore/SIMT overlap candidates."""

from __future__ import annotations

import json
import os
import re
import time
import unittest
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
    fused_rmsnorm_rope,
)
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb_batched

try:
    import deep_gemm
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor
except Exception:  # pragma: no cover - exercised on hosts without DeepGEMM
    deep_gemm = None
    get_mn_major_tma_aligned_packed_ue8m0_tensor = None


N_HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
N_GROUPS = 8
HEADS_PER_GROUP = N_HEADS // N_GROUPS
WO_A_RANK = int(os.environ.get("DSV4_OVERLAP_WO_A_RANK", "1024"))
WO_B_OUT = int(os.environ.get("DSV4_OVERLAP_WO_B_OUT", "2048"))
Q_LORA_RANK = int(os.environ.get("DSV4_OVERLAP_Q_LORA_RANK", "1536"))
MODEL_DIM = int(os.environ.get("DSV4_OVERLAP_MODEL_DIM", "4096"))
IDX_HEADS = int(os.environ.get("DSV4_OVERLAP_INDEX_HEADS", "64"))
IDX_HEAD_DIM = int(os.environ.get("DSV4_OVERLAP_INDEX_HEAD_DIM", "128"))
IDX_OUT = IDX_HEADS * IDX_HEAD_DIM


@dataclass
class BenchResult:
    section: str
    case: str
    M: int
    tile_m: str
    baseline_us: float
    candidate_us: float
    speedup: float
    valid: bool = True
    invalid_reason: str = ""
    max_abs: Optional[float] = None
    mean_abs: Optional[float] = None
    measure_method: str = "cuda_event"
    baseline_trace: str = ""
    candidate_trace: str = ""
    baseline_kernel_span_us: float = 0.0
    candidate_kernel_span_us: float = 0.0
    baseline_kernel_sum_us: float = 0.0
    candidate_kernel_sum_us: float = 0.0
    baseline_kernel_union_us: float = 0.0
    candidate_kernel_union_us: float = 0.0
    baseline_kernel_overlap_us: float = 0.0
    candidate_kernel_overlap_us: float = 0.0
    baseline_idle_gap_us: float = 0.0
    candidate_idle_gap_us: float = 0.0
    baseline_kernel_count: int = 0
    candidate_kernel_count: int = 0
    baseline_stream_count: int = 0
    candidate_stream_count: int = 0

    def __post_init__(self):
        if self.baseline_kernel_span_us == 0.0:
            self.baseline_kernel_span_us = self.baseline_us
        if self.candidate_kernel_span_us == 0.0:
            self.candidate_kernel_span_us = self.candidate_us


@dataclass
class TimelineStats:
    span_us: float
    kernel_sum_us: float
    kernel_union_us: float
    kernel_overlap_us: float
    idle_gap_us: float
    kernel_count: int
    stream_count: int
    trace_path: str


def _parse_int_list(name: str, default: Iterable[int]) -> List[int]:
    value = os.environ.get(name)
    if not value:
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_tile_list(name: str, default: Iterable[object]) -> List[object]:
    value = os.environ.get(name)
    if not value:
        return list(default)
    out: List[object] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append("no_split" if raw in ("0", "none", "no_split") else int(raw))
    return out


def _default_m_list() -> List[int]:
    if os.environ.get("DSV4_OVERLAP_FULL", "0") == "1":
        return [1, 16, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    return [1, 16, 256, 1024]


def _default_tile_list() -> List[object]:
    if os.environ.get("DSV4_OVERLAP_FULL", "0") == "1":
        return ["no_split", 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    return ["no_split", 64, 256, 1024]


def _iters_for_m(M: int) -> int:
    env = os.environ.get("DSV4_OVERLAP_ITERS")
    if env:
        return int(env)
    if M <= 256:
        return 80
    if M <= 4096:
        return 25
    return 8


def _warmup() -> int:
    return int(os.environ.get("DSV4_OVERLAP_WARMUP", "15"))


def _profile_enabled() -> bool:
    return os.environ.get("DSV4_OVERLAP_PROFILE", "0") == "1"


def _profile_iters_for_m(M: int) -> int:
    return int(os.environ.get("DSV4_OVERLAP_PROFILE_ITERS", "1"))


def _trace_dir() -> str:
    path = os.environ.get("DSV4_OVERLAP_TRACE_DIR")
    if path:
        return path
    report_path = os.environ.get("DSV4_OVERLAP_JSON")
    if report_path:
        return os.path.splitext(report_path)[0] + "_traces"
    return "/tmp/dsv4_tc_simt_overlap_traces"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _merged_interval_duration(intervals: List[Tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def _parse_timeline(trace_path: str, iters: int) -> TimelineStats:
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    kernels = [
        e
        for e in events
        if e.get("ph") == "X"
        and "kernel" in str(e.get("cat", ""))
        and "ts" in e
        and "dur" in e
    ]
    if not kernels:
        raise AssertionError(f"profile trace has no GPU kernel events: {trace_path}")

    intervals = [(float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in kernels]
    span_us = max(end for _, end in intervals) - min(start for start, _ in intervals)
    kernel_sum_us = sum(float(e["dur"]) for e in kernels)
    kernel_union_us = _merged_interval_duration(intervals)
    streams = {
        str(e.get("args", {}).get("stream", e.get("tid", "")))
        for e in kernels
    }
    scale = max(iters, 1)
    return TimelineStats(
        span_us=span_us / scale,
        kernel_sum_us=kernel_sum_us / scale,
        kernel_union_us=kernel_union_us / scale,
        kernel_overlap_us=(kernel_sum_us - kernel_union_us) / scale,
        idle_gap_us=(span_us - kernel_union_us) / scale,
        kernel_count=len(kernels) // scale,
        stream_count=len(streams),
        trace_path=trace_path,
    )


def _profile_one(fn: Callable[[], torch.Tensor], label: str, M: int) -> TimelineStats:
    warmup = _warmup()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    iters = _profile_iters_for_m(M)
    trace_dir = _trace_dir()
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, _safe_name(label) + ".json")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function(label):
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    return _parse_timeline(trace_path, iters)


def _measure_pair(
    baseline: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
    *,
    section: str,
    case: str,
    M: int,
    tile_name: str,
) -> Tuple[float, float, Dict[str, object]]:
    if _profile_enabled():
        prefix = f"{section}_{case}_M{M}_tile{tile_name}"
        base_stats = _profile_one(baseline, prefix + "_baseline", M)
        cand_stats = _profile_one(candidate, prefix + "_candidate", M)
        return (
            base_stats.span_us,
            cand_stats.span_us,
            {
                "measure_method": "torch_profiler_kernel_span",
                "baseline_trace": base_stats.trace_path,
                "candidate_trace": cand_stats.trace_path,
                "baseline_kernel_span_us": base_stats.span_us,
                "candidate_kernel_span_us": cand_stats.span_us,
                "baseline_kernel_sum_us": base_stats.kernel_sum_us,
                "candidate_kernel_sum_us": cand_stats.kernel_sum_us,
                "baseline_kernel_union_us": base_stats.kernel_union_us,
                "candidate_kernel_union_us": cand_stats.kernel_union_us,
                "baseline_kernel_overlap_us": base_stats.kernel_overlap_us,
                "candidate_kernel_overlap_us": cand_stats.kernel_overlap_us,
                "baseline_idle_gap_us": base_stats.idle_gap_us,
                "candidate_idle_gap_us": cand_stats.idle_gap_us,
                "baseline_kernel_count": base_stats.kernel_count,
                "candidate_kernel_count": cand_stats.kernel_count,
                "baseline_stream_count": base_stats.stream_count,
                "candidate_stream_count": cand_stats.stream_count,
            },
        )

    iters = _iters_for_m(M)
    return (
        _bench(baseline, iters=iters, warmup=_warmup()),
        _bench(candidate, iters=iters, warmup=_warmup()),
        {"measure_method": "cuda_event_loop"},
    )


def _make_freqs(rows: int, rd: int = ROPE_DIM) -> torch.Tensor:
    ang = torch.rand(rows, rd // 2, device="cuda") * 6.28
    return torch.polar(torch.ones_like(ang), ang).to(torch.complex64).contiguous()


def _bench(fn: Callable[[], torch.Tensor], *, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: List[float] = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    samples.sort()
    return samples[len(samples) // 2]


def _diff(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    d = (a.float() - b.float()).abs()
    return float(d.max().item()), float(d.mean().item())


def _is_deepgemm_env_error(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "NVCC compilation failed" in msg
        or "DeepGEMM" in msg
        or "deep_gemm" in msg
        or "No such file or directory" in msg and "g++" in msg
    )


def _make_wo_a_weight(G: int, R: int, K: int, seed: int = 0):
    assert deep_gemm is not None
    assert get_mn_major_tma_aligned_packed_ue8m0_tensor is not None
    torch.manual_seed(seed)
    w_fp32 = torch.randn(G * R, K, dtype=torch.float32, device="cuda") * 0.3
    weight_fp8 = w_fp32.to(torch.float8_e4m3fn).view(G, R, K).contiguous()
    scale_bytes = torch.randint(
        115, 125, (G * R // 128, K // 128), dtype=torch.uint8, device="cuda"
    )
    scale_raw = scale_bytes.view(torch.float8_e8m0fnu).contiguous()
    scale_fp32 = scale_raw.float().view(G, R // 128, K // 128)
    idx = torch.arange(R, device="cuda") // 128
    scale_rep = scale_fp32.index_select(-2, idx).contiguous()
    s_stk = get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)
    return weight_fp8, s_stk


def _make_scale_buf(n_groups: int, rows: int, packed_sf_k: int, device) -> torch.Tensor:
    tma_m = ((rows + 3) // 4) * 4
    return torch.empty(
        n_groups * packed_sf_k * tma_m, dtype=torch.int32, device=device
    ).as_strided((n_groups, rows, packed_sf_k), (packed_sf_k * tma_m, 1, tma_m))


def _wo_a_einsum(
    o_fp8: torch.Tensor,
    o_scale: torch.Tensor,
    w_stk: torch.Tensor,
    s_stk: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (w_stk, s_stk),
        out,
        recipe=(1, 1, 128),
    )
    return out


def _run_out_proj_baseline(
    o: torch.Tensor,
    freqs: torch.Tensor,
    w_a: torch.Tensor,
    s_a: torch.Tensor,
    w_b: torch.Tensor,
) -> torch.Tensor:
    M = o.shape[0]
    fp8, scale = fused_inv_rope_fp8_quant(
        o,
        freqs,
        n_groups=N_GROUPS,
        heads_per_group=HEADS_PER_GROUP,
        nope_dim=NOPE_DIM,
        rope_head_dim=ROPE_DIM,
    )
    wo_a = torch.empty(M, N_GROUPS, WO_A_RANK, dtype=torch.bfloat16, device=o.device)
    _wo_a_einsum(fp8, scale, w_a, s_a, wo_a)
    return F.linear(wo_a.reshape(M, N_GROUPS * WO_A_RANK), w_b)


def _run_out_proj_overlap(
    o: torch.Tensor,
    freqs: torch.Tensor,
    w_a: torch.Tensor,
    s_a: torch.Tensor,
    w_b: torch.Tensor,
    tile_m: int,
    ring_depth: int = 2,
) -> torch.Tensor:
    M = o.shape[0]
    if tile_m >= M:
        return _run_out_proj_baseline(o, freqs, w_a, s_a, w_b)

    simt_stream = torch.cuda.Stream(device=o.device)
    tc_stream = torch.cuda.Stream(device=o.device)
    current = torch.cuda.current_stream(o.device)
    simt_stream.wait_stream(current)
    tc_stream.wait_stream(current)

    K = HEADS_PER_GROUP * HEAD_DIM
    packed_sf_k = HEADS_PER_GROUP
    fp8_ring = [
        torch.empty((N_GROUPS, tile_m, K), dtype=torch.float8_e4m3fn, device=o.device)
        for _ in range(ring_depth)
    ]
    scale_ring = [
        _make_scale_buf(N_GROUPS, tile_m, packed_sf_k, o.device) for _ in range(ring_depth)
    ]
    wo_a_ring = [
        torch.empty((tile_m, N_GROUPS, WO_A_RANK), dtype=torch.bfloat16, device=o.device)
        for _ in range(ring_depth)
    ]
    quant_done = [torch.cuda.Event(blocking=False) for _ in range(ring_depth)]
    tc_done = [torch.cuda.Event(blocking=False) for _ in range(ring_depth)]
    out = torch.empty(M, WO_B_OUT, dtype=torch.bfloat16, device=o.device)

    for tile_idx, start in enumerate(range(0, M, tile_m)):
        end = min(start + tile_m, M)
        n = end - start
        slot = tile_idx % ring_depth
        if tile_idx >= ring_depth:
            simt_stream.wait_event(tc_done[slot])
        with torch.cuda.stream(simt_stream):
            fused_inv_rope_fp8_quant(
                o[start:end],
                freqs[start:end],
                n_groups=N_GROUPS,
                heads_per_group=HEADS_PER_GROUP,
                nope_dim=NOPE_DIM,
                rope_head_dim=ROPE_DIM,
                fp8_buf=fp8_ring[slot],
                scale_buf=scale_ring[slot],
            )
            quant_done[slot].record(simt_stream)
        with torch.cuda.stream(tc_stream):
            tc_stream.wait_event(quant_done[slot])
            wo_a_view = wo_a_ring[slot][:n]
            _wo_a_einsum(
                fp8_ring[slot][:, :n].transpose(0, 1),
                scale_ring[slot][:, :n].transpose(0, 1),
                w_a,
                s_a,
                wo_a_view,
            )
            out[start:end] = F.linear(wo_a_view.reshape(n, N_GROUPS * WO_A_RANK), w_b)
            tc_done[slot].record(tc_stream)

    current.wait_stream(simt_stream)
    current.wait_stream(tc_stream)
    return out


def _bench_out_projection(M: int, tile: object) -> Optional[BenchResult]:
    if deep_gemm is None:
        return None
    torch.manual_seed(1000 + M)
    o = (torch.randn(M, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.3).contiguous()
    freqs = _make_freqs(M)
    w_a, s_a = _make_wo_a_weight(N_GROUPS, WO_A_RANK, HEADS_PER_GROUP * HEAD_DIM, seed=1)
    w_b = torch.randn(WO_B_OUT, N_GROUPS * WO_A_RANK, dtype=torch.bfloat16, device="cuda") * 0.02
    baseline = lambda: _run_out_proj_baseline(o, freqs, w_a, s_a, w_b)
    if tile == "no_split":
        candidate = baseline
        tile_name = "no_split"
    else:
        tile_i = int(tile)
        tile_name = str(tile_i)
        candidate = lambda: _run_out_proj_overlap(o, freqs, w_a, s_a, w_b, tile_i)

    ref = baseline()
    cand = candidate()
    torch.cuda.synchronize()
    max_abs, mean_abs = _diff(ref, cand)
    if max_abs > 8e-2:
        raise AssertionError(f"out projection mismatch M={M} tile={tile_name}: {max_abs}")
    baseline_us, candidate_us, measure_extra = _measure_pair(
        baseline,
        candidate,
        section="out_projection",
        case="inv_rope_quant_woa_wob",
        M=M,
        tile_name=tile_name,
    )
    return BenchResult(
        "out_projection",
        "inv_rope_quant_woa_wob",
        M,
        tile_name,
        baseline_us,
        candidate_us,
        baseline_us / candidate_us if candidate_us > 0 else 0.0,
        max_abs=max_abs,
        mean_abs=mean_abs,
        **measure_extra,
    )


def _qk_tensors(M: int):
    torch.manual_seed(2000 + M)
    x = torch.randn(M, MODEL_DIM, dtype=torch.bfloat16, device="cuda") * 0.1
    qr = torch.randn(M, Q_LORA_RANK, dtype=torch.bfloat16, device="cuda") * 0.1
    wq_b = torch.randn(N_HEADS * HEAD_DIM, Q_LORA_RANK, dtype=torch.bfloat16, device="cuda") * 0.02
    wkv = torch.randn(HEAD_DIM, MODEL_DIM, dtype=torch.bfloat16, device="cuda") * 0.02
    kv_norm = torch.rand(HEAD_DIM, dtype=torch.bfloat16, device="cuda") + 0.5
    freqs = _make_freqs(M)
    return x, qr, wq_b, wkv, kv_norm, freqs


def _run_qk_serial(x, qr, wq_b, wkv, kv_norm, freqs):
    M = x.shape[0]
    q = F.linear(qr, wq_b).view(M, 1, N_HEADS, HEAD_DIM).contiguous()
    q = fused_rmsnorm_rope(q, None, freqs, ROPE_DIM)
    kv = F.linear(x, wkv).view(M, 1, HEAD_DIM).contiguous()
    kv = fused_rmsnorm_rope(kv, kv_norm, freqs, ROPE_DIM)
    return q, kv


def _run_qk_tc_tc_overlap(x, qr, wq_b, wkv, kv_norm, freqs, resources=None):
    M = x.shape[0]
    if resources is None:
        resources = {
            "s_q": torch.cuda.Stream(device=x.device),
            "s_k": torch.cuda.Stream(device=x.device),
            "ev_q": torch.cuda.Event(blocking=False),
            "ev_k": torch.cuda.Event(blocking=False),
        }
    s_q = resources["s_q"]
    s_k = resources["s_k"]
    cur = torch.cuda.current_stream(x.device)
    s_q.wait_stream(cur)
    s_k.wait_stream(cur)
    ev_q = resources["ev_q"]
    ev_k = resources["ev_k"]
    with torch.cuda.stream(s_q):
        q_raw = F.linear(qr, wq_b)
        ev_q.record(s_q)
    with torch.cuda.stream(s_k):
        kv_raw = F.linear(x, wkv)
        ev_k.record(s_k)
    cur.wait_event(ev_q)
    cur.wait_event(ev_k)
    q = fused_rmsnorm_rope(q_raw.view(M, 1, N_HEADS, HEAD_DIM).contiguous(), None, freqs, ROPE_DIM)
    kv = fused_rmsnorm_rope(kv_raw.view(M, 1, HEAD_DIM).contiguous(), kv_norm, freqs, ROPE_DIM)
    return q, kv


def _run_qk_tc_simt_overlap(x, qr, wq_b, wkv, kv_norm, freqs, resources=None):
    M = x.shape[0]
    if resources is None:
        resources = {
            "s_q": torch.cuda.Stream(device=x.device),
            "s_simt": torch.cuda.Stream(device=x.device),
            "s_k": torch.cuda.Stream(device=x.device),
            "ev_q_raw": torch.cuda.Event(blocking=False),
            "ev_q_done": torch.cuda.Event(blocking=False),
            "ev_k_raw": torch.cuda.Event(blocking=False),
        }
    s_q = resources["s_q"]
    s_simt = resources["s_simt"]
    s_k = resources["s_k"]
    cur = torch.cuda.current_stream(x.device)
    for s in (s_q, s_simt, s_k):
        s.wait_stream(cur)
    ev_q_raw = resources["ev_q_raw"]
    ev_q_done = resources["ev_q_done"]
    ev_k_raw = resources["ev_k_raw"]
    with torch.cuda.stream(s_q):
        q_raw = F.linear(qr, wq_b)
        ev_q_raw.record(s_q)
    with torch.cuda.stream(s_simt):
        s_simt.wait_event(ev_q_raw)
        q = fused_rmsnorm_rope(q_raw.view(M, 1, N_HEADS, HEAD_DIM).contiguous(), None, freqs, ROPE_DIM)
        ev_q_done.record(s_simt)
    with torch.cuda.stream(s_k):
        kv_raw = F.linear(x, wkv)
        ev_k_raw.record(s_k)
    cur.wait_event(ev_q_done)
    cur.wait_event(ev_k_raw)
    kv = fused_rmsnorm_rope(kv_raw.view(M, 1, HEAD_DIM).contiguous(), kv_norm, freqs, ROPE_DIM)
    return q, kv


def _bench_qk(M: int, candidate_name: str) -> BenchResult:
    x, qr, wq_b, wkv, kv_norm, freqs = _qk_tensors(M)
    baseline = lambda: _run_qk_serial(x, qr, wq_b, wkv, kv_norm, freqs)
    persistent = candidate_name.endswith("_persistent")
    base_name = candidate_name.replace("_persistent", "")
    resources = None
    if persistent and base_name == "tc_tc":
        resources = {
            "s_q": torch.cuda.Stream(device=x.device),
            "s_k": torch.cuda.Stream(device=x.device),
            "ev_q": torch.cuda.Event(blocking=False),
            "ev_k": torch.cuda.Event(blocking=False),
        }
    elif persistent and base_name == "tc_simt":
        resources = {
            "s_q": torch.cuda.Stream(device=x.device),
            "s_simt": torch.cuda.Stream(device=x.device),
            "s_k": torch.cuda.Stream(device=x.device),
            "ev_q_raw": torch.cuda.Event(blocking=False),
            "ev_q_done": torch.cuda.Event(blocking=False),
            "ev_k_raw": torch.cuda.Event(blocking=False),
        }
    if base_name == "tc_tc":
        candidate = lambda: _run_qk_tc_tc_overlap(x, qr, wq_b, wkv, kv_norm, freqs, resources)
    else:
        candidate = lambda: _run_qk_tc_simt_overlap(x, qr, wq_b, wkv, kv_norm, freqs, resources)
    q_ref, kv_ref = baseline()
    q_c, kv_c = candidate()
    torch.cuda.synchronize()
    max_q, mean_q = _diff(q_ref, q_c)
    max_k, mean_k = _diff(kv_ref, kv_c)
    max_abs = max(max_q, max_k)
    if max_abs > 5e-2:
        raise AssertionError(f"qk mismatch M={M} {candidate_name}: {max_abs}")
    baseline_us, candidate_us, measure_extra = _measure_pair(
        baseline,
        candidate,
        section="qk_preprocess",
        case=candidate_name,
        M=M,
        tile_name="n/a",
    )
    return BenchResult(
        "qk_preprocess",
        candidate_name,
        M,
        "n/a",
        baseline_us,
        candidate_us,
        baseline_us / candidate_us if candidate_us > 0 else 0.0,
        max_abs=max_abs,
        mean_abs=max(mean_q, mean_k),
        **measure_extra,
    )


def _run_two_linears_serial(x, w1, w2):
    return F.linear(x, w1), F.linear(x, w2)


def _run_two_linears_overlap(x, w1, w2, resources=None):
    if resources is None:
        resources = {
            "s1": torch.cuda.Stream(device=x.device),
            "s2": torch.cuda.Stream(device=x.device),
            "ev1": torch.cuda.Event(blocking=False),
            "ev2": torch.cuda.Event(blocking=False),
        }
    s1 = resources["s1"]
    s2 = resources["s2"]
    cur = torch.cuda.current_stream(x.device)
    s1.wait_stream(cur)
    s2.wait_stream(cur)
    ev1 = resources["ev1"]
    ev2 = resources["ev2"]
    with torch.cuda.stream(s1):
        y1 = F.linear(x, w1)
        ev1.record(s1)
    with torch.cuda.stream(s2):
        y2 = F.linear(x, w2)
        ev2.record(s2)
    cur.wait_event(ev1)
    cur.wait_event(ev2)
    return y1, y2


def _bench_compressor(M: int, persistent: bool = False) -> BenchResult:
    torch.manual_seed(3000 + M)
    out_dim = 2 * HEAD_DIM
    x = torch.randn(M, MODEL_DIM, dtype=torch.bfloat16, device="cuda") * 0.1
    wkv = torch.randn(out_dim, MODEL_DIM, dtype=torch.bfloat16, device="cuda") * 0.02
    wgate = torch.randn(out_dim, MODEL_DIM, dtype=torch.bfloat16, device="cuda") * 0.02
    baseline = lambda: _run_two_linears_serial(x, wkv, wgate)
    resources = (
        {
            "s1": torch.cuda.Stream(device=x.device),
            "s2": torch.cuda.Stream(device=x.device),
            "ev1": torch.cuda.Event(blocking=False),
            "ev2": torch.cuda.Event(blocking=False),
        }
        if persistent
        else None
    )
    candidate = lambda: _run_two_linears_overlap(x, wkv, wgate, resources)
    a0, b0 = baseline()
    a1, b1 = candidate()
    torch.cuda.synchronize()
    max_abs = max(_diff(a0, a1)[0], _diff(b0, b1)[0])
    if max_abs != 0.0:
        raise AssertionError(f"compressor linear mismatch M={M}: {max_abs}")
    case = "wkv_wgate_tc_tc_persistent" if persistent else "wkv_wgate_tc_tc"
    baseline_us, candidate_us, measure_extra = _measure_pair(
        baseline,
        candidate,
        section="compressor",
        case=case,
        M=M,
        tile_name="n/a",
    )
    return BenchResult(
        "compressor",
        case,
        M,
        "n/a",
        baseline_us,
        candidate_us,
        baseline_us / candidate_us if candidate_us > 0 else 0.0,
        max_abs=max_abs,
        mean_abs=0.0,
        **measure_extra,
    )


def _run_indexer_serial(x, qr, wq_b, weights_proj, freqs):
    M = x.shape[0]
    q = F.linear(qr, wq_b).view(M, 1, IDX_HEADS, IDX_HEAD_DIM).contiguous()
    apply_rotary_emb_batched(q[..., -ROPE_DIM:], freqs)
    weights = F.linear(x, weights_proj)
    return q, weights


def _run_indexer_overlap(x, qr, wq_b, weights_proj, freqs, resources=None):
    M = x.shape[0]
    if resources is None:
        resources = {
            "s_q": torch.cuda.Stream(device=x.device),
            "s_w": torch.cuda.Stream(device=x.device),
            "ev_q": torch.cuda.Event(blocking=False),
            "ev_w": torch.cuda.Event(blocking=False),
        }
    s_q = resources["s_q"]
    s_w = resources["s_w"]
    cur = torch.cuda.current_stream(x.device)
    s_q.wait_stream(cur)
    s_w.wait_stream(cur)
    ev_q = resources["ev_q"]
    ev_w = resources["ev_w"]
    with torch.cuda.stream(s_q):
        q = F.linear(qr, wq_b).view(M, 1, IDX_HEADS, IDX_HEAD_DIM).contiguous()
        apply_rotary_emb_batched(q[..., -ROPE_DIM:], freqs)
        ev_q.record(s_q)
    with torch.cuda.stream(s_w):
        weights = F.linear(x, weights_proj)
        ev_w.record(s_w)
    cur.wait_event(ev_q)
    cur.wait_event(ev_w)
    return q, weights


def _bench_indexer(M: int, persistent: bool = False) -> BenchResult:
    torch.manual_seed(4000 + M)
    x = torch.randn(M, MODEL_DIM, dtype=torch.bfloat16, device="cuda") * 0.1
    qr = torch.randn(M, Q_LORA_RANK, dtype=torch.bfloat16, device="cuda") * 0.1
    wq_b = torch.randn(IDX_OUT, Q_LORA_RANK, dtype=torch.bfloat16, device="cuda") * 0.02
    weights_proj = torch.randn(IDX_HEADS, MODEL_DIM, dtype=torch.bfloat16, device="cuda") * 0.02
    freqs = _make_freqs(M)
    baseline = lambda: _run_indexer_serial(x, qr, wq_b, weights_proj, freqs)
    resources = (
        {
            "s_q": torch.cuda.Stream(device=x.device),
            "s_w": torch.cuda.Stream(device=x.device),
            "ev_q": torch.cuda.Event(blocking=False),
            "ev_w": torch.cuda.Event(blocking=False),
        }
        if persistent
        else None
    )
    candidate = lambda: _run_indexer_overlap(x, qr, wq_b, weights_proj, freqs, resources)
    q0, w0 = baseline()
    q1, w1 = candidate()
    torch.cuda.synchronize()
    max_abs = max(_diff(q0, q1)[0], _diff(w0, w1)[0])
    if max_abs > 1e-2:
        raise AssertionError(f"indexer mismatch M={M}: {max_abs}")
    case = "wq_b_rope_weights_proj_persistent" if persistent else "wq_b_rope_weights_proj"
    baseline_us, candidate_us, measure_extra = _measure_pair(
        baseline,
        candidate,
        section="indexer",
        case=case,
        M=M,
        tile_name="n/a",
    )
    return BenchResult(
        "indexer",
        case,
        M,
        "n/a",
        baseline_us,
        candidate_us,
        baseline_us / candidate_us if candidate_us > 0 else 0.0,
        max_abs=max_abs,
        mean_abs=0.0,
        **measure_extra,
    )


def _print_table(results: List[BenchResult], section: str) -> None:
    rows = [r for r in results if r.section == section]
    if not rows:
        print(f"\n[{section}] no results")
        return
    print(f"\n[{section}]")
    print("  {:>8} {:>10} {:>20} {:>12} {:>12} {:>9}".format("M", "tile", "case", "base_us", "cand_us", "speedup"))
    for r in rows:
        print(
            "  {:8d} {:>10} {:>20} {:12.2f} {:12.2f} {:8.3f}x".format(
                r.M, r.tile_m, r.case, r.baseline_us, r.candidate_us, r.speedup
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DSV4TcSimtOverlapPerfTest(unittest.TestCase):
    def test_overlap_report(self):
        m_list = _parse_int_list("DSV4_OVERLAP_M_LIST", _default_m_list())
        tile_list = _parse_tile_list("DSV4_OVERLAP_TILE_LIST", _default_tile_list())
        results: List[BenchResult] = []
        started = time.time()

        skip_out_projection = os.environ.get("DSV4_OVERLAP_SKIP_OUT_PROJECTION", "0") == "1"
        if skip_out_projection:
            print("[out_projection] skipped: DSV4_OVERLAP_SKIP_OUT_PROJECTION=1")
        elif deep_gemm is None:
            print("[out_projection] skipped: deep_gemm is not available")
        else:
            for M in m_list:
                for tile in tile_list:
                    if tile != "no_split" and int(tile) < 1:
                        continue
                    try:
                        result = _bench_out_projection(M, tile)
                    except torch.cuda.OutOfMemoryError as exc:
                        torch.cuda.empty_cache()
                        result = BenchResult(
                            "out_projection",
                            "inv_rope_quant_woa_wob",
                            M,
                            str(tile),
                            0.0,
                            0.0,
                            0.0,
                            valid=False,
                            invalid_reason=f"OOM: {exc}",
                        )
                    except RuntimeError as exc:
                        if not _is_deepgemm_env_error(exc):
                            raise
                        result = BenchResult(
                            "out_projection",
                            "inv_rope_quant_woa_wob",
                            M,
                            str(tile),
                            0.0,
                            0.0,
                            0.0,
                            valid=False,
                            invalid_reason=f"DeepGEMM/JIT unavailable: {exc}",
                        )
                    if result is not None:
                        results.append(result)
                    torch.cuda.empty_cache()

        for M in m_list:
            for case in ("tc_tc", "tc_tc_persistent", "tc_simt", "tc_simt_persistent"):
                try:
                    results.append(_bench_qk(M, case))
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    results.append(
                        BenchResult(
                            "qk_preprocess",
                            case,
                            M,
                            "n/a",
                            0.0,
                            0.0,
                            0.0,
                            valid=False,
                            invalid_reason=f"OOM: {exc}",
                        )
                    )
            for persistent in (False, True):
                try:
                    results.append(_bench_compressor(M, persistent))
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    results.append(
                        BenchResult(
                            "compressor",
                            "wkv_wgate_tc_tc_persistent" if persistent else "wkv_wgate_tc_tc",
                            M,
                            "n/a",
                            0.0,
                            0.0,
                            0.0,
                            valid=False,
                            invalid_reason=f"OOM: {exc}",
                        )
                    )
            for persistent in (False, True):
                try:
                    results.append(_bench_indexer(M, persistent))
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    results.append(
                        BenchResult(
                            "indexer",
                            "wq_b_rope_weights_proj_persistent" if persistent else "wq_b_rope_weights_proj",
                            M,
                            "n/a",
                            0.0,
                            0.0,
                            0.0,
                            valid=False,
                            invalid_reason=f"OOM: {exc}",
                        )
                    )

        for section in ("out_projection", "qk_preprocess", "compressor", "indexer"):
            _print_table([r for r in results if r.valid], section)

        payload: Dict[str, object] = {
            "elapsed_sec": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(),
            "m_list": m_list,
            "tile_list": [str(x) for x in tile_list],
            "dims": {
                "n_heads": N_HEADS,
                "head_dim": HEAD_DIM,
                "rope_dim": ROPE_DIM,
                "n_groups": N_GROUPS,
                "wo_a_rank": WO_A_RANK,
                "wo_b_out": WO_B_OUT,
                "q_lora_rank": Q_LORA_RANK,
                "model_dim": MODEL_DIM,
                "idx_heads": IDX_HEADS,
                "idx_head_dim": IDX_HEAD_DIM,
            },
            "results": [asdict(r) for r in results],
        }
        report_path = os.environ.get("DSV4_OVERLAP_JSON")
        if report_path:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            print(f"\nWrote overlap report JSON: {report_path}")

        self.assertTrue(results or deep_gemm is None)


if __name__ == "__main__":
    unittest.main()
