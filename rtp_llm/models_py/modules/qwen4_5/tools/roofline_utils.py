#!/usr/bin/env python3
"""Empirical L20A SM100 hardware roofline calibration and analytic Qwen4.5-Flash FLOP/Byte model.

Shared across Prefill (`bench_prefill_roofline.py`) and Decode (`bench_decode_roofline.py`)
tracks so MFU% and MBU% use identical, strictly non-inflated hardware & algorithmic accounting.

Conventions:
  - FLOPs are counted using 2 * M * K * N per matrix multiply (exact FMA = 2 FLOPs), in base-10
    TFLOP/s (1e12 FLOP/s) to match NVIDIA / CUDA hardware conventions.
  - Memory bandwidth is counted in base-10 GB/s (1e9 bytes/s).
  - Model Algorithmic FLOPs (`algo_flops`) and Bytes (`algo_bytes`) measure the minimum
    mathematical work required by the network (counting weights & replicated modules once,
    never multiplying by `tp_size` for redundant execution).
  - Cluster Executed FLOPs/Bytes (`exec_flops`, `exec_bytes`) additionally track actual GPU
    work performed under a given parallelism (`tp4_ep4` replicates MoE latent/shared &
    FullAttn K/V/Indexer across 4 GPUs, whereas `dp4_ep4` executes them once).
  - Two whole-model MFU metrics are reported:
      * `mfu_bf16_pct`: `algo_flops / (elapsed_s * cluster_bf16_tflops * 1e12) * 100`.
      * `mfu_aware_pct`: `sum_i (op_i.algo_flops / cluster_peak_for(op_i.precision)) / elapsed_s * 100`
        (precision-aware Speed-of-Light compute utilization across BF16/FP8/MXFP4/FP4 ops).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Hardware Roofline Calibration & Persistence
# ---------------------------------------------------------------------------


@dataclass
class HardwareRoofline:
    """Calibrated single-GPU and cluster ceilings for L20A (SM100)."""

    gpu_name: str
    sm_count: int
    compute_capability: str
    world_size: int
    bf16_tflops: float
    fp8_tflops: float
    mxfp4_tflops: float
    fp4_tflops: float
    hbm_gbps: float
    nvlink5_allreduce_gbps: float = 0.0
    nvlink5_allreduce_algbw_gbps: float = 0.0
    nvlink5_allreduce_small_us: float = 0.0
    nvlink5_alltoall_gbps: float = 0.0
    nvlink5_alltoall_algbw_gbps: float = 0.0
    nvlink5_alltoall_small_us: float = 0.0
    calibrated_at_epoch: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def cluster_bf16_tflops(self) -> float:
        return self.bf16_tflops * self.world_size

    @property
    def cluster_fp8_tflops(self) -> float:
        return self.fp8_tflops * self.world_size

    @property
    def cluster_mxfp4_tflops(self) -> float:
        return self.mxfp4_tflops * self.world_size

    @property
    def cluster_fp4_tflops(self) -> float:
        return self.fp4_tflops * self.world_size

    @property
    def cluster_hbm_gbps(self) -> float:
        return self.hbm_gbps * self.world_size

    def peak_tflops_for_precision(self, precision: str, *, per_gpu: bool = False) -> float:
        """Return empirical TFLOPS ceiling for `bf16`, `fp8`, `mxfp4`, or `fp4`."""
        p = precision.lower()
        if p in ("fp4", "nvfp4", "w4a4"):
            val = self.fp4_tflops if self.fp4_tflops > 0 else (
                self.mxfp4_tflops if self.mxfp4_tflops > 0 else (
                    self.fp8_tflops if self.fp8_tflops > 0 else self.bf16_tflops
                )
            )
        elif p in ("mxfp4", "w4a8", "fp8_fp4"):
            val = self.mxfp4_tflops if self.mxfp4_tflops > 0 else (
                self.fp8_tflops if self.fp8_tflops > 0 else self.bf16_tflops
            )
        elif p in ("fp8", "w8a8", "bf16_fp8"):
            val = self.fp8_tflops if self.fp8_tflops > 0 else self.bf16_tflops
        else:
            val = self.bf16_tflops
        return val if per_gpu else val * self.world_size

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cluster_bf16_tflops"] = round(self.cluster_bf16_tflops, 3)
        d["cluster_fp8_tflops"] = round(self.cluster_fp8_tflops, 3)
        d["cluster_mxfp4_tflops"] = round(self.cluster_mxfp4_tflops, 3)
        d["cluster_fp4_tflops"] = round(self.cluster_fp4_tflops, 3)
        d["cluster_hbm_gbps"] = round(self.cluster_hbm_gbps, 3)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HardwareRoofline":
        fields = {
            f: data[f]
            for f in (
                "gpu_name",
                "sm_count",
                "compute_capability",
                "world_size",
                "bf16_tflops",
                "fp8_tflops",
                "mxfp4_tflops",
                "fp4_tflops",
                "hbm_gbps",
                "nvlink5_allreduce_gbps",
                "nvlink5_allreduce_algbw_gbps",
                "nvlink5_allreduce_small_us",
                "nvlink5_alltoall_gbps",
                "nvlink5_alltoall_algbw_gbps",
                "nvlink5_alltoall_small_us",
                "calibrated_at_epoch",
                "details",
            )
            if f in data
        }
        return cls(**fields)

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return p

    @classmethod
    def load_json(cls, path: str | Path) -> "HardwareRoofline":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _time_cuda_op_ms(fn: Any, *, warmup: int = 10, iters: int = 30) -> float:
    """Time a CUDA callable using median CUDA event duration in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(float(s.elapsed_time(e)) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def _calibrate_bf16_gemm(device: torch.device, *, warmup: int, iters: int) -> Tuple[float, Dict[str, Any]]:
    """Measure peak single-GPU BF16 TensorCore TFLOPS via torch.mm and deep_gemm.bf16_gemm_nt."""
    shapes = (
        (4096, 4096, 4096),
        (8192, 8192, 4096),
        (8192, 4096, 8192),
        (16384, 5120, 2560),
    )
    records: List[Dict[str, Any]] = []
    peak = 0.0
    try:
        import deep_gemm  # type: ignore
    except Exception:
        deep_gemm = None

    for m, n, k in shapes:
        a = torch.randn((m, k), dtype=torch.bfloat16, device=device)
        b_col = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        b_nt = b_col.t()
        out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
        flops = 2.0 * m * n * k

        ms_cublas = _time_cuda_op_ms(lambda: torch.mm(a, b_nt, out=out), warmup=warmup, iters=iters)
        tf_cublas = flops / (ms_cublas * 1e-3) / 1e12
        peak = max(peak, tf_cublas)
        rec: Dict[str, Any] = {
            "shape": [m, n, k],
            "torch_mm_ms": round(ms_cublas, 4),
            "torch_mm_tflops": round(tf_cublas, 2),
        }

        if deep_gemm is not None and hasattr(deep_gemm, "bf16_gemm_nt"):
            try:
                ms_dg = _time_cuda_op_ms(
                    lambda: deep_gemm.bf16_gemm_nt(a, b_col, out),
                    warmup=warmup,
                    iters=iters,
                )
                tf_dg = flops / (ms_dg * 1e-3) / 1e12
                peak = max(peak, tf_dg)
                rec["deep_gemm_ms"] = round(ms_dg, 4)
                rec["deep_gemm_tflops"] = round(tf_dg, 2)
            except Exception as exc:
                rec["deep_gemm_err"] = str(exc)
        records.append(rec)
    return round(peak, 2), {"shapes": records}


def _calibrate_fp8_gemm(device: torch.device, *, warmup: int, iters: int) -> Tuple[float, Dict[str, Any]]:
    """Measure peak single-GPU FP8 E4M3 TensorCore TFLOPS."""
    shapes = (
        (4096, 4096, 4096),
        (8192, 8192, 4096),
        (8192, 4096, 8192),
        (16384, 5120, 2560),
    )
    records: List[Dict[str, Any]] = []
    peak = 0.0
    try:
        import deep_gemm  # type: ignore
    except Exception:
        deep_gemm = None

    for m, n, k in shapes:
        a_bf = torch.randn((m, k), dtype=torch.bfloat16, device=device)
        b_bf = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        a_fp8 = a_bf.to(torch.float8_e4m3fn)
        b_fp8 = b_bf.to(torch.float8_e4m3fn)
        b_col = b_fp8.t()
        scale_a = torch.ones((1,), dtype=torch.float32, device=device)
        scale_b = torch.ones((1,), dtype=torch.float32, device=device)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
        flops = 2.0 * m * n * k
        rec: Dict[str, Any] = {"shape": [m, n, k]}

        try:
            ms_sm = _time_cuda_op_ms(
                lambda: torch._scaled_mm(
                    a_fp8,
                    b_col,
                    scale_a=scale_a,
                    scale_b=scale_b,
                    out_dtype=torch.bfloat16,
                    use_fast_accum=True,
                ),
                warmup=warmup,
                iters=iters,
            )
            tf_sm = flops / (ms_sm * 1e-3) / 1e12
            peak = max(peak, tf_sm)
            rec["scaled_mm_ms"] = round(ms_sm, 4)
            rec["scaled_mm_tflops"] = round(tf_sm, 2)
        except Exception as exc:
            rec["scaled_mm_err"] = str(exc)

        if deep_gemm is not None and hasattr(deep_gemm, "fp8_gemm_nt"):
            try:
                sa_dg = torch.ones((k // 128, m), dtype=torch.float32, device=device).t()
                sb_dg = torch.ones((n // 128, k // 128), dtype=torch.float32, device=device)
                ms_dg = _time_cuda_op_ms(
                    lambda: deep_gemm.fp8_gemm_nt((a_fp8, sa_dg), (b_fp8, sb_dg), out),
                    warmup=warmup,
                    iters=iters,
                )
                tf_dg = flops / (ms_dg * 1e-3) / 1e12
                peak = max(peak, tf_dg)
                rec["deep_gemm_ms"] = round(ms_dg, 4)
                rec["deep_gemm_tflops"] = round(tf_dg, 2)
            except Exception as exc:
                rec["deep_gemm_err"] = str(exc)
        records.append(rec)
    return round(peak, 2), {"shapes": records}


def _calibrate_mxfp4_gemm(device: torch.device, *, warmup: int, iters: int) -> Tuple[float, Dict[str, Any]]:
    """Measure peak single-GPU W4A8 (`FP8 x MXFP4`) TFLOPS via DeepGEMM SM100 kernels."""
    details: Dict[str, Any] = {"shapes": []}
    peak = 0.0
    try:
        import deep_gemm  # type: ignore
        from rtp_llm.models_py.modules.qwen4_5.mxfp4_ops import (
            prepare_fp4_weight_scale_for_deepgemm,
        )
    except Exception as exc:
        details["unavailable"] = str(exc)
        return 0.0, details

    # 1. Dense FP8 x MXFP4 (no group imbalance)
    if hasattr(deep_gemm, "fp8_fp4_gemm_nt"):
        for m, n, k in ((4096, 5120, 4096), (8192, 5120, 4096), (8192, 8192, 4096)):
            try:
                a_fp8 = torch.randn((m, k), dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
                a_sc = torch.full((k // 128, m), 0x7F7F7F7F, dtype=torch.int32, device=device).t()
                w_u8 = torch.randint(0, 256, (1, n, k // 2), dtype=torch.uint8, device=device)
                w_sc = torch.full((1, n, k // 32), 127, dtype=torch.uint8, device=device)
                w_sc_dg = prepare_fp4_weight_scale_for_deepgemm(w_sc, n, k, num_groups=1)[0]
                out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
                ms = _time_cuda_op_ms(
                    lambda: deep_gemm.fp8_fp4_gemm_nt(
                        (a_fp8, a_sc),
                        (w_u8[0].view(torch.int8), w_sc_dg),
                        out,
                        recipe_a=(1, 32),
                        recipe_b=(1, 32),
                    ),
                    warmup=warmup,
                    iters=iters,
                )
                tf = (2.0 * m * n * k) / (ms * 1e-3) / 1e12
                peak = max(peak, tf)
                details["shapes"].append(
                    {"kernel": "deep_gemm.fp8_fp4_gemm_nt", "shape": [m, n, k], "ms": round(ms, 4), "tflops": round(tf, 2)}
                )
            except Exception as exc:
                details["shapes"].append({"kernel": "deep_gemm.fp8_fp4_gemm_nt", "shape": [m, n, k], "err": str(exc)})

    # 2. Grouped MoE FP8 x MXFP4 (E=16, active_m=512 each -> saturated SM100 TensorCores)
    if hasattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_masked"):
        e_g, m_g, n_g, k_g = 16, 512, 5120, 2560
        try:
            a_fp8 = torch.randn((e_g, m_g, k_g), dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
            a_sc = torch.full((e_g, k_g // 128, m_g), 0x7F7F7F7F, dtype=torch.int32, device=device).transpose(1, 2)
            w_u8 = torch.randint(0, 256, (e_g, n_g, k_g // 2), dtype=torch.uint8, device=device)
            w_sc = torch.full((e_g, n_g, k_g // 32), 127, dtype=torch.uint8, device=device)
            w_sc_dg = prepare_fp4_weight_scale_for_deepgemm(w_sc, n_g, k_g, num_groups=e_g)
            masked_m = torch.full((e_g,), m_g, dtype=torch.int32, device=device)
            out = torch.empty((e_g, m_g, n_g), dtype=torch.bfloat16, device=device)
            ms = _time_cuda_op_ms(
                lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
                    (a_fp8, a_sc),
                    (w_u8.view(torch.int8), w_sc_dg),
                    out,
                    masked_m,
                    m_g,
                    recipe_a=(1, 32),
                    recipe_b=(1, 32),
                ),
                warmup=warmup,
                iters=iters,
            )
            tf = (2.0 * e_g * m_g * n_g * k_g) / (ms * 1e-3) / 1e12
            peak = max(peak, tf)
            details["shapes"].append(
                {
                    "kernel": "deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked",
                    "shape": [e_g, m_g, n_g, k_g],
                    "ms": round(ms, 4),
                    "tflops": round(tf, 2),
                }
            )
        except Exception as exc:
            details["shapes"].append({"kernel": "m_grouped_fp8_fp4_gemm_nt_masked", "err": str(exc)})

    return round(peak, 2), details


def _ensure_cuda_include_cpath() -> None:
    """Include CUDA headers from site-packages so FlashInfer JIT compiles find cublasLt.h."""
    import glob

    inc_candidates = [
        "/opt/conda310/lib/python3.10/site-packages/nvidia/cu13/include",
        "/opt/conda310/lib/python3.10/site-packages/nvidia/cublas/include",
        "/usr/local/cuda/include",
    ]
    inc_candidates.extend(sorted(glob.glob("/opt/conda*/lib/python*/site-packages/nvidia/*/include")))
    existing: list[str] = []
    for p in inc_candidates:
        if os.path.isdir(p) and p not in existing:
            existing.append(p)
    if not existing:
        return
    cur = os.environ.get("CPATH", "")
    parts = [p for p in cur.split(":") if p]
    for p in existing:
        if p not in parts:
            parts.append(p)
    joined = ":".join(parts)
    os.environ["CPATH"] = joined
    os.environ["CPLUS_INCLUDE_PATH"] = joined


def _calibrate_fp4_gemm(device: torch.device, *, warmup: int, iters: int) -> Tuple[float, Dict[str, Any]]:
    """Measure peak single-GPU W4A4 NVFP4 TFLOPS via FlashInfer `mm_fp4` if supported."""
    details: Dict[str, Any] = {"runs": []}
    peak = 0.0
    _ensure_cuda_include_cpath()
    try:
        import flashinfer  # type: ignore
    except Exception as exc:
        details["unavailable"] = str(exc)
        return 0.0, details

    if not hasattr(flashinfer, "nvfp4_quantize") or not hasattr(flashinfer, "mm_fp4"):
        details["unavailable"] = "flashinfer.mm_fp4 missing"
        return 0.0, details

    sf_layout = getattr(flashinfer, "SfLayout", None)
    layout_128x4 = getattr(sf_layout, "layout_128x4", None) if sf_layout is not None else None
    global_sf = torch.ones((1,), dtype=torch.float32, device=device)
    alpha = torch.ones((1,), dtype=torch.float32, device=device)

    for m, n, k in ((4096, 4096, 4096), (8192, 8192, 4096)):
        a_bf = torch.randn((m, k), dtype=torch.bfloat16, device=device)
        b_bf = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        q_kw = {"sfLayout": layout_128x4} if layout_128x4 is not None else {}
        try:
            a_fp4, a_sf = flashinfer.nvfp4_quantize(a_bf, global_sf, **q_kw)
            b_fp4, b_sf = flashinfer.nvfp4_quantize(b_bf, global_sf, **q_kw)
        except Exception as exc:
            details["runs"].append({"shape": [m, n, k], "quant_err": str(exc)})
            continue

        b_t = b_fp4.t()
        for backend in ("cutlass", "trtllm", "cudnn", "auto"):
            try:
                out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
                ms = _time_cuda_op_ms(
                    lambda: flashinfer.mm_fp4(
                        a_fp4,
                        b_t,
                        a_sf,
                        b_sf,
                        alpha=alpha,
                        out_dtype=torch.bfloat16,
                        out=out,
                        backend=backend,
                        use_nvfp4=True,
                    ),
                    warmup=min(warmup, 5),
                    iters=min(iters, 20),
                )
                tf = (2.0 * m * n * k) / (ms * 1e-3) / 1e12
                peak = max(peak, tf)
                details["runs"].append(
                    {"shape": [m, n, k], "backend": backend, "ms": round(ms, 4), "tflops": round(tf, 2)}
                )
            except Exception as exc:
                details["runs"].append({"shape": [m, n, k], "backend": backend, "err": str(exc)})

        # Also probe deep_gemm.cublaslt_nvfp4_gemm_nt on the quantized NVFP4 tensors if available
        try:
            import deep_gemm  # type: ignore

            if hasattr(deep_gemm, "cublaslt_nvfp4_gemm_nt"):
                out_dg = torch.empty((m, n), dtype=torch.bfloat16, device=device)
                ms_dg = _time_cuda_op_ms(
                    lambda: deep_gemm.cublaslt_nvfp4_gemm_nt((a_fp4, a_sf), (b_fp4, b_sf), out_dg),
                    warmup=min(warmup, 5),
                    iters=min(iters, 20),
                )
                tf_dg = (2.0 * m * n * k) / (ms_dg * 1e-3) / 1e12
                peak = max(peak, tf_dg)
                details["runs"].append(
                    {"shape": [m, n, k], "backend": "deep_gemm.cublaslt_nvfp4", "ms": round(ms_dg, 4), "tflops": round(tf_dg, 2)}
                )
        except Exception as exc:
            details["runs"].append({"shape": [m, n, k], "backend": "deep_gemm.cublaslt_nvfp4", "err": str(exc)})
    return round(peak, 2), details


def _calibrate_hbm_bw(device: torch.device, *, warmup: int = 10, iters: int = 30) -> Tuple[float, Dict[str, Any]]:
    """Measure sustained single-GPU HBM3e bandwidth (GB/s = 1e9 B/s) on 512 MiB buffers."""
    n_bytes = 512 * 1024 * 1024
    n_elems = n_bytes // 4  # float32
    a = torch.ones((n_elems,), dtype=torch.float32, device=device)
    b = torch.ones((n_elems,), dtype=torch.float32, device=device)
    c = torch.empty((n_elems,), dtype=torch.float32, device=device)

    ms_copy = _time_cuda_op_ms(lambda: c.copy_(a), warmup=warmup, iters=iters)
    gbps_copy = (2.0 * n_bytes) / (ms_copy * 1e-3) / 1e9

    ms_triad = _time_cuda_op_ms(lambda: torch.add(a, b, out=c), warmup=warmup, iters=iters)
    gbps_triad = (3.0 * n_bytes) / (ms_triad * 1e-3) / 1e9

    peak = max(gbps_copy, gbps_triad)
    return round(peak, 2), {
        "buffer_bytes": n_bytes,
        "copy_2x_ms": round(ms_copy, 4),
        "copy_2x_gbps": round(gbps_copy, 2),
        "triad_3x_ms": round(ms_triad, 4),
        "triad_3x_gbps": round(gbps_triad, 2),
    }


def _calibrate_nvlink5(
    device: torch.device,
    *,
    warmup: int = 10,
    iters: int = 30,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Measure 4-GPU NVLink5 AllReduce & AllToAll busBW/algBW across message sizes."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return {
            "nvlink5_allreduce_gbps": 0.0,
            "nvlink5_allreduce_algbw_gbps": 0.0,
            "nvlink5_allreduce_small_us": 0.0,
            "nvlink5_alltoall_gbps": 0.0,
            "nvlink5_alltoall_algbw_gbps": 0.0,
            "nvlink5_alltoall_small_us": 0.0,
        }, {"distributed": False}

    w = dist.get_world_size()
    details: Dict[str, Any] = {"world_size": w, "allreduce": [], "alltoall": []}

    # AllReduce sweep (BF16 buffers)
    ar_sizes = (16 * 1024, 512 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024, 128 * 1024 * 1024)
    ar_peak_busbw = 0.0
    ar_peak_algbw = 0.0
    ar_small_us = 0.0
    for nbytes in ar_sizes:
        buf = torch.ones((nbytes // 2,), dtype=torch.bfloat16, device=device)
        dist.barrier()
        ms = _time_cuda_op_ms(lambda: dist.all_reduce(buf), warmup=warmup, iters=iters)
        # Take max rank time across cluster
        t_max = torch.tensor([ms], dtype=torch.float64, device=device)
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
        max_ms = float(t_max.item())
        algbw = nbytes / (max_ms * 1e-3) / 1e9
        busbw = algbw * (2.0 * (w - 1) / w)
        ar_peak_algbw = max(ar_peak_algbw, algbw)
        ar_peak_busbw = max(ar_peak_busbw, busbw)
        if nbytes == 16 * 1024:
            ar_small_us = max_ms * 1000.0
        details["allreduce"].append(
            {
                "backend": "nccl",
                "bytes": nbytes,
                "max_ms": round(max_ms, 4),
                "algbw_gbps": round(algbw, 2),
                "busbw_gbps": round(busbw, 2),
            }
        )

    # Probe FlashInfer/symm-mem FastAllReduce on Decode hidden sizes (16KB, 512KB)
    try:
        from rtp_llm.models_py.modules.qwen4_5.fast_comm import get_qwen4_5_fast_all_reduce

        far = get_qwen4_5_fast_all_reduce(dist.group.WORLD, device=device)
        for tokens in (1, 8, 64, 512):
            hidden = 5120
            x = torch.ones((tokens, hidden), dtype=torch.bfloat16, device=device)
            nbytes = x.numel() * 2
            dist.barrier()
            ms = _time_cuda_op_ms(lambda: far(x), warmup=warmup, iters=iters)
            t_max = torch.tensor([ms], dtype=torch.float64, device=device)
            dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
            max_ms = float(t_max.item())
            algbw = nbytes / (max_ms * 1e-3) / 1e9
            busbw = algbw * (2.0 * (w - 1) / w)
            if tokens == 1:
                ar_small_us = min(ar_small_us, max_ms * 1000.0) if ar_small_us > 0 else max_ms * 1000.0
            far_backend = getattr(far, "active_backend", None) or getattr(far, "backend", None) or "fast_ar"
            details["allreduce"].append(
                {
                    "backend": f"fast_ar:{far_backend}",
                    "tokens": tokens,
                    "bytes": nbytes,
                    "max_ms": round(max_ms, 4),
                    "algbw_gbps": round(algbw, 2),
                    "busbw_gbps": round(busbw, 2),
                }
            )
    except Exception as exc:
        details["fast_ar_err"] = str(exc)

    # AllToAll sweep (BF16 buffers, total bytes per rank)
    a2a_sizes = (64 * 1024, 1 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024)
    a2a_peak_busbw = 0.0
    a2a_peak_algbw = 0.0
    a2a_small_us = 0.0
    for nbytes in a2a_sizes:
        elems = ((nbytes // 2) // w) * w
        inp = torch.ones((elems,), dtype=torch.bfloat16, device=device)
        out = torch.empty_like(inp)
        dist.barrier()
        ms = _time_cuda_op_ms(lambda: dist.all_to_all_single(out, inp), warmup=warmup, iters=iters)
        t_max = torch.tensor([ms], dtype=torch.float64, device=device)
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
        max_ms = float(t_max.item())
        algbw = (elems * 2) / (max_ms * 1e-3) / 1e9
        busbw = algbw * ((w - 1) / w)
        a2a_peak_algbw = max(a2a_peak_algbw, algbw)
        a2a_peak_busbw = max(a2a_peak_busbw, busbw)
        if nbytes == 64 * 1024:
            a2a_small_us = max_ms * 1000.0
        details["alltoall"].append(
            {
                "bytes": elems * 2,
                "max_ms": round(max_ms, 4),
                "algbw_gbps": round(algbw, 2),
                "busbw_gbps": round(busbw, 2),
            }
        )

    return {
        "nvlink5_allreduce_gbps": round(ar_peak_busbw, 2),
        "nvlink5_allreduce_algbw_gbps": round(ar_peak_algbw, 2),
        "nvlink5_allreduce_small_us": round(ar_small_us, 2),
        "nvlink5_alltoall_gbps": round(a2a_peak_busbw, 2),
        "nvlink5_alltoall_algbw_gbps": round(a2a_peak_algbw, 2),
        "nvlink5_alltoall_small_us": round(a2a_small_us, 2),
    }, details


def calibrate_l20a_roofline(
    *,
    device: Optional[torch.device] = None,
    json_path: Optional[str | Path] = None,
    world_size: Optional[int] = None,
    warmup: int = 8,
    iters: int = 25,
) -> HardwareRoofline:
    """Calibrate L20A SM100 compute, HBM3e, and (when distributed) 4-GPU NVLink5 ceilings."""
    dev = device or torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(dev)
    dist_ws = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    ws = int(world_size or dist_ws)

    bf16_tf, bf16_det = _calibrate_bf16_gemm(dev, warmup=warmup, iters=iters)
    fp8_tf, fp8_det = _calibrate_fp8_gemm(dev, warmup=warmup, iters=iters)
    mx_tf, mx_det = _calibrate_mxfp4_gemm(dev, warmup=warmup, iters=iters)
    fp4_tf, fp4_det = _calibrate_fp4_gemm(dev, warmup=warmup, iters=iters)
    hbm_bw, hbm_det = _calibrate_hbm_bw(dev, warmup=warmup, iters=iters)
    nv_metrics, nv_det = _calibrate_nvlink5(dev, warmup=warmup, iters=iters)

    cap = torch.cuda.get_device_capability(dev)
    hr = HardwareRoofline(
        gpu_name=str(props.name),
        sm_count=int(props.multi_processor_count),
        compute_capability=f"{cap[0]}.{cap[1]}",
        world_size=ws,
        bf16_tflops=bf16_tf,
        fp8_tflops=fp8_tf,
        mxfp4_tflops=mx_tf,
        fp4_tflops=fp4_tf,
        hbm_gbps=hbm_bw,
        nvlink5_allreduce_gbps=nv_metrics["nvlink5_allreduce_gbps"],
        nvlink5_allreduce_algbw_gbps=nv_metrics["nvlink5_allreduce_algbw_gbps"],
        nvlink5_allreduce_small_us=nv_metrics["nvlink5_allreduce_small_us"],
        nvlink5_alltoall_gbps=nv_metrics["nvlink5_alltoall_gbps"],
        nvlink5_alltoall_algbw_gbps=nv_metrics["nvlink5_alltoall_algbw_gbps"],
        nvlink5_alltoall_small_us=nv_metrics["nvlink5_alltoall_small_us"],
        details={
            "bf16": bf16_det,
            "fp8": fp8_det,
            "mxfp4": mx_det,
            "fp4": fp4_det,
            "hbm": hbm_det,
            "nvlink5": nv_det,
        },
    )
    if json_path:
        is_rank0 = not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)
        if is_rank0:
            hr.save_json(json_path)
    return hr


# ---------------------------------------------------------------------------
# Qwen4.5-Flash 5L Analytic FLOP / Byte Roofline Model
# ---------------------------------------------------------------------------


PARALLEL_MODES: Dict[str, Tuple[int, int, int, int]] = {
    # mode -> (dp, tp, ep, sp)
    "tp4_ep4": (1, 4, 4, 1),
    "dp4_ep4": (4, 1, 4, 1),
    "dp2_tp2_ep4": (2, 2, 4, 1),
    "sp4_tp4_ep4": (1, 4, 4, 4),
    "tp1_ep1": (1, 1, 1, 1),
}

# Dense precision weights/activations and routed MoE precision presets
DTYPE_PRESETS: Dict[str, Tuple[str, str]] = {
    # dtype_mode -> (dense_precision, routed_moe_precision)
    "bf16": ("bf16", "bf16"),
    "mxfp4": ("bf16", "mxfp4"),
    "w4a8": ("bf16", "mxfp4"),
    "fp8": ("fp8", "mxfp4"),
    "fp8_mxfp4": ("fp8", "mxfp4"),
    "w8a8_mxfp4": ("fp8", "mxfp4"),
    "fp8_fp4": ("fp8", "fp4"),
    "nvfp4": ("fp4", "fp4"),
    "w4a4": ("fp4", "fp4"),
}


@dataclass(frozen=True)
class Qwen45ModelSpec:
    """Exact layer/projection structure extracted from Target QSA or DFlash2 config."""

    hidden_size: int = 5120
    vocab_size: int = 248320
    num_hidden_layers: int = 5
    intermediate_size: int = 12800
    routed_expert_hidden_size: int = 2560
    moe_intermediate_size: int = 2560
    shared_expert_intermediate_size: int = 1280
    num_experts: int = 640
    num_experts_per_tok: int = 10
    hc_count: int = 4
    hc_lowrank: int = 128
    # Layer counts (Target)
    n_gdn_normal: int = 2
    n_gdn_consumer: int = 1
    n_full_producer: int = 1
    n_full_consumer: int = 1
    n_dense_mlp: int = 1
    n_latent_moe: int = 4
    n_ple_layers: int = 1
    n_hc_blocks: int = 11  # 2 per layer + 1 final mixer
    n_output_sconv: int = 10  # 2 per layer (attn + ffn)
    # Full Attention + QSA dims
    num_attention_heads: int = 32
    num_key_value_heads: int = 1
    head_dim: int = 256
    attn_output_gate: bool = True
    indexer_n_heads: int = 8
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_compress_ratio: int = 4
    indexer_budget: int = 2048
    # GDN dims
    linear_num_key_heads: int = 32
    linear_key_head_dim: int = 64
    linear_num_value_heads: int = 128
    linear_value_head_dim: int = 64
    linear_conv_kernel_dim: int = 4
    # PLE dims
    ple_embed_dim: int = 5120
    ple_conv_kernel_size: int = 4


@dataclass(frozen=True)
class DFlash2ModelSpec:
    """Exact 5-layer DFlash2 non-causal draft model dimensions."""

    num_hidden_layers: int = 5
    hidden_size: int = 5120
    vocab_size: int = 248320
    num_attention_heads: int = 32
    num_key_value_heads: int = 1
    head_dim: int = 256
    swa_window_size: int = 2048
    n_swa_layers: int = 4
    n_full_layers: int = 1
    n_target_capture_layers: int = 5
    hc_count: int = 4
    hc_lowrank: int = 128
    conv_kernel_size: int = 2
    conv_group_size: int = 16
    routed_expert_hidden_size: int = 2560
    moe_intermediate_size: int = 2560
    shared_expert_intermediate_size: int = 1280
    num_experts: int = 640
    num_experts_per_tok: int = 10
    selector_rank: int = 256
    selector_top_k: int = 16


def model_spec_from_arch_config(arch_cfg: Any = None) -> Qwen45ModelSpec:
    """Build exact :class:`Qwen45ModelSpec` from a ``Qwen4_5ArchConfig`` or config dict."""
    if arch_cfg is None:
        return Qwen45ModelSpec()
    if isinstance(arch_cfg, Qwen45ModelSpec):
        return arch_cfg

    if isinstance(arch_cfg, Mapping):
        from rtp_llm.config.qwen4_5_config import parse_qwen4_5_config_json

        arch_cfg = parse_qwen4_5_config_json(dict(arch_cfg), strict=False)

    from rtp_llm.config.qwen4_5_config import validate_qwen4_5_arch_config

    share_map = validate_qwen4_5_arch_config(arch_cfg)
    n_layers = int(arch_cfg.num_hidden_layers)
    n_gdn_norm = 0
    n_gdn_cons = 0
    n_full_prod = 0
    n_full_cons = 0
    n_dense = 0
    n_moe = 0
    n_sconv = 0
    for lid in range(n_layers):
        ltype = arch_cfg.layer_types[lid]
        is_cons = lid in share_map
        if ltype == "linear_attention":
            if is_cons:
                n_gdn_cons += 1
            else:
                n_gdn_norm += 1
            if arch_cfg.has_output_sconv("linear_attn_output_sconv", is_consumer=is_cons):
                n_sconv += 1
        else:
            if is_cons:
                n_full_cons += 1
            else:
                n_full_prod += 1
            if arch_cfg.has_output_sconv("full_attn_output_sconv", is_consumer=is_cons):
                n_sconv += 1
        if arch_cfg.has_output_sconv("ffn_output_sconv", is_consumer=is_cons):
            n_sconv += 1
        if lid < int(arch_cfg.first_k_dense_replace):
            n_dense += 1
        else:
            n_moe += 1

    return Qwen45ModelSpec(
        hidden_size=int(arch_cfg.hidden_size),
        vocab_size=int(arch_cfg.vocab_size),
        num_hidden_layers=n_layers,
        intermediate_size=int(arch_cfg.intermediate_size),
        routed_expert_hidden_size=int(arch_cfg.routed_expert_hidden_size),
        moe_intermediate_size=int(arch_cfg.moe_intermediate_size),
        shared_expert_intermediate_size=int(arch_cfg.shared_expert_intermediate_size),
        num_experts=int(arch_cfg.num_experts),
        num_experts_per_tok=int(arch_cfg.num_experts_per_tok),
        hc_count=int(arch_cfg.hc_count),
        hc_lowrank=int(arch_cfg.hc_lowrank),
        n_gdn_normal=n_gdn_norm,
        n_gdn_consumer=n_gdn_cons,
        n_full_producer=n_full_prod,
        n_full_consumer=n_full_cons,
        n_dense_mlp=n_dense,
        n_latent_moe=n_moe,
        n_ple_layers=len(arch_cfg.ple_layer_ids),
        n_hc_blocks=2 * n_layers + 1,
        n_output_sconv=n_sconv,
        num_attention_heads=int(arch_cfg.num_attention_heads),
        num_key_value_heads=int(arch_cfg.num_key_value_heads),
        head_dim=int(arch_cfg.head_dim),
        attn_output_gate=bool(arch_cfg.attn_output_gate),
        indexer_n_heads=int(arch_cfg.indexer_n_heads),
        indexer_kv_heads=int(arch_cfg.indexer_kv_heads),
        indexer_head_dim=int(arch_cfg.indexer_head_dim),
        indexer_compress_ratio=int(arch_cfg.indexer_compress_ratio),
        indexer_budget=int(arch_cfg.indexer_budget),
        linear_num_key_heads=int(arch_cfg.linear_num_key_heads),
        linear_key_head_dim=int(arch_cfg.linear_key_head_dim),
        linear_num_value_heads=int(arch_cfg.linear_num_value_heads),
        linear_value_head_dim=int(arch_cfg.linear_value_head_dim),
        linear_conv_kernel_dim=int(arch_cfg.linear_conv_kernel_dim),
        ple_embed_dim=int(arch_cfg.ple_embed_dim),
        ple_conv_kernel_size=int(arch_cfg.ple_conv_kernel_size),
    )


def _weight_bytes_per_elem(precision: str) -> float:
    p = precision.lower()
    if p in ("fp4", "nvfp4", "w4a4"):
        # 4-bit value + FP8 E4M3 scale per 16 elements
        return 0.5 + 1.0 / 16.0
    if p in ("mxfp4", "w4a8", "fp8_fp4"):
        # 4-bit value + UE8M0 scale per 32 elements
        return 0.5 + 1.0 / 32.0
    if p in ("fp8", "w8a8", "bf16_fp8"):
        return 1.0
    return 2.0


def _expected_active_experts(num_active_tokens: int, num_experts: int, top_k: int) -> float:
    if num_active_tokens <= 0 or num_experts <= 0 or top_k <= 0:
        return 0.0
    if top_k >= num_experts:
        return float(num_experts)
    # Without replacement within one token: 1 - C(E-1, k)/C(E, k) = top_k / E per token
    p_hit = float(top_k) / float(num_experts)
    return float(num_experts) * (1.0 - (1.0 - p_hit) ** float(num_active_tokens))


def _causal_qsa_kv_sum(
    prefix_len: int,
    seq_len: int,
    budget: int,
    *,
    is_prefill: bool,
    avg_ctx: int,
) -> float:
    """Total attended KV tokens summed across all `seq_len` query positions of ONE sequence."""
    if seq_len <= 0:
        return 0.0
    if not is_prefill:
        eff_kv = float(min(max(1, avg_ctx), budget)) if budget > 0 else float(max(1, avg_ctx))
        return float(seq_len) * eff_kv

    start = max(0, prefix_len)
    end = start + seq_len - 1
    if budget <= 0:
        # Dense causal: sum_{p=start..end} (p + 1)
        return 0.5 * float(seq_len) * float((start + 1) + (end + 1))

    # Split at threshold p + 1 <= budget -> p <= budget - 1
    cap_p = min(end, budget - 1)
    total = 0.0
    if cap_p >= start:
        cnt = cap_p - start + 1
        total += 0.5 * float(cnt) * float((start + 1) + (cap_p + 1))
    if end > cap_p:
        cnt = end - max(start - 1, cap_p)
        total += float(cnt) * float(budget)
    return total


@dataclass
class OperatorMetrics:
    """Analytic FLOP/Byte and empirical efficiency metrics for one operator category."""

    name: str
    precision: str
    algo_flops: float
    algo_weight_bytes: float
    algo_act_kv_bytes: float
    exec_flops: float
    exec_weight_bytes: float
    comm_bytes: float = 0.0
    measured_ms: Optional[float] = None
    peak_tflops: Optional[float] = None
    peak_hbm_gbps: Optional[float] = None
    peak_comm_gbps: Optional[float] = None

    @property
    def algo_bytes(self) -> float:
        return self.algo_weight_bytes + self.algo_act_kv_bytes

    @property
    def exec_bytes(self) -> float:
        return self.exec_weight_bytes + self.algo_act_kv_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.algo_flops / max(self.algo_bytes, 1.0)

    def roofline_min_ms(self) -> Optional[float]:
        if self.peak_tflops is None or self.peak_hbm_gbps is None:
            return None
        comp_ms = (self.algo_flops / max(self.peak_tflops * 1e12, 1.0)) * 1e3
        mem_ms = (self.algo_bytes / max(self.peak_hbm_gbps * 1e9, 1.0)) * 1e3
        comm_ms = 0.0
        if self.comm_bytes > 0 and self.peak_comm_gbps and self.peak_comm_gbps > 0:
            comm_ms = (self.comm_bytes / (self.peak_comm_gbps * 1e9)) * 1e3
        return max(comp_ms, mem_ms) + comm_ms

    def to_dict(self) -> Dict[str, Any]:
        mfu_pct = None
        mbu_pct = None
        if self.comm_bytes > 0 and self.algo_flops == 0:
            bound = "comm"
        elif self.peak_tflops and self.peak_hbm_gbps and self.algo_bytes > 0:
            ridge = (self.peak_tflops * 1e12) / (self.peak_hbm_gbps * 1e9)
            bound = "compute" if self.arithmetic_intensity >= ridge else "memory"
        else:
            bound = "compute"
        if self.measured_ms is not None and self.measured_ms > 0:
            sec = self.measured_ms * 1e-3
            if self.peak_tflops and self.peak_tflops > 0 and self.algo_flops > 0:
                mfu_pct = (self.algo_flops / (sec * self.peak_tflops * 1e12)) * 100.0
            if self.peak_hbm_gbps and self.peak_hbm_gbps > 0 and self.algo_bytes > 0:
                mbu_pct = (self.algo_bytes / (sec * self.peak_hbm_gbps * 1e9)) * 100.0
            elif self.peak_comm_gbps and self.peak_comm_gbps > 0 and self.comm_bytes > 0:
                mbu_pct = (self.comm_bytes / (sec * self.peak_comm_gbps * 1e9)) * 100.0

        min_ms = self.roofline_min_ms()
        return {
            "name": self.name,
            "precision": self.precision,
            "algo_gflops": round(self.algo_flops / 1e9, 4),
            "algo_mb": round(self.algo_bytes / 1e6, 4),
            "algo_weight_mb": round(self.algo_weight_bytes / 1e6, 4),
            "algo_act_kv_mb": round(self.algo_act_kv_bytes / 1e6, 4),
            "exec_gflops": round(self.exec_flops / 1e9, 4),
            "comm_mb": round(self.comm_bytes / 1e6, 4),
            "arithmetic_intensity": round(self.arithmetic_intensity, 2),
            "bound_regime": bound,
            "roofline_min_ms": round(min_ms, 4) if min_ms is not None else None,
            "measured_ms": round(self.measured_ms, 4) if self.measured_ms is not None else None,
            "mfu_pct": round(mfu_pct, 2) if mfu_pct is not None else None,
            "mbu_pct": round(mbu_pct, 2) if mbu_pct is not None else None,
        }


@dataclass
class WorkloadRooflineSummary:
    """Whole-model Qwen4.5-Flash 5L FLOP/Byte accounting and achieved efficiency."""

    batch_size: int
    seq_len_per_step: int
    total_tokens: int
    avg_ctx: int
    prefix_len: int
    parallel_mode: str
    dtype_mode: str
    world_size: int
    algo_flops: float
    algo_weight_bytes: float
    algo_act_kv_bytes: float
    exec_flops: float
    exec_weight_bytes: float
    comm_bytes: float
    operators: Dict[str, OperatorMetrics]
    elapsed_ms: Optional[float] = None
    roofline: Optional[HardwareRoofline] = None
    kv_dtype: str = "bf16"
    dflash_kv_dtype: Optional[str] = None
    indexer_kv_dtype: str = "bf16"

    @property
    def algo_bytes(self) -> float:
        return self.algo_weight_bytes + self.algo_act_kv_bytes

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "seq_len_per_step": self.seq_len_per_step,
            "total_tokens": self.total_tokens,
            "avg_ctx": self.avg_ctx,
            "prefix_len": self.prefix_len,
            "parallel_mode": self.parallel_mode,
            "dtype_mode": self.dtype_mode,
            "kv_dtype": self.kv_dtype,
            "dflash_kv_dtype": self.dflash_kv_dtype or self.kv_dtype,
            "indexer_kv_dtype": self.indexer_kv_dtype,
            "world_size": self.world_size,
            "algo_tflops": round(self.algo_flops / 1e12, 4),
            "algo_mb": round(self.algo_bytes / 1e6, 4),
            "algo_weight_mb": round(self.algo_weight_bytes / 1e6, 4),
            "algo_act_kv_mb": round(self.algo_act_kv_bytes / 1e6, 4),
            "exec_tflops": round(self.exec_flops / 1e12, 4),
            "replication_compute_ratio": round(self.exec_flops / max(self.algo_flops, 1.0), 3),
            "comm_mb": round(self.comm_bytes / 1e6, 4),
            "arithmetic_intensity": round(self.algo_flops / max(self.algo_bytes, 1.0), 2),
            "elapsed_ms": round(self.elapsed_ms, 4) if self.elapsed_ms is not None else None,
            "operators": {k: v.to_dict() for k, v in self.operators.items()},
        }
        if self.roofline is not None:
            active_gpus = max(1, int(self.world_size))
            bf16_cluster = max(self.roofline.bf16_tflops * active_gpus, 1e-6)
            hbm_cluster = max(self.roofline.hbm_gbps * active_gpus, 1e-6)
            ideal_comp_bf16_ms = (self.algo_flops / (bf16_cluster * 1e12)) * 1e3
            ideal_comp_aware_ms = sum(
                (
                    op.algo_flops
                    / max(
                        self.roofline.peak_tflops_for_precision(op.precision, per_gpu=True)
                        * active_gpus
                        * 1e12,
                        1.0,
                    )
                )
                * 1e3
                for op in self.operators.values()
            )
            ideal_hbm_ms = (self.algo_bytes / (hbm_cluster * 1e9)) * 1e3
            sol_sum_ms = sum((op.roofline_min_ms() or 0.0) for op in self.operators.values())
            out["ideal_compute_bf16_ms"] = round(ideal_comp_bf16_ms, 4)
            out["ideal_compute_aware_ms"] = round(ideal_comp_aware_ms, 4)
            out["ideal_hbm_ms"] = round(ideal_hbm_ms, 4)
            out["sol_sum_ms"] = round(sol_sum_ms, 4)

            if self.elapsed_ms is not None and self.elapsed_ms > 0:
                sec = self.elapsed_ms * 1e-3
                out["tokens_per_s"] = round(self.total_tokens / sec, 2)
                out["achieved_tflops"] = round((self.algo_flops / sec) / 1e12, 3)
                out["achieved_gbps"] = round((self.algo_bytes / sec) / 1e9, 2)
                out["mfu_bf16_pct"] = round((ideal_comp_bf16_ms / self.elapsed_ms) * 100.0, 2)
                out["mfu_aware_pct"] = round((ideal_comp_aware_ms / self.elapsed_ms) * 100.0, 2)
                out["mbu_pct"] = round((ideal_hbm_ms / self.elapsed_ms) * 100.0, 2)
                out["sol_efficiency_pct"] = round((sol_sum_ms / self.elapsed_ms) * 100.0, 2)
        return out


def _hc_block_accounting(
    hidden_size: int,
    hc_count: int,
    hc_lowrank: int,
    n_blocks: int,
    tokens: float,
) -> Tuple[int, float]:
    """Return ``(weight_elems, flops)`` for Qwen4.5 4-stream Hyper-Connections.

    Matches :class:`Qwen4_5HyperConnection`:
      - Layer 0 ``attn_hc`` (head): ``input_mix_weight_up`` is ``[H, hc_lowrank]``
      - Final layer ``mlp_hc``: omits ``input_mix_weight_down`` (``emits_latent=False``)
      - Final ``hyper_connection_mixer``: omits ``block_inject_weight`` (``use_combine=False``)
    """
    if n_blocks <= 0:
        return 0, 0.0
    hc_h = hc_count * hidden_size
    n_down = max(0, n_blocks - 1) if n_blocks >= 2 else n_blocks
    n_inj = max(0, n_blocks - 1) if n_blocks >= 2 else n_blocks
    down_elems = n_down * hc_lowrank * hc_h
    up_elems = (hidden_size * hc_lowrank) + max(0, n_blocks - 1) * (hc_h * hc_lowrank)
    inj_elems = n_inj * hc_count * hc_h
    norm_elems = n_blocks * 2 * hc_h
    gemm_elems = down_elems + up_elems + inj_elems
    # GEMM (down/up/inject) + grouped RMSNorm / stream combine elementwise work
    flops = 2.0 * tokens * gemm_elems + n_blocks * 6.0 * tokens * hc_h
    return gemm_elems + norm_elems, flops


def _normalize_kv_dtype(kv_dtype: str) -> Tuple[str, float]:
    k = str(kv_dtype or "bf16").strip().lower()
    if any(tag in k for tag in ("fp8", "f8", "e4m3", "e5m2")):
        return "fp8_e4m3", 1.0
    return "bf16", 2.0


def compute_qwen45_workload_metrics(
    arch_config: Any = None,
    *,
    batch_size: int,
    seq_len_per_step: int,
    avg_ctx: int = 0,
    parallel_mode: str = "tp4_ep4",
    dtype_mode: str = "mxfp4",
    kv_dtype: str = "bf16",
    dflash_kv_dtype: Optional[str] = None,
    indexer_kv_dtype: str = "bf16",
    world_size: int = 4,
    is_prefill: bool = True,
    prefix_len: int = 0,
    logits_tokens: Optional[int] = None,
    include_dflash2: bool = False,
    dflash_spec: Any = None,
    dflash_tokens_per_step: Optional[int] = None,
    dflash_commit_tokens: Optional[int] = None,
    roofline: Optional[HardwareRoofline] = None,
    elapsed_ms: Optional[float] = None,
    op_times_ms: Optional[Mapping[str, float]] = None,
) -> WorkloadRooflineSummary:
    """Compute exact algorithmic + executed FLOPs, Bytes, MFU%, and MBU% for Qwen4.5-Flash 5L."""
    spec = model_spec_from_arch_config(arch_config)
    pm = parallel_mode.lower()
    if pm not in PARALLEL_MODES:
        raise ValueError(f"Unsupported parallel_mode={parallel_mode!r}; expected {tuple(PARALLEL_MODES)}")
    dp, tp, ep, sp = PARALLEL_MODES[pm]

    dm = dtype_mode.lower()
    if dm not in DTYPE_PRESETS:
        raise ValueError(f"Unsupported dtype_mode={dtype_mode!r}; expected {tuple(DTYPE_PRESETS)}")
    dense_prec, moe_prec = DTYPE_PRESETS[dm]
    dense_wb = _weight_bytes_per_elem(dense_prec)
    moe_wb = _weight_bytes_per_elem(moe_prec)
    bf16_b = 2.0
    kv_norm, kv_b = _normalize_kv_dtype(kv_dtype)
    df_kv_norm, df_kv_b = _normalize_kv_dtype(dflash_kv_dtype if dflash_kv_dtype is not None else kv_norm)
    idx_kv_norm, idx_kv_b = _normalize_kv_dtype(indexer_kv_dtype)

    B = int(batch_size)
    T = int(seq_len_per_step)
    M = B * T
    eff_ctx = int(avg_ctx) if avg_ctx > 0 else (prefix_len + T)
    if logits_tokens is None:
        logits_tokens = B if is_prefill else M

    active_gpus = max(1, int(world_size))
    H = spec.hidden_size
    op_map = dict(op_times_ms or {})

    def _make_op(
        name: str,
        precision: str,
        algo_flops: float,
        algo_w_bytes: float,
        algo_akv_bytes: float,
        *,
        rep_factor: float = 1.0,
        exec_flops: Optional[float] = None,
        exec_w_bytes: Optional[float] = None,
        comm_bytes: float = 0.0,
        comm_is_a2a: bool = False,
    ) -> OperatorMetrics:
        peak_tf = (
            roofline.peak_tflops_for_precision(precision, per_gpu=True) * active_gpus
            if roofline
            else None
        )
        peak_hbm = (roofline.hbm_gbps * active_gpus) if roofline else None
        peak_comm = None
        if roofline and comm_bytes > 0:
            peak_comm = roofline.nvlink5_alltoall_gbps if comm_is_a2a else roofline.nvlink5_allreduce_gbps
        return OperatorMetrics(
            name=name,
            precision=precision,
            algo_flops=algo_flops,
            algo_weight_bytes=algo_w_bytes,
            algo_act_kv_bytes=algo_akv_bytes,
            exec_flops=(algo_flops * rep_factor) if exec_flops is None else exec_flops,
            exec_weight_bytes=(algo_w_bytes * rep_factor) if exec_w_bytes is None else exec_w_bytes,
            comm_bytes=comm_bytes,
            measured_ms=op_map.get(name),
            peak_tflops=peak_tf,
            peak_hbm_gbps=peak_hbm,
            peak_comm_gbps=peak_comm,
        )

    ops: Dict[str, OperatorMetrics] = {}

    # 1. Embedding + Layer 1 PLE
    dense_rep = max(1.0, float(tp) / float(sp)) if tp > 1 else 1.0
    ple_w_elems = spec.n_ple_layers * (
        spec.ple_embed_dim * (spec.hc_count * H) + spec.ple_embed_dim * H
    )
    ple_flops = spec.n_ple_layers * (
        2.0 * M * spec.ple_embed_dim * (spec.hc_count * H + H)
        + 2.0 * M * (spec.hc_count * H) * spec.ple_conv_kernel_size
    )
    ple_act_bytes = M * H * bf16_b * 2.0 + spec.n_ple_layers * (
        M * spec.ple_embed_dim * 1.0 + M * spec.hc_count * H * bf16_b * 2.0
    )
    ops["embed_ple"] = _make_op(
        "embed_ple",
        "bf16",
        ple_flops,
        ple_w_elems * bf16_b,
        ple_act_bytes,
        rep_factor=dense_rep,
    )

    # 2. Hyper-Connections (11 blocks) & Output ShortConvs (10 blocks)
    hc_w_elems, hc_flops = _hc_block_accounting(
        H, spec.hc_count, spec.hc_lowrank, spec.n_hc_blocks, float(M)
    )
    sconv_w_elems = spec.n_output_sconv * (H * spec.linear_conv_kernel_dim)
    sconv_flops = spec.n_output_sconv * (2.0 * M * H * spec.linear_conv_kernel_dim)
    hc_sconv_act = M * spec.hc_count * H * bf16_b * 2.0 * spec.n_hc_blocks + (
        B * H * max(0, spec.linear_conv_kernel_dim - 1) * bf16_b * 2.0 * spec.n_output_sconv
    )
    ops["hc_and_output_sconv"] = _make_op(
        "hc_and_output_sconv",
        "bf16",
        hc_flops + sconv_flops,
        (hc_w_elems + sconv_w_elems) * bf16_b,
        hc_sconv_act,
        rep_factor=dense_rep,
    )

    # 3. Gated DeltaNet (GDN) layers: Projections + Non-GEMM Conv/Recurrence
    k_dim = spec.linear_num_key_heads * spec.linear_key_head_dim
    v_dim = spec.linear_num_value_heads * spec.linear_value_head_dim
    gdn_norm_qkvz = 2 * k_dim + 2 * v_dim  # 20480
    gdn_norm_ba = 2 * spec.linear_num_value_heads  # 256
    gdn_cons_qz = k_dim + v_dim  # 10240
    gdn_proj_w_elems = (
        spec.n_gdn_normal * (H * gdn_norm_qkvz + H * gdn_norm_ba + v_dim * H)
        + spec.n_gdn_consumer * (H * gdn_cons_qz + v_dim * H)
    )
    gdn_proj_flops = 2.0 * M * gdn_proj_w_elems
    ops["gdn_proj"] = _make_op(
        "gdn_proj",
        dense_prec,
        gdn_proj_flops,
        gdn_proj_w_elems * dense_wb,
        M * (H + v_dim) * bf16_b * (spec.n_gdn_normal + spec.n_gdn_consumer),
        rep_factor=1.0,  # TP-sharded across heads
    )

    gdn_conv_ch_norm = 2 * k_dim + v_dim  # 12288
    gdn_conv_ch_cons = k_dim  # 2048
    n_gdn_total = spec.n_gdn_normal + spec.n_gdn_consumer
    # Both source/normal AND consumer GDN layers keep their own SSM state and run full delta-rule core
    gdn_core_flops = (
        n_gdn_total * (6.0 * M * spec.linear_num_value_heads * spec.linear_key_head_dim * spec.linear_value_head_dim)
        + spec.n_gdn_normal * (2.0 * M * gdn_conv_ch_norm * spec.linear_conv_kernel_dim)
        + spec.n_gdn_consumer * (2.0 * M * gdn_conv_ch_cons * spec.linear_conv_kernel_dim)
    )
    ssm_state_bytes = n_gdn_total * B * (
        spec.linear_num_value_heads * spec.linear_key_head_dim * spec.linear_value_head_dim * 4.0 * 2.0
    )
    gdn_conv_state_bytes = B * (
        spec.n_gdn_normal * gdn_conv_ch_norm + spec.n_gdn_consumer * gdn_conv_ch_cons
    ) * max(0, spec.linear_conv_kernel_dim - 1) * bf16_b * 2.0
    ops["gdn_core"] = _make_op(
        "gdn_core",
        "bf16",
        gdn_core_flops,
        (spec.n_gdn_normal * gdn_conv_ch_norm + spec.n_gdn_consumer * gdn_conv_ch_cons) * spec.linear_conv_kernel_dim * bf16_b,
        ssm_state_bytes + gdn_conv_state_bytes + M * v_dim * bf16_b * 2.0 * n_gdn_total,
        rep_factor=1.0,  # TP-sharded across heads
    )

    # 4. Full Attention + QSA Projections
    q_out = spec.num_attention_heads * spec.head_dim * (2 if spec.attn_output_gate else 1)  # 16384
    kv_out = spec.num_key_value_heads * spec.head_dim  # 256
    o_in = spec.num_attention_heads * spec.head_dim  # 8192
    idx_qk_out = (spec.indexer_n_heads + spec.indexer_kv_heads) * spec.indexer_head_dim  # 1152

    # q_proj/o_proj are TP-sharded; k/v_proj (producers only) and index_qk_proj (producer + consumer) replicate in TP
    n_full_total = spec.n_full_producer + spec.n_full_consumer
    fa_sharded_w = n_full_total * (H * q_out + o_in * H)
    fa_repl_w = spec.n_full_producer * (H * 2 * kv_out) + n_full_total * (H * idx_qk_out)
    fa_proj_flops = 2.0 * M * (fa_sharded_w + fa_repl_w)
    fa_proj_w_bytes = (fa_sharded_w + fa_repl_w) * dense_wb
    fa_repl_tp = tp if tp > 1 else 1
    ops["full_attn_proj"] = _make_op(
        "full_attn_proj",
        dense_prec,
        fa_proj_flops,
        fa_proj_w_bytes,
        M * (H + o_in) * bf16_b * n_full_total,
        exec_flops=2.0 * M * (fa_sharded_w + fa_repl_w * fa_repl_tp),
        exec_w_bytes=(fa_sharded_w + fa_repl_w * fa_repl_tp) * dense_wb,
    )

    # 5. Full Attention + QSA Indexer + Sparse Core
    # Both producer (Layer 2) and consumer (Layer 4) run QSA Indexer scoring + Top-K against Layer 2's idx_kv_pool
    kv_sum_full = _causal_qsa_kv_sum(prefix_len, T, budget=0, is_prefill=is_prefill, avg_ctx=eff_ctx)
    kv_sum_sparse = _causal_qsa_kv_sum(
        prefix_len, T, budget=spec.indexer_budget, is_prefill=is_prefill, avg_ctx=eff_ctx
    )
    idx_flops = n_full_total * B * (
        2.0 * (kv_sum_full / max(1, spec.indexer_compress_ratio)) * spec.indexer_n_heads * spec.indexer_head_dim
    )
    qsa_attn_flops = n_full_total * B * (
        4.0 * kv_sum_sparse * spec.num_attention_heads * spec.head_dim
    )
    # Only producer writes new K/V + Indexer K; both producer & consumer read Indexer K + selected K/V
    kv_write_bytes = spec.n_full_producer * M * (
        2 * kv_out * kv_b + spec.indexer_head_dim * idx_kv_b / max(1, spec.indexer_compress_ratio)
    )
    kv_read_bytes = B * (
        n_full_total * (kv_sum_full / max(1, spec.indexer_compress_ratio)) * spec.indexer_head_dim * idx_kv_b
        + n_full_total * kv_sum_sparse * (2 * kv_out * kv_b)
    )
    # Only QSA Indexer & producer KV SConv replicate in TP=4; 32 Q-head QSA Attention is TP-sharded (8/GPU)
    fa_idx_rep = float(tp) if tp > 1 else 1.0
    fa_sconv_w_bytes = spec.n_full_producer * (2 * kv_out * spec.linear_conv_kernel_dim) * bf16_b
    ops["full_attn_qsa_core"] = _make_op(
        "full_attn_qsa_core",
        "bf16",
        idx_flops + qsa_attn_flops,
        fa_sconv_w_bytes,
        kv_write_bytes + kv_read_bytes,
        exec_flops=qsa_attn_flops + idx_flops * fa_idx_rep,
        exec_w_bytes=fa_sconv_w_bytes * fa_idx_rep,
    )

    # 6. Dense MLP (Layer 0)
    dense_w_elems = spec.n_dense_mlp * (3 * H * spec.intermediate_size)
    ops["dense_mlp"] = _make_op(
        "dense_mlp",
        dense_prec,
        2.0 * M * dense_w_elems,
        dense_w_elems * dense_wb,
        M * (H + spec.intermediate_size) * bf16_b * spec.n_dense_mlp,
        rep_factor=1.0,  # TP-sharded in MLP
    )

    # 7. LatentMoE Replicated Latent + Router + Shared Expert (Layers 1..4)
    L_lat = spec.routed_expert_hidden_size
    lat_shared_w = spec.n_latent_moe * (
        3 * H * L_lat  # latent_down, latent_gate, latent_up
        + L_lat * spec.num_experts  # router gate
        + 3 * H * spec.shared_expert_intermediate_size  # shared expert
    )
    ops["moe_latent_shared"] = _make_op(
        "moe_latent_shared",
        dense_prec,
        2.0 * M * lat_shared_w,
        lat_shared_w * dense_wb,
        M * (H + L_lat) * bf16_b * spec.n_latent_moe,
        rep_factor=dense_rep,
    )

    # 8. LatentMoE Routed Experts (EP=4 sharded across 4 ranks)
    routed_per_expert_elems = 3 * L_lat * spec.moe_intermediate_size
    e_act = _expected_active_experts(M, spec.num_experts, spec.num_experts_per_tok)
    routed_flops = spec.n_latent_moe * (2.0 * M * spec.num_experts_per_tok * routed_per_expert_elems)
    routed_w_bytes = spec.n_latent_moe * e_act * routed_per_expert_elems * moe_wb
    routed_act_bytes = spec.n_latent_moe * M * spec.num_experts_per_tok * (
        L_lat * (1.0 if moe_prec in ("mxfp4", "fp8", "fp4") else bf16_b) + L_lat * bf16_b
    )
    ops["moe_routed"] = _make_op(
        "moe_routed",
        moe_prec,
        routed_flops,
        routed_w_bytes,
        routed_act_bytes,
        rep_factor=1.0,  # Cleanly EP-sharded across 4 ranks!
    )

    # 9. LM Head
    lm_w_elems = H * spec.vocab_size
    lm_prec = dense_prec if dm in ("fp8", "fp8_mxfp4", "fp8_fp4") else "bf16"
    ops["lm_head"] = _make_op(
        "lm_head",
        lm_prec,
        2.0 * logits_tokens * lm_w_elems,
        lm_w_elems * _weight_bytes_per_elem(lm_prec),
        logits_tokens * (H + spec.vocab_size) * bf16_b,
        rep_factor=1.0,
    )

    # 10. Optional DFlash2 5L Draft Model (Decode Commit + Propose track)
    ds: Optional[DFlash2ModelSpec] = None
    Md = 0
    if include_dflash2:
        ds = dflash_spec if isinstance(dflash_spec, DFlash2ModelSpec) else DFlash2ModelSpec()
        Md = int(dflash_tokens_per_step) if dflash_tokens_per_step is not None else M
        M_commit = int(dflash_commit_tokens) if dflash_commit_tokens is not None else B

        d_q_size = ds.num_attention_heads * ds.head_dim
        d_kv_size = ds.num_key_value_heads * ds.head_dim
        # Commit track (runs on M_commit committed rows): fc [25600, 5120] + commit_feature_rows k/v
        d_fc_w = ds.n_target_capture_layers * ds.hidden_size * ds.hidden_size
        d_commit_kv_w = ds.num_hidden_layers * (ds.hidden_size * d_kv_size * 2)
        d_commit_flops = 2.0 * M_commit * (d_fc_w + d_commit_kv_w)

        # Propose track dense GEMMs (runs on Md = B * query_width tokens)
        d_layer_proj_w = ds.num_hidden_layers * (
            ds.hidden_size * d_q_size * 2  # q_proj + output_gate_proj
            + ds.hidden_size * d_kv_size * 2  # k_proj + v_proj
            + d_q_size * ds.hidden_size  # o_proj
            + 3 * ds.hidden_size * ds.routed_expert_hidden_size  # latent down/gate/up
            + ds.routed_expert_hidden_size * ds.num_experts  # router
            + 3 * ds.hidden_size * ds.shared_expert_intermediate_size  # shared expert
        )
        d_propose_flops = 2.0 * Md * d_layer_proj_w

        # 10 DFlashGroupedConv blocks (attn + mlp per layer)
        n_d_conv = 2 * ds.num_hidden_layers
        d_num_groups = ds.hidden_size // max(1, ds.conv_group_size)
        d_conv_proj_w = n_d_conv * ds.hidden_size * (2 * ds.conv_kernel_size * d_num_groups)
        d_conv_base_w = n_d_conv * 2 * ds.conv_kernel_size * ds.hidden_size
        d_conv_flops = 2.0 * Md * d_conv_proj_w + n_d_conv * 4.0 * ds.conv_kernel_size * Md * ds.hidden_size
        d_conv_act_bytes = n_d_conv * Md * (
            2 * ds.conv_kernel_size * d_num_groups + ds.hidden_size * 2.0
        ) * bf16_b

        # 11 Hyper-Connection blocks (exact Qwen4_5HyperConnection shapes)
        d_hc_w, d_hc_flops = _hc_block_accounting(
            ds.hidden_size, ds.hc_count, ds.hc_lowrank, 2 * ds.num_hidden_layers + 1, float(Md)
        )

        # CandidateSelector: hidden_projection [H, rank] + sparse top_k=16 codebook lookup & [16, 16] edge matmul
        d_sel_proj_w = ds.hidden_size * ds.selector_rank
        d_sel_flops = (
            2.0 * Md * d_sel_proj_w
            + Md * ds.selector_top_k * ds.selector_rank
            + 2.0 * Md * (ds.selector_top_k ** 2) * ds.selector_rank
        )
        d_sel_codebook_bytes = 2.0 * Md * ds.selector_top_k * ds.selector_rank * bf16_b

        # DFlash2 Attention Core: d_attn_kv ALREADY sums all 5 layers (4*SWA + 1*Full)
        d_attn_kv = (
            ds.n_swa_layers * min(eff_ctx, ds.swa_window_size)
            + ds.n_full_layers * eff_ctx
        )
        d_attn_flops = 4.0 * Md * d_attn_kv * ds.num_attention_heads * ds.head_dim
        d_kv_write_bytes = ds.num_hidden_layers * M_commit * 2 * d_kv_size * df_kv_b
        d_kv_read_bytes = B * d_attn_kv * 2 * d_kv_size * df_kv_b

        # DFlash2 LM head matches Target lm_head precision (BF16 unless quantized)
        d_lm_w = ds.hidden_size * ds.vocab_size
        d_lm_wb = _weight_bytes_per_elem(lm_prec)
        d_lm_flops = 2.0 * Md * d_lm_w

        d_routed_per_e = 3 * ds.routed_expert_hidden_size * ds.moe_intermediate_size
        d_e_act = _expected_active_experts(Md, ds.num_experts, ds.num_experts_per_tok)
        d_routed_flops = ds.num_hidden_layers * (2.0 * Md * ds.num_experts_per_tok * d_routed_per_e)

        # Unique static weights: fc is read when M_commit > 0; k/v_proj is shared with d_layer_proj_w
        d_static_dense_w = (d_fc_w if M_commit > 0 else 0) + d_layer_proj_w
        d_dense_w_bytes = (
            d_static_dense_w * dense_wb
            + (d_conv_proj_w + d_conv_base_w + d_hc_w + d_sel_proj_w) * bf16_b
            + d_lm_w * d_lm_wb
        )
        d_moe_w_bytes = ds.num_hidden_layers * d_e_act * d_routed_per_e * moe_wb
        d_akv_bytes = (
            d_kv_write_bytes
            + d_kv_read_bytes
            + d_sel_codebook_bytes
            + d_conv_act_bytes
            + Md * (ds.hidden_size + ds.vocab_size) * bf16_b
        )
        ops["dflash2_attn_dense"] = _make_op(
            "dflash2_attn_dense",
            dense_prec,
            d_commit_flops + d_propose_flops + d_conv_flops + d_hc_flops + d_sel_flops + d_attn_flops + d_lm_flops,
            d_dense_w_bytes,
            d_akv_bytes,
            rep_factor=1.0,
        )
        ops["dflash2_moe_routed"] = _make_op(
            "dflash2_moe_routed",
            moe_prec,
            d_routed_flops,
            d_moe_w_bytes,
            ds.num_hidden_layers * Md * ds.num_experts_per_tok * ds.routed_expert_hidden_size * 3.0,
            rep_factor=1.0,
        )

    # 11. Inter-GPU Collectives (Per-GPU NVLink5 Bus Bandwidth Traffic)
    comm_ar_bytes = 0.0
    comm_a2a_bytes = 0.0
    M_local = float(M) / float(max(1, dp))
    Md_local = float(Md) / float(max(1, dp))
    bus_scale_ar = 2.0 * (active_gpus - 1) / float(active_gpus) if active_gpus > 1 else 0.0
    bus_scale_a2a = (active_gpus - 1) / float(active_gpus) if active_gpus > 1 else 0.0
    if tp > 1:
        # GDN + FullAttn + DenseMLP AllReduces in H=5120 space over local tokens
        n_tp_ar_h = n_gdn_total + n_full_total + spec.n_dense_mlp
        comm_ar_bytes += n_tp_ar_h * M_local * H * bf16_b * (2.0 * (tp - 1) / float(tp))
        # PLE shard AllReduce across ep ranks + LatentMoE EP AllReduce in latent_dim=2560
        comm_ar_bytes += spec.n_ple_layers * M_local * H * bf16_b * bus_scale_ar
        comm_ar_bytes += spec.n_latent_moe * M_local * L_lat * bf16_b * bus_scale_ar
    elif ep > 1:
        # dp4_ep4: PLE EP AllReduce + per-GPU AllToAll dispatch (1B) + combine (2B) on M_local tokens
        comm_ar_bytes += spec.n_ple_layers * M_local * H * bf16_b * bus_scale_ar
        comm_a2a_bytes += spec.n_latent_moe * M_local * spec.num_experts_per_tok * L_lat * (1.0 + bf16_b) * bus_scale_a2a

    if include_dflash2 and ds is not None:
        if tp > 1:
            # 5 Attn TP AllReduces (H=5120) + 5 MoE EP AllReduces (L_lat=2560)
            comm_ar_bytes += ds.num_hidden_layers * Md_local * ds.hidden_size * bf16_b * (2.0 * (tp - 1) / float(tp))
            comm_ar_bytes += ds.num_hidden_layers * Md_local * ds.routed_expert_hidden_size * bf16_b * bus_scale_ar
            # vocab_parallel_topk AllGathers: top_k float32 values + int64 ids per rank
            comm_ar_bytes += Md_local * ds.selector_top_k * (4.0 + 8.0) * (float(tp - 1) / float(tp))
        elif ep > 1:
            comm_a2a_bytes += ds.num_hidden_layers * Md_local * ds.num_experts_per_tok * ds.routed_expert_hidden_size * (1.0 + bf16_b) * bus_scale_a2a

    if comm_ar_bytes > 0:
        ops["comm_allreduce"] = _make_op("comm_allreduce", "bf16", 0.0, 0.0, 0.0, comm_bytes=comm_ar_bytes, comm_is_a2a=False)
    if comm_a2a_bytes > 0:
        ops["comm_alltoall"] = _make_op("comm_alltoall", "bf16", 0.0, 0.0, 0.0, comm_bytes=comm_a2a_bytes, comm_is_a2a=True)

    return WorkloadRooflineSummary(
        batch_size=B,
        seq_len_per_step=T,
        total_tokens=M,
        avg_ctx=eff_ctx,
        prefix_len=prefix_len,
        parallel_mode=pm,
        dtype_mode=dm,
        kv_dtype=kv_norm,
        dflash_kv_dtype=df_kv_norm,
        indexer_kv_dtype=idx_kv_norm,
        world_size=active_gpus,
        algo_flops=sum(o.algo_flops for o in ops.values()),
        algo_weight_bytes=sum(o.algo_weight_bytes for o in ops.values()),
        algo_act_kv_bytes=sum(o.algo_act_kv_bytes for o in ops.values()),
        exec_flops=sum(o.exec_flops for o in ops.values()),
        exec_weight_bytes=sum(o.exec_weight_bytes for o in ops.values()),
        comm_bytes=sum(o.comm_bytes for o in ops.values()),
        operators=ops,
        elapsed_ms=elapsed_ms,
        roofline=roofline,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate L20A SM100 Hardware Roofline & inspect 5L workload FLOPs/Bytes")
    parser.add_argument("--calibrate", action="store_true", help="Run empirical GPU/NVLink5 microbenchmarks")
    parser.add_argument("--out", default="", help="Optional JSON path to save calibrated HardwareRoofline")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--show-workload", action="store_true", help="Print sample 5L Prefill 20K & Decode 20K workload FLOPs/Bytes")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)

    hr: Optional[HardwareRoofline] = None
    if args.calibrate:
        hr = calibrate_l20a_roofline(
            device=torch.device("cuda", local_rank),
            json_path=args.out or None,
            warmup=args.warmup,
            iters=args.iters,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("HARDWARE_ROOFLINE:", json.dumps(hr.to_dict(), indent=2))

    if args.show_workload:
        if not dist.is_initialized() or dist.get_rank() == 0:
            pf_0 = compute_qwen45_workload_metrics(
                batch_size=1, seq_len_per_step=20480, avg_ctx=20480, parallel_mode="tp4_ep4", dtype_mode="mxfp4", is_prefill=True, prefix_len=0, roofline=hr
            )
            pf_90 = compute_qwen45_workload_metrics(
                batch_size=4, seq_len_per_step=2048, avg_ctx=20480, parallel_mode="tp4_ep4", dtype_mode="mxfp4", is_prefill=True, prefix_len=18432, roofline=hr
            )
            dec_20k = compute_qwen45_workload_metrics(
                batch_size=32, seq_len_per_step=8, avg_ctx=20480, parallel_mode="tp4_ep4", dtype_mode="mxfp4", is_prefill=False, include_dflash2=True, roofline=hr
            )
            print("PREFILL_0PCT_20K:", json.dumps(pf_0.to_dict(), indent=2))
            print("PREFILL_90PCT_B4:", json.dumps(pf_90.to_dict(), indent=2))
            print("DECODE_20K_B32_V8:", json.dumps(dec_20k.to_dict(), indent=2))

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
