"""K3.1 NVFP4 quantization A/B, not a model/GEMM benchmark.

Run the SM103 Bazel target with the repository's remote-execution config.
Inputs are seeded synthetic tensors, not checkpoint weights or traffic samples.
K=7168: KDA/DSA input and AttnRes; K=1536: DSA Q-B; K=128: KDA F_b.
Weight rows=1024 comes from nvfp4_tensor.py's CPU streaming chunk. Activation
rows sweep decode, the Linear fast-path boundary (512/513), and large prefill.
K=16/32/96 are interface/alignment checks, not claimed model dimensions.

Candidates are the RTP C++ external-scale entry and RTP's Triton dynamic entry.
Production weight/Linear entrypoints are also checked with FlashInfer quant
disabled. FlashInfer is the oracle;
removing its GEMM, attention or other dependencies is outside this experiment.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models.kimi_k3.nvfp4_tensor import (
    nvfp4_global_scale,
    quantize_nvfp4_weight,
    unswizzle_nvfp4_scales,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
    CudaFp4GEMMLinear,
)
from rtp_llm.models_py.triton_kernels.common.nvfp4 import (
    quantize_dynamic_nvfp4,
    quantize_nvfp4_from_partials,
    quantize_strided_nvfp4,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


@contextmanager
def forbid_flashinfer_quantization():
    with patch(
        "flashinfer.fp4_quantize", side_effect=AssertionError("FlashInfer quant called")
    ) as forbidden:
        # Patch both a lazy import and a module-level imported alias. The old
        # Linear implementation used the latter, which patching flashinfer
        # alone would not intercept.
        with patch(
            CudaFp4GEMMLinear.__module__ + ".fp4_quantize", forbidden, create=True
        ):
            yield


def buffers(x):
    m, k = x.shape
    return (
        torch.empty((m, k // 2), dtype=torch.uint8, device=x.device),
        torch.empty(
            (((m + 127) // 128) * 128, ((k // 16 + 3) // 4) * 4),
            dtype=torch.uint8,
            device=x.device,
        ),
    )


def native(x, inverse, out=None):
    out = buffers(x) if out is None else out
    rtp_llm_ops.nvfp4_quantize(x, inverse, *out)
    return out


def dequantize(codes, scales, outer, k):
    logical = unswizzle_nvfp4_scales(scales, codes.shape[0], k // 16)
    sf = logical.contiguous().view(torch.float8_e4m3fn).float()
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=codes.device,
    )
    nibbles = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    return lookup[nibbles] * sf.repeat_interleave(16, -1) * outer.reshape(-1, 1)


def compare(x, left, right, outer, right_outer=None):
    m, k = x.shape
    ls = unswizzle_nvfp4_scales(left[1], m, k // 16)
    rs = unswizzle_nvfp4_scales(right[1], m, k // 16)
    lq = dequantize(*left, outer, k)
    rq = dequantize(*right, outer if right_outer is None else right_outer, k)
    err = (lq - x.float()).square().mean().sqrt()
    rms = x.float().square().mean().sqrt().clamp_min(1e-30)
    return {
        "code_byte_mismatches": (left[0] != right[0]).sum().item(),
        "logical_scale_byte_mismatches": (ls != rs).sum().item(),
        "dequant_max_abs_difference": (lq - rq).abs().max().item(),
        "candidate_relative_rmse": (err / rms).item(),
        "reference_relative_rmse": (
            (rq - x.float()).square().mean().sqrt() / rms
        ).item(),
        "candidate_finite": bool(torch.isfinite(lq).all().item()),
    }


def latency(fn, *, repeats=9, iterations=40):
    # Compilation, allocation of input tensors and warmup are outside timing.
    # Callers distinguish the allocating API and the preallocated kernel API.
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    ordered = sorted(samples)
    result = {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
        "samples_us": samples,
        "iterations_per_sample": iterations,
        "method": "CUDA events, eager launches, 5 warmups; no JIT compilation",
    }
    # Capture repeated calls in one graph to remove Python launch gaps from
    # the GPU-throughput measurement. Captured allocation is NOT host allocator
    # latency; keep the eager number above and identify both measurements.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            graph_output = fn()
    graph.replay()
    torch.cuda.synchronize()
    graph_samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        graph_samples.append(start.elapsed_time(end) * 1000 / iterations)
    del graph_output
    result["cuda_graph_median_us"] = statistics.median(graph_samples)
    result["cuda_graph_samples_us"] = graph_samples
    return result


def make_input(m, k, dtype, layout="dense", pattern="random"):
    stride = k if layout == "dense" else k + 17
    offset = 1 if layout == "misaligned" else 0
    storage = torch.randn((m * stride + offset,), device="cuda", dtype=dtype)
    x = storage[offset:].as_strided((m, k), (stride, 1))
    if pattern == "zero":
        x.zero_()
    elif pattern == "tiny":
        x.mul_(1e-12)
    elif pattern == "underflow":
        x.mul_(1e-12)
        x[0, :16] = 1
    elif pattern == "ties":
        # Deliberate half-way values; alternating signs test nibble packing.
        v = torch.tensor(
            [
                0,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5,
                6,
                -0.25,
                -0.75,
                -1.25,
                -1.75,
                -2.5,
                -3.5,
                -5,
            ],
            device="cuda",
            dtype=dtype,
        )
        x.copy_(v.repeat(m, k // 16))
    return x


def pack_case(
    m, k, dtype, *, layout="dense", pattern="random", role="activation", timed=True
):
    from flashinfer import fp4_quantize

    x = make_input(m, k, dtype, layout, pattern)
    outer = nvfp4_global_scale(x)
    inverse = outer.reciprocal()
    out = buffers(x)
    start = time.perf_counter()
    reference = fp4_quantize(x.contiguous(), inverse)
    torch.cuda.synchronize()
    cold_wall_s = time.perf_counter() - start
    candidate = native(x, inverse, out)
    torch.cuda.synchronize()
    result = {
        "case": f"{role}/{m}x{k}/{dtype}/{layout}/{pattern}",
        "shape": [m, k],
        "dtype": str(dtype),
        "stride": list(x.stride()),
        "role": role,
        "pattern": pattern,
        "scale_mode": "one shared tensor scale",
        "flashinfer_first_call_wall_s": cold_wall_s,
        **compare(x, candidate, reference, outer),
    }
    # Require exact logical bytes for the external-scale replacement, including
    # rounding/underflow diagnostics. Report all failures instead of stopping
    # at the first shape so one mismatch cannot hide missing coverage.
    result["pass"] = (
        result["code_byte_mismatches"] == 0
        and result["logical_scale_byte_mismatches"] == 0
        and result["candidate_finite"]
    )
    if timed:
        result["timing"] = {
            "native_pack_preallocated": latency(lambda: native(x, inverse, out)),
            "native_pack_allocating": latency(lambda: native(x, inverse)),
            "flashinfer_pack_allocating": latency(
                lambda: fp4_quantize(x.contiguous(), inverse)
            ),
            "native_with_calibration": latency(
                lambda: native(x, nvfp4_global_scale(x).reciprocal())
            ),
            "flashinfer_with_calibration": latency(
                lambda: fp4_quantize(x.contiguous(), nvfp4_global_scale(x).reciprocal())
            ),
        }
        if dtype == torch.bfloat16 and k % 64 == 0:
            fused = quantize_strided_nvfp4(x)
            result["existing_native_fused_amax"] = {
                **compare(x, fused[:2], reference, fused.scale, outer),
                "global_scale_abs_difference": (fused.scale - outer).abs().max().item(),
                "timing": latency(lambda: quantize_strided_nvfp4(x)),
            }
            if x.is_contiguous():
                # AttnRes/DSA RMSNorm supply maxima. This measures their
                # existing packing stage, not the producer or RMSNorm itself.
                count = torch.cuda.get_device_properties(x.device).multi_processor_count
                row_max = x.abs().amax(dim=1).float()
                partials = (
                    torch.nn.functional.pad(
                        row_max, (0, ((m + count - 1) // count) * count - m)
                    )
                    .view(count, -1)
                    .amax(dim=1)
                    .contiguous()
                )
                packed = quantize_nvfp4_from_partials(x, partials)
                result["existing_triton_producer_amax"] = {
                    **compare(x, packed[:2], reference, packed.scale, outer),
                    "global_scale_abs_difference": (packed.scale - outer)
                    .abs()
                    .max()
                    .item(),
                    "timing": latency(
                        lambda: quantize_nvfp4_from_partials(x, partials)
                    ),
                }
    if dtype == torch.bfloat16 and k % 64 == 0 and x.is_contiguous():
        dynamic = quantize_dynamic_nvfp4(x)
        dynamic_metrics = compare(x, dynamic[:2], reference, dynamic.scale, outer)
        scale_diff = (dynamic.scale - outer).abs().max().item()
        result["rtp_triton_dynamic"] = {
            **dynamic_metrics,
            "global_scale_abs_difference": scale_diff,
        }
        if timed:
            result["rtp_triton_dynamic"]["timing"] = latency(
                lambda: quantize_dynamic_nvfp4(x)
            )
        result["pass"] = result["pass"] and (
            dynamic_metrics["code_byte_mismatches"] == 0
            and dynamic_metrics["logical_scale_byte_mismatches"] == 0
            and dynamic_metrics["candidate_finite"]
            and scale_diff == 0
        )
    return result


def rowwise_case(m=32, k=7168):
    from flashinfer import fp4_quantize

    x = make_input(m, k, torch.bfloat16)
    outer = x.abs().amax(dim=1).float().clamp_min(1e-8) / (6 * 448)
    inverse = outer.reciprocal()
    candidate = native(x, inverse)
    # A loop is a correctness oracle only, never a claimed performance peer
    # for AttnRes's fused rowwise producer.
    codes, logical = [], []
    for row in range(m):
        q, s = fp4_quantize(x[row : row + 1], inverse[row : row + 1])
        codes.append(q)
        logical.append(unswizzle_nvfp4_scales(s, 1, k // 16))
    from rtp_llm.models.kimi_k3.nvfp4_tensor import swizzle_nvfp4_scales

    ref = torch.cat(codes), swizzle_nvfp4_scales(torch.cat(logical))
    metrics = compare(x, candidate, ref, outer)
    return {
        "case": "attnres/rowwise/32x7168/BF16",
        "shape": [m, k],
        "scale_mode": "one scale per row; same semantics as decode AttnRes",
        **metrics,
        "pass": metrics["code_byte_mismatches"] == 0
        and metrics["logical_scale_byte_mismatches"] == 0,
        "timing": latency(lambda: native(x, inverse, candidate)),
        "scope": "packing only; fused AttnRes producer is not replaced or timed here",
    }


def streaming_case(k, dtype):
    from flashinfer import fp4_quantize

    # Two complete 1024-row chunks and a ragged tail. One scale must be
    # retained for all chunks; per-chunk recalibration would change weights.
    x = make_input(2305, k, dtype)
    x[0, 0] = 32
    outer = nvfp4_global_scale(x)
    inverse = outer.reciprocal()
    reference = fp4_quantize(x, inverse)
    candidate = buffers(x)
    for begin in range(0, x.shape[0], 1024):
        end = min(begin + 1024, x.shape[0])
        q, s = native(x[begin:end], inverse)
        candidate[0][begin:end].copy_(q)
        candidate[1][begin : begin + s.shape[0]].copy_(s)
    metrics = compare(x, candidate, reference, outer)
    # Also cross the real CPU-staging boundary, but exclude H2D/D2H from the
    # kernel latency measurements above (PCIe is not quantization compute).
    cpu = x.cpu()
    q_cpu = torch.empty_like(candidate[0], device="cpu")
    s_cpu = torch.empty_like(candidate[1], device="cpu")
    for begin in range(0, x.shape[0], 1024):
        end = min(begin + 1024, x.shape[0])
        q, s = native(cpu[begin:end].to("cuda"), inverse)
        q_cpu[begin:end].copy_(q)
        s_cpu[begin : begin + s.shape[0]].copy_(s)
    cpu_match = torch.equal(q_cpu, reference[0].cpu()) and torch.equal(
        unswizzle_nvfp4_scales(s_cpu, 2305, k // 16),
        unswizzle_nvfp4_scales(reference[1], 2305, k // 16).cpu(),
    )
    return {
        "case": f"weight_stream/2305x{k}/{dtype}",
        "shape": [2305, k],
        "chunk_rows": 1024,
        "shared_inverse_scale": inverse.item(),
        **metrics,
        "cpu_staging_exact": cpu_match,
        "pass": metrics["code_byte_mismatches"] == 0
        and metrics["logical_scale_byte_mismatches"] == 0
        and cpu_match,
    }


def buffer_contract_case():
    """Exercise row/K tails with guard bytes, without the FlashInfer oracle."""
    checked = []
    for dtype in (torch.bfloat16, torch.float16):
        for m, k in ((0, 128), (1, 16), (129, 32), (129, 96), (17, 1536), (4097, 1536)):
            x = make_input(m, k, dtype, layout="misaligned")
            sizes = (m * k // 2, ((m + 127) // 128 * 128) * ((k // 16 + 3) // 4 * 4))
            storage = [
                torch.full((n + 64,), 0xA5, dtype=torch.uint8, device="cuda")
                for n in sizes
            ]
            q = storage[0][32:-32].view(m, k // 2)
            sf = storage[1][32:-32].view((m + 127) // 128 * 128, (k // 16 + 3) // 4 * 4)
            native(x, torch.ones(1, device="cuda"), (q, sf))
            for s in storage:
                if not bool(((s[:32] == 0xA5) & (s[-32:] == 0xA5)).all().item()):
                    raise AssertionError(f"Guard overwritten for {m}x{k}/{dtype}")
            if m:
                logical = unswizzle_nvfp4_scales(sf, sf.shape[0], sf.shape[1])
                padding = torch.ones_like(logical, dtype=torch.bool)
                padding[:m, : k // 16] = False
                if not bool((logical[padding] == 0).all().item()):
                    raise AssertionError(
                        f"Scale padding is not zero for {m}x{k}/{dtype}"
                    )
            checked.append({"shape": [m, k], "dtype": str(dtype), "offset_elements": 1})
    return {"case": "native/buffer_contract", "checked": checked, "pass": True}


def dynamic_alignment_case():
    from flashinfer import fp4_quantize

    x = make_input(1, 7168, torch.bfloat16, layout="misaligned")
    outer = nvfp4_global_scale(x)
    # Match the existing Linear fallback's handling of misaligned singleton
    # slices. The new dynamic entry performs its own 32-byte alignment copy.
    ref = fp4_quantize(
        x.clone(memory_format=torch.contiguous_format), outer.reciprocal()
    )
    dynamic = quantize_dynamic_nvfp4(x)
    metrics = compare(x, dynamic[:2], ref, dynamic.scale, outer)
    scale_diff = (dynamic.scale - outer).abs().max().item()
    return {
        "case": "rtp_dynamic/misaligned_singleton/1x7168/BF16",
        **metrics,
        "global_scale_abs_difference": scale_diff,
        "pass": metrics["code_byte_mismatches"] == 0
        and metrics["logical_scale_byte_mismatches"] == 0
        and scale_diff == 0,
        "timing": latency(lambda: quantize_dynamic_nvfp4(x)),
    }


def production_entrypoints_case():
    from flashinfer import fp4_quantize

    checked = []
    for dtype in (torch.bfloat16, torch.float16):
        for source_device in ("cpu", "cuda"):
            source = torch.randn(2305, 128, dtype=dtype, device=source_device)
            outer = nvfp4_global_scale(source)
            gpu_source = source.to("cuda")
            gpu_outer = outer.to("cuda")
            reference = fp4_quantize(gpu_source, gpu_outer.reciprocal())
            with forbid_flashinfer_quantization():
                codes, scales, actual_outer = quantize_nvfp4_weight(
                    source, device=torch.device("cuda")
                )
            metrics = compare(
                gpu_source, (codes.to("cuda"), scales.to("cuda")), reference, gpu_outer
            )
            assert (
                metrics["code_byte_mismatches"] == 0
                and metrics["logical_scale_byte_mismatches"] == 0
            ), metrics
            assert torch.equal(actual_outer, outer)
            checked.append(f"weight/{source_device}/{dtype}/2305x128")
    for rows, width, dtype, layout in (
        (1, 128, torch.bfloat16, "dense"),
        (513, 7168, torch.bfloat16, "dense"),
        (513, 1536, torch.bfloat16, "strided"),
        (4096, 1536, torch.bfloat16, "dense"),
        (17, 1536, torch.float16, "dense"),
    ):
        x = make_input(rows, width, dtype, layout)
        outer = nvfp4_global_scale(x)
        reference = fp4_quantize(x.contiguous(), outer.reciprocal())
        with forbid_flashinfer_quantization():
            actual = CudaFp4GEMMLinear.quantize_dynamic(x)
        metrics = compare(x, actual[:2], reference, actual.scale, outer)
        assert (
            metrics["code_byte_mismatches"] == 0
            and metrics["logical_scale_byte_mismatches"] == 0
        ), metrics
        assert torch.equal(actual.scale, outer)
        checked.append(f"linear/{rows}x{width}/{dtype}/{layout}")
    return {"case": "production/no_flashinfer_quant", "checked": checked, "pass": True}


def main():
    output = (
        Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"))
        / "k31_nvfp4_quant.json"
    )
    report = {
        "schema": 1,
        "cases": [],
        "failures": [],
        "scope": "synthetic operator microbenchmark; no full-model accuracy/performance claim",
        "width_sources": {
            "7168": "KimiK3AttentionResidual BF16 SM10x specialization; KDA/DSA input",
            "1536": "DSA Q LoRA projection; native scaled_fp4_quant specialization",
            "128": "KDA F_a -> F_b latent; kda_attention.py fast path",
        },
        "non_goals": [
            "FlashInfer GEMM removal",
            "MoE MXFP4",
            "FP8 KV cache",
            "full AttnRes/RMSNorm producer timing",
        ],
    }

    def run(fn):
        try:
            result = fn()
            report["cases"].append(result)
            print(json.dumps({k: result[k] for k in ("case", "pass")}), flush=True)
            if not result["pass"]:
                report["failures"].append(result["case"])
        except Exception:
            report["failures"].append(traceback.format_exc())
            print(report["failures"][-1], flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU: benchmark did not run")
        capability = torch.cuda.get_device_capability()
        if capability != (10, 3):
            raise RuntimeError(f"Expected SM103 remote pool, got {capability}")
        torch.manual_seed(20261118)
        report["environment"] = {
            "gpu": torch.cuda.get_device_name(),
            "capability": capability,
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
            "cpu_arch": platform.machine(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": importlib.metadata.version("flashinfer-python"),
            "triton": importlib.metadata.version("triton"),
            "source_public": os.environ.get(
                "RTP_QUANT_BENCH_PUBLIC_SHA", "unspecified"
            ),
            "source_internal": os.environ.get(
                "RTP_QUANT_BENCH_INTERNAL_SHA", "unspecified"
            ),
            "source_manifest_sha256": os.environ.get(
                "RTP_QUANT_BENCH_SOURCE_SHA256", "unspecified"
            ),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "quantization": "group16 E2M1 + E4M3 128x4 + external FP32 inverse scale",
        }
        nvcc = Path("/usr/local/cuda/bin/nvcc")
        if nvcc.is_file():
            report["environment"]["nvcc"] = subprocess.run(
                [str(nvcc), "--version"], check=True, capture_output=True, text=True
            ).stdout.strip()
        run(buffer_contract_case)
        # Compile the FlashInfer oracle once and record a real failure if its
        # own dependencies cannot compile. No silent skip or patched oracle.
        from flashinfer import fp4_quantize

        start = time.perf_counter()
        fp4_quantize(
            torch.ones((128, 128), device="cuda", dtype=torch.bfloat16),
            torch.ones(1, device="cuda"),
        )
        torch.cuda.synchronize()
        report["flashinfer_initialization_wall_s"] = time.perf_counter() - start
        for dtype in (torch.bfloat16, torch.float16):
            for k in (128, 1536, 7168):
                run(
                    lambda k=k, dtype=dtype: pack_case(
                        1024, k, dtype, role="weight_chunk"
                    )
                )
                run(lambda k=k, dtype=dtype: streaming_case(k, dtype))
        for k in (128, 1536, 7168):
            for m in (1, 32, 128, 512, 513, 4096, 32768):
                run(lambda m=m, k=k: pack_case(m, k, torch.bfloat16))
            for layout in ("strided", "misaligned"):
                run(
                    lambda k=k, layout=layout: pack_case(
                        128, k, torch.bfloat16, layout=layout
                    )
                )
        # FP16 is a loader-supported source dtype. Exercise both dtypes through
        # the wide-row traversal, not only through the 1024-row CPU chunks.
        for k in (1536, 7168):
            run(
                lambda k=k: pack_case(
                    4096, k, torch.float16, role="weight_large_interface"
                )
            )
        for dtype in (torch.bfloat16, torch.float16):
            for k in (16, 32, 96):
                run(
                    lambda k=k, dtype=dtype: pack_case(
                        129, k, dtype, role="interface", timed=False
                    )
                )
            for pattern in ("zero", "tiny", "underflow", "ties"):
                run(
                    lambda dtype=dtype, pattern=pattern: pack_case(
                        129, 128, dtype, pattern=pattern, timed=False
                    )
                )
        run(rowwise_case)
        run(dynamic_alignment_case)
        run(production_entrypoints_case)
    except Exception:
        report["failures"].append(traceback.format_exc())
        print(report["failures"][-1], flush=True)
    finally:
        report["pass"] = not report["failures"] and bool(report["cases"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {output}; PASS={report['pass']}", flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
