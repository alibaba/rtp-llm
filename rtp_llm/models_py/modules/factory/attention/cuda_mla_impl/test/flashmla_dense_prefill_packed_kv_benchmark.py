import argparse
import gc
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple
from unittest import mock

_LOCAL_DEEP_GEMM_PATH = os.environ.get("RTP_LOCAL_DEEP_GEMM_PATH")
if _LOCAL_DEEP_GEMM_PATH:
    sys.path.insert(0, _LOCAL_DEEP_GEMM_PATH)

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    MlaFlashMLAPrefillOp,
)
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
    CudaF16Linear,
)


def measure_cuda_events(
    fn: Callable[[], object], warmup: int, iterations: int
) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def max_abs_diff_chunked(actual: torch.Tensor, expected: torch.Tensor) -> float:
    result = 0.0
    for start in range(0, actual.shape[0], 4096):
        end = min(start + 4096, actual.shape[0])
        result = max(
            result,
            float(
                (actual[start:end].float() - expected[start:end].float())
                .abs()
                .max()
            ),
        )
    return result


def allocated_peak_gib(fn: Callable[[], object]) -> Tuple[float, object]:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    output = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - baseline) / (1024**3), output


def profile_cuda(name: str, fn: Callable[[], object], output_dir: Path) -> None:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        fn()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(output_dir / f"{name}.json"))
    (output_dir / f"{name}.txt").write_text(
        profiler.key_averages().table(
            sort_by="cuda_time_total", max_name_column_width=160, row_limit=40
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-tokens", type=int, default=1_024_000)
    parser.add_argument("--query-tokens", type=int, default=20_480)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--attention-iterations", type=int, default=3)
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("this benchmark requires SM100")

    import flash_mla.cuda as flash_mla_cuda

    torch.manual_seed(789)
    device = torch.device("cuda")
    heads = 12
    kv_lora_rank = 512
    k_nope_dim = 128
    k_pe_dim = 64
    v_dim = 128

    compressed_kv = torch.randn(
        (args.kv_tokens, kv_lora_rank), device=device, dtype=torch.bfloat16
    )
    k_pe = torch.randn(
        (args.kv_tokens, k_pe_dim), device=device, dtype=torch.bfloat16
    )
    q = torch.randn(
        (args.query_tokens, heads, k_nope_dim + k_pe_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    checkpoint_weight = torch.randn(
        (kv_lora_rank, heads * (k_nope_dim + v_dim)),
        device=device,
        dtype=torch.bfloat16,
    )
    linear = CudaF16Linear(checkpoint_weight)
    head_splits = (k_nope_dim, k_pe_dim, v_dim)

    import deep_gemm

    if not callable(deep_gemm.bf16_gemm_nt_skip_head_mid):
        raise RuntimeError("DeepGEMM bf16_gemm_nt_skip_head_mid is required")

    op = object.__new__(MlaFlashMLAPrefillOp)
    op.flash_mla_cuda = flash_mla_cuda
    op.num_heads = heads
    op.kv_lora_rank = kv_lora_rank
    op.qk_nope_head_dim = k_nope_dim
    op.qk_rope_head_dim = k_pe_dim
    op.v_head_dim = v_dim
    op.weights = [{}]
    op.quant_config = None
    op.qo_indptr = torch.tensor(
        [0, args.query_tokens], device=device, dtype=torch.int32
    )
    op.kv_indptr = torch.tensor(
        [0, args.kv_tokens], device=device, dtype=torch.int32
    )
    op.max_q_len = args.query_tokens
    op.max_kv_len = args.kv_tokens
    op.scale = (k_nope_dim + k_pe_dim) ** -0.5

    def baseline_project_kv() -> Tuple[torch.Tensor, torch.Tensor]:
        expanded_dim = k_nope_dim + v_dim
        kv = linear(compressed_kv).view(args.kv_tokens, heads, expanded_dim)
        k = compressed_kv.new_empty(
            args.kv_tokens, heads, k_nope_dim + k_pe_dim
        )
        k[..., :k_nope_dim].copy_(kv[..., :k_nope_dim])
        k[..., k_nope_dim:].copy_(k_pe.view(args.kv_tokens, 1, k_pe_dim))
        return k, kv[..., k_nope_dim:]

    def packed_project_kv() -> Tuple[torch.Tensor, torch.Tensor]:
        return op._project_kv(compressed_kv, k_pe, 0)

    result = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "shape": {
            "kv_tokens": args.kv_tokens,
            "query_tokens": args.query_tokens,
            "heads": heads,
            "kv_lora_rank": kv_lora_rank,
            "head_splits": list(head_splits),
            "weight_stride": list(linear.weight.stride()),
        },
        "timing": {},
        "peak_activation_gib": {},
        "precision": {},
    }
    if _LOCAL_DEEP_GEMM_PATH:
        result["deep_gemm_path"] = deep_gemm.__file__
        if not deep_gemm.__file__.startswith(_LOCAL_DEEP_GEMM_PATH):
            raise RuntimeError(f"unexpected DeepGEMM package: {deep_gemm.__file__}")

    with mock.patch.object(
        LinearFactory, "create_linear_from_weights", return_value=linear
    ):
        # Compile and initialize both backends before sampling allocator peaks.
        # Otherwise the first-call cuBLAS/DeepGEMM state is counted as activation.
        for projection in (baseline_project_kv, packed_project_kv):
            warmup_kv = projection()
            torch.cuda.synchronize()
            del warmup_kv
        gc.collect()

        baseline_peak, baseline_kv = allocated_peak_gib(baseline_project_kv)
        baseline_k, baseline_v = baseline_kv
        result["timing"]["baseline_projection"] = measure_cuda_events(
            baseline_project_kv, args.warmup, args.iterations
        )
        result["peak_activation_gib"]["baseline_projection"] = baseline_peak

        packed_peak, packed_kv = allocated_peak_gib(packed_project_kv)
        packed_k, packed_v = packed_kv
        result["timing"]["packed_projection"] = measure_cuda_events(
            packed_project_kv, args.warmup, args.iterations
        )
        result["peak_activation_gib"]["packed_projection"] = packed_peak

        result["precision"]["k_max_abs"] = max_abs_diff_chunked(
            packed_k, baseline_k
        )
        result["precision"]["v_max_abs"] = max_abs_diff_chunked(
            packed_v, baseline_v
        )
        result["layout"] = {
            "baseline_k_stride": list(baseline_k.stride()),
            "baseline_v_stride": list(baseline_v.stride()),
            "packed_k_stride": list(packed_k.stride()),
            "packed_v_stride": list(packed_v.stride()),
            "packed_kv_share_storage": packed_k.untyped_storage().data_ptr()
            == packed_v.untyped_storage().data_ptr(),
        }

        if result["precision"]["k_max_abs"] != 0.0:
            raise AssertionError(result["precision"])
        if result["precision"]["v_max_abs"] != 0.0:
            raise AssertionError(result["precision"])

        if not args.skip_attention:
            baseline_out = op._dense_attention(q, baseline_k, baseline_v)
            packed_out = op._dense_attention(q, packed_k, packed_v)
            torch.cuda.synchronize()
            result["precision"]["attention_max_abs"] = max_abs_diff_chunked(
                packed_out, baseline_out
            )
            if result["precision"]["attention_max_abs"] != 0.0:
                raise AssertionError(result["precision"])

            result["timing"]["baseline_attention"] = measure_cuda_events(
                lambda: op._dense_attention(q, baseline_k, baseline_v),
                args.warmup,
                args.attention_iterations,
            )
            result["timing"]["packed_attention"] = measure_cuda_events(
                lambda: op._dense_attention(q, packed_k, packed_v),
                args.warmup,
                args.attention_iterations,
            )

            del baseline_out, packed_out
            del baseline_k, baseline_v, packed_k, packed_v, baseline_kv, packed_kv
            gc.collect()
            torch.cuda.synchronize()

            result["timing"]["baseline_projection_attention"] = measure_cuda_events(
                lambda: op._dense_attention(q, *baseline_project_kv()),
                args.warmup,
                args.attention_iterations,
            )
            result["timing"]["packed_projection_attention"] = measure_cuda_events(
                lambda: op._dense_attention(q, *packed_project_kv()),
                args.warmup,
                args.attention_iterations,
            )

            baseline_e2e_ms = result["timing"][
                "baseline_projection_attention"
            ]["median_ms"]
            packed_e2e_ms = result["timing"]["packed_projection_attention"][
                "median_ms"
            ]
            result["projection_attention_speedup"] = (
                baseline_e2e_ms / packed_e2e_ms
            )

        baseline_ms = result["timing"]["baseline_projection"]["median_ms"]
        packed_ms = result["timing"]["packed_projection"]["median_ms"]
        result["projection_speedup"] = baseline_ms / packed_ms

        if args.profile:
            profile_cuda(
                "baseline_projection", baseline_project_kv, args.output_dir
            )
            profile_cuda("packed_projection", packed_project_kv, args.output_dir)

    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
