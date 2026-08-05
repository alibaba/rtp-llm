"""DeepGEMM FP8/FP4 MQA logits benchmark for CP=8 zigzag rank 0.

The workload matches GLM-5 indexer prefill:
  * 32 query heads, head_dim=128
  * one causal sequence
  * rank-local Q length = full KV length / 8
  * rank 0 owns the first and last half-chunks under zigzag CP
"""

import statistics

import torch

CP_SIZE = 8
NUM_HEADS = 32
HEAD_DIM = 128
WARMUP = 10
MEASURE = 30
KV_LENS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
PACKED_UE8M0_ONE = 0x7F7F7F7F


def rank0_zigzag_positions(kv_len: int) -> torch.Tensor:
    q_len = kv_len // CP_SIZE
    half = q_len // 2
    return torch.cat(
        (
            torch.arange(half, dtype=torch.int32, device="cuda"),
            torch.arange(kv_len - half, kv_len, dtype=torch.int32, device="cuda"),
        )
    )


def benchmark(call) -> tuple[float, float, float]:
    for _ in range(WARMUP):
        out = call()
        del out
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE)]
    for start, end in zip(starts, ends):
        start.record()
        out = call()
        end.record()
        del out
    torch.cuda.synchronize()

    elapsed_us = sorted(
        start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)
    )
    return (
        statistics.median(elapsed_us),
        elapsed_us[int(0.9 * (len(elapsed_us) - 1))],
        min(elapsed_us),
    )


def run_shape(kv_len: int) -> dict[str, float]:
    import deep_gemm

    q_len = kv_len // CP_SIZE
    positions = rank0_zigzag_positions(kv_len)
    ks = torch.zeros(q_len, dtype=torch.int32, device="cuda")
    ke = positions + 1
    weights = torch.ones(q_len, NUM_HEADS, dtype=torch.float32, device="cuda")

    q_fp8 = torch.zeros(
        q_len,
        NUM_HEADS,
        HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    k_fp8 = torch.zeros(kv_len, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    k_scale_fp8 = torch.ones(kv_len, dtype=torch.float32, device="cuda")

    fp8_median, fp8_p90, fp8_min = benchmark(
        lambda: deep_gemm.fp8_fp4_mqa_logits(
            (q_fp8, None),
            (k_fp8, k_scale_fp8),
            weights,
            ks,
            ke,
            clean_logits=False,
        )
    )
    legacy_fp8_median, legacy_fp8_p90, legacy_fp8_min = benchmark(
        lambda: deep_gemm.fp8_mqa_logits(
            q_fp8,
            (k_fp8, k_scale_fp8),
            weights,
            ks,
            ke,
            clean_logits=False,
        )
    )
    del q_fp8, k_fp8, k_scale_fp8
    torch.cuda.empty_cache()

    q_fp4 = torch.zeros(
        q_len,
        NUM_HEADS,
        HEAD_DIM // 2,
        dtype=torch.int8,
        device="cuda",
    )
    q_scale_fp4 = torch.full(
        (q_len, NUM_HEADS),
        PACKED_UE8M0_ONE,
        dtype=torch.int32,
        device="cuda",
    )
    k_fp4 = torch.zeros(kv_len, HEAD_DIM // 2, dtype=torch.int8, device="cuda")
    k_scale_fp4 = torch.full(
        (kv_len,),
        PACKED_UE8M0_ONE,
        dtype=torch.int32,
        device="cuda",
    )

    fp4_median, fp4_p90, fp4_min = benchmark(
        lambda: deep_gemm.fp8_fp4_mqa_logits(
            (q_fp4, q_scale_fp4),
            (k_fp4, k_scale_fp4),
            weights,
            ks,
            ke,
            clean_logits=False,
        )
    )
    del q_fp4, q_scale_fp4, k_fp4, k_scale_fp4
    del positions, ks, ke, weights
    torch.cuda.empty_cache()

    return {
        "kv_len": kv_len,
        "q_len": q_len,
        "fp8_us": fp8_median,
        "legacy_fp8_us": legacy_fp8_median,
        "fp4_us": fp4_median,
        "speedup": fp8_median / fp4_median,
        "fp8_p90_us": fp8_p90,
        "legacy_fp8_p90_us": legacy_fp8_p90,
        "fp4_p90_us": fp4_p90,
        "fp8_min_us": fp8_min,
        "legacy_fp8_min_us": legacy_fp8_min,
        "fp4_min_us": fp4_min,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() < (10, 0):
        raise RuntimeError("FP4 MQA logits requires Blackwell SM100+")

    import deep_gemm

    print(f"deep_gemm={deep_gemm.__file__}")
    print(
        "kv_len,q_len,fp8_unified_us,fp8_legacy_us,fp4_us,speedup,"
        "fp8_p90_us,fp8_legacy_p90_us,fp4_p90_us,"
        "fp8_min_us,fp8_legacy_min_us,fp4_min_us"
    )
    for kv_len in KV_LENS:
        row = run_shape(kv_len)
        print(
            f"{row['kv_len']},{row['q_len']},"
            f"{row['fp8_us']:.2f},{row['legacy_fp8_us']:.2f},"
            f"{row['fp4_us']:.2f},"
            f"{row['speedup']:.3f},"
            f"{row['fp8_p90_us']:.2f},{row['legacy_fp8_p90_us']:.2f},"
            f"{row['fp4_p90_us']:.2f},"
            f"{row['fp8_min_us']:.2f},{row['legacy_fp8_min_us']:.2f},"
            f"{row['fp4_min_us']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
