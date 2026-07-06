"""Correctness and performance test for Sm120Fp8GroupedGemmExecutor.

Run with:
    python rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/test/sm120_fp8_grouped_gemm_test.py
"""

import time

import torch


def _skip_if_not_sm120():
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12):
        print("SKIP: SM120 hardware required")
        return True
    return False


def _make_fp8_weights(E, N, K, device="cuda"):
    w13_fp8 = (torch.rand(E, N, K, device=device, dtype=torch.float32) * 2 - 1).to(
        torch.float8_e4m3fn
    )
    w13_scale = (
        torch.ones(E, N // 128, K // 128, device=device, dtype=torch.float32) * 0.1
    )
    w2_fp8 = (torch.rand(E, K, N // 2, device=device, dtype=torch.float32) * 2 - 1).to(
        torch.float8_e4m3fn
    )
    w2_scale = (
        torch.ones(E, K // 128, (N // 2) // 128, device=device, dtype=torch.float32)
        * 0.1
    )
    return w13_fp8, w13_scale, w2_fp8, w2_scale


def _run_grouped_gemm_executor(E, K, N, M, top_k, device="cuda"):
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
        ExpertForwardPayload,
        ExpertTokensMetadata,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_fp8_grouped_gemm_executor import (
        Sm120Fp8GroupedGemmExecutor,
    )
    from rtp_llm.ops.compute_ops import trt_fp8_quantize_128

    w13_fp8, w13_scale, w2_fp8, w2_scale = _make_fp8_weights(E, N, K, device)

    hidden_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1
    hidden_fp8, hidden_scale = trt_fp8_quantize_128(hidden_bf16, False)

    topk_ids = torch.zeros(M, top_k, device=device, dtype=torch.int64)
    for i in range(M):
        topk_ids[i, 0] = i % E
        for k in range(1, top_k):
            topk_ids[i, k] = (i + k) % E
    topk_weights = torch.ones(M, top_k, device=device, dtype=torch.float32) / top_k

    expert_num_tokens = torch.zeros(E, device=device, dtype=torch.int32)
    for expert_id in range(E):
        expert_num_tokens[expert_id] = (topk_ids == expert_id).sum().item()

    executor = Sm120Fp8GroupedGemmExecutor.__new__(Sm120Fp8GroupedGemmExecutor)
    executor.ep_size = 1
    executor.ep_rank = 0
    executor.num_experts = E
    executor.num_experts_per_partition = E
    executor.top_k = top_k
    executor.w13_weight = w13_fp8
    executor.w2_weight = w2_fp8
    executor.w13_scale = w13_scale
    executor.w2_scale = w2_scale
    executor.E = E
    executor.N = N
    executor.K = K
    executor.inter_size = N // 2
    executor.BLOCK_SIZE = 128
    executor.EXPERT_ALIGNMENT = 128

    payload = ExpertForwardPayload(
        expert_x=hidden_fp8,
        expert_x_scale=hidden_scale,
        expert_topk_ids=topk_ids,
        expert_topk_weights=topk_weights,
        expert_tokens_meta=ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expected_m=max(1, M * top_k // E),
        ),
    )
    return executor.execute(
        payload,
        activation="silu",
        expert_map=None,
        a2_scale=None,
        apply_router_weight_on_input=False,
        extra_expert_args=None,
    )


def test_invoke_grouped_gemm_basic():
    from rtp_llm.models_py.triton_kernels.moe.fp8_grouped_gemm import (
        invoke_sm120_fp8_grouped_gemm,
    )

    E, max_T, K, N = 4, 128, 256, 512
    A = torch.zeros(E, max_T, K, device="cuda", dtype=torch.float8_e4m3fn)
    A_sf = torch.ones(E, max_T, K // 128, device="cuda", dtype=torch.float32)
    B = torch.zeros(E, N, K, device="cuda", dtype=torch.float8_e4m3fn)
    B_sf = torch.ones(E, N // 128, K // 128, device="cuda", dtype=torch.float32)
    expert_num_tokens = torch.tensor([16, 32, 8, 64], device="cuda", dtype=torch.int32)
    C = torch.empty(E, max_T, N, device="cuda", dtype=torch.bfloat16)

    invoke_sm120_fp8_grouped_gemm(A, A_sf, B, B_sf, expert_num_tokens, C)

    assert C.shape == (E, max_T, N)
    assert not C.isnan().any(), "Output contains NaN"
    assert not C.isinf().any(), "Output contains Inf"
    print("  test_invoke_grouped_gemm_basic PASS")


def test_grouped_gemm_zero_input():
    from rtp_llm.models_py.triton_kernels.moe.fp8_grouped_gemm import (
        invoke_sm120_fp8_grouped_gemm,
    )

    E, max_T, K, N = 2, 128, 256, 512
    A = torch.zeros(E, max_T, K, device="cuda", dtype=torch.float8_e4m3fn)
    A_sf = torch.ones(E, max_T, K // 128, device="cuda", dtype=torch.float32)
    B = torch.zeros(E, N, K, device="cuda", dtype=torch.float8_e4m3fn)
    B_sf = torch.ones(E, N // 128, K // 128, device="cuda", dtype=torch.float32)
    expert_num_tokens = torch.tensor([64, 64], device="cuda", dtype=torch.int32)
    C = torch.ones(E, max_T, N, device="cuda", dtype=torch.bfloat16)

    invoke_sm120_fp8_grouped_gemm(A, A_sf, B, B_sf, expert_num_tokens, C)

    for e in range(E):
        n_tok = expert_num_tokens[e].item()
        assert (
            C[e, :n_tok].abs().max().item() == 0.0
        ), f"Expert {e}: expected zero output for zero inputs"
    print("  test_grouped_gemm_zero_input PASS")


def test_grouped_gemm_empty_expert():
    from rtp_llm.models_py.triton_kernels.moe.fp8_grouped_gemm import (
        invoke_sm120_fp8_grouped_gemm,
    )

    E, max_T, K, N = 2, 128, 256, 512
    A = torch.randn(E, max_T, K, device="cuda").to(torch.float8_e4m3fn)
    A_sf = torch.ones(E, max_T, K // 128, device="cuda", dtype=torch.float32)
    B = torch.randn(E, N, K, device="cuda").to(torch.float8_e4m3fn)
    B_sf = torch.ones(E, N // 128, K // 128, device="cuda", dtype=torch.float32)
    expert_num_tokens = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    sentinel = 999.0
    C = torch.full((E, max_T, N), sentinel, device="cuda", dtype=torch.bfloat16)

    invoke_sm120_fp8_grouped_gemm(A, A_sf, B, B_sf, expert_num_tokens, C)

    assert (C[0] == sentinel).all(), "Expert 0 (empty) rows should be unchanged"
    print("  test_grouped_gemm_empty_expert PASS")


def test_executor_output_finite(M, E, K, N, top_k):
    result = _run_grouped_gemm_executor(E, K, N, M, top_k)
    out = result.fused_expert_output
    assert out.shape == (M, K), f"Shape mismatch: {out.shape} vs ({M}, {K})"
    assert not out.isnan().any(), "Output contains NaN"
    assert not out.isinf().any(), "Output contains Inf"
    print(f"  test_executor_output_finite M={M} E={E} K={K} N={N} top_k={top_k} PASS")


def test_benchmark(E=128, K=2048, N=6144, top_k=8):
    WARMUP, RUNS = 5, 20
    print(f"\n  Benchmark: E={E} K={K} N={N} top_k={top_k}")
    print(f"  {'M':>6}  {'grouped_ms':>12}  {'note'}")
    for label, M in [
        ("decode_1", 1),
        ("decode_4", 4),
        ("decode_8", 8),
        ("decode_16", 16),
        ("decode_32", 32),
        ("prefill_64", 64),
        ("prefill_128", 128),
    ]:
        for _ in range(WARMUP):
            _run_grouped_gemm_executor(E, K, N, M, top_k)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(RUNS):
            _run_grouped_gemm_executor(E, K, N, M, top_k)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / RUNS * 1000
        print(f"  {M:>6}  {ms:>12.3f}ms  ({label})")


if __name__ == "__main__":
    if _skip_if_not_sm120():
        exit(0)

    print(
        f"GPU: {torch.cuda.get_device_name(0)}, SM: {torch.cuda.get_device_capability()}"
    )
    print()
    print("=== Kernel unit tests ===")
    test_invoke_grouped_gemm_basic()
    test_grouped_gemm_zero_input()
    test_grouped_gemm_empty_expert()

    print()
    print("=== Executor correctness tests ===")
    # K must be divisible by 512 (ep_gather BLOCK_D=512 constraint)
    # N must be divisible by 256 (two halves each need 128 alignment)
    for M, E, K, N, top_k in [
        (8, 8, 512, 1024, 2),
        (32, 8, 512, 1024, 2),
        (8, 16, 1024, 2048, 4),
    ]:
        test_executor_output_finite(M, E, K, N, top_k)

    print()
    print("=== Performance benchmark (Qwen3-30B-A3B dims) ===")
    test_benchmark()

    print()
    print("ALL TESTS PASSED")
