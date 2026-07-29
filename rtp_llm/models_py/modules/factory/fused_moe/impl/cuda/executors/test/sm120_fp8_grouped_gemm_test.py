"""Correctness tests for Sm120Fp8GroupedGemmExecutor.

Run with:
    python rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/test/sm120_fp8_grouped_gemm_test.py
"""

import math
import unittest

import torch

from rtp_llm.models_py.utils.arch import is_sm12x

SM120_AVAILABLE = torch.cuda.is_available() and is_sm12x()


def _ceil_div(a, b):
    return math.ceil(a / b)


def _make_fp8_weights(E, N, K, device="cuda"):
    w13_fp8 = (torch.rand(E, N, K, device=device, dtype=torch.float32) * 2 - 1).to(
        torch.float8_e4m3fn
    )
    w13_scale = (
        torch.ones(
            E,
            _ceil_div(N, 128),
            _ceil_div(K, 128),
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    )
    w2_fp8 = (torch.rand(E, K, N // 2, device=device, dtype=torch.float32) * 2 - 1).to(
        torch.float8_e4m3fn
    )
    w2_scale = (
        torch.ones(
            E,
            _ceil_div(K, 128),
            _ceil_div(N // 2, 128),
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    )
    return w13_fp8, w13_scale, w2_fp8, w2_scale


def _make_grouped_gemm_executor_case(
    E, K, N, M, top_k, device="cuda", config_expert_num=None
):
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
        MoEConfigAdapter,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
        ExpertForwardPayload,
        ExpertTokensMetadata,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
        FusedMoEQuantConfig,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_fp8_grouped_gemm_executor import (
        Sm120Fp8GroupedGemmExecutor,
    )
    from rtp_llm.ops import MoeConfig, ParallelismConfig
    from rtp_llm.ops.compute_ops import trt_fp8_quantize_128
    from rtp_llm.utils.model_weight import W

    w13_fp8, w13_scale, w2_fp8, w2_scale = _make_fp8_weights(E, N, K, device)

    hidden_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1
    hidden_fp8, hidden_scale = trt_fp8_quantize_128(hidden_bf16, False)

    topk_ids = torch.zeros(M, top_k, device=device, dtype=torch.int64)
    for i in range(M):
        topk_ids[i, 0] = i % E
        for k in range(1, top_k):
            topk_ids[i, k] = (i + k) % E
    topk_weights = torch.arange(
        1, top_k + 1, device=device, dtype=torch.float32
    ).expand(M, -1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    expert_num_tokens = torch.zeros(E, device=device, dtype=torch.int32)
    for expert_id in range(E):
        expert_num_tokens[expert_id] = (topk_ids == expert_id).sum().item()

    model_config = ModelConfig()
    model_config.expert_num = E if config_expert_num is None else config_expert_num
    model_config.moe_k = top_k
    parallelism_config = ParallelismConfig()
    parallelism_config.ep_size = 1
    moe_config = MoeConfig()
    moe_config.masked_max_token_num = max(256, M * top_k)
    config = MoEConfigAdapter(model_config, parallelism_config, moe_config)
    executor = Sm120Fp8GroupedGemmExecutor(
        config,
        FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128]),
        {
            W.moe_w1: w13_fp8,
            W.moe_w2: w2_fp8,
            W.moe_s1: w13_scale,
            W.moe_s2: w2_scale,
        },
    )

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
    return executor, payload


def _execute_grouped_gemm_executor(executor, payload):
    return executor.execute(
        payload,
        activation="silu",
        expert_map=None,
        a2_scale=None,
        apply_router_weight_on_input=False,
        extra_expert_args=None,
    )


def _run_grouped_gemm_executor(E, K, N, M, top_k, device="cuda"):
    executor, payload = _make_grouped_gemm_executor_case(E, K, N, M, top_k, device)
    return _execute_grouped_gemm_executor(executor, payload)


def _check_grouped_gemm_config_selector(test_case):
    from rtp_llm.models_py.triton_kernels.moe.fp8_grouped_gemm import (
        GEMM_ROLE_DOWN,
        GEMM_ROLE_GATE_UP,
        Sm120Fp8GroupedGemmConfig,
        _select_sm120_fp8_grouped_gemm_config,
    )

    cases = [
        (
            128,
            6144,
            GEMM_ROLE_GATE_UP,
            None,
            Sm120Fp8GroupedGemmConfig(16, 128, 128, 4, 3),
        ),
        (
            128,
            2048,
            GEMM_ROLE_DOWN,
            768,
            Sm120Fp8GroupedGemmConfig(16, 128, 128, 4, 3),
        ),
        (
            1024,
            2048,
            GEMM_ROLE_DOWN,
            768,
            Sm120Fp8GroupedGemmConfig(16, 128, 128, 4, 3),
        ),
        (
            2020,
            2048,
            GEMM_ROLE_DOWN,
            768,
            Sm120Fp8GroupedGemmConfig(64, 128, 128, 8, 3),
        ),
        (
            512,
            2048,
            GEMM_ROLE_DOWN,
            1600,
            Sm120Fp8GroupedGemmConfig(64, 128, 128, 8, 3),
        ),
        (
            512,
            6144,
            GEMM_ROLE_GATE_UP,
            None,
            Sm120Fp8GroupedGemmConfig(64, 128, 128, 8, 3),
        ),
        (
            1024,
            6144,
            GEMM_ROLE_GATE_UP,
            None,
            Sm120Fp8GroupedGemmConfig(64, 128, 128, 8, 3),
        ),
        (
            1024,
            4096,
            GEMM_ROLE_DOWN,
            2048,
            Sm120Fp8GroupedGemmConfig(64, 128, 128, 8, 3),
        ),
    ]
    for tuning_token_count, n, role, k, expected in cases:
        config = _select_sm120_fp8_grouped_gemm_config(tuning_token_count, n, role, k)
        test_case.assertEqual(config, expected)
        test_case.assertEqual(config.block_n, 128)
        test_case.assertEqual(config.block_k, 128)

    with test_case.assertRaisesRegex(
        ValueError, "Unsupported SM120 FP8 grouped GEMM role"
    ):
        _select_sm120_fp8_grouped_gemm_config(128, 6144, "invalid")


def _check_invoke_grouped_gemm_basic(test_case):
    from rtp_llm.models_py.triton_kernels.moe.fp8_grouped_gemm import (
        invoke_sm120_fp8_grouped_gemm,
    )

    E, max_T, K, N = 4, 128, 256, 512
    A = torch.randn(E, max_T, K, device="cuda").to(torch.float8_e4m3fn)
    A_sf = torch.ones(E, max_T, K // 128, device="cuda", dtype=torch.float32)
    B = torch.randn(E, N, K, device="cuda").to(torch.float8_e4m3fn)
    B_sf = torch.ones(E, N // 128, K // 128, device="cuda", dtype=torch.float32)
    expert_num_tokens = torch.tensor([16, 32, 8, 64], device="cuda", dtype=torch.int32)
    C = torch.empty(E, max_T, N, device="cuda", dtype=torch.bfloat16)

    invoke_sm120_fp8_grouped_gemm(A, A_sf, B, B_sf, expert_num_tokens, C)

    test_case.assertEqual(C.shape, (E, max_T, N))
    for expert_id, token_count in enumerate(expert_num_tokens.tolist()):
        valid = C[expert_id, :token_count]
        test_case.assertFalse(valid.isnan().any(), f"Expert {expert_id}: NaN")
        test_case.assertFalse(valid.isinf().any(), f"Expert {expert_id}: Inf")


def _check_grouped_gemm_zero_input(test_case):
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
        test_case.assertEqual(
            C[e, :n_tok].abs().max().item(),
            0.0,
            f"Expert {e}: expected zero output for zero inputs",
        )


def _check_grouped_gemm_empty_expert(test_case):
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

    test_case.assertTrue(
        (C[0] == sentinel).all(), "Expert 0 (empty) rows should be unchanged"
    )


def _check_grouped_gemm_raw_tail_n(test_case):
    from rtp_llm.models_py.triton_kernels.moe.fp8_grouped_gemm import (
        invoke_sm120_fp8_grouped_gemm,
    )

    E, max_T, K, N = 4, 128, 256, 3120
    A = torch.randn(E, max_T, K, device="cuda").to(torch.float8_e4m3fn)
    A_sf = torch.ones(E, max_T, _ceil_div(K, 128), device="cuda", dtype=torch.float32)
    B = torch.randn(E, N, K, device="cuda").to(torch.float8_e4m3fn)
    B_sf = torch.ones(
        E,
        _ceil_div(N, 128),
        _ceil_div(K, 128),
        device="cuda",
        dtype=torch.float32,
    )
    expert_num_tokens = torch.tensor([16, 32, 8, 64], device="cuda", dtype=torch.int32)
    C = torch.empty(E, max_T, N, device="cuda", dtype=torch.bfloat16)

    invoke_sm120_fp8_grouped_gemm(A, A_sf, B, B_sf, expert_num_tokens, C)

    for e in range(E):
        n_tok = expert_num_tokens[e].item()
        test_case.assertFalse(C[e, :n_tok].isnan().any(), f"Expert {e}: NaN")
        test_case.assertFalse(C[e, :n_tok].isinf().any(), f"Expert {e}: Inf")


def _check_grouped_gemm_nonuniform_scales_reference(test_case):
    from rtp_llm.models_py.triton_kernels.moe.fp8_grouped_gemm import (
        invoke_sm120_fp8_grouped_gemm,
    )

    E, max_T, K, N = 2, 128, 256, 256
    torch.manual_seed(7)
    A = (torch.randn(E, max_T, K, device="cuda") * 0.125).to(torch.float8_e4m3fn)
    B = (torch.randn(E, N, K, device="cuda") * 0.125).to(torch.float8_e4m3fn)
    A_sf = (
        torch.tensor([0.25, 0.5], device="cuda")
        .view(1, 1, 2)
        .expand(E, max_T, 2)
        .contiguous()
    )
    B_sf = (
        torch.tensor([[0.125, 0.25], [0.5, 1.0]], device="cuda", dtype=torch.float32)
        .view(1, 2, 2)
        .expand(E, 2, 2)
        .contiguous()
    )
    expert_num_tokens = torch.tensor([3, 0], device="cuda", dtype=torch.int32)
    C = torch.full((E, max_T, N), float("nan"), device="cuda", dtype=torch.bfloat16)

    invoke_sm120_fp8_grouped_gemm(A, A_sf, B, B_sf, expert_num_tokens, C)

    a_dequant = A[0, :3].float().view(3, 2, 128) * A_sf[0, :3, :, None]
    b_dequant = B[0].float().view(2, 128, 2, 128)
    b_dequant = b_dequant * B_sf[0, :, None, :, None]
    reference = a_dequant.reshape(3, K) @ b_dequant.reshape(N, K).T
    torch.testing.assert_close(C[0, :3].float(), reference, rtol=0.03, atol=0.08)
    test_case.assertTrue(C[1].isnan().all(), "empty expert rows must remain untouched")


def _check_executor_output_finite(test_case, M, E, K, N, top_k):
    result = _run_grouped_gemm_executor(E, K, N, M, top_k)
    out = result.fused_expert_output
    test_case.assertEqual(out.shape, (M, K))
    test_case.assertFalse(out.isnan().any(), "Output contains NaN")
    test_case.assertFalse(out.isinf().any(), "Output contains Inf")


def _dequant_blockwise(tensor, scales):
    row_scale = scales.repeat_interleave(128, dim=-2)
    element_scale = row_scale.repeat_interleave(128, dim=-1)
    return tensor.float() * element_scale[..., : tensor.shape[-2], : tensor.shape[-1]]


def _executor_fp32_reference(executor, payload):
    hidden = payload.expert_x.float().view(payload.expert_x.shape[0], -1, 128)
    hidden = (hidden * payload.expert_x_scale[..., None]).flatten(1)
    w13 = _dequant_blockwise(executor.w13_weight, executor.w13_scale)
    w2 = _dequant_blockwise(executor.w2_weight, executor.w2_scale)
    output = torch.zeros_like(hidden)
    for token_id in range(hidden.shape[0]):
        for route_id in range(payload.expert_topk_ids.shape[1]):
            expert_id = int(payload.expert_topk_ids[token_id, route_id])
            gate_up = hidden[token_id] @ w13[expert_id].T
            up, gate = gate_up.chunk(2)
            intermediate = up * torch.nn.functional.silu(gate)
            expert_output = intermediate @ w2[expert_id].T
            output[token_id] += (
                payload.expert_topk_weights[token_id, route_id] * expert_output
            )
    return output


class TestSm120Fp8GroupedGemmConfig(unittest.TestCase):
    def test_config_selector(self):
        _check_grouped_gemm_config_selector(self)


@unittest.skipUnless(SM120_AVAILABLE, "SM120 hardware required")
class TestSm120Fp8GroupedGemm(unittest.TestCase):
    def test_kernel_edge_cases_and_reference(self):
        for case in (
            _check_invoke_grouped_gemm_basic,
            _check_grouped_gemm_zero_input,
            _check_grouped_gemm_empty_expert,
            _check_grouped_gemm_raw_tail_n,
            _check_grouped_gemm_nonuniform_scales_reference,
        ):
            with self.subTest(case=case.__name__):
                case(self)

    def test_executor_shapes_and_tail_batches(self):
        for shape in (
            (1, 8, 512, 1024, 2),
            (8, 8, 512, 1024, 2),
            (129, 8, 512, 1024, 2),
            (2020, 8, 512, 1024, 2),
            (8, 16, 1024, 2048, 4),
        ):
            with self.subTest(shape=shape):
                _check_executor_output_finite(self, *shape)

    def test_unsupported_execute_options_fail_fast(self):
        executor, payload = _make_grouped_gemm_executor_case(8, 512, 1024, 8, 2)
        cases = (
            ({"activation": "gelu"}, "only SiLU"),
            ({"expert_map": torch.arange(8, device="cuda")}, "expert_map"),
            ({"a2_scale": torch.ones(1, device="cuda")}, "a2_scale"),
            ({"apply_router_weight_on_input": True}, "during gather"),
        )
        defaults = dict(
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                args = defaults | overrides
                with self.assertRaisesRegex(ValueError, message):
                    executor.execute(payload, **args)

    def test_weight_expert_count_must_match_ep_partition(self):
        with self.assertRaisesRegex(ValueError, "weight expert count"):
            _make_grouped_gemm_executor_case(8, 512, 1024, 8, 2, config_expert_num=16)

    def test_executor_matches_fp32_reference(self):
        executor, payload = _make_grouped_gemm_executor_case(4, 512, 1024, 8, 2)
        reference = _executor_fp32_reference(executor, payload)
        actual = _execute_grouped_gemm_executor(executor, payload).fused_expert_output
        torch.testing.assert_close(actual.float(), reference, rtol=0.15, atol=0.02)
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), reference.flatten(), dim=0
        )
        self.assertGreater(cosine.item(), 0.99)

    def test_siglu_alias_uses_silu_kernel(self):
        executor, payload = _make_grouped_gemm_executor_case(8, 512, 1024, 8, 2)
        silu_result = _execute_grouped_gemm_executor(
            executor, payload
        ).fused_expert_output
        siglu_result = executor.execute(
            payload,
            activation="SiGLU",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        )
        torch.testing.assert_close(siglu_result.fused_expert_output, silu_result)

    def test_routed_token_limit_fails_before_scratch_allocation(self):
        executor, payload = _make_grouped_gemm_executor_case(8, 512, 1024, 8, 2)
        executor.masked_max_token_num = 1
        with self.assertRaisesRegex(ValueError, "masked_max_token_num"):
            _execute_grouped_gemm_executor(executor, payload)


if __name__ == "__main__":
    unittest.main()
