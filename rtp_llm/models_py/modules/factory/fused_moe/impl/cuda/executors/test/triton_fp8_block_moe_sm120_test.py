import os
import unittest
from unittest.mock import patch

import torch
import triton.language as tl

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    requant_weight_ue8m0,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import per_block_cast_to_fp8
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
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
    DeepGemmHybridExecutor,
    get_sm120_triton_fp8_config,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    invoke_fused_moe_kernel,
    moe_align_block_size_torch,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


class TritonFp8BlockMoeSm120Test(unittest.TestCase):
    NUM_EXPERTS = 8
    TOP_K = 2
    NUM_TOKENS = 16
    HIDDEN_SIZE = 512
    INTERMEDIATE_SIZE = 256

    def setUp(self) -> None:
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("SM120-only optimization")
        torch.manual_seed(20260830)
        self.config = self._make_config()
        self.executor = self._make_executor()

    def _make_config(
        self,
        *,
        num_experts: int | None = None,
        top_k: int | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        enable_cuda_graph: bool = True,
    ) -> MoEConfigAdapter:
        num_experts = num_experts or self.NUM_EXPERTS
        top_k = top_k or self.TOP_K
        hidden_size = hidden_size or self.HIDDEN_SIZE
        intermediate_size = intermediate_size or self.INTERMEDIATE_SIZE
        model_config = ModelConfig()
        model_config.quant_config = Fp8BlockWiseQuantConfig()
        model_config.data_type = "bf16"
        model_config.expert_num = num_experts
        model_config.moe_k = top_k
        model_config.hidden_size = hidden_size
        model_config.moe_inter_size = intermediate_size
        model_config.activation_type = "SiGLU"

        parallelism_config = ParallelismConfig()
        parallelism_config.world_size = 1
        parallelism_config.local_world_size = 1
        parallelism_config.tp_size = 1
        parallelism_config.ep_size = 1

        moe_config = MoeConfig()
        moe_config.use_all_gather = True
        moe_config.use_deepep_moe = False
        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            enable_cuda_graph=enable_cuda_graph,
        )

    def _make_executor(
        self,
        *,
        config: MoEConfigAdapter | None = None,
        num_experts: int | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        vary_k_group_scales: bool = False,
    ) -> DeepGemmHybridExecutor:
        config = config or self.config
        num_experts = num_experts or self.NUM_EXPERTS
        hidden_size = hidden_size or self.HIDDEN_SIZE
        intermediate_size = intermediate_size or self.INTERMEDIATE_SIZE
        gate_up = torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        down = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Make adjacent 128-row output groups use intentionally different
        # scales. This catches incorrect indexing of DeepGEMM's expanded,
        # packed UE8M0 N dimension.
        def grouped_scales(size: int) -> torch.Tensor:
            group_ids = torch.arange(
                (size + 127) // 128, device="cuda", dtype=torch.float32
            )
            return torch.pow(2.0, group_ids.remainder(7) - 4).repeat_interleave(128)[
                :size
            ]

        gate_up.mul_(grouped_scales(2 * intermediate_size).view(1, -1, 1))
        down.mul_(grouped_scales(hidden_size).view(1, -1, 1))
        if vary_k_group_scales:
            # Deliberately make adjacent 128-wide K groups and packed words
            # differ. K=2048 and K=768 then exercise words 0..3 and 0..1.
            gate_up.mul_(grouped_scales(hidden_size).view(1, 1, -1))
            down.mul_(grouped_scales(intermediate_size).view(1, 1, -1))
        w13, s13, w2, s2 = [], [], [], []
        for expert in range(num_experts):
            q, scale = per_block_cast_to_fp8(gate_up[expert], use_ue8m0=False)
            w13.append(q)
            s13.append(scale)
            q, scale = per_block_cast_to_fp8(down[expert], use_ue8m0=False)
            w2.append(q)
            s2.append(scale)
        w13, s13 = requant_weight_ue8m0(torch.stack(w13), torch.stack(s13))
        w2, s2 = requant_weight_ue8m0(torch.stack(w2), torch.stack(s2))
        return DeepGemmHybridExecutor(
            config=config,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128]
            ),
            weights={
                W.moe_w1: w13,
                W.moe_s1: s13,
                W.moe_w2: w2,
                W.moe_s2: s2,
            },
        )

    def _make_payload(
        self,
        offset: int,
        num_tokens: int | None = None,
        topk_ids_dtype: torch.dtype | None = None,
        executor: DeepGemmHybridExecutor | None = None,
        num_experts: int | None = None,
        top_k: int | None = None,
        hidden_size: int | None = None,
        vary_k_group_scales: bool = False,
    ) -> ExpertForwardPayload:
        executor = executor or self.executor
        num_tokens = self.NUM_TOKENS if num_tokens is None else num_tokens
        num_experts = num_experts or self.NUM_EXPERTS
        top_k = top_k or self.TOP_K
        hidden_size = hidden_size or self.HIDDEN_SIZE
        topk_ids_dtype = topk_ids_dtype or executor.topk_ids_dtype
        hidden = torch.randn(
            num_tokens,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        if vary_k_group_scales:
            group_ids = torch.arange(
                (hidden_size + 127) // 128, device="cuda", dtype=torch.float32
            )
            scales = torch.pow(2.0, group_ids.remainder(7) - 4).repeat_interleave(128)[
                :hidden_size
            ]
            hidden.mul_(scales)
        hidden_fp8, hidden_scale = sgl_per_token_group_quant_fp8(
            hidden,
            group_size=128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        tokens = torch.arange(num_tokens, device="cuda")
        topk_ids = torch.stack(
            tuple(
                (tokens + offset + 3 * route) % num_experts for route in range(top_k)
            ),
            dim=1,
        ).to(topk_ids_dtype)
        topk_weights = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)
        topk_weights /= topk_weights.sum(dim=1, keepdim=True)
        counts = torch.bincount(topk_ids.view(-1).long(), minlength=num_experts).to(
            torch.int32
        )
        return ExpertForwardPayload(
            expert_x=hidden_fp8,
            expert_x_scale=hidden_scale,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=counts, expert_num_tokens_cpu=None
            ),
        )

    def _triton(self, payload: ExpertForwardPayload) -> torch.Tensor:
        return self.executor.execute_triton_fp8(
            payload,
            activation="SiGLU",
            apply_router_weight_on_input=False,
        ).fused_expert_output

    def _deepgemm(self, payload: ExpertForwardPayload) -> torch.Tensor:
        return self.executor.execute_contiguous(
            payload,
            activation="SiGLU",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        ).fused_expert_output

    def _assert_matches_deepgemm(
        self, actual: torch.Tensor, expected: torch.Tensor
    ) -> None:
        diff = actual.float() - expected.float()
        relative_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(
            expected.float()
        )
        relative_max = diff.abs().max() / expected.float().abs().max()
        self.assertLess(float(relative_l2), 2e-2)
        self.assertLess(float(relative_max), 2e-2)

    def test_matches_deepgemm(self) -> None:
        payload = self._make_payload(offset=0)
        self.assertEqual(payload.expert_topk_ids.dtype, self.executor.topk_ids_dtype)
        actual = self._triton(payload)
        expected = self._deepgemm(payload)
        self._assert_matches_deepgemm(actual, expected)

    def test_int32_topk_ids_match_deepgemm(self) -> None:
        payload = self._make_payload(offset=0, topk_ids_dtype=torch.int32)
        actual = self._triton(payload)
        expected = self._deepgemm(payload)
        self._assert_matches_deepgemm(actual, expected)

    def test_boundary_tokens_match_deepgemm(self) -> None:
        for num_tokens in (1, 32):
            with self.subTest(num_tokens=num_tokens):
                payload = self._make_payload(offset=0, num_tokens=num_tokens)
                actual = self._triton(payload)
                expected = self._deepgemm(payload)
                self._assert_matches_deepgemm(actual, expected)

    def test_tuned_configs_match_deepgemm(self) -> None:
        payload = self._make_payload(offset=0)
        expected = self._deepgemm(payload)
        config_pairs = (
            (
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 3,
                },
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 2,
                },
            ),
            (
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 3,
                },
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 2,
                },
            ),
        )
        for gate_up_config, down_config in config_pairs:
            with self.subTest(gate_up=gate_up_config, down=down_config), patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
                "deepgemm_hybrid_executor.get_sm120_triton_fp8_config",
                side_effect=(gate_up_config, down_config),
            ):
                actual = self._triton(payload)
                self._assert_matches_deepgemm(actual, expected)

    def test_qwen_config_selection(self) -> None:
        gate_up = get_sm120_triton_fp8_config(1, 128, 1536, 2048, 8)
        down = get_sm120_triton_fp8_config(32, 128, 2048, 768, 8)
        self.assertEqual(gate_up["BLOCK_SIZE_M"], 16)
        self.assertEqual(gate_up["BLOCK_SIZE_N"], 128)
        self.assertEqual(gate_up["BLOCK_SIZE_K"], 128)
        self.assertEqual(gate_up["num_stages"], 3)
        self.assertEqual(down["BLOCK_SIZE_M"], 16)
        self.assertEqual(down["BLOCK_SIZE_N"], 128)
        self.assertEqual(down["BLOCK_SIZE_K"], 128)
        self.assertEqual(down["num_stages"], 2)

        outside_dispatch = get_sm120_triton_fp8_config(33, 128, 1536, 2048, 8)
        self.assertEqual(outside_dispatch["BLOCK_SIZE_N"], 64)

    def test_generic_config_is_not_mutated(self) -> None:
        shared = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 3,
        }
        original = dict(shared)
        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
            "deepgemm_hybrid_executor.get_triton_moe_config",
            return_value=shared,
        ):
            selected = get_sm120_triton_fp8_config(1, 8, 512, 512, 2)
        self.assertEqual(shared, original)
        self.assertIsNot(selected, shared)
        self.assertEqual(selected["BLOCK_SIZE_M"], 16)

    def test_alignment_capacity_covers_routing_boundaries(self) -> None:
        cases = (
            ("active_expert_upper_bound", torch.arange(8), 128, 8),
            ("concentrated", torch.zeros(32), 8, 9),
            ("block_exact", torch.zeros(16), 1, 1),
            ("block_plus_one", torch.zeros(17), 1, 2),
        )
        for name, flat_ids, num_experts, expected_capacity_blocks in cases:
            topk_ids = flat_ids.to(device="cuda", dtype=torch.int64).view(-1, 1)
            sorted_token_ids, expert_ids, padded = moe_align_block_size_torch(
                topk_ids, block_size=16, num_experts=num_experts
            )
            with self.subTest(name=name):
                self.assertEqual(
                    sorted_token_ids.numel(), expected_capacity_blocks * 16
                )
                self.assertEqual(expert_ids.numel(), expected_capacity_blocks)
                self.assertEqual(expert_ids.numel() * 16, sorted_token_ids.numel())
                self.assertLessEqual(int(padded.item()), sorted_token_ids.numel())

    def test_dispatches_triton_for_one_through_32_tokens(self) -> None:
        call_args = {
            "activation": "SiGLU",
            "expert_map": None,
            "a2_scale": None,
            "apply_router_weight_on_input": False,
            "extra_expert_args": None,
        }
        for num_tokens, graph_phase, use_triton in (
            (0, True, False),
            (1, False, False),
            (1, True, True),
            (32, True, True),
            (33, True, False),
        ):
            payload = ExpertForwardPayload(
                expert_x=torch.empty(
                    num_tokens,
                    self.HIDDEN_SIZE,
                    device="cuda",
                    dtype=torch.float8_e4m3fn,
                )
            )
            triton_result = object()
            deepgemm_result = object()
            with self.subTest(num_tokens=num_tokens, graph_phase=graph_phase), patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
                "deepgemm_hybrid_executor._is_cuda_graph_warmup_or_capture",
                return_value=graph_phase,
            ), patch.object(
                self.executor, "execute_triton_fp8", return_value=triton_result
            ) as triton, patch.object(
                self.executor, "execute_contiguous", return_value=deepgemm_result
            ) as deepgemm:
                result = self.executor.execute(payload, **call_args)
                self.assertIs(result, triton_result if use_triton else deepgemm_result)
                self.assertEqual(triton.call_count, int(use_triton))
                self.assertEqual(deepgemm.call_count, int(not use_triton))

    def test_cuda_graph_warmup_flag_dispatches_triton(self) -> None:
        payload = ExpertForwardPayload(
            expert_x=torch.empty(
                1,
                self.HIDDEN_SIZE,
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
        )
        fast_result = object()
        with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
            "deepgemm_hybrid_executor.torch.cuda.is_current_stream_capturing",
            return_value=False,
        ), patch.object(
            self.executor, "execute_triton_fp8", return_value=fast_result
        ) as triton, patch.object(
            self.executor, "execute_contiguous"
        ) as deepgemm:
            result = self.executor.execute(
                payload,
                activation="SiGLU",
                expert_map=None,
                a2_scale=None,
                apply_router_weight_on_input=False,
                extra_expert_args=None,
            )
        self.assertIs(result, fast_result)
        triton.assert_called_once()
        deepgemm.assert_not_called()

    def test_unsupported_contracts_fall_back_to_deepgemm(self) -> None:
        payload = ExpertForwardPayload(
            expert_x=torch.empty(
                1,
                self.HIDDEN_SIZE,
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
        )
        base_args = {
            "activation": "SiGLU",
            "expert_map": None,
            "a2_scale": None,
            "apply_router_weight_on_input": False,
            "extra_expert_args": None,
        }
        cases = (
            ("activation", {"activation": "GeGLU"}, 1, 32),
            ("router_weight", {"apply_router_weight_on_input": True}, 1, 32),
            (
                "expert_map",
                {"expert_map": torch.arange(self.NUM_EXPERTS, device="cuda")},
                1,
                32,
            ),
            ("ep_size", {}, 2, 32),
            ("disabled", {}, 1, 0),
        )
        for name, overrides, ep_size, max_tokens in cases:
            call_args = {**base_args, **overrides}
            fallback_result = object()
            with self.subTest(name=name), patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
                "deepgemm_hybrid_executor._is_cuda_graph_warmup_or_capture",
                return_value=True,
            ), patch.object(self.executor, "ep_size", ep_size), patch.object(
                self.executor, "sm120_triton_max_tokens", max_tokens
            ), patch.object(
                self.executor, "execute_triton_fp8"
            ) as triton, patch.object(
                self.executor,
                "execute_contiguous",
                return_value=fallback_result,
            ) as deepgemm:
                result = self.executor.execute(payload, **call_args)
                self.assertIs(result, fallback_result)
                triton.assert_not_called()
                deepgemm.assert_called_once()

    def test_sm120_env_is_not_parsed_for_inapplicable_executor(self) -> None:
        weights = {
            W.moe_w1: self.executor.w13_weight,
            W.moe_s1: self.executor.w13_weight_scale_inv,
            W.moe_w2: self.executor.w2_weight,
            W.moe_s2: self.executor.w2_weight_scale_inv,
        }
        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
            "deepgemm_hybrid_executor.get_sm",
            return_value=(9, 0),
        ), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
            "deepgemm_hybrid_executor._get_sm120_triton_max_tokens",
            side_effect=AssertionError("SM120 env must not be parsed"),
        ) as get_max_tokens:
            executor = DeepGemmHybridExecutor(
                config=self.config,
                quant_config=self.executor.quant_config,
                weights=weights,
            )
        get_max_tokens.assert_not_called()
        self.assertEqual(executor.sm120_triton_max_tokens, 0)

    def test_qwen_k_groups_match_deepgemm_and_graph_replay(self) -> None:
        # Keep E small for CI memory/runtime, but use the exact Qwen gate/down
        # K dimensions. Their 16/6 K groups span 4/2 packed UE8M0 words.
        num_experts, top_k = 8, 8
        hidden_size, intermediate_size = 2048, 768
        config = self._make_config(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        executor = self._make_executor(
            config=config,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vary_k_group_scales=True,
        )
        payload = self._make_payload(
            offset=0,
            num_tokens=1,
            executor=executor,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            vary_k_group_scales=True,
        )
        # Make the intended coverage mechanically explicit: a kernel stuck on
        # packed word zero must observe different scale bits for later words.
        for name, packed_scale, packed_words in (
            ("gate activation", payload.expert_x_scale, 4),
            ("gate weight", executor.w13_weight_scale_inv, 4),
            ("down weight", executor.w2_weight_scale_inv, 2),
        ):
            with self.subTest(scale=name):
                self.assertGreaterEqual(packed_scale.shape[-1], packed_words)
                self.assertFalse(
                    torch.equal(packed_scale[..., 0], packed_scale[..., 1])
                )
        gate_config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
        down_config = {**gate_config, "num_stages": 2}

        def select_config(
            _m: int, _e: int, n: int, k: int, _top_k: int
        ) -> dict[str, int]:
            if (n, k) == (2 * intermediate_size, hidden_size):
                return dict(gate_config)
            if (n, k) == (hidden_size, intermediate_size):
                return dict(down_config)
            raise AssertionError(f"unexpected GEMM shape N={n}, K={k}")

        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
            "deepgemm_hybrid_executor.get_sm120_triton_fp8_config",
            side_effect=select_config,
        ):
            expected = executor.execute_contiguous(
                payload,
                activation="SiGLU",
                expert_map=None,
                a2_scale=None,
                apply_router_weight_on_input=False,
                extra_expert_args=None,
            ).fused_expert_output
            actual = executor.execute_triton_fp8(
                payload,
                activation="SiGLU",
                apply_router_weight_on_input=False,
            ).fused_expert_output
            self._assert_matches_deepgemm(actual, expected)

            # The eager call above compiles/warms the static routing and Triton
            # kernels. The public executor entry must select Triton only while
            # the stream is actually being captured.
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = executor.execute(
                    payload,
                    activation="SiGLU",
                    expert_map=None,
                    a2_scale=None,
                    apply_router_weight_on_input=False,
                    extra_expert_args=None,
                ).fused_expert_output

            replay = self._make_payload(
                offset=1,
                num_tokens=1,
                executor=executor,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                vary_k_group_scales=True,
            )
            payload.expert_x.copy_(replay.expert_x)
            payload.expert_x_scale.copy_(replay.expert_x_scale)
            payload.expert_topk_ids.copy_(replay.expert_topk_ids)
            payload.expert_topk_weights.copy_(replay.expert_topk_weights)
            payload.expert_tokens_meta.expert_num_tokens.copy_(
                replay.expert_tokens_meta.expert_num_tokens
            )
            graph.replay()
            captured = graph_output.clone()
            expected = executor.execute_contiguous(
                payload,
                activation="SiGLU",
                expert_map=None,
                a2_scale=None,
                apply_router_weight_on_input=False,
                extra_expert_args=None,
            ).fused_expert_output
            self._assert_matches_deepgemm(captured, expected)

    def test_invalid_fp8_scale_tile_config_is_rejected(self) -> None:
        payload = self._make_payload(offset=0, num_tokens=1)
        base_config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
        for key, value, message in (
            ("BLOCK_SIZE_N", 96, "must divide FP8 scale group_n"),
            ("BLOCK_SIZE_K", 256, "must divide FP8 scale group_k"),
        ):
            config = {**base_config, key: value}
            with self.subTest(key=key), patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
                "deepgemm_hybrid_executor.get_sm120_triton_fp8_config",
                return_value=config,
            ), self.assertRaisesRegex(AssertionError, message):
                self._triton(payload)

    def test_non_fp8_kernel_path_accepts_omitted_scales(self) -> None:
        num_tokens, top_k, num_experts, k_size, n_size = 4, 2, 4, 32, 64
        activations = torch.randn(
            num_tokens, k_size, device="cuda", dtype=torch.bfloat16
        )
        weights = torch.randn(
            num_experts, n_size, k_size, device="cuda", dtype=torch.bfloat16
        )
        topk_ids = torch.tensor(
            [[0, 1], [2, 3], [1, 3], [0, 2]],
            device="cuda",
            dtype=torch.int64,
        )
        topk_weights = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)
        sorted_ids, expert_ids, padded = moe_align_block_size_torch(
            topk_ids, block_size=16, num_experts=num_experts
        )
        output = torch.empty(
            num_tokens * top_k, n_size, device="cuda", dtype=torch.bfloat16
        )
        invoke_fused_moe_kernel(
            activations,
            weights,
            output,
            topk_weights.view(-1),
            topk_ids.view(-1),
            sorted_ids,
            expert_ids,
            padded,
            True,
            top_k,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 2,
            },
            tl.bfloat16,
        )

        reference = torch.empty_like(output)
        flat_ids = topk_ids.view(-1)
        flat_weights = topk_weights.view(-1)
        for route in range(num_tokens * top_k):
            expert = int(flat_ids[route])
            reference[route] = (
                activations[route // top_k].float()
                @ weights[expert].float().T
                * flat_weights[route]
            ).to(torch.bfloat16)
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)

    def test_cuda_graph_replay(self) -> None:
        payload = self._make_payload(offset=0)
        # Compile and allocate before capture through the explicit fast path.
        self._triton(payload)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = self.executor.execute(
                payload,
                activation="SiGLU",
                expert_map=None,
                a2_scale=None,
                apply_router_weight_on_input=False,
                extra_expert_args=None,
            ).fused_expert_output

        replay = self._make_payload(offset=2)
        payload.expert_x.copy_(replay.expert_x)
        payload.expert_x_scale.copy_(replay.expert_x_scale)
        payload.expert_topk_ids.copy_(replay.expert_topk_ids)
        payload.expert_topk_weights.copy_(replay.expert_topk_weights)
        payload.expert_tokens_meta.expert_num_tokens.copy_(
            replay.expert_tokens_meta.expert_num_tokens
        )
        graph.replay()
        captured = graph_output.clone()
        eager = self._triton(payload)
        torch.testing.assert_close(captured, eager, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
