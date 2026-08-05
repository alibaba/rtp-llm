# SPDX-License-Identifier: Apache-2.0
import random
import unittest
from typing import Dict, List, Optional, Tuple

import torch

from rtp_llm.utils.model_weight import W


def _skip_reason() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    major = torch.cuda.get_device_capability()[0]
    if major != 12:
        return "b12x NVFP4 MoE requires sm_120/sm_121 (compute capability 12.x)"
    return ""


class B12xFp4ExecutorTestBase:
    """Shared shape constants and NVFP4 weight/reference generation.

    Holds no test methods: subclasses declare their own so that changing one
    case's assertions never silently replaces another's.
    """

    NUM_EXPERTS = 16
    TOP_K = 4
    NUM_TOKENS = 16
    HIDDEN_SIZE = 2048  # H
    MOE_INTERMEDIATE_SIZE = 768  # I

    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = 448.0
    E4M3_MIN_SUBNORMAL = 2.0**-9
    BLOCK_SIZE = 16
    WEIGHT_SCALE = 0.1

    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)

    def _generate_config(
        self,
        *,
        enable_cuda_graph: bool = False,
        max_num_tokens: Optional[int] = None,
    ):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
            MoEConfigAdapter,
        )
        from rtp_llm.ops import MoeConfig, ParallelismConfig

        model_config = ModelConfig()
        model_config.attn_config.head_num = 2
        model_config.attn_config.size_per_head = 128
        model_config.num_layers = 2
        model_config.max_seq_len = 2048
        model_config.vocab_size = 500000
        model_config.expert_num = self.NUM_EXPERTS
        model_config.hidden_size = self.HIDDEN_SIZE
        model_config.moe_inter_size = self.MOE_INTERMEDIATE_SIZE
        model_config.moe_k = self.TOP_K

        parallelism_config = ParallelismConfig()
        parallelism_config.world_size = 1
        parallelism_config.dp_size = 1
        parallelism_config.tp_size = 1
        parallelism_config.ep_size = 1
        parallelism_config.dp_rank = 0
        parallelism_config.tp_rank = 0
        parallelism_config.ep_rank = 0
        parallelism_config.world_rank = 0
        parallelism_config.local_rank = 0
        parallelism_config.local_world_size = 1

        moe_config = MoeConfig()
        moe_config.ll_num_max_token = (
            self.NUM_TOKENS if max_num_tokens is None else max_num_tokens
        )

        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            enable_cuda_graph=enable_cuda_graph,
        )

    def _quant_expert(
        self, w: torch.Tensor, fp4_quantize
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize one [M, K] bf16 weight to NVFP4."""
        M, K = w.shape
        global_scale = (
            self.FLOAT8_E4M3_MAX * self.FLOAT4_E2M1_MAX
        ) / w.abs().amax().to(torch.float32)
        w_fp4, w_sf = fp4_quantize(
            w.cuda(), global_scale.cuda(), self.BLOCK_SIZE, False, False
        )
        w_sf = w_sf.view(torch.float8_e4m3fn).reshape(M, K // self.BLOCK_SIZE)
        return w_fp4, w_sf, global_scale

    def _generate_weights(
        self, fp4_quantize, w2_expert_scales: Optional[List[float]] = None
    ):
        from rtp_llm.config.moe_config import Fp4MoeOp
        from rtp_llm.device.device_impl import prepare_static_weights_for_fp4_moe

        E = self.NUM_EXPERTS
        H = self.HIDDEN_SIZE
        I = self.MOE_INTERMEDIATE_SIZE

        w13_bf16 = (
            torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16)
            * self.WEIGHT_SCALE
        )
        w2_bf16 = (
            torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16)
            * self.WEIGHT_SCALE
        )
        if w2_expert_scales is not None:
            if len(w2_expert_scales) != E:
                raise ValueError(
                    f"expected {E} w2 expert scales, got {len(w2_expert_scales)}"
                )
            scales = torch.tensor(
                w2_expert_scales, device="cuda", dtype=torch.bfloat16
            ).reshape(E, 1, 1)
            w2_bf16.mul_(scales)

        w13_fp4_l, w13_sf_l, w13_gs_l = [], [], []
        w2_fp4_l, w2_sf_l, w2_gs_l = [], [], []
        for e in range(E):
            fp4, sf, gs = self._quant_expert(w13_bf16[e], fp4_quantize)
            w13_fp4_l.append(fp4)
            w13_sf_l.append(sf)
            w13_gs_l.append(gs)
            fp4, sf, gs = self._quant_expert(w2_bf16[e], fp4_quantize)
            w2_fp4_l.append(fp4)
            w2_sf_l.append(sf)
            w2_gs_l.append(gs)

        w13_fp4 = torch.stack(w13_fp4_l)  # [E, 2I, H//2] uint8
        w2_fp4 = torch.stack(w2_fp4_l)  # [E, H, I//2] uint8
        w13_sf_linear = torch.stack(w13_sf_l)  # [E, 2I, H//16] fp8
        w2_sf_linear = torch.stack(w2_sf_l)  # [E, H, I//16] fp8
        w13_gs = torch.stack(w13_gs_l).reshape(E)  # [E]
        w2_gs = torch.stack(w2_gs_l).reshape(E)  # [E]

        # Exercise the production boundary that owns the B12X weight ordering
        # and blockscale layout. Keep ref_pack from the preprocessed tensors so
        # an accidental gate/up swap here cannot also mutate the reference.
        w13_fp4, w13_sf_sw = prepare_static_weights_for_fp4_moe(
            Fp4MoeOp.B12X.value, W.moe_w1, W.moe_s1, w13_fp4, w13_sf_linear
        )
        w2_fp4, w2_sf_sw = prepare_static_weights_for_fp4_moe(
            Fp4MoeOp.B12X.value, W.moe_w2, W.moe_s2, w2_fp4, w2_sf_linear
        )

        weights: Dict[str, torch.Tensor] = {
            W.moe_w1: w13_fp4,
            W.moe_w2: w2_fp4,
            W.moe_s1: w13_sf_sw,
            W.moe_s2: w2_sf_sw,
            W.moe_w1_s2: 1.0 / w13_gs,  # weight_scale_2 (w13)
            W.moe_w2_s2: 1.0 / w2_gs,  # weight_scale_2 (w2)
        }
        ref_pack = {
            "w13_fp4": w13_fp4_l,
            "w13_sf": w13_sf_l,
            "w13_gs": w13_gs_l,
            "w2_fp4": w2_fp4_l,
            "w2_sf": w2_sf_l,
            "w2_gs": w2_gs_l,
        }
        return weights, ref_pack

    def _generate_fold_boundary_weights(
        self, *, all_scales_underflow: bool
    ) -> Dict[str, torch.Tensor]:
        """Construct already-swizzled scales on a deterministic e4m3 boundary."""
        from rtp_llm.utils.model_weight import W

        E = self.NUM_EXPERTS
        H = self.HIDDEN_SIZE
        I = self.MOE_INTERMEDIATE_SIZE
        device = "cuda"
        scale_dtype = torch.float8_e4m3fn

        w1_sf = torch.ones(
            (E, 2 * I, H // self.BLOCK_SIZE),
            dtype=scale_dtype,
            device=device,
        )
        w2_sf = torch.ones(
            (E, H, I // self.BLOCK_SIZE),
            dtype=scale_dtype,
            device=device,
        )
        if all_scales_underflow:
            w1_sf.fill_(self.E4M3_MIN_SUBNORMAL)
            w2_sf.fill_(self.E4M3_MIN_SUBNORMAL)
        else:
            w1_sf.view(-1)[0] = self.E4M3_MIN_SUBNORMAL
            w2_sf.view(-1)[0] = self.E4M3_MIN_SUBNORMAL

        # 2^-9 * 0.25 = 2^-11, below half of e4m3's minimum subnormal,
        # so the selected entries deterministically round to zero.
        weight_scale_2 = torch.full((E,), 0.25, dtype=torch.float32, device=device)
        return {
            W.moe_w1: torch.zeros((E, 2 * I, H // 2), dtype=torch.uint8, device=device),
            W.moe_w2: torch.zeros((E, H, I // 2), dtype=torch.uint8, device=device),
            W.moe_s1: w1_sf,
            W.moe_s2: w2_sf,
            W.moe_w1_s2: weight_scale_2,
            W.moe_w2_s2: weight_scale_2.clone(),
        }

    def _dequant_expert(
        self, fp4, sf_linear, global_scale, M, K, e2m1_and_ufp8sf_scale_to_float
    ) -> torch.Tensor:
        """Dequantize one expert back to bf16 via flashinfer's official inverse."""
        deq = e2m1_and_ufp8sf_scale_to_float(
            fp4.cpu(),
            sf_linear.cpu().view(torch.uint8).reshape(-1),
            (1.0 / global_scale).cpu(),
            self.BLOCK_SIZE,
            1,  # ufp8_type
            False,  # is_sf_swizzled_layout — matches how we quantized
        )
        return deq.reshape(M, K).to(torch.bfloat16).cuda()

    def _ref_moe(
        self,
        hidden_states: torch.Tensor,
        ref_pack: dict,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        e2m1_and_ufp8sf_scale_to_float,
    ) -> torch.Tensor:
        E = self.NUM_EXPERTS
        H = self.HIDDEN_SIZE
        I = self.MOE_INTERMEDIATE_SIZE

        w13_deq: List[torch.Tensor] = []
        w2_deq: List[torch.Tensor] = []
        for e in range(E):
            w13_deq.append(
                self._dequant_expert(
                    ref_pack["w13_fp4"][e],
                    ref_pack["w13_sf"][e],
                    ref_pack["w13_gs"][e],
                    2 * I,
                    H,
                    e2m1_and_ufp8sf_scale_to_float,
                )
            )
            w2_deq.append(
                self._dequant_expert(
                    ref_pack["w2_fp4"][e],
                    ref_pack["w2_sf"][e],
                    ref_pack["w2_gs"][e],
                    H,
                    I,
                    e2m1_and_ufp8sf_scale_to_float,
                )
            )

        xf = hidden_states.to(torch.float32)
        out = torch.zeros(
            hidden_states.shape[0], H, device=hidden_states.device, dtype=torch.float32
        )
        for t in range(hidden_states.shape[0]):
            for k in range(self.TOP_K):
                e = int(topk_ids[t, k].item())
                wgt = float(topk_weights[t, k].item())
                g = xf[t] @ w13_deq[e].to(torch.float32).t()  # [2I]
                up = g[:I]
                gate = g[I:]
                inter = torch.nn.functional.silu(gate) * up  # [I]
                o = inter @ w2_deq[e].to(torch.float32).t()  # [H]
                out[t] += wgt * o
        return out

    def _run_semantic_case(
        self,
        *,
        w2_expert_scales: Optional[List[float]] = None,
        routed_experts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from flashinfer import e2m1_and_ufp8sf_scale_to_float, fp4_quantize

        from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
            ExpertForwardPayload,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
            FusedMoEQuantConfig,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_fp4_executor import (
            B12xFp4Executor,
        )

        config = self._generate_config()
        weights, ref_pack = self._generate_weights(fp4_quantize, w2_expert_scales)
        hidden_states = (
            torch.randn(
                self.NUM_TOKENS, self.HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
            )
            * 0.1
        )
        if routed_experts is None:
            router_logits = torch.randn(
                self.NUM_TOKENS, self.NUM_EXPERTS, device="cuda"
            )
            probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(probs, self.TOP_K, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_ids = routed_experts.to(device="cuda")
            topk_weights = torch.ones(
                topk_ids.shape, device="cuda", dtype=torch.float32
            )
        topk_weights = topk_weights.to(torch.float32)

        executor = B12xFp4Executor(
            config,
            FusedMoEQuantConfig(
                quant_dtype=torch.uint8,
                block_shape=[self.BLOCK_SIZE, self.BLOCK_SIZE],
            ),
            weights,
        )
        self.assertIs(weights[W.moe_s1], executor.w1_sf_mma)
        self.assertIs(weights[W.moe_s2], executor.w2_sf_mma)
        self.assertNotIn(W.moe_w1_s2, weights)
        self.assertNotIn(W.moe_w2_s2, weights)
        topk_ids = topk_ids.to(executor.topk_ids_dtype)
        payload = ExpertForwardPayload(
            expert_x=hidden_states,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )
        actual = executor.execute(
            payload, "silu", None, None, False, None
        ).fused_expert_output.to(torch.float32)
        ref = self._ref_moe(
            hidden_states,
            ref_pack,
            topk_ids,
            topk_weights,
            e2m1_and_ufp8sf_scale_to_float,
        )
        return actual, ref

    def _assert_semantic_accuracy(
        self,
        actual: torch.Tensor,
        ref: torch.Tensor,
        *,
        min_cosine: float = 0.98,
        min_row_cosine: float = 0.95,
        max_relative_l2: float = 0.30,
        max_row_relative_l2: float = 0.45,
    ) -> None:
        self.assertEqual(tuple(actual.shape), (self.NUM_TOKENS, self.HIDDEN_SIZE))
        self.assertTrue(torch.isfinite(actual).all(), "output has non-finite values")

        actual_f32 = actual.to(torch.float32)
        ref_f32 = ref.to(torch.float32)
        cosine = torch.nn.functional.cosine_similarity(
            actual_f32.flatten(), ref_f32.flatten(), dim=0
        ).item()
        row_cosine = torch.nn.functional.cosine_similarity(actual_f32, ref_f32, dim=-1)
        relative_l2 = (
            torch.linalg.vector_norm(actual_f32 - ref_f32)
            / torch.linalg.vector_norm(ref_f32)
        ).item()
        row_relative_l2 = torch.linalg.vector_norm(
            actual_f32 - ref_f32, dim=-1
        ) / torch.linalg.vector_norm(ref_f32, dim=-1).clamp_min(1e-12)

        self.assertGreaterEqual(
            cosine,
            min_cosine,
            f"global cosine {cosine:.4f} < {min_cosine}",
        )
        self.assertGreaterEqual(
            row_cosine.min().item(),
            min_row_cosine,
            f"worst-row cosine {row_cosine.min().item():.4f} < {min_row_cosine}",
        )
        self.assertLessEqual(
            relative_l2,
            max_relative_l2,
            f"relative L2 {relative_l2:.4f} > {max_relative_l2}",
        )
        self.assertLessEqual(
            row_relative_l2.max().item(),
            max_row_relative_l2,
            "worst-row relative L2 "
            f"{row_relative_l2.max().item():.4f} > {max_row_relative_l2}",
        )


class B12xFp4ExecutorTest(B12xFp4ExecutorTestBase, unittest.TestCase):
    """Numerical accuracy of the b12x NVFP4 path against a dequantized
    bf16 reference MoE."""

    def test_b12x_fp4_executor_semantic(self):
        reason = _skip_reason()
        if reason:
            self.skipTest(reason)

        actual, ref = self._run_semantic_case()
        self._assert_semantic_accuracy(actual, ref)


class B12xFp4PerExpertScaleTest(B12xFp4ExecutorTestBase, unittest.TestCase):
    """Catch missing, duplicated, or cross-expert weight_scale_2 folds."""

    NUM_EXPERTS = 4
    TOP_K = 1
    NUM_TOKENS = 4

    def test_b12x_fp4_executor_preserves_per_expert_magnitude(self):
        reason = _skip_reason()
        if reason:
            self.skipTest(reason)

        actual, ref = self._run_semantic_case(
            w2_expert_scales=[0.25, 0.5, 1.0, 2.0],
            routed_experts=torch.arange(self.NUM_EXPERTS).reshape(-1, 1),
        )
        self._assert_semantic_accuracy(actual, ref)


class B12xFp4ExecutorSmallWeightTest(B12xFp4ExecutorTestBase, unittest.TestCase):
    """Deterministic coverage on both sides of the folded-scale energy limit."""

    NUM_EXPERTS = 4
    HIDDEN_SIZE = 256
    MOE_INTERMEDIATE_SIZE = 128

    def test_b12x_fp4_executor_rejects_underflowing_scales(self):
        reason = _skip_reason()
        if reason:
            self.skipTest(reason)

        from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
            FusedMoEQuantConfig,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_fp4_executor import (
            B12xFp4Executor,
        )

        config = self._generate_config()
        weights = self._generate_fold_boundary_weights(all_scales_underflow=True)
        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,
            block_shape=[self.BLOCK_SIZE, self.BLOCK_SIZE],
        )
        with self.assertRaisesRegex(ValueError, "total scale energy"):
            B12xFp4Executor(config, quant_config, weights)

    def test_b12x_fp4_executor_warns_for_negligible_underflow(self):
        reason = _skip_reason()
        if reason:
            self.skipTest(reason)

        from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
            FusedMoEQuantConfig,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_fp4_executor import (
            B12xFp4Executor,
        )

        config = self._generate_config()
        weights = self._generate_fold_boundary_weights(all_scales_underflow=False)
        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,
            block_shape=[self.BLOCK_SIZE, self.BLOCK_SIZE],
        )
        with self.assertLogs(B12xFp4Executor.__module__, level="WARNING") as logs:
            executor = B12xFp4Executor(config, quant_config, weights)

        messages = "\n".join(logs.output)
        self.assertIn("w1 blockscale entries underflowed e4m3", messages)
        self.assertIn("w2 blockscale entries underflowed e4m3", messages)
        self.assertEqual(executor.local_num_experts, self.NUM_EXPERTS)


class B12xFp4ExecutorCudaGraphTest(B12xFp4ExecutorTestBase, unittest.TestCase):
    """Smoke coverage for the wrapper's preallocated CUDA Graph path."""

    NUM_EXPERTS = 4
    NUM_TOKENS = 8

    def test_b12x_fp4_executor_cuda_graph_capture(self):
        reason = _skip_reason()
        if reason:
            self.skipTest(reason)

        from flashinfer import fp4_quantize

        from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
            ExpertForwardPayload,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
            FusedMoEQuantConfig,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_fp4_executor import (
            B12xFp4Executor,
        )

        config = self._generate_config(
            enable_cuda_graph=True, max_num_tokens=2 * self.NUM_TOKENS
        )
        weights, _ = self._generate_weights(fp4_quantize)
        expert_x = torch.randn(
            self.NUM_TOKENS,
            self.HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        topk_ids = (
            torch.arange(self.NUM_TOKENS * self.TOP_K, device="cuda")
            .reshape(self.NUM_TOKENS, self.TOP_K)
            .remainder(self.NUM_EXPERTS)
        )
        topk_weights = torch.full(
            (self.NUM_TOKENS, self.TOP_K),
            1.0 / self.TOP_K,
            device="cuda",
            dtype=torch.float32,
        )
        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,
            block_shape=[self.BLOCK_SIZE, self.BLOCK_SIZE],
        )
        # Production constructs FusedMoe under torch.inference_mode(); mutable
        # FlashInfer workspaces still need to be normal tensors for replay.
        with torch.inference_mode():
            executor = B12xFp4Executor(config, quant_config, weights)
        topk_ids = topk_ids.to(executor.topk_ids_dtype)
        payload = ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

        # Warm up JIT and cache weight views before capture.
        expected = executor.execute(
            payload, "silu", None, None, False, None
        ).fused_expert_output.clone()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = executor.execute(
                payload, "silu", None, None, False, None
            ).fused_expert_output
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(tuple(output.shape), (self.NUM_TOKENS, self.HIDDEN_SIZE))
        self.assertTrue(torch.isfinite(output).all())
        output_f32 = output.to(torch.float32)
        expected_f32 = expected.to(torch.float32)
        cosine = torch.nn.functional.cosine_similarity(
            output_f32.flatten(), expected_f32.flatten(), dim=0
        ).item()
        relative_l2 = (
            torch.linalg.vector_norm(output_f32 - expected_f32)
            / torch.linalg.vector_norm(expected_f32)
        ).item()
        self.assertGreaterEqual(
            cosine,
            0.999,
            f"CUDA Graph replay cosine {cosine:.6f} < 0.999",
        )
        self.assertLessEqual(
            relative_l2,
            0.02,
            f"CUDA Graph replay relative L2 {relative_l2:.6f} > 0.02",
        )

    def test_b12x_fp4_executor_large_prefill_uses_eager_fallback(self):
        reason = _skip_reason()
        if reason:
            self.skipTest(reason)

        from flashinfer import fp4_quantize

        from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
            ExpertForwardPayload,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
            FusedMoEQuantConfig,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_fp4_executor import (
            B12xFp4Executor,
        )

        graph_capacity = self.NUM_TOKENS
        num_prefill_tokens = graph_capacity + 1
        config = self._generate_config(
            enable_cuda_graph=True, max_num_tokens=graph_capacity
        )
        weights, _ = self._generate_weights(fp4_quantize)
        expert_x = torch.randn(
            num_prefill_tokens,
            self.HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        topk_ids = (
            torch.arange(num_prefill_tokens * self.TOP_K, device="cuda")
            .reshape(num_prefill_tokens, self.TOP_K)
            .remainder(self.NUM_EXPERTS)
            .to(torch.int32)
        )
        topk_weights = torch.full(
            (num_prefill_tokens, self.TOP_K),
            1.0 / self.TOP_K,
            device="cuda",
            dtype=torch.float32,
        )
        executor = B12xFp4Executor(
            config,
            FusedMoEQuantConfig(
                quant_dtype=torch.uint8,
                block_shape=[self.BLOCK_SIZE, self.BLOCK_SIZE],
            ),
            weights,
        )
        payload = ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

        output = executor.execute(
            payload, "silu", None, None, False, None
        ).fused_expert_output

        self.assertGreater(num_prefill_tokens, executor._b12x_moe.max_num_tokens)
        self.assertIsNotNone(executor._b12x_moe_eager)
        self.assertEqual(tuple(output.shape), (num_prefill_tokens, self.HIDDEN_SIZE))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
