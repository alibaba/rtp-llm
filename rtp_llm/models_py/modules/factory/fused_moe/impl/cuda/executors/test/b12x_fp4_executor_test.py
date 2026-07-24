# SPDX-License-Identifier: Apache-2.0
import random
import unittest
from typing import Dict, List, Tuple

import torch


def _skip_reason() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    major = torch.cuda.get_device_capability()[0]
    if major != 12:
        return "b12x NVFP4 MoE requires sm_120/sm_121 (compute capability 12.x)"
    return ""


class B12xFp4ExecutorTestBase:
    NUM_EXPERTS = 16
    TOP_K = 4
    NUM_TOKENS = 16
    HIDDEN_SIZE = 2048  # H
    MOE_INTERMEDIATE_SIZE = 768  # I

    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = 448.0
    BLOCK_SIZE = 16

    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)

    def _generate_config(self):
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
        moe_config.ll_num_max_token = self.NUM_TOKENS

        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
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

    def _generate_weights(self, fp4_quantize, swizzle_blockscale):
        from rtp_llm.utils.model_weight import W

        E = self.NUM_EXPERTS
        H = self.HIDDEN_SIZE
        I = self.MOE_INTERMEDIATE_SIZE

        w13_bf16 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.1
        w2_bf16 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.1

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

        w13_sf_sw = swizzle_blockscale(w13_sf_linear)
        w2_sf_sw = swizzle_blockscale(w2_sf_linear)

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

    def test_b12x_fp4_executor_semantic(self):
        reason = _skip_reason()
        if reason:
            self.skipTest(reason)

        from flashinfer import e2m1_and_ufp8sf_scale_to_float, fp4_quantize

        from rtp_llm.device.device_impl import CudaImpl
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
        weights, ref_pack = self._generate_weights(
            fp4_quantize, CudaImpl.swizzle_blockscale
        )
        hidden_states = (
            torch.randn(
                self.NUM_TOKENS, self.HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
            )
            * 0.1
        )
        router_logits = torch.randn(self.NUM_TOKENS, self.NUM_EXPERTS, device="cuda")
        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(probs, self.TOP_K, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(torch.float32)

        payload = ExpertForwardPayload(
            expert_x=hidden_states,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,
            block_shape=[self.BLOCK_SIZE, self.BLOCK_SIZE],
        )
        executor = B12xFp4Executor(config, quant_config, weights)
        forward_payload = executor.execute(payload, "silu", None, None, False, None)
        actual = forward_payload.fused_expert_output.to(torch.float32)

        ref = self._ref_moe(
            hidden_states,
            ref_pack,
            topk_ids,
            topk_weights,
            e2m1_and_ufp8sf_scale_to_float,
        )

        self.assertEqual(tuple(actual.shape), (self.NUM_TOKENS, self.HIDDEN_SIZE))
        self.assertFalse(torch.isnan(actual).any(), "output has NaNs")
        self.assertFalse(torch.isinf(actual).any(), "output has Infs")

        cos = torch.nn.functional.cosine_similarity(
            actual.flatten(), ref.flatten(), dim=0
        ).item()
        row_cos = torch.nn.functional.cosine_similarity(actual, ref, dim=-1)
        max_abs = (actual - ref).abs().max().item()

        print(
            f"[b12x-fp4] global cos={cos:.4f} "
            f"min_row_cos={row_cos.min().item():.4f} "
            f"max_abs_diff={max_abs:.4f}"
        )

        self.assertGreaterEqual(cos, 0.98, f"global cosine {cos:.4f} < 0.98")
        self.assertGreaterEqual(
            row_cos.min().item(),
            0.95,
            f"worst-row cosine {row_cos.min().item():.4f} < 0.95",
        )


class B12xFp4ExecutorTest(B12xFp4ExecutorTestBase, unittest.TestCase):
    pass


if __name__ == "__main__":
    reason = _skip_reason()
    if reason:
        print(f"SKIP: {reason}")
        raise SystemExit(0)
    unittest.main()
