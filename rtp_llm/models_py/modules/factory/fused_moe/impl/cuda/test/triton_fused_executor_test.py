"""Correctness test for TritonFusedMoeExecutor (sglang-style fused_moe kernel)."""

import itertools
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    FusedMoe,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.router.batched_data_router import (
    BatchedDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
    TritonFusedMoeExecutor,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig, RuntimeConfig
from rtp_llm.utils.model_weight import W


def torch_reference_moe(hidden_states, w1, w2, topk_weights, topk_ids):
    """Reference MoE implementation using pure torch (no fusion)."""
    M, K = hidden_states.shape
    num_experts = w1.shape[0]
    inter_size = w2.shape[2]
    N = w1.shape[1]  # 2 * inter_size

    final = torch.zeros(M, K, dtype=hidden_states.dtype, device=hidden_states.device)
    expert_mask = F.one_hot(topk_ids.long(), num_classes=num_experts).permute(2, 1, 0)

    for e in range(num_experts):
        idx, top_x = torch.where(expert_mask[e])
        if top_x.numel() == 0:
            continue
        top_x_list = top_x.tolist()
        idx_list = idx.tolist()
        weights = topk_weights[top_x_list, idx_list, None]
        x = hidden_states[top_x_list]

        # GEMM1 + SiGLU activation
        h = x @ w1[e].T  # [tokens, 2*inter]
        gate = h[:, inter_size:]
        value = h[:, :inter_size]
        h2 = (F.silu(gate.float()) * value.float()).to(hidden_states.dtype)

        # GEMM2 + weighted sum
        out = h2 @ w2[e].T
        final.index_add_(0, top_x, (out * weights).to(hidden_states.dtype))

    return final


class TritonFusedMoeExecutorTest(TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            raise SkipTest("CUDA not available")
        torch.set_default_device("cuda")

    def _make_config(self, hidden_size, num_experts, top_k, inter_size, num_tokens):
        model_config = ModelConfig()
        model_config.attn_config.head_num = 4
        model_config.attn_config.size_per_head = 64
        model_config.num_layers = 2
        model_config.max_seq_len = 2048
        model_config.vocab_size = 32000
        model_config.hidden_size = hidden_size
        model_config.expert_num = num_experts
        model_config.moe_k = top_k
        model_config.inter_size = inter_size
        model_config.has_moe_norm = True
        model_config.activation_type = "SiGLU"

        par = ParallelismConfig()
        par.ep_size = 1
        par.ep_rank = 0
        par.tp_size = 1
        par.tp_rank = 0
        par.dp_size = 1
        par.dp_rank = 0
        par.world_size = 1
        par.world_rank = 0
        par.local_world_size = 1

        moe_config = MoeConfig()
        moe_config.ll_num_max_token = num_tokens

        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=par,
            moe_config=moe_config,
        )

    def _run_test(self, M, hidden_size, num_experts, top_k, inter_size, dtype):
        torch.manual_seed(42)
        scale = 0.1

        config = self._make_config(hidden_size, num_experts, top_k, inter_size, M)

        w1 = (
            torch.randn(
                num_experts, inter_size * 2, hidden_size, dtype=dtype, device="cuda"
            )
            * scale
        )
        w2 = (
            torch.randn(
                num_experts, hidden_size, inter_size, dtype=dtype, device="cuda"
            )
            * scale
        )

        executor = TritonFusedMoeExecutor(
            config=config,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
            weights={W.moe_w1: w1, W.moe_w2: w2},
        )

        hidden = torch.randn(M, hidden_size, dtype=dtype, device="cuda") * scale
        gating = torch.randn(M, num_experts, device="cuda")
        _, topk_ids = torch.topk(gating, top_k, dim=1)
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = torch.softmax(torch.randn(M, top_k, device="cuda"), dim=1)

        # Run executor
        payload = ExpertForwardPayload(
            expert_x=hidden,
            expert_x_scale=None,
            expert_x_origin_dtype=dtype,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
            expert_tokens_meta=None,
        )
        result = executor.execute(
            payload=payload,
            activation="SiGLU",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        )
        output = result.fused_expert_output

        # Reference
        ref = torch_reference_moe(hidden, w1, w2, topk_weights, topk_ids)

        # Compare
        self.assertEqual(output.shape, ref.shape)
        max_diff = (output.float() - ref.float()).abs().max().item()
        rel_err = max_diff / (ref.float().abs().max().item() + 1e-6)
        self.assertLess(
            rel_err,
            0.05,
            f"M={M} E={num_experts} top_k={top_k}: rel_err={rel_err:.4f} max_diff={max_diff:.6f}",
        )

    def test_basic_configs(self):
        """Test across representative configurations."""
        configs = [
            # (M, hidden, experts, top_k, inter, dtype)
            (1, 256, 4, 2, 128, torch.bfloat16),
            (8, 512, 8, 2, 256, torch.bfloat16),
            (64, 1024, 8, 2, 512, torch.bfloat16),
            (256, 1024, 8, 2, 512, torch.bfloat16),
            (1024, 512, 4, 2, 256, torch.bfloat16),
            # fp16
            (64, 512, 8, 2, 256, torch.float16),
            (256, 256, 4, 2, 128, torch.float16),
        ]
        for M, hs, E, tk, inter, dt in configs:
            with self.subTest(
                M=M, hidden=hs, experts=E, top_k=tk, inter=inter, dtype=dt
            ):
                self._run_test(M, hs, E, tk, inter, dt)

    def test_dsv2_lite_dims(self):
        """Test DSv2-Lite actual dimensions (E=64, inter=1408, top_k=6)."""
        for M in [1, 8, 64, 256]:
            with self.subTest(M=M):
                self._run_test(M, 2048, 64, 6, 1408, torch.bfloat16)

    def test_edge_cases(self):
        """Test edge cases: M=1, large top_k, non-power-of-2."""
        self._run_test(1, 256, 4, 1, 128, torch.bfloat16)
        self._run_test(1, 512, 8, 4, 256, torch.bfloat16)
        self._run_test(7, 384, 6, 3, 192, torch.bfloat16)  # odd sizes


if __name__ == "__main__":
    main()
