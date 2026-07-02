"""End-to-end correctness test for DeepGemmBf16HybridExecutor.

Validates the full bf16 DeepEP-normal expert path (scatter -> deepgemm grouped GEMM
-> silu_and_mul -> deepgemm grouped GEMM -> gather with router weight) against a
plain-torch reference, for BOTH runtime paths:

- masked     (token_num <= masked_max_token_num): 3D [E, alignment, K] layout
- contiguous (token_num >  masked_max_token_num): flat [Σ align(ei, 128), K] layout

and for BOTH single-rank (ep_size=1) and multi-rank Expert Parallelism (ep_size>1).

Unlike DeepGemmMaskedExecutor (which consumes a pre-grouped 3D payload), the hybrid
executor consumes a FLAT [M, K] DeepEP-normal payload and does the scatter itself, so
this test builds a flat payload and a flat reference rather than reusing
fused_moe_executor_test_util.

EP coverage (ep_size>1) without a real multi-process DeepEP buffer: a true pass through
DeepepNormalRouter.prepare/finalize requires NVSHMEM all-to-all across processes and is
not single-process testable. Instead we simulate the post-dispatch state — the payload's
expert_topk_ids carry GLOBAL expert ids and only this rank's per-local-expert token
counts / weights are provided — so the executor's _to_local_expert_ids mapping (global ->
local id + masking experts outside [start_expert_id, end_expert_id]) is exercised, and
the reference computes ONLY this rank's in-partition contribution (the per-rank partial
that DeepEP combine would later sum across ranks). This is the same simulate-EP pattern
used by deepep_normal_executor_test.py.

Requires the optional `deep_gemm` package; skips otherwise. Tagged H20 so internal CI
routes it to an SM>=9 worker where deep_gemm is available; open_skip keeps it out of the
open-source public CI lane.
"""

import unittest
from typing import Dict, Tuple

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm_bf16_grouped,
)
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
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_bf16_hybrid_executor import (
    DeepGemmBf16HybridExecutor,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.utils.model_weight import W


class DeepGemmBf16HybridExecutorTestBase:
    NUM_EXPERTS = 8
    TOP_K = 4
    HIDDEN_SIZE = 2048
    MOE_INTERMEDIATE_SIZE = 768  # N = 2 * 768 = 1536
    MASKED_MAX_TOKEN_NUM = 256

    @property
    def N(self) -> int:
        return self.MOE_INTERMEDIATE_SIZE * 2

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        if not has_deep_gemm_bf16_grouped():
            self.skipTest("deep_gemm bf16 grouped GEMM kernels required")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def _generate_config(self, ep_size: int = 1, ep_rank: int = 0) -> MoEConfigAdapter:
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
        parallelism_config.world_size = ep_size
        parallelism_config.dp_size = 1
        parallelism_config.tp_size = 1
        parallelism_config.ep_size = ep_size
        parallelism_config.dp_rank = 0
        parallelism_config.tp_rank = 0
        parallelism_config.ep_rank = ep_rank
        parallelism_config.world_rank = ep_rank
        parallelism_config.local_world_size = ep_size

        moe_config = MoeConfig()
        moe_config.masked_max_token_num = self.MASKED_MAX_TOKEN_NUM
        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
        )

    def _make_payload_and_weights(
        self, token_num: int, ep_size: int = 1, ep_rank: int = 0
    ) -> Tuple[ExpertForwardPayload, Dict[str, torch.Tensor], torch.Tensor]:
        device = "cuda"
        K = self.HIDDEN_SIZE
        N = self.N
        num_experts = self.NUM_EXPERTS
        num_local_experts = num_experts // ep_size
        start_expert_id = ep_rank * num_local_experts

        expert_x = (
            torch.rand((token_num, K), device=device, dtype=torch.float32) * 0.1 - 0.05
        ).to(torch.bfloat16)

        # Each token routes to TOP_K *distinct* GLOBAL experts. With ep_size>1 only the
        # subset falling in [start_expert_id, start_expert_id + num_local_experts) is owned
        # by this rank; the executor masks out the rest (mirrors the real dispatched state).
        topk_ids = torch.empty(
            (token_num, self.TOP_K), device=device, dtype=torch.int32
        )
        for t in range(token_num):
            topk_ids[t] = torch.randperm(num_experts, device=device)[: self.TOP_K].to(
                torch.int32
            )
        topk_weights = torch.rand(
            (token_num, self.TOP_K), device=device, dtype=torch.float32
        )

        # Per-LOCAL-expert token counts (length num_local_experts): how many tokens route
        # to each expert this rank owns. Distinct top-k => each token contributes <= 1 per
        # expert, so .any(dim=1) matches what ep_scatter places.
        counts = torch.zeros(num_local_experts, device=device, dtype=torch.int32)
        for local_e in range(num_local_experts):
            g = start_expert_id + local_e
            counts[local_e] = (topk_ids == g).any(dim=1).sum().to(torch.int32)

        payload = ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=None,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=counts,
                expert_num_tokens_cpu=counts.tolist(),
            ),
        )

        # Weights are per-LOCAL-expert ([num_local_experts, ...]); the executor indexes them
        # by local id, and the reference uses weights[local] for global expert start+local.
        weights = {
            W.moe_w1: (
                torch.rand(
                    (num_local_experts, N, K), device=device, dtype=torch.float32
                )
                * 2
                - 1
            ).to(torch.bfloat16),
            W.moe_w2: (
                torch.rand(
                    (num_local_experts, K, N // 2), device=device, dtype=torch.float32
                )
                * 2
                - 1
            ).to(torch.bfloat16),
            W.moe_s1: None,
            W.moe_s2: None,
        }
        # Keep a pristine copy of the inputs for the reference (executor disposes tensors).
        ref_input = expert_x.clone()
        return payload, weights, ref_input

    def _reference(
        self,
        expert_x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        ep_size: int = 1,
        ep_rank: int = 0,
    ) -> torch.Tensor:
        """Plain-torch flat MoE reference, gate/value split matching the deepgemm path.

        Computes only the experts THIS rank owns (global id in
        [start_expert_id, start_expert_id + num_local_experts)); other experts contribute
        nothing here (DeepEP combine sums the per-rank partials across ranks).
        """
        M, K = expert_x.shape
        N = self.N
        num_local_experts = self.NUM_EXPERTS // ep_size
        start_expert_id = ep_rank * num_local_experts
        w1 = weights[W.moe_w1]
        w2 = weights[W.moe_w2]
        out = torch.zeros((M, K), device=expert_x.device, dtype=torch.float32)
        for local_e in range(num_local_experts):
            g = start_expert_id + local_e
            sel = topk_ids == g  # [M, TOP_K]
            if not bool(sel.any()):
                continue
            tok_idx, k_idx = sel.nonzero(as_tuple=True)
            x = expert_x[tok_idx]  # [n, K] bf16
            ws1 = x @ w1[local_e].transpose(0, 1)  # [n, N]
            gate = ws1[..., N // 2 :].to(torch.float32)
            value = ws1[..., : N // 2].to(torch.float32)
            gate = gate * (1.0 / (1.0 + torch.exp(-gate)))
            ws2 = (gate * value).to(torch.bfloat16)  # [n, N//2]
            down = ws2 @ w2[local_e].transpose(0, 1)  # [n, K]
            weight = topk_weights[tok_idx, k_idx].unsqueeze(1).to(torch.float32)
            out.index_add_(0, tok_idx, down.to(torch.float32) * weight)
        return out.to(torch.bfloat16)

    def _run_path(
        self,
        token_num: int,
        expect_masked: bool,
        ep_size: int = 1,
        ep_rank: int = 0,
    ) -> None:
        config = self._generate_config(ep_size=ep_size, ep_rank=ep_rank)
        self.assertEqual(
            token_num <= config.masked_max_token_num,
            expect_masked,
            "token_num does not select the intended path",
        )

        payload, weights, ref_input = self._make_payload_and_weights(
            token_num, ep_size=ep_size, ep_rank=ep_rank
        )
        ref_output = self._reference(
            ref_input,
            payload.expert_topk_ids,
            payload.expert_topk_weights,
            weights,
            ep_size=ep_size,
            ep_rank=ep_rank,
        )

        executor = DeepGemmBf16HybridExecutor(
            config, FusedMoEQuantConfig(quant_dtype=None), weights
        )
        combine_payload = executor.execute(payload, "silu", None, None, False, None)
        out = combine_payload.fused_expert_output

        self.assertEqual(out.shape, ref_output.shape)
        # bf16 grouped GEMM + fp32-accumulated topk-weighted gather vs torch reference;
        # looser than the single-expert masked executor test (<0.003).
        diff = calc_diff(out, ref_output)
        self.assertLess(
            diff,
            0.01,
            f"output diff {diff} too large (token_num={token_num}, "
            f"ep_size={ep_size}, ep_rank={ep_rank})",
        )

    # ---- single rank (ep_size=1) ----
    def test_masked_path(self):
        # token_num <= masked_max_token_num -> 3D masked layout (decode)
        self._run_path(token_num=128, expect_masked=True)

    def test_contiguous_path(self):
        # token_num > masked_max_token_num -> flat contiguous layout (prefill)
        self._run_path(token_num=512, expect_masked=False)

    # ---- multi-rank Expert Parallelism (ep_size=2) ----
    # Exercises _to_local_expert_ids: global->local id mapping + masking experts outside
    # this rank's [start_expert_id, end_expert_id] partition.
    def test_masked_path_ep2_rank0(self):
        self._run_path(token_num=128, expect_masked=True, ep_size=2, ep_rank=0)

    def test_contiguous_path_ep2_rank0(self):
        self._run_path(token_num=512, expect_masked=False, ep_size=2, ep_rank=0)

    def test_masked_path_ep2_rank1(self):
        # ep_rank=1 -> non-zero start_expert_id (4); validates the local-id offset.
        self._run_path(token_num=128, expect_masked=True, ep_size=2, ep_rank=1)

    def test_contiguous_path_ep2_rank1(self):
        self._run_path(token_num=512, expect_masked=False, ep_size=2, ep_rank=1)

    # ---- empty rank (DeepEP small-batch / skewed routing -> 0 tokens) ----
    def _run_empty_rank(self, ep_size: int, ep_rank: int) -> None:
        config = self._generate_config(ep_size=ep_size, ep_rank=ep_rank)
        payload, weights, _ = self._make_payload_and_weights(
            token_num=0, ep_size=ep_size, ep_rank=ep_rank
        )
        executor = DeepGemmBf16HybridExecutor(
            config, FusedMoEQuantConfig(quant_dtype=None), weights
        )
        # Must not launch 0-grid scatter / 0-size DeepGEMM; returns an empty,
        # same-shape [0, K] output.
        out = executor.execute(payload, "silu", None, None, False, None)
        self.assertEqual(tuple(out.fused_expert_output.shape), (0, self.HIDDEN_SIZE))
        self.assertEqual(out.fused_expert_output.dtype, torch.bfloat16)

    def test_empty_rank(self):
        self._run_empty_rank(ep_size=1, ep_rank=0)

    def test_empty_rank_ep2(self):
        self._run_empty_rank(ep_size=2, ep_rank=1)


class DeepGemmBf16HybridExecutorTest(
    DeepGemmBf16HybridExecutorTestBase, unittest.TestCase
):
    pass


if __name__ == "__main__":
    unittest.main()
