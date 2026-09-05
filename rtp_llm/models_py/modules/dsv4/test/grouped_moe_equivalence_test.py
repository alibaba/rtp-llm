"""Real-GPU numerical coverage for grouped DSV4 MoE execution.

The test builds the production ``MoE`` layer from current ``W.v4_*`` tensor
descriptors. It compares the automatically available grouped-FP4
implementation against the explicit local-loop rollback path, including the
standalone shared expert used by the production single-rank model, and
exercises the zero-token boundary without mocks or fake strategies.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.moe_layer import Dsv4MoeLayer as MoE
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4 import (
    GroupedFp4Executor,
    _has_fp8_fp4_grouped_kernel,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop import (
    LocalLoopExecutor,
)
from rtp_llm.utils.model_weight import W


def _make_layer_weights(
    experts: int,
    dim: int,
    inter_dim: int,
    device: str,
    *,
    stable_routing: bool = False,
    routing_offset: int = 0,
    include_shared: bool = True,
) -> dict:
    def packed_fp4(out_dim: int, in_dim: int) -> torch.Tensor:
        return torch.randint(
            -10,
            10,
            (experts, out_dim, in_dim // 2),
            dtype=torch.int8,
            device=device,
        )

    def fp4_scale(out_dim: int, in_dim: int) -> torch.Tensor:
        # Exercise the real block-scale layout instead of letting every block
        # collapse to the same scale. Values close to the neutral UE8M0 byte
        # 127 keep the output large enough for relative-error checks to matter.
        return torch.randint(
            124,
            127,
            (experts, out_dim, in_dim // 32),
            dtype=torch.uint8,
            device=device,
        ).view(torch.float8_e8m0fnu)

    router_w = torch.randn(experts, dim, dtype=torch.bfloat16, device=device)
    router_bias = torch.zeros(experts, dtype=torch.float32, device=device)
    if stable_routing:
        # Hash routing keeps both implementations on identical routes while
        # covering every expert; a monotonic bias would exercise only the same
        # top-k experts in every layer.
        router_w.zero_()

    weights = {
        W.v4_router_w: router_w,
        W.v4_router_bias: router_bias,
        W.v4_routed_w1_w: packed_fp4(inter_dim, dim),
        W.v4_routed_w1_s: fp4_scale(inter_dim, dim),
        W.v4_routed_w2_w: packed_fp4(dim, inter_dim),
        W.v4_routed_w2_s: fp4_scale(dim, inter_dim),
        W.v4_routed_w3_w: packed_fp4(inter_dim, dim),
        W.v4_routed_w3_s: fp4_scale(inter_dim, dim),
    }
    if stable_routing:
        token_ids = torch.arange(32, dtype=torch.int32, device=device).unsqueeze(1)
        topk_slots = torch.arange(4, dtype=torch.int32, device=device).unsqueeze(0)
        weights[W.v4_router_tid2eid] = (
            token_ids * 4 + topk_slots + routing_offset
        ).remainder(experts)
    weights.update(
        {
            W.v4_shared_w13_w: torch.randn(
                2 * inter_dim,
                dim,
                dtype=torch.bfloat16,
                device=device,
            ).to(torch.float8_e4m3fn),
            W.v4_shared_w13_s: torch.randint(
                124,
                127,
                (2 * inter_dim // 128, dim // 128),
                dtype=torch.uint8,
                device=device,
            ).view(torch.float8_e8m0fnu),
            W.v4_shared_w2_w: torch.randn(
                dim,
                inter_dim,
                dtype=torch.bfloat16,
                device=device,
            ).to(torch.float8_e4m3fn),
            W.v4_shared_w2_s: torch.randint(
                124,
                127,
                (dim // 128, inter_dim // 128),
                dtype=torch.uint8,
                device=device,
            ).view(torch.float8_e8m0fnu),
        }
    )
    if not include_shared:
        for key in (
            W.v4_shared_w13_w,
            W.v4_shared_w13_s,
            W.v4_shared_w2_w,
            W.v4_shared_w2_s,
        ):
            weights.pop(key)
    return weights


def _clone_weights(weights: dict) -> dict:
    return {key: value.clone() for key, value in weights.items()}


class GroupedMoEExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA is required by this dedicated SM100 Bazel target"
            )
        if not _has_fp8_fp4_grouped_kernel():
            raise AssertionError(
                "SM100 grouped FP8xFP4 DeepGEMM kernel is required by this "
                "dedicated SM100 Bazel target"
            )

    def _build_moe(
        self,
        layer_weights: dict,
        strategy: str,
        *,
        layer_id: int = 3,
        hash_routing: bool = False,
        n_shared_experts: int = 1,
    ) -> MoE:
        return MoE(
            layer_id=layer_id,
            dim=512,
            moe_inter_dim=256,
            n_routed_experts=16,
            n_activated_experts=4,
            n_shared_experts=n_shared_experts,
            score_func="sqrtsoftplus",
            route_scale=1.0,
            swiglu_limit=10.0,
            n_hash_layers=4 if hash_routing else 0,
            vocab_size=32,
            layer_weights=layer_weights,
            ep_size=1,
            ep_rank=0,
            max_tokens_per_rank=128,
            strategy=strategy,
        )

    def test_grouped_matches_local_loop_and_handles_empty_rank(self):
        torch.manual_seed(20260901)
        weights = _make_layer_weights(16, 512, 256, "cuda")
        grouped = self._build_moe(_clone_weights(weights), "grouped_fp4")
        local = self._build_moe(_clone_weights(weights), "local_loop")

        self.assertIsInstance(grouped.fused_moe.fused_experts, GroupedFp4Executor)
        self.assertIsInstance(local.fused_moe.fused_experts, LocalLoopExecutor)
        for moe in (grouped, local):
            self.assertIsNotNone(moe._moe.shared_experts)
            self.assertIsNotNone(moe._moe._shared_executor)

        x = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
        input_ids = torch.arange(8, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            grouped_out = grouped(x, input_ids).clone()
            local_out = local(x, input_ids)

        self.assertNotEqual(grouped_out.data_ptr(), local_out.data_ptr())
        self.assertEqual(grouped_out.shape, x.shape)
        self.assertEqual(local_out.shape, x.shape)
        diff = (grouped_out.float() - local_out.float()).abs()
        self.assertGreater(grouped_out.float().abs().mean().item(), 1.0e-3)
        self.assertGreater(local_out.float().abs().mean().item(), 1.0e-3)
        scale = local_out.float().abs().mean().item() + 1e-6
        self.assertLess(
            diff.mean().item() / scale,
            0.03,
            "grouped_fp4 diverged from the local_loop rollback path",
        )

        empty_x = torch.empty(0, 512, dtype=torch.bfloat16, device="cuda")
        empty_ids = torch.empty(0, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            self.assertEqual(grouped(empty_x, empty_ids).shape, empty_x.shape)
            self.assertEqual(local(empty_x, empty_ids).shape, empty_x.shape)

    def test_routed_only_grouped_matches_local_loop(self):
        torch.manual_seed(20260902)
        weights = _make_layer_weights(16, 512, 256, "cuda", include_shared=False)
        grouped = self._build_moe(
            _clone_weights(weights), "grouped_fp4", n_shared_experts=0
        )
        local = self._build_moe(
            _clone_weights(weights), "local_loop", n_shared_experts=0
        )
        for moe in (grouped, local):
            self.assertIsNone(moe._moe.shared_experts)
            self.assertIsNone(moe._moe._shared_executor)

        x = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
        input_ids = torch.arange(8, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            grouped_out = grouped(x, input_ids).clone()
            local_out = local(x, input_ids)

        diff = (grouped_out.float() - local_out.float()).abs()
        scale = local_out.float().abs().mean().item() + 1e-6
        self.assertGreater(scale, 1.0e-3)
        # This denominator contains routed output only, so the bound cannot be
        # diluted by the identical shared-expert contribution.
        self.assertLess(
            diff.mean().item() / scale,
            0.04,
            "routed-only grouped_fp4 diverged from local_loop",
        )

    def test_four_layer_chain_has_no_systematic_error_compounding(self):
        grouped_layers = []
        local_layers = []
        for layer_id in range(4):
            torch.manual_seed(20262000 + layer_id)
            weights = _make_layer_weights(
                16,
                512,
                256,
                "cuda",
                stable_routing=True,
                routing_offset=layer_id * 4,
            )
            grouped_layers.append(
                self._build_moe(
                    _clone_weights(weights),
                    "grouped_fp4",
                    layer_id=layer_id,
                    hash_routing=True,
                )
            )
            local_layers.append(
                self._build_moe(
                    _clone_weights(weights),
                    "local_loop",
                    layer_id=layer_id,
                    hash_routing=True,
                )
            )

        torch.manual_seed(20262010)
        grouped_out = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
        local_out = grouped_out.clone()
        input_ids = torch.arange(8, dtype=torch.long, device="cuda")
        for layer in grouped_layers:
            covered = set(layer.gate.tid2eid[input_ids].flatten().cpu().tolist())
            self.assertEqual(covered, set(range(16)))
        with torch.inference_mode():
            for grouped, local in zip(grouped_layers, local_layers):
                # Mirror the transformer boundary around each MoE: pre-norm
                # each branch independently, then carry the residual forward.
                # This keeps the chained regression sensitive to systematic
                # kernel drift without turning tiny rounding differences into
                # unbounded synthetic-MLP magnitude growth.
                grouped_in = F.rms_norm(grouped_out.float(), (512,)).to(torch.bfloat16)
                local_in = F.rms_norm(local_out.float(), (512,)).to(torch.bfloat16)
                grouped_delta = grouped(grouped_in, input_ids).clone()
                local_delta = local(local_in, input_ids).clone()
                self.assertNotEqual(grouped_delta.data_ptr(), local_delta.data_ptr())
                grouped_out = (grouped_out + grouped_delta).to(torch.bfloat16).clone()
                self.assertGreater(grouped_delta.float().abs().mean().item(), 1.0e-3)
                self.assertGreater(local_delta.float().abs().mean().item(), 1.0e-3)
                local_out = (local_out + local_delta).to(torch.bfloat16).clone()
                self.assertNotEqual(grouped_out.data_ptr(), local_out.data_ptr())

        self.assertTrue(torch.isfinite(grouped_out).all())
        self.assertTrue(torch.isfinite(local_out).all())
        diff = (grouped_out.float() - local_out.float()).abs()
        scale = local_out.float().abs().mean().item() + 1e-6
        self.assertLess(
            diff.mean().item() / scale,
            0.05,
            "four-layer grouped_fp4 error compounded beyond the rollback baseline",
        )


if __name__ == "__main__":
    unittest.main()
