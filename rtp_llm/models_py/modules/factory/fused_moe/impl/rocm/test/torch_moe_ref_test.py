"""CPU-only regression tests for ``torch_moe_ref``.

The executor's ROCm end-to-end tests (``rocm_fp8_fused_moe_test.py``) only
run on ROCm CI. The original ``torch_moe_ref`` implementation referenced
``weighted`` outside the if-branch that defined it, so calling with
``apply_router_weight_on_input=True`` raised ``UnboundLocalError``. We want
that bug to be regression-tested in any CI that can run plain PyTorch CPU,
not only on ROCm hardware.

These tests construct tiny payloads / weights on CPU in fp32 and exercise
both branches of ``apply_router_weight_on_input``.
"""

import unittest
from unittest import SkipTest

import torch

try:
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
        ExpertForwardPayload,
        ExpertTokensMetadata,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.deepep_normal_fused_moe_executor import (
        torch_moe_ref,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # aiter / librtp_compute_ops.so not present
    _IMPORT_ERROR = exc


def _make_payload(M: int, D: int, E: int, top_k: int) -> "ExpertForwardPayload":
    hidden_states = torch.randn(M, D, dtype=torch.float32) * 0.05
    topk_ids = torch.topk(torch.rand(M, E), top_k, dim=1).indices.to(torch.int32)
    topk_weights = torch.softmax(torch.randn(M, top_k, dtype=torch.float32), dim=-1)
    return ExpertForwardPayload(
        expert_x=hidden_states,
        expert_x_origin_dtype=hidden_states.dtype,
        expert_x_scale=None,
        expert_tokens_meta=ExpertTokensMetadata(None, None, None),
        expert_topk_ids=topk_ids,
        expert_topk_weights=topk_weights,
    )


class TorchMoeRefRouterWeightTest(unittest.TestCase):
    """PR #882 review #4: ensure both router-weight branches run end-to-end."""

    M, D, N, E, TOP_K = 4, 16, 16, 4, 2

    def setUp(self):
        if _IMPORT_ERROR is not None:
            raise SkipTest(f"deepep executor import failed: {_IMPORT_ERROR}")
        torch.manual_seed(7)

    def _weights(self):
        w1 = torch.randn(self.E, 2 * self.N, self.D, dtype=torch.float32) * 0.02
        w2 = torch.randn(self.E, self.D, self.N, dtype=torch.float32) * 0.02
        return w1, w2

    def _run(self, apply_router_weight_on_input: bool, top_k: int) -> torch.Tensor:
        payload = _make_payload(self.M, self.D, self.E, top_k)
        # apply_router_weight_on_input is only meaningful with top_k == 1 in
        # the production path, but the ref accepts any top_k - we still want
        # to exercise the code path with top_k == 1 to mirror real usage.
        if top_k == 1:
            payload.expert_topk_ids = payload.expert_topk_ids[:, :1]
            payload.expert_topk_weights = payload.expert_topk_weights[:, :1]

        w1, w2 = self._weights()
        return torch_moe_ref(
            payload=payload,
            activation="silu",
            global_num_experts=self.E,
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=None,
            w1=w1,
            w2=w2,
        )

    def test_apply_router_weight_on_input_false_runs(self):
        out = self._run(apply_router_weight_on_input=False, top_k=self.TOP_K)
        self.assertEqual(out.shape, (self.M, self.D))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_apply_router_weight_on_input_true_runs(self):
        # This path used to raise UnboundLocalError because ``weighted`` was
        # only assigned inside ``if apply_router_weight_on_input == False``.
        out = self._run(apply_router_weight_on_input=True, top_k=1)
        self.assertEqual(out.shape, (self.M, self.D))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_two_branches_agree_when_weights_are_one(self):
        """With top_k=1 and topk_weight=1.0, both branches must give the
        same final output (the two branches differ only in whether the
        weighting is applied to the input or the output, and 1.0 makes
        them numerically equivalent)."""
        payload = _make_payload(self.M, self.D, self.E, top_k=1)
        payload.expert_topk_ids = payload.expert_topk_ids[:, :1]
        payload.expert_topk_weights = torch.ones_like(payload.expert_topk_weights[:, :1])

        w1, w2 = self._weights()
        kwargs = dict(
            payload=payload,
            activation="silu",
            global_num_experts=self.E,
            expert_map=None,
            a2_scale=None,
            extra_expert_args=None,
            w1=w1,
            w2=w2,
        )
        out_false = torch_moe_ref(apply_router_weight_on_input=False, **kwargs)
        out_true = torch_moe_ref(apply_router_weight_on_input=True, **kwargs)
        torch.testing.assert_close(out_false, out_true, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
