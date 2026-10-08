"""V4.1 FP32 routing and shared/routed FFN composition contracts."""

import dataclasses
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source
from rtp_llm.models.test.dsv41_collective_test import collective, dispatch
from rtp_llm.models.test.dsv41_expert_test import (
    expert,
    shared_reference,
    shared_weights,
)
from rtp_llm.models.test.dsv41_moe_test import (
    MODULES,
    local_reference,
    moe,
    routed,
    routed_weights,
)

with patch.dict(
    sys.modules,
    {
        **MODULES,
        "rtp_llm.models_py.modules.dsv4.v41_collective": collective,
    },
):
    ffn = load_codec_source("_dsv41_feed_forward_test", "v41_feed_forward.py")


def route_reference(
    x, matrix, bias, valid, topk=2, temperature=0.7, scale=1.5, normalize=True
):
    scores = F.softplus(
        torch.where(valid[:, None], x, 0).float() @ matrix.float().T / temperature
    ).sqrt()
    ids = (scores + bias).topk(topk, dim=-1).indices
    weights = scores.gather(1, ids)
    if normalize and topk > 1:
        weights /= weights.sum(-1, keepdim=True) + 1e-20
    return (
        torch.where(valid[:, None], weights * scale, 0),
        torch.where(valid[:, None], ids, -1).int(),
    )


def fixture():
    generator = torch.Generator().manual_seed(4145)
    matrix = (torch.randn(3, 64, generator=generator) / 8).bfloat16()
    bias = torch.tensor([0.5, -0.5, 0.1])
    router = ffn.V41Router(matrix, bias, topk=2, gate_temp=0.7, max_rows=8)
    shared = shared_weights()
    tensors = routed_weights()
    local = moe.V41LocalMoE(
        routed.V41RoutedExperts(tensors, backend="eager"),
        max_tokens=8,
        masked_token_limit=2,
    )
    op = ffn.V41FeedForward(
        14, router, expert.V41SharedExpert(shared, backend="eager"), local
    )
    x = torch.randn(4, 64, generator=generator).bfloat16()
    valid = torch.tensor([True, False, True, True])
    x[~valid] = float("nan")
    return op, matrix, bias, shared, tensors, x, valid


class V41FeedForwardTest(unittest.TestCase):
    def test_fp32_temperature_bias_selection_and_unbiased_weights(self):
        op, matrix, bias, _, _, x, valid = fixture()
        for normalize in (True, False):
            for topk in (1, 2, 3):
                router = ffn.V41Router(
                    matrix, bias, topk=topk, gate_temp=0.7, norm_topk_prob=normalize
                )
                weights, ids = router(x, valid)
                expected, expected_ids = route_reference(
                    x, matrix, bias, valid, topk=topk, normalize=normalize
                )
                self.assertEqual(weights.dtype, torch.float32)
                self.assertEqual(ids.dtype, torch.int32)
                torch.testing.assert_close(weights, expected, rtol=0, atol=0)
                torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        router = ffn.V41Router(
            torch.zeros(3, 64, dtype=torch.bfloat16),
            torch.tensor([100.0, 1.0, -10.0]),
            topk=2,
        )
        weights, ids = router(x, valid)
        self.assertEqual(ids[0].tolist(), [0, 1])
        torch.testing.assert_close(
            weights[0], torch.tensor([0.75, 0.75]), rtol=0, atol=0
        )

    def test_normalization_epsilon_is_checkpoint_value(self):
        matrix = torch.full((2, 1), -80.0, dtype=torch.float32)
        x, valid = torch.ones(1, 1, dtype=torch.bfloat16), torch.ones(
            1, dtype=torch.bool
        )
        router = ffn.V41Router(matrix, torch.zeros(2), topk=2, gate_temp=1.0)
        actual, _ = router(x, valid)
        score = F.softplus(torch.tensor(-80.0)).sqrt()
        torch.testing.assert_close(
            actual,
            torch.full((1, 2), score / (2 * score + 1e-20) * 1.5),
            rtol=0,
            atol=0,
        )
        self.assertGreater(actual.min().item(), 0.7)

    def test_full_ffn_matches_independent_expert_sum_and_shared_equations(self):
        op, matrix, bias, shared, tensors, x, valid = fixture()
        weights, ids = route_reference(x, matrix, bias, valid)
        partial = local_reference(x, weights, ids, valid, tensors, 0)
        expected = (partial + shared_reference(x, shared, valid).float()).bfloat16()
        actual = op(x, valid)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(bool((actual[~valid] == 0).all()))
        chunks = torch.cat([op(x[i : i + 2], valid[i : i + 2]) for i in (0, 2)])
        torch.testing.assert_close(chunks, expected, rtol=0, atol=0)
        order = torch.tensor([3, 1, 0, 2])
        torch.testing.assert_close(
            op(x[order], valid[order])[order.argsort()], expected, rtol=0, atol=0
        )

    def test_empty_and_all_dead_batches(self):
        op, _, _, _, _, x, valid = fixture()
        self.assertEqual(op(x[:0], valid[:0]).shape, (0, 64))
        x.fill_(float("inf"))
        self.assertTrue(bool((op(x, torch.zeros_like(valid)) == 0).all()))

    def test_router_rejects_malformed_and_nonfinite_live_inputs(self):
        op, matrix, bias, _, _, x, valid = fixture()
        for kwargs in (
            {"topk": 0},
            {"topk": 4},
            {"gate_temp": 0},
            {"route_scale": float("nan")},
            {"max_rows": False},
            {"norm_topk_prob": 1},
        ):
            with self.assertRaises(ValueError):
                ffn.V41Router(matrix, bias, **kwargs)
        with self.assertRaisesRegex(ValueError, "finite"):
            ffn.V41Router(matrix, bias * float("inf"))
        for bad_x, bad_mask in (
            (x.float(), valid),
            (x.T, valid),
            (x.repeat(3, 1), valid.repeat(3)),
            (x, valid.int()),
            (x.requires_grad_(), valid),
        ):
            with self.assertRaises(ValueError):
                op.router(bad_x, bad_mask)
        x = x.detach()
        x[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            op.router(x, valid)

    def test_ffn_rejects_partial_experts_and_mismatched_geometry(self):
        op, _, _, _, _, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "layer"):
            ffn.V41FeedForward(40, op.router, op.shared, op.routed)
        op.routed.expert_start_id = 1
        with self.assertRaisesRegex(ValueError, "complete"):
            ffn.V41FeedForward(14, op.router, op.shared, op.routed)
        op.routed.expert_start_id = 0
        op.router.max_rows = 9
        with self.assertRaisesRegex(ValueError, "geometry"):
            ffn.V41FeedForward(14, op.router, op.shared, op.routed)

    def test_collective_empty_rank_still_enters_transport_and_checks_plan(self):
        plan = collective.V41EPPlan(
            dispatch.V41DispatchLayout(128, 2, 8),
            "decode",
            0,
            14,
            tuple(range(8)),
            "ffn-test",
            2,
            intermediate_dim=128,
        )
        local = moe.V41LocalMoE(
            routed.V41RoutedExperts(
                routed_weights(experts=1, dim=128, inter=128), backend="eager"
            )
        )
        transport = collective.V41CollectiveMoE(local, plan, object())
        router = ffn.V41Router(torch.zeros(8, 128), torch.zeros(8), topk=2, max_rows=2)
        shared = expert.V41SharedExpert(
            shared_weights(dim=128, inter=128), backend="eager"
        )
        op = ffn.V41FeedForward(14, router, shared, transport)
        with patch.object(
            transport, "forward", return_value=torch.zeros(0, 128)
        ) as forward:
            self.assertEqual(
                op(
                    torch.empty(0, 128, dtype=torch.bfloat16),
                    torch.empty(0, dtype=torch.bool),
                ).shape,
                (0, 128),
            )
            forward.assert_called_once()
        transport.plan = dataclasses.replace(plan, layer_id=13)
        with self.assertRaisesRegex(ValueError, "plan"):
            ffn.V41FeedForward(14, router, shared, transport)

    def test_loader_binding_keeps_weight_dictionary_on_success_and_failure(self):
        from rtp_llm.utils.model_weight import W

        op, matrix, bias, shared, _, x, valid = fixture()
        tensors = {
            W.v4_router_w: matrix,
            W.v4_router_bias: bias,
            **{getattr(W, "v4_shared_" + key): value for key, value in shared.items()},
        }
        original = dict(tensors)
        actual = ffn.build_v41_feed_forward(
            14, tensors, op.routed, max_rows=8, topk=2, gate_temp=0.7, backend="eager"
        )
        torch.testing.assert_close(actual(x, valid), op(x, valid), rtol=0, atol=0)
        self.assertTrue(all(tensors[key] is value for key, value in original.items()))
        tensors[W.v4_router_bias] = bias.bfloat16()
        with self.assertRaises(ValueError):
            ffn.build_v41_feed_forward(14, tensors, op.routed, topk=2, backend="eager")
        self.assertEqual(tensors.keys(), original.keys())


if __name__ == "__main__":
    unittest.main()
