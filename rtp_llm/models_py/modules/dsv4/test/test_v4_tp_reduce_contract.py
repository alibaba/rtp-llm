"""TP4 ordering and output ownership; kernel/collective leaves are CPU oracles."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.platforms.ppu.models.dsv4 import ppu_tp_moe


class PpuTPReduceContractTest(unittest.TestCase):
    def model(self, capacity=3):
        model = ppu_tp_moe.PpuTPMoE.__new__(ppu_tp_moe.PpuTPMoE)
        nn.Module.__init__(model)
        model.layer_id = 0
        model.dim = 2
        model.max_tokens_per_rank = capacity
        model.route_scale = 1.5
        events = []

        def gate(x, ids, *, include_route_scale=True):
            self.assertFalse(include_route_scale)
            events.append("unscaled_gate")
            return torch.ones(x.size(0), 1), ids.reshape(-1, 1)

        def shared(x):
            events.append("shared")
            return x * 2

        def routed(x, weights, indices):
            events.append("routed")
            self.assertTrue(torch.equal(weights, torch.ones_like(weights)))
            return x.float() * 0.37

        def combine(routed, shared, scale):
            events.append("combine")
            self.assertEqual(
                (routed.dtype, shared.dtype), (torch.float32, torch.bfloat16)
            )
            return ((routed * scale).bfloat16().float() + shared.float()).bfloat16()

        def reduce(x):
            events.append("tp_reduce")
            self.assertEqual(x.dtype, torch.bfloat16)
            return x.mul_(4)

        model.gate = gate
        model.shared_experts = shared
        model._strategy = routed
        model._reduce = reduce
        return model, events, combine

    def test_scale_shared_add_then_one_bf16_reduce_per_chunk(self):
        for batch, chunks in ((1, 1), (3, 1), (7, 3)):
            model, events, combine = self.model()
            x = torch.arange(batch * 2, dtype=torch.bfloat16).reshape(batch, 2)
            with patch.object(
                ppu_tp_moe, "combine_tp_partials", side_effect=combine
            ), patch("torch.cuda.is_current_stream_capturing", return_value=False):
                actual = model(x, torch.arange(batch))
            expected = (
                (x.float() * 0.37 * 1.5).bfloat16().float() + (x * 2).float()
            ).bfloat16() * 4
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(
                events,
                ["unscaled_gate", "shared", "routed", "combine", "tp_reduce"] * chunks,
            )

    def test_retained_outputs_do_not_alias_other_calls(self):
        model, _, combine = self.model()
        with patch.object(
            ppu_tp_moe, "combine_tp_partials", side_effect=combine
        ), patch("torch.cuda.is_current_stream_capturing", return_value=False):
            first = model(torch.ones(7, 2, dtype=torch.bfloat16), torch.arange(7))
            expected = first.clone()
            second = model(torch.zeros(7, 2, dtype=torch.bfloat16), torch.arange(7))
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_decode_flag_rejected_before_any_collective(self):
        model, events, _ = self.model()
        with self.assertRaisesRegex(RuntimeError, "Prefill only"):
            model(
                torch.ones(1, 2, dtype=torch.bfloat16),
                torch.arange(1),
                is_decode_forward=True,
            )
        self.assertFalse(events)


if __name__ == "__main__":
    unittest.main()
