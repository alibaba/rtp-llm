"""PPU checks for checkpoint-exact repacking and activation quantization."""

import unittest

import torch
import torch.nn.functional as F
from rtp_llm.device.device_type import is_ppu


@unittest.skipUnless(is_ppu(), "Requires PPU")
class A8MoeTest(unittest.TestCase):
    def test_tp_compaction_has_stable_duplicate_order(self):
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_grouped_a8 import _compact_routes_for_tp

        ids = torch.tensor([[2, 0, 2, -1], [0, 1, 3, 1]], device="cuda")
        expected = (
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 2, 2, 0, 0],
            [[4, 0, 5, -1], [1, 2, -1, 3]],
            [2, 2, 2],
        )
        for _ in range(3):
            for actual, values in zip(_compact_routes_for_tp(ids, 3), expected):
                self.assertEqual(actual.cpu().tolist(), values)

    def test_a8_output_shards_are_not_reduced_twice(self):
        from unittest.mock import patch
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as recorder
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_tp_moe import PpuTPMoE

        moe = PpuTPMoE.__new__(PpuTPMoE)
        torch.nn.Module.__init__(moe)
        moe.layer_id, moe._a8_experts, moe._shared_phases = 0, True, False
        x = torch.ones(5, 128, device="cuda", dtype=torch.bfloat16)
        moe.gate = lambda *args, **kwargs: (None, None)
        moe._forward_serial = lambda *args: x.float() * 3
        def forbidden_reduce(_):
            self.fail("A8 strategy already gathered full output shards")
        moe._reduce = forbidden_reduce
        with patch.object(recorder, "should_record_layer", return_value=False):
            actual = moe._forward_local_chunk(x, torch.zeros(5, device="cuda", dtype=torch.long))
        torch.testing.assert_close(actual, x * 3, rtol=0, atol=0)

    def test_tp_gather_keeps_fp32_until_reduction(self):
        from rtp_llm.models_py.kernels.ppu.moe.ppu_moe_exact_gather import (
            gather_and_combine_local_loop_compatible,
        )

        for tokens in (1, 6, 1024):
            ids = torch.tensor([[5, 0, 3, 1, 4, 2]], device="cuda", dtype=torch.int64).repeat(tokens, 1)
            inverse = torch.arange(tokens * 6, device="cuda", dtype=torch.int64).reshape(tokens, 6)
            weights = torch.ones(tokens, 6, device="cuda", dtype=torch.float32)
            down = torch.zeros(tokens, 6, 512, device="cuda", dtype=torch.bfloat16)
            down[:, 0] = 1.0
            down[:, 1] = 1 / 256
            shared = torch.full((tokens, 512), -1.0, device="cuda", dtype=torch.bfloat16)
            for keep in (False, True):
                out = torch.empty_like(shared, dtype=torch.float32 if keep else torch.bfloat16)

                def run():
                    gather_and_combine_local_loop_compatible(
                        down.reshape(-1, 512), ids, weights, inverse, shared, out, 1.0,
                        keep_fp32=keep,
                    )

                run()
                expected = torch.full_like(out, 1 / 256 if keep else 0)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_repack_all_nibbles_and_scale_span(self):
        from rtp_llm.models_py.kernels.ppu.moe.ppu_a8_moe import repack_fp4_to_fp8

        raw = (
            torch.arange(256, dtype=torch.int64)
            .to(torch.uint8)
            .repeat(64)
            .reshape(1, 128, 128)
        )
        scales = (torch.arange(8)[None, None, :].expand(1, 128, 8) + 115).to(
            torch.uint8
        )
        q, s = repack_fp4_to_fp8(raw.cuda(), scales.cuda().view(torch.float8_e8m0fnu))
        nibble = torch.stack((raw & 15, raw >> 4), -1).flatten(-2).long()
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        reference = lut[nibble] * (2.0 ** (scales.float() - 127)).repeat_interleave(
            32, -1
        )
        decoded = q.cpu().float() * s.cpu().repeat_interleave(128, 1).repeat_interleave(
            128, 2
        )
        torch.testing.assert_close(decoded, reference, rtol=0, atol=0)
        bad = scales.clone()
        bad[0, 0, 0] = 90
        with self.assertRaisesRegex(ValueError, "losslessly"):
            repack_fp4_to_fp8(raw.cuda(), bad.cuda().view(torch.float8_e8m0fnu))

    def test_activation_pow2_and_zero_rows(self):
        from rtp_llm.models_py.kernels.ppu.moe.ppu_a8_moe import quantize_a8

        torch.manual_seed(7)
        x = torch.randn(5, 512, dtype=torch.bfloat16)
        x[0].zero_()
        q, s = quantize_a8(x.cuda())
        groups = x.float().reshape(5, 4, 128)
        ref_s = 2.0 ** torch.ceil(
            torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448)
        )
        ref_q = (groups / ref_s[..., None]).to(torch.float8_e4m3fn).reshape_as(x)
        torch.testing.assert_close(s.cpu(), ref_s, rtol=0, atol=0)
        torch.testing.assert_close(q.cpu().float(), ref_q.float(), rtol=0, atol=0)




@unittest.skipUnless(is_ppu(), "Requires PPU")
class WeightedA8RoundingTest(unittest.TestCase):
    @staticmethod
    def run_weighted(inputs, limit):
        from rtp_llm.models_py.kernels.ppu.moe.ppu_a8_moe import weighted_nopad_a8
        return weighted_nopad_a8(*inputs, limit)

    # Real FP8 midpoint cases; each also carries its block's maximum pair.
    CASES = (
        (0.50390625, 0.349609375, 1.2265625, -0.87890625, 0.2228645384311676),
        (-0.71875, -0.62109375, 0.341796875, -1.0390625, 0.2812357544898987),
        (0.72265625, -0.1298828125, 1.0546875, -0.73046875, 0.22458505630493164),
        (0.87109375, -0.5703125, 0.87109375, -0.5703125, 0.23491975665092468),
    )

    def inputs(self, rows, inter, offset=0):
        gu = torch.zeros((rows, 2 * inter), device="cpu", dtype=torch.bfloat16)
        weights = torch.empty((rows, 1), device="cpu", dtype=torch.float32)
        for row in range(rows):
            gate, up, maximum_gate, maximum_up, weight = self.CASES[(row + offset) % len(self.CASES)]
            gu[row, 0], gu[row, 1] = maximum_gate, gate
            gu[row, inter], gu[row, inter + 1] = maximum_up, up
            weights[row, 0] = weight
        source = torch.arange(rows, device="cuda", dtype=torch.int64)
        experts = torch.zeros(rows, device="cuda", dtype=torch.int32)
        indices = torch.zeros((rows, 1), device="cuda", dtype=torch.int64)
        counts = torch.tensor([rows], device="cuda", dtype=torch.int32)
        return gu.cuda(), source, experts, indices, weights.cuda(), counts

    def reference(self, inputs, limit):
        gu, _, _, _, weights, _ = inputs
        gate, up = gu.float().chunk(2, -1)
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        value = (weights * (F.silu(gate) * up)).bfloat16()
        blocks = value.float().unflatten(-1, (-1, 128))
        scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448)))
        quant = (blocks / scale[..., None]).to(torch.float8_e4m3fn).flatten(-2)
        return quant, scale

    def check_output(self, actual, expected):
        self.assertTrue(torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8)))
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_rounding_boundaries_and_shapes(self):
        self.assertTrue(torch.cuda.is_available(), "This test requires a PPU worker")
        for rows, inter, limit in ((1, 512, 0.0), (3, 512, 0.0), (7, 2048, 1.0), (129, 2048, 0.0)):
            with self.subTest(rows=rows, inter=inter, limit=limit):
                inputs = self.inputs(rows, inter)
                expected = self.reference(inputs, limit)
                self.check_output(self.run_weighted(inputs, limit), expected)
                if rows == 3:
                    self.assertEqual(expected[0][:, 1].float().tolist(), [52.0, 160.0, -30.0])

    def test_graph_uses_changed_inputs(self):
        inputs = self.inputs(3, 512)
        self.run_weighted(inputs, 0.0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = self.run_weighted(inputs, 0.0)
        for offset in (1, 2, 0):
            changed = self.inputs(3, 512, offset)
            inputs[0].copy_(changed[0])
            inputs[4].copy_(changed[4])
            graph.replay()
            self.check_output(result, self.reference(inputs, 0.0))



if __name__ == "__main__":
    unittest.main()
