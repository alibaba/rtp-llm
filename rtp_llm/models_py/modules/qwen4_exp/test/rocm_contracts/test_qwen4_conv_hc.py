"""Qwen4 TP4 conv and native-FP8 HC contracts on actual ROCm hardware.

Checkpoint dimensions: H4/HV12/D128 => conv width 2560, kernel4, page128;
HC hidden2560, mult4, lowrank320. No checkpoint or distributed process needed.
"""

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.hybrid.gated_residual import (
    GatedResidual,
    inject_into_residual,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)


@unittest.skipUnless(
    torch.version.hip and torch.cuda.is_available(), "ROCm GPU required"
)
class Qwen4ConvHcTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261025)
        self.dim = (2 * 4 + 12) * 128
        self.weight = (torch.randn(self.dim, 4, device="cuda") * 0.1).bfloat16()

    def reference(self, tokens, history=None):
        x = tokens.float().T
        if history is None:
            x = F.pad(x, (3, 0))
        else:
            x = torch.cat((history.float().T, x), dim=1)
        return F.silu(F.conv1d(x, self.weight.float()[:, None], groups=self.dim)).T.to(
            tokens.dtype
        )

    def test_prefill_page_checkpoints_then_decode(self):
        lengths = [128, 257]
        tokens = torch.randn(
            sum(lengths), self.dim, device="cuda", dtype=torch.bfloat16
        )
        # Hybrid row stride and token-major state layout, with untouched tail.
        storage = torch.full(
            (10, self.dim * 3 + 32), 17.0, device="cuda", dtype=tokens.dtype
        )
        states = storage[:, : self.dim * 3].view(10, 3, self.dim).transpose(1, 2)
        table_storage = torch.tensor(
            [[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]], device="cuda", dtype=torch.int32
        )
        table = table_storage[:, :4]
        cu = torch.tensor([0, 128, 385], device="cuda", dtype=torch.int32)
        prefixes = torch.zeros(2, device="cuda", dtype=torch.int32)
        out = causal_conv1d_fn(
            tokens.T, self.weight, None, states, cu, table, prefixes, 128
        )
        offset = 0
        for row, length in enumerate(lengths):
            seq = tokens[offset : offset + length]
            torch.testing.assert_close(
                out.T[offset : offset + length],
                self.reference(seq),
                atol=0.01,
                rtol=0.02,
            )
            for page, end in enumerate(range(128, length + 128, 128)):
                end = min(end, length)
                torch.testing.assert_close(
                    states[table[row, page]].T, seq[end - 3 : end], atol=0, rtol=0
                )
            offset += length
        x = torch.randn(2, self.dim, device="cuda", dtype=tokens.dtype)
        decode = causal_conv1d_update(
            x,
            states,
            self.weight,
            activation="silu",
            block_map=table,
            sequence_lengths=torch.tensor([129, 258], device="cuda", dtype=torch.int32),
            seq_size_per_block=128,
        )
        for row, end in enumerate([128, 385]):
            torch.testing.assert_close(
                decode[row : row + 1],
                self.reference(x[row : row + 1], tokens[end - 3 : end]),
                atol=0.01,
                rtol=0.02,
            )
        torch.testing.assert_close(
            storage[:, -32:], torch.full_like(storage[:, -32:], 17), atol=0, rtol=0
        )

    def test_decode_graph_refreshes_inputs_lengths_and_pages(self):
        batch = 16
        x = torch.randn(batch, self.dim, device="cuda", dtype=torch.bfloat16)
        states = torch.randn(100, 3, self.dim, device="cuda", dtype=x.dtype).transpose(
            1, 2
        )
        lengths = torch.full((batch,), 128, device="cuda", dtype=torch.int32)
        table = torch.arange(1, 1 + batch * 4, device="cuda", dtype=torch.int32).view(
            batch, 4
        )

        def run():
            return causal_conv1d_update(
                x,
                states,
                self.weight,
                activation="silu",
                block_map=table,
                sequence_lengths=lengths,
                seq_size_per_block=128,
            )

        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for length in (129, 257):
            x.normal_()
            states.normal_()
            table.copy_(table.roll(1, 0))
            lengths.fill_(length)
            before = states.clone()
            histories = before[table[:, (length - 2) // 128]].transpose(1, 2)
            expected = torch.cat(
                [self.reference(x[i : i + 1], histories[i]) for i in range(batch)]
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
            for row in range(batch):
                written = states[table[row, (length - 1) // 128]].T
                torch.testing.assert_close(
                    written,
                    torch.cat((histories[row, 1:], x[row : row + 1])),
                    atol=0,
                    rtol=0,
                )

    def test_hc_native_fp8_prefill_decode_and_graph(self):
        def weight(shape):
            return (torch.randn(shape) * 0.05).to(torch.float8_e4m3fn).cuda()

        gamma = torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.02
        down, up, gate = weight((320, 10240)), weight((10240, 320)), weight((4, 10240))
        scale = torch.tensor([0.125], device="cuda")
        for role, rows in (("PREFILL", 129), ("DECODE", 16)):
            module = GatedResidual(
                gamma,
                down,
                up,
                gate,
                hc_mult=4,
                norm_eps=1e-6,
                weight_scales=[scale] * 3,
                role=role,
            )
            self.assertTrue(module.down_projection.use_scaled_mm)
            self.assertTrue(module.up_projection.use_scaled_mm)
            self.assertIsNone(module.down_projection.compute_weight)
            self.assertEqual(module.down_projection.weight.dtype, torch.float8_e4m3fnuz)
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)

            def run():
                mixed, residual, gates = module(x)
                return mixed, gates, inject_into_residual(residual, mixed, gates)

            actual = run()
            norm = x.float().view(rows, 4, 2560)
            norm = (
                norm * torch.rsqrt(norm.square().mean(-1, keepdim=True) + 1e-6)
            ).flatten(1)
            norm = (norm * (1 + gamma.float())).bfloat16()
            d = F.linear(norm, (down.float() * scale).bfloat16())
            logits = F.linear(F.silu(d / 4), (up.float() * scale).bfloat16())
            expected_mix = (
                logits.sigmoid().view(rows, 4, 2560) * norm.view(rows, 4, 2560)
            ).mean(1)
            expected_gate = (
                2 * (F.linear(norm, (gate.float() * scale).bfloat16()) / 4).sigmoid()
            )
            torch.testing.assert_close(actual[0], expected_mix, atol=0.02, rtol=0.03)
            torch.testing.assert_close(actual[1], expected_gate, atol=0.01, rtol=0.02)
            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            x.normal_()
            gamma.normal_(0, 0.03)
            expected = run()
            graph.replay()
            torch.cuda.synchronize()
            for result, reference in zip(captured, expected):
                torch.testing.assert_close(result, reference, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
