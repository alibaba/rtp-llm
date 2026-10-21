import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.base.common.fp8_linear import (
    make_projection,
    project_fp8_pair,
)
from rtp_llm.models_py.modules.hybrid.gated_residual import grouped_rms_norm
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpPLELayer
from rtp_llm.models_py.modules.qwen4_exp.ple.ple_conv_triton import try_ple_conv
from rtp_llm.models_py.modules.qwen4_exp.ple.ple_gate_triton import try_ple_gate


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PleGateTest(unittest.TestCase):
    def tensors(self, rows):
        torch.manual_seed(7300 + rows)
        shapes = [(1, rows, 10240), (1, rows, 2560), (1, rows, 10240)]
        return [torch.randn(s, device="cuda", dtype=torch.bfloat16) for s in shapes] + [
            torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07
            for _ in range(3)
        ]

    def reference(self, key, value, hidden, gk, gq, gc):
        kn = grouped_rms_norm(key, gk, 2560, 1e-6).unflatten(-1, (4, 2560))
        qn = grouped_rms_norm(hidden, gq, 2560, 1e-6).unflatten(-1, (4, 2560))
        gate = (kn * qn).sum(-1, keepdim=True) / math.sqrt(2560)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
        return gated, grouped_rms_norm(gated, gc, 2560, 1e-6)

    def test_all_materialization_boundaries(self):
        for rows in (1, 4, 16, 127, 128, 257, 4096):
            tensors = self.tensors(rows)
            for actual, expected in zip(
                try_ple_gate(*tensors, 1e-6), self.reference(*tensors)
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_batched_page_quant_and_projected_gate_are_exact(self):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            scaled_fp8_per_tensor_quant,
        )
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple_page_projection import (
            project_pages_pair,
            quantize_pages,
        )

        layer = self.layer()
        for rows in (128, 129, 257, 4093, 4095, 4096, 12287, 12288):
            torch.manual_seed(7820 + rows)
            x = torch.randn(1, rows, 2560, device="cuda", dtype=torch.bfloat16)
            x[:, : min(rows, 128)] *= 0.125
            quantized, scales = quantize_pages(
                x.squeeze(0), layer.key_projection.scale, layer.value_projection.scale
            )
            keys, values = [], []
            for start in range(0, rows, 128):
                page = x[:, start : start + 128].contiguous()
                padded = torch.empty(
                    ((page.shape[1] + 15) // 16 * 16, 2560),
                    device="cuda",
                    dtype=torch.float8_e4m3fn,
                )
                q, scale = scaled_fp8_per_tensor_quant(page.squeeze(0), output=padded)
                torch.testing.assert_close(
                    quantized[start : start + q.shape[0]].view(torch.uint8),
                    q.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    scales[0][start // 128 : start // 128 + 1], scale, rtol=0, atol=0
                )
                keys.append(layer.key_projection(page))
                values.append(layer.value_projection(page))
            key, value, sf = project_pages_pair(
                x, layer.key_projection, layer.value_projection
            )
            hidden = torch.randn(1, rows, 10240, device="cuda", dtype=torch.bfloat16)
            expected = try_ple_gate(
                torch.cat(keys, 1),
                torch.cat(values, 1),
                hidden,
                layer.norm_key,
                layer.norm_query,
                layer.norm_conv,
                layer.norm_eps,
            )
            actual = try_ple_gate(
                key,
                value,
                hidden,
                layer.norm_key,
                layer.norm_query,
                layer.norm_conv,
                layer.norm_eps,
                projection_scales=sf,
            )
            for candidate, reference in zip(actual, expected):
                torch.testing.assert_close(candidate, reference, rtol=0, atol=0)

    def test_shared_page_quant_keeps_both_projection_scales(self):
        for rows in (1, 17, 128):
            x = torch.randn(1, rows, 2560, device="cuda", dtype=torch.bfloat16)
            projections = [
                make_projection(
                    torch.randn(n, 2560, device="cuda").to(torch.float8_e4m3fn),
                    torch.tensor([scale], device="cuda"),
                )
                for n, scale in ((10240, 0.03), (2560, 0.07))
            ]
            expected = [p(x) for p in projections]
            actual = project_fp8_pair(x, *projections)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_batched_graph_conv_keeps_prefix_and_candidate_states(self):
        for batch, queries in ((1, 1), (1, 4), (4, 4), (8, 4)):
            key, value, hidden, gk, gq, gc = self.tensors(batch * queries)
            key = key.reshape(batch, queries, 10240)
            value = value.reshape(batch, queries, 2560)
            hidden = hidden.reshape(batch, queries, 10240)
            initial = torch.randn(batch, 9, 10240, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(10240, 1, 4, device="cuda", dtype=torch.bfloat16) * 0.1

            def call():
                gated, normed = try_ple_gate(key, value, hidden, gk, gq, gc, 1e-6)
                return try_ple_conv(gated, normed, initial, weight, 3), normed

            for _ in range(3):
                call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual, candidate = call()
            for seed in (7331, 7332):
                torch.manual_seed(seed)
                key.normal_()
                value.normal_()
                hidden.normal_()
                initial.normal_()
                before = initial.clone()
                gated, normed = self.reference(key, value, hidden, gk, gq, gc)
                expected = gated + F.silu(
                    F.conv1d(
                        torch.cat((initial, normed), 1).transpose(1, 2),
                        weight,
                        groups=10240,
                        dilation=3,
                    )
                ).transpose(1, 2)
                graph.replay()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(candidate, normed, rtol=0, atol=0)
                torch.testing.assert_close(initial, before, rtol=0, atol=0)

    def layer(self):
        def w(shape):
            return (torch.randn(shape, device="cuda") * 0.02).to(torch.float8_e4m3fn)

        return Qwen4ExpPLELayer(
            SimpleNamespace(ngram_size=3, context_len=2),
            w((10240, 2560)),
            w((2560, 2560)),
            torch.randn(10240, 1, 4, device="cuda", dtype=torch.bfloat16) * 0.1,
            *[
                torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07
                for _ in range(3)
            ],
            hc_mult=4,
            hidden_size=2560,
            conv_kernel_size=4,
            norm_eps=1e-6,
            projection_scales=[
                torch.tensor([0.03], device="cuda"),
                torch.tensor([0.07], device="cuda"),
            ],
        )

    def serial(self, layer, hidden, tokens, embeddings, initial):
        state = initial
        outputs, snapshots = [], []
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.ple.ple_gate_triton.try_ple_gate",
            return_value=None,
        ):
            for start in range(0, hidden.shape[1], 128):
                gated, normed = layer._gated_values(
                    hidden[:, start : start + 128],
                    tokens,
                    embeddings=embeddings[:, start : start + 128].contiguous(),
                )
                if state is None:
                    output = gated + layer._short_conv(normed)
                    state = layer.prefill_conv_state(normed)
                else:
                    history = torch.cat((state, normed), 1)
                    output = gated + F.silu(
                        F.conv1d(
                            history.transpose(1, 2),
                            layer.conv_weight,
                            groups=10240,
                            dilation=3,
                        )
                    ).transpose(1, 2)
                    state = history[:, -9:]
                outputs.append(output)
                snapshots.append(state)
        return torch.cat(outputs, 1), snapshots

    def test_all_page_snapshots_tail_and_prefix(self):
        layer = self.layer()
        for length in (1, 7, 127, 128, 129, 257, 4096):
            hidden = torch.randn(1, length, 10240, device="cuda", dtype=torch.bfloat16)
            embeddings = torch.randn(
                1, length, 2560, device="cuda", dtype=torch.bfloat16
            )
            tokens = torch.zeros(1, length + 2, device="cuda", dtype=torch.int64)
            for initial in (
                None,
                torch.randn(1, 9, 10240, device="cuda", dtype=torch.bfloat16),
            ):
                expected, states = self.serial(
                    layer, hidden, tokens, embeddings, initial
                )
                actual, snapshots = layer.prefill_page_snapshots(
                    hidden, tokens, 128, initial, embeddings=embeddings
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(len(states), len(snapshots))
                for a, b in zip(snapshots, states):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
