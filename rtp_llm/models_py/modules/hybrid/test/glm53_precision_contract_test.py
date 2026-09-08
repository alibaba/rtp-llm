"""Independent FP32 oracles for GLM router and four-tap convolution."""

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.glm53_router import Glm53FP32Router
from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d_fn
from rtp_llm.models_py.triton_kernels.kimi_kda.glm53_short_conv import (
    glm53_kda_short_conv_decode,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Glm53PrecisionContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(530909)
        self.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.old_tf32

    def test_router_fp32_logits_routing_and_graph(self):
        # Match serialized BF16 values, but promote before the projection.
        weight = (
            torch.randn(4096, 288, device="cuda", dtype=torch.bfloat16).float() / 64
        )
        router = Glm53FP32Router(weight)
        bias = torch.randn(288, device="cuda") * 0.01
        for batch in (1, 7, 48, 384, 8193):
            x = torch.randn(batch, 4096, device="cuda", dtype=torch.bfloat16)
            actual = router(x)
            expected = F.linear(x.float(), weight.T.contiguous())
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=4e-6)
            ids = (actual.sigmoid() + bias).topk(8, dim=-1).indices.sort(-1).values
            ref_ids = (
                (expected.sigmoid() + bias).topk(8, dim=-1).indices.sort(-1).values
            )
            torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
        x = torch.randn(48, 4096, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                router(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = router(x)
        x.normal_()
        graph.replay()
        torch.testing.assert_close(actual, router(x), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            Glm53FP32Router(weight.bfloat16())

    def test_prefill_convolution_fp32_before_rounding(self):
        channels = 384
        lengths = [1, 3, 127, 128, 129, 4097]
        x = torch.randn(sum(lengths), channels, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16).float()
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        actual = causal_conv1d_fn(
            x.T,
            w,
            None,
            None,
            cu,
            None,
            torch.zeros(len(lengths), device="cuda", dtype=torch.int32),
            128,
            activation="silu",
        ).T
        reference = []
        for begin, end in zip(offsets[:-1], offsets[1:]):
            taps = F.pad(x[begin:end].float().T, (3, 0)).unfold(-1, 4, 1)
            reference.append(F.silu((taps * w[:, None]).sum(-1)).T.bfloat16())
        expected = torch.cat(reference)
        torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=2e-6)
        self.assertLess(
            (
                (actual.float() - expected.float()).norm() / expected.float().norm()
            ).item(),
            1e-4,
        )

    def test_prefill_convolution_fp32_cached_prefix_and_page_boundaries(self):
        channels, page_size = 384, 128
        lengths = [1, 3, 127, 128, 129, 4097]
        prefixes = [128, 0, 256, 0, 128, 128]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        x = torch.randn(sum(lengths), channels, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16).float()
        pages_per_request = 34
        page_count = len(lengths) * pages_per_request
        table = torch.randperm(page_count, device="cuda", dtype=torch.int32).reshape(
            len(lengths), pages_per_request
        )
        # Cache layout matches production: the channel dimension is contiguous.
        state = torch.randn(
            page_count, 3, channels, device="cuda", dtype=torch.bfloat16
        )
        initial = state.clone()
        expected_state = state.clone()
        reference = []
        for row, (length, prefix) in enumerate(zip(lengths, prefixes)):
            history = (
                initial[table[row, (prefix - 1) // page_size]]
                if prefix
                else torch.zeros(3, channels, device="cuda", dtype=x.dtype)
            )
            tokens = torch.cat([history, x[offsets[row] : offsets[row + 1]]])
            taps = tokens.float().T.unfold(-1, 4, 1)
            reference.append(F.silu((taps * w[:, None]).sum(-1)).T.bfloat16())
            for token in range(length):
                if (prefix + token + 1) % page_size == 0 or token + 1 == length:
                    page = table[row, (prefix + token) // page_size]
                    expected_state[page] = tokens[token + 1 : token + 4]
        actual = causal_conv1d_fn(
            x.T,
            w,
            None,
            state.transpose(1, 2),
            torch.tensor(offsets, device="cuda", dtype=torch.int32),
            table,
            torch.tensor(prefixes, device="cuda", dtype=torch.int32),
            page_size,
            activation="silu",
        ).T
        expected = torch.cat(reference)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=2e-6)
        self.assertLess(
            (
                (actual.float() - expected.float()).norm() / expected.float().norm()
            ).item(),
            1e-4,
        )
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)

    def test_decode_convolution_fp32_and_cache(self):
        batch, channels = 48, 3 * 64 * 128
        x = torch.randn(batch, channels, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16).float()
        state = torch.randn(batch + 1, 3, channels, device="cuda", dtype=torch.bfloat16)
        initial = state.clone()
        table = torch.randperm(batch, device="cuda", dtype=torch.int32).add(1)[:, None]
        lengths = torch.full((batch,), 64, device="cuda", dtype=torch.int32)
        taps = torch.cat([initial[table[:, 0]].float(), x[:, None].float()], dim=1)
        expected = F.silu((taps * w.T[None]).sum(1)).bfloat16()
        actual = torch.cat(
            glm53_kda_short_conv_decode(x, w, state, table, lengths, 128), dim=-1
        )
        torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=2e-6)
        self.assertLess(
            (
                (actual.float() - expected.float()).norm() / expected.float().norm()
            ).item(),
            1e-4,
        )
        torch.testing.assert_close(
            state[table[:, 0]], taps[:, 1:].bfloat16(), rtol=0, atol=0
        )
        torch.testing.assert_close(state[0], initial[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
