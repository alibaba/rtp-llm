"""Real FP4 MQA kernels: compressed columns, BF16 reduction and graph replay."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _indexer_score as backend
from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer import _fp4_rows_torch
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_decode_indexer import (
    _check_graph_replay,
    _check_paged_owner_layout,
    _decode_fp4,
)


def make_contiguous(heads, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(5, heads, 128, generator=generator, device="cuda").bfloat16()
    k = torch.randn(577, 128, generator=generator, device="cuda").bfloat16()
    q_payload, q_sf, _ = _fp4_rows_torch(q)
    k_payload, k_sf, _ = _fp4_rows_torch(k)
    return [
        q_payload.reshape(5, heads, 64),
        q_sf.reshape(5, heads),
        k_payload,
        k_sf,
        torch.randn(5, heads, generator=generator, device="cuda") / heads,
        torch.tensor([0, 5, 259, 13, 64], dtype=torch.int32, device="cuda"),
        torch.tensor([1, 257, 577, 513, 64], dtype=torch.int32, device="cuda"),
    ]


def reference_contiguous(inputs, width):
    q, q_sf, k, k_sf, weights, starts, ends = inputs
    rows, heads = q.shape[:2]
    q_values = _decode_fp4(q.reshape(-1, 64), q_sf.reshape(-1)).reshape(
        rows, heads, 128
    )
    k_values = _decode_fp4(k, k_sf).reshape(-1, 128)
    dots = torch.einsum("mhd,kd->mhk", q_values.double(), k_values.double())
    dots = dots.relu_().bfloat16().double()
    weights = weights.bfloat16().double()
    # Four independent BF16 FMA lanes, followed by the documented pair sum.
    lanes = [torch.zeros_like(dots[:, 0]) for _ in range(4)]
    for head in range(heads):
        lane = head % 4
        lanes[lane] = (
            (dots[:, head] * weights[:, head, None] + lanes[lane]).bfloat16().double()
        )
    even = (lanes[0] + lanes[2]).bfloat16().double()
    odd = (lanes[1] + lanes[3]).bfloat16().double()
    scores = (even + odd).bfloat16().float()
    output = torch.full((rows, width or k.shape[0]), -torch.inf, device="cuda")
    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        destination = slice(0, end - start) if width else slice(start, end)
        output[row, destination] = scores[row, start:end]
    return output


class MqaScoreCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These are required native tests, not a capability-probe skip lane.
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("FP4 MQA tests require a real SM100/SM103 device")
        if (
            not backend.has_fp8_fp4_mqa_logits()
            or not backend.has_fp8_fp4_paged_mqa_logits()
        ):
            raise RuntimeError("Installed DeepGEMM lacks the required FP4 MQA kernels")
        print("DeepGEMM native package:", backend._deep_gemm.__file__)
        print("Contiguous ABI:", backend._deep_gemm.fp8_fp4_mqa_logits.__doc__)
        print("Paged ABI:", backend._deep_gemm.fp8_fp4_paged_mqa_logits.__doc__)

    def assert_scores(self, actual, expected):
        visible = torch.isfinite(expected)
        self.assertTrue(torch.equal(torch.isfinite(actual), visible))
        self.assertTrue(torch.isneginf(actual[~visible]).all())
        error = (actual[visible] - expected[visible]).float()
        denominator = expected[visible].square().mean().sqrt().clamp_min(1e-6)
        relative_rms = error.square().mean().sqrt() / denominator
        print(
            f"relative_rms={relative_rms.item():.7g} max_abs={error.abs().max().item():.7g}"
        )
        # Same BF16 numerical oracle threshold as the existing paged tests.
        self.assertLess(relative_rms.item(), 0.0001)

    def test_contiguous_nonzero_starts_and_both_column_layouts(self):
        for heads in (32, 64):
            inputs = make_contiguous(heads, 71)
            for width in (0, 500):
                with self.subTest(heads=heads, width=width):
                    actual = backend.fp8_fp4_mqa_indexer_score(
                        *inputs, max_seqlen_k=width
                    )
                    self.assert_scores(actual, reference_contiguous(inputs, width))

    def test_contiguous_graph_replay_changes_payload_scales_weights_and_bounds(self):
        for width in (0, 500):
            with self.subTest(width=width):
                inputs = make_contiguous(32, 81)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        backend.fp8_fp4_mqa_indexer_score(*inputs, max_seqlen_k=width)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = backend.fp8_fp4_mqa_indexer_score(
                        *inputs, max_seqlen_k=width
                    )
                for step in range(3):
                    updated = make_contiguous(32, 91 + step)
                    updated[5].copy_(torch.tensor([3, 64, 256, 31, 1], device="cuda"))
                    updated[6].copy_(
                        torch.tensor([3, 129, 573 - step, 531, 257], device="cuda")
                    )
                    for destination, source in zip(inputs, updated):
                        destination.copy_(source)
                    output.fill_(torch.nan)
                    graph.replay()
                    self.assert_scores(output, reference_contiguous(inputs, width))

    def test_paged_ragged_requests_and_speculative_lengths(self):
        _check_paged_owner_layout(128, 128, 1, 2, 6)
        _check_paged_owner_layout(128, 64, 2, 2, 6)

    def test_paged_graph_replay_changes_cache_table_and_live_lengths(self):
        _check_graph_replay(128, 1)
        _check_graph_replay(64, 2)


if __name__ == "__main__":
    unittest.main()
