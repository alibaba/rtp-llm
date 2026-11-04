import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha import (
    sparse_paged_gqa_attn,
)
from rtp_llm.models_py.modules.qwen4_exp.test.reference.sparse_paged import _reference


class ChunkPagedGqaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("SM120 is required")

    def fixture(self, fp8=False):
        torch.manual_seed(1773)
        b, h, s, d = 2, 3, 17, 256
        q = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
        cache = torch.randn(7, 2, 1, 128, d, device="cuda", dtype=torch.bfloat16)
        table = torch.tensor([[3, 6, 1], [5, 2, 4]], device="cuda", dtype=torch.int32)
        lengths = torch.full((b, s), 367, device="cuda", dtype=torch.int32)
        lengths[1, -1] = 0
        selected = torch.full((b, s, 2051), -1, device="cuda", dtype=torch.int32)
        selected[..., :11] = torch.arange(11, device="cuda", dtype=torch.int32) * 31
        selected[1, -1] = -1
        scale = None
        if fp8:
            cache = cache.to(torch.float8_e4m3fn)
            scale = torch.rand(7, 2, 1, 128, device="cuda") * 0.8 + 0.2
        return q, cache, table, lengths, selected, scale

    def run_attention(self, args, **kwargs):
        q, cache, table, lengths, selected, scale = args
        return sparse_paged_gqa_attn(
            q, cache, table, lengths, selected, page_size=128, kv_scale=scale, **kwargs
        )

    def test_grouped_chunk_matches_independent_float_reference(self):
        for fp8 in (False, True):
            args = self.fixture(fp8)
            q, cache, table, lengths, selected, scale = args
            decoded = cache.float()
            if scale is not None:
                decoded = decoded * scale[..., None]
            expected = _reference(q, decoded, table, lengths, selected)
            with patch.dict(os.environ, RTP_QWEN4_PAGED_CHUNK_IMPL="grouped"):
                actual = self.run_attention(args)
            # Independent FP64 probes give the same relative L2 error as the
            # online path (~0.00166); the rounded BF16 results can differ by
            # one ULP. Keep a BF16-relative bound plus an absolute zero floor.
            torch.testing.assert_close(actual, expected, rtol=8e-3, atol=2e-3)
            with patch.dict(os.environ, RTP_QWEN4_PAGED_CHUNK_IMPL="online"):
                legacy = self.run_attention(args)
            torch.testing.assert_close(actual, legacy, rtol=8e-3, atol=2e-3)
            torch.testing.assert_close(
                actual[1, :, -1], torch.zeros_like(actual[1, :, -1])
            )

    def test_packed_count_gate_and_replay_consume_live_metadata(self):
        args = list(self.fixture(True))
        q, cache, table, lengths, selected, scale = args
        counts = (selected >= 0).sum(-1, keepdim=True).int()
        args[4] = torch.cat([selected, counts], dim=-1)
        gate = torch.randn(2, 17, 3, 256, device="cuda", dtype=torch.bfloat16)
        options = dict(selected_has_count=True, output_gate=gate)
        with patch.dict(os.environ, RTP_QWEN4_PAGED_CHUNK_IMPL="grouped"):
            for _ in range(3):
                self.run_attention(args, **options)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = self.run_attention(args, graph_capture=True, **options)
            for gain in (0.25, 1.0, 2.0):
                q.normal_().mul_(gain)
                scale.uniform_(0.2, 1.0)
                graph.replay()
                with patch.dict(os.environ, RTP_QWEN4_PAGED_CHUNK_IMPL="online"):
                    expected = self.run_attention(args, **options)
                torch.testing.assert_close(actual, expected, rtol=8e-3, atol=2e-3)
            args[4][..., :-1].fill_(-1)
            args[4][..., -1].zero_()
            lengths.zero_()
            graph.replay()
            torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
        graph.reset()

    def test_trusted_path_skips_redundant_value_scan_and_keeps_shape_checks(self):
        args = self.fixture(True)
        expected = self.run_attention(args)
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha._validate_sparse_paged_indices",
            side_effect=AssertionError("redundant selected scan"),
        ):
            actual = self.run_attention(args, validate_selected=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        invalid_shape = list(args)
        invalid_shape[3] = invalid_shape[3][:, :-1]
        with self.assertRaisesRegex(ValueError, "kv_lens must"):
            self.run_attention(invalid_shape, validate_selected=False)
        invalid_value = list(args)
        invalid_value[4] = invalid_value[4].clone()
        invalid_value[4][0, 0, 0] = 367
        with self.assertRaisesRegex(ValueError, "outside its row"):
            self.run_attention(invalid_value)


if __name__ == "__main__":
    unittest.main()
