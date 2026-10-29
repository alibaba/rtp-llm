"""Regression for MTP's multi-axis config with scalar decode lengths."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.ops import AttentionConfigs, RopeStyle
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeKVCacheDecodeOp,
    _get_fused_rope_kvcache,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DecodeRopeStrideTest(unittest.TestCase):
    def config(self, factor=3, style=RopeStyle.Base):
        cfg = AttentionConfigs()
        cfg.head_num, cfg.kv_head_num, cfg.size_per_head = 12, 1, 256
        cfg.kernel_tokens_per_block, cfg.max_seq_len = 128, 8192
        cfg.need_rope_kv_cache, cfg.use_logn_attn = True, False
        cfg.rope_config.style = style
        cfg.rope_config.index_factor, cfg.rope_config.dim = factor, 64
        cfg.rope_config.base, cfg.rope_config.scale = 10000000.0, 1.0
        cfg.rope_config.mrope_dim1 = 11
        cfg.rope_config.mrope_dim2 = 11
        cfg.rope_config.mrope_dim3 = 10
        return cfg

    def case(self, factor=3, style=RopeStyle.Base):
        torch.manual_seed(82018)
        cfg = self.config(factor, style)
        op = FusedRopeKVCacheDecodeOp(cfg)
        # A defined poison tail makes the old stride fail deterministically,
        # without requiring an illegal access to a different allocation.
        storage = torch.full((24,), 4096, dtype=torch.int32, device="cuda")
        lengths = storage[:8]
        lengths.copy_(
            torch.tensor([3, 7, 15, 127, 128, 129, 4095, 8190], device="cuda")
        )
        op._sequence_lengths_device = lengths
        table = torch.arange(1, 9, device="cuda", dtype=torch.int32)[:, None]
        offset = _get_fused_rope_kvcache().convert_offset_to_block_array(
            table.expand(8, 67).contiguous()
        )
        params = SimpleNamespace(
            sequence_lengths=lengths,
            position_ids=torch.arange(1, 25, device="cuda", dtype=torch.int32),
            kv_cache_offset=offset,
            kv_cache_offset_h=None,
        )
        cache = SimpleNamespace(
            kv_cache_base=torch.full((9, 2, 1, 128, 256), 0.25, device="cuda").to(
                torch.float8_e4m3fn
            ),
            kv_scale_base=torch.full((9, 256), 0.375, device="cuda"),
        )
        qkv = torch.randn(8, 3584, device="cuda", dtype=torch.bfloat16)
        return cfg, op, params, cache, qkv

    def clone_cache(self, cache):
        return SimpleNamespace(
            kv_cache_base=cache.kv_cache_base.clone(),
            kv_scale_base=cache.kv_scale_base.clone(),
        )

    def scalar_reference(self, params, cache, qkv):
        outputs = []
        for row in range(8):
            op = FusedRopeKVCacheDecodeOp(self.config(factor=1))
            lengths = params.sequence_lengths[row : row + 1].clone()
            op._sequence_lengths_device = lengths
            one = SimpleNamespace(
                sequence_lengths=lengths,
                position_ids=None,
                kv_cache_offset=params.kv_cache_offset[row : row + 1],
                kv_cache_offset_h=None,
            )
            outputs.append(op.forward(qkv[row : row + 1], cache, one))
        return torch.cat(outputs)

    def assert_cache_equal(self, actual, expected):
        self.assertTrue(
            torch.equal(
                actual.kv_cache_base.view(torch.uint8),
                expected.kv_cache_base.view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(actual.kv_scale_base, expected.kv_scale_base))

    def test_cached_batch_matches_independent_scalar_rows_and_all_cache_bytes(self):
        cfg, op, params, cache, qkv = self.case()
        expected_cache = self.clone_cache(cache)
        expected = self.scalar_reference(params, expected_cache, qkv)
        actual = op.forward(qkv, cache, params)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_cache_equal(cache, expected_cache)
        self.assertEqual(cfg.rope_config.index_factor, 3)

    def test_two_graphs_refresh_lengths_positions_and_page_tables(self):
        cases = [self.case(), self.case()]
        captures = []
        for _, op, params, cache, qkv in cases:
            for _ in range(2):
                op.forward(qkv, cache, params)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = op.forward(qkv, cache, params)
            captures.append((graph, output))
        for generation in range(6):
            for case, (graph, output) in zip(cases, captures):
                _, _, params, cache, qkv = case
                params.sequence_lengths.add_(1)
                params.sequence_lengths.remainder_(8192)
                params.position_ids.add_(3)
                params.kv_cache_offset.copy_(params.kv_cache_offset.flip(0))
                qkv.mul_(0.75)
                expected_cache = self.clone_cache(cache)
                expected = self.scalar_reference(params, expected_cache, qkv)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                self.assert_cache_equal(cache, expected_cache)

    def test_mrope_preserves_explicit_three_axis_stride_even_with_base_cache(self):
        # Initialize Base's process-global trig cache before using MRoPE with
        # the same base/dimension; check_rope_cache alone also accepts it.
        _, base_op, base_params, base_cache, base_qkv = self.case()
        base_op.forward(base_qkv, base_cache, base_params)
        cfg, op, params, cache, qkv = self.case(style=RopeStyle.Mrope)
        expected_cache = self.clone_cache(cache)
        with patch(
            "rtp_llm.ops.fused_rope_kvcache_op.check_rope_cache", return_value=False
        ):
            expected = op.forward(qkv, expected_cache, params)
        actual = op.forward(qkv, cache, params)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_cache_equal(cache, expected_cache)
        self.assertEqual(cfg.rope_config.index_factor, 3)


if __name__ == "__main__":
    unittest.main()
