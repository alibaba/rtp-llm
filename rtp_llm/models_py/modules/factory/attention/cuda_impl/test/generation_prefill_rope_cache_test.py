import unittest
from types import SimpleNamespace

import torch

from rtp_llm.ops import AttentionConfigs, RopeStyle, get_rope_cache_once
from rtp_llm.ops.compute_ops import get_typemeta
from rtp_llm.ops.fused_rope_kvcache_op import FusedRopeKVCachePrefillOpQKVOut


class GenerationPrefillRopeCacheTest(unittest.TestCase):
    def test_mixed_request_maxima_do_not_overestimate_replay_positions(self):
        configs = SimpleNamespace(rope_config=SimpleNamespace(style=RopeStyle.Base))
        op = FusedRopeKVCachePrefillOpQKVOut(configs)
        op._graph_rope_cache = torch.empty(8192, 1)
        params = SimpleNamespace(
            input_lengths=torch.tensor([1, 5], dtype=torch.int32),
            prefix_lengths=torch.tensor([8188, 128], dtype=torch.int32),
        )
        cache = op._graph_rope_cache
        op._prepare_graph_rope_cache(params)
        self.assertIs(op._graph_rope_cache, cache)
        params.prefix_lengths[1] = 8188
        with self.assertRaisesRegex(RuntimeError, "captured RoPE cache"):
            op._prepare_graph_rope_cache(params)

    @staticmethod
    def inputs(length, *, graph):
        cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
        return SimpleNamespace(
            is_prefill=True,
            is_cuda_graph=graph,
            context_parallel_info=None,
            kv_cache_kernel_block_id_device=None,
            combo_position_ids=None,
            padding_offset=None,
            cu_seqlens_device=cu,
            cu_kv_seqlens_device=cu,
            input_lengths=torch.tensor([length], dtype=torch.int32),
            prefix_lengths=torch.zeros(1, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
            context_total_kv_length=length,
            dtype=get_typemeta(torch.empty(0, dtype=torch.bfloat16)),
        )

    def test_total_bucket_owns_bounded_cache_and_preserves_real_positions(self):
        configs = AttentionConfigs()
        configs.head_num, configs.kv_head_num, configs.size_per_head = 2, 1, 128
        configs.max_seq_len = 1024
        configs.kernel_tokens_per_block = 16
        configs.rope_config.style = RopeStyle.Base
        configs.rope_config.dim = 128
        configs.rope_config.base = 10000
        configs.rope_config.scale = 1.0
        op = FusedRopeKVCachePrefillOpQKVOut(configs)
        inputs = self.inputs(6144, graph=True)
        params = op.prepare(inputs)
        original = get_rope_cache_once(configs.rope_config, configs.max_seq_len)
        self.assertEqual(original.data.shape[0], 1024)
        original_data = original.data.clone()
        self.assertEqual(op._graph_rope_cache.shape[0], 6144)
        torch.testing.assert_close(
            op._graph_rope_cache[:1024], original_data, atol=0, rtol=0
        )
        # The global once cache and all coefficients for real requests stay intact.
        torch.testing.assert_close(original.data, original_data, atol=0, rtol=0)
        primed_op = FusedRopeKVCachePrefillOpQKVOut(configs)
        primed_op.prepare(inputs)
        torch.testing.assert_close(
            primed_op._graph_rope_cache[:1024], original_data, atol=0, rtol=0
        )
        ptr = op._graph_rope_cache.data_ptr()
        torch.manual_seed(20261031)
        qkv = torch.randn(6144, 512, dtype=torch.bfloat16, device="cuda")
        eager_op = FusedRopeKVCachePrefillOpQKVOut(configs)
        eager_params = eager_op.prepare(self.inputs(1024, graph=False))
        expected = eager_op.forward(qkv[:1024].clone(), None, eager_params).clone()

        for _ in range(3):
            op.forward(qkv.clone(), None, params)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        graph_input = qkv.clone()
        with torch.cuda.graph(graph):
            result = op.forward(graph_input, None, params)
        for _ in range(2):
            graph_input.copy_(qkv)
            op.prepare(inputs)
            graph.replay()
            torch.testing.assert_close(result[:1024], expected, atol=0, rtol=0)
            self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(op._graph_rope_cache.data_ptr(), ptr)
        with self.assertRaisesRegex(RuntimeError, "captured RoPE cache"):
            op.prepare(self.inputs(6145, graph=True))


if __name__ == "__main__":
    unittest.main()
