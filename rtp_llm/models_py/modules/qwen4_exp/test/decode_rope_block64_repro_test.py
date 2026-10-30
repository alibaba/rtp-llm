"""Native decode RoPE/cache replay checks at Qwen4's 64-token page boundary.

This is a component regression, not an MTP executor or distributed-engine test.
The scalar-row native reference isolates batch/graph metadata handling; it does
not provide an independent mathematical implementation of RoPE or FP8 casting.
"""

import unittest
from types import SimpleNamespace

import torch
from libth_transformer_config import AttentionConfigs, RopeStyle

from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeKVCacheDecodeOp,
    _get_fused_rope_kvcache,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DecodeRopeBlock64ReproTest(unittest.TestCase):
    SEED = 20261030
    PAGE_SIZE = 64
    MAX_SEQ_LEN = 32768

    def config(self, factor=3):
        cfg = AttentionConfigs()
        cfg.head_num, cfg.kv_head_num, cfg.size_per_head = 12, 1, 256
        cfg.kernel_tokens_per_block = self.PAGE_SIZE
        cfg.max_seq_len = self.MAX_SEQ_LEN
        cfg.need_rope_kv_cache, cfg.use_logn_attn = True, False
        cfg.rope_config.style = RopeStyle.Base
        cfg.rope_config.index_factor, cfg.rope_config.dim = factor, 64
        cfg.rope_config.base, cfg.rope_config.scale = 10000000.0, 1.0
        return cfg

    def case(self, batch, graph_index=0):
        torch.manual_seed(self.SEED + batch + 100 * graph_index)
        op = FusedRopeKVCacheDecodeOp(self.config())
        lengths = torch.full((batch,), 62, dtype=torch.int32, device="cuda")
        inputs = SimpleNamespace(sequence_lengths=lengths, is_cuda_graph=False)
        owned = op.refresh_sequence_lengths(inputs)
        columns = self.MAX_SEQ_LEN // self.PAGE_SIZE
        # A permutation makes adjacent logical pages physically non-contiguous.
        table = torch.randperm(batch * columns, device="cuda").to(torch.int32)
        table = (table + 1).reshape(batch, columns)
        offset = _get_fused_rope_kvcache().convert_offset_to_block_array(table)
        params = SimpleNamespace(
            sequence_lengths=owned,
            position_ids=torch.arange(batch * 3, device="cuda", dtype=torch.int32),
            kv_cache_offset=offset,
            kv_cache_offset_h=None,
        )
        pages = batch * columns + 1
        cache = SimpleNamespace(
            kv_cache_base=torch.full(
                (pages, 2, 1, self.PAGE_SIZE, 256),
                0.25,
                dtype=torch.float32,
                device="cuda",
            ).to(torch.float8_e4m3fn),
            kv_scale_base=torch.full((pages, 2 * self.PAGE_SIZE), 0.375, device="cuda"),
        )
        qkv = torch.randn(batch, 3584, device="cuda", dtype=torch.bfloat16)
        return op, inputs, params, table, cache, qkv

    def clone_cache(self, cache):
        return SimpleNamespace(
            kv_cache_base=cache.kv_cache_base.clone(),
            kv_scale_base=cache.kv_scale_base.clone(),
        )

    def scalar_reference(self, params, cache, qkv):
        outputs = []
        for row in range(qkv.shape[0]):
            op = FusedRopeKVCacheDecodeOp(self.config(factor=1))
            op._sequence_lengths_device = params.sequence_lengths[row : row + 1].clone()
            one = SimpleNamespace(
                sequence_lengths=op._sequence_lengths_device,
                position_ids=None,
                kv_cache_offset=params.kv_cache_offset[row : row + 1],
                kv_cache_offset_h=None,
            )
            outputs.append(op.forward(qkv[row : row + 1], cache, one))
        return torch.cat(outputs)

    def assert_all_equal(self, output, expected, cache, expected_cache):
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertTrue(
            torch.equal(
                cache.kv_cache_base.view(torch.uint8),
                expected_cache.kv_cache_base.view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(cache.kv_scale_base, expected_cache.kv_scale_base))

    def refresh(self, case, position, generation):
        op, inputs, params, table, _, qkv = case
        pointer = params.sequence_lengths.data_ptr()
        inputs.sequence_lengths.copy_(
            torch.full_like(inputs.sequence_lengths, position)
        )
        owned = op.refresh_sequence_lengths(inputs, forbid_reallocation=True)
        self.assertIs(owned, params.sequence_lengths)
        self.assertEqual(owned.data_ptr(), pointer)
        # Both metadata buffers retain their captured addresses.
        params.position_ids.add_(3)
        new_table = table if generation % 2 == 0 else table.flip(1)
        params.kv_cache_offset.copy_(
            _get_fused_rope_kvcache().convert_offset_to_block_array(new_table)
        )
        qkv.mul_(0.875)

    def test_eager_batch_and_all_cache_bytes_across_page_boundaries(self):
        positions = [62, 63, 64, 65, 127, 128, 129, 32764, 32765, 32766, 32767]
        for batch in (1, 2, 4):
            case = self.case(batch)
            op, _, params, _, cache, qkv = case
            for generation, position in enumerate(positions):
                with self.subTest(batch=batch, position=position):
                    self.refresh(case, position, generation)
                    expected_cache = self.clone_cache(cache)
                    expected = self.scalar_reference(params, expected_cache, qkv)
                    actual = op.forward(qkv, cache, params)
                    torch.cuda.synchronize()
                    self.assert_all_equal(actual, expected, cache, expected_cache)

    def test_two_graphs_refresh_page64_metadata_for_three_draft_steps(self):
        # 62→63→64→65 explicitly crosses a page over gamma=3 draft steps.
        positions = [62, 63, 64, 65, 127, 128, 129, 32764, 32765, 32766, 32767]
        for batch in (1, 2, 4):
            cases = [self.case(batch, graph_index=index) for index in range(2)]
            captures = []
            for op, _, params, _, cache, qkv in cases:
                for _ in range(2):
                    op.forward(qkv, cache, params)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = op.forward(qkv, cache, params)
                captures.append((graph, output))
            for generation, position in enumerate(positions):
                for index, (case, (graph, output)) in enumerate(zip(cases, captures)):
                    with self.subTest(batch=batch, graph=index, position=position):
                        graph_position = (
                            position if index == 0 else positions[-1 - generation]
                        )
                        self.refresh(case, graph_position, generation + index)
                        _, _, params, _, cache, qkv = case
                        expected_cache = self.clone_cache(cache)
                        expected = self.scalar_reference(params, expected_cache, qkv)
                        graph.replay()
                        torch.cuda.synchronize()
                        self.assert_all_equal(output, expected, cache, expected_cache)


if __name__ == "__main__":
    unittest.main()
