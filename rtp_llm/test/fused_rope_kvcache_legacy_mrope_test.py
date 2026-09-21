"""Numerical CUDA regressions for legacy mRoPE positions and physical slots."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.ops import AttentionConfigs, KvCacheDataType, RopeStyle
from rtp_llm.ops import fused_rope_kvcache_op as ops


def reference_mrope(tensor, position_ids, rope_dim=64):
    # Qwen3.5 [11, 11, 10]: temporal frequencies with interleaved H/W slots.
    pairs = rope_dim // 2
    axes = torch.zeros(pairs, dtype=torch.long, device=tensor.device)
    axes[1 : 3 * 11 : 3] = 1
    axes[2 : 3 * 10 : 3] = 2
    inv_freq = 10000 ** (
        -torch.arange(pairs, dtype=torch.float32, device=tensor.device) / pairs
    )
    angles = position_ids[:, axes].float() * inv_freq
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
    value = tensor.float()
    low, high = value[..., :pairs], value[..., pairs:rope_dim]
    return torch.cat(
        (low * cos - high * sin, high * cos + low * sin, value[..., rope_dim:]),
        dim=-1,
    ).to(tensor.dtype)


@unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda, "CUDA required")
class LegacyMropeDecodeTest(unittest.TestCase):
    def make_case(self, host_lengths=False):
        torch.manual_seed(381)
        cfg = AttentionConfigs()
        cfg.head_num = 4
        cfg.kv_head_num = 2
        cfg.size_per_head = 256
        cfg.tokens_per_block = 64
        cfg.kernel_tokens_per_block = 64
        cfg.max_seq_len = 512
        cfg.dtype = torch.bfloat16
        cfg.kv_cache_dtype = KvCacheDataType.BASE
        cfg.use_logn_attn = False
        cfg.rope_config.style = RopeStyle.Mrope
        cfg.rope_config.dim = 64
        cfg.rope_config.base = 10000
        cfg.rope_config.scale = 1.0
        cfg.rope_config.index_factor = 3
        cfg.rope_config.mrope_dim1 = 11
        cfg.rope_config.mrope_dim2 = 11
        cfg.rope_config.mrope_dim3 = 10
        cfg.rope_config.mrope_interleaved = True

        lengths = torch.tensor([63, 64, 129], dtype=torch.int32)
        lengths = lengths.pin_memory() if host_lengths else lengths.cuda()
        positions = torch.tensor(
            [[7, 2, 5], [11, 8, 3], [19, 4, 13]],
            dtype=torch.int32,
            device="cuda",
        )
        # Non-sequential pages catch accidental direct indexing by token or batch.
        table = torch.tensor(
            [[6, 3, 9], [1, 8, 5], [7, 2, 4]], dtype=torch.int32, device="cuda"
        )
        cu = torch.arange(4, dtype=torch.int32, device="cuda")
        params = ops.FusedRopeAttnParams(
            kv_cache_offset=ops._get_fused_rope_kvcache().convert_offset_to_block_array(
                table
            ),
            kv_cache_offset_h=None,
            padding_offset=torch.zeros(3, dtype=torch.int32, device="cuda"),
            position_ids=positions,
            cu_seqlens=cu,
            cu_kv_seqlens=cu,
            input_lengths=torch.ones(3, dtype=torch.int32, device="cuda"),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=lengths,
            max_seq_len=0,
            max_prefix_length=0,
            context_total_kv_length=0,
            decode_plan=True,
            attn_type=torch.bfloat16,
        )
        q = torch.randn(3, 4, 256, dtype=torch.bfloat16, device="cuda") * 0.25
        k = torch.randn(3, 2, 256, dtype=torch.bfloat16, device="cuda") * 0.25
        v = torch.randn_like(k) * 0.25
        qkv = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
        pool = torch.full(
            (10, 2, 2, 64, 256), 0.125, dtype=torch.bfloat16, device="cuda"
        )
        kv_cache = SimpleNamespace(
            kv_cache_base=pool,
            kv_scale_base=None,
        )
        return cfg, params, table, qkv, q, k, v, kv_cache

    def check_result(self, result, case):
        cfg, params, table, _, q, k, v, kv_cache = case
        expected_q = reference_mrope(q, params.position_ids)
        expected_k = reference_mrope(k, params.position_ids)
        torch.testing.assert_close(result, expected_q, atol=0.01, rtol=0.01)
        torch.testing.assert_close(result[..., 64:], q[..., 64:], atol=0, rtol=0)

        # Compare the complete cache, including untouched pages/token slots.
        actual = kv_cache.kv_cache_base
        expected = torch.full(actual.shape, 0.125, device="cuda")
        touched = torch.zeros(actual.shape[:-1], dtype=torch.bool, device="cuda")
        for batch, length in enumerate(params.sequence_lengths.cpu().tolist()):
            page = int(table[batch, length // 64])
            slot = length % 64
            expected[page, 0, :, slot] = expected_k[batch].float()
            expected[page, 1, :, slot] = v[batch].float()
            touched[page, :, :, slot] = True
        expected = expected.to(actual.dtype).float()
        torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.01)
        torch.testing.assert_close(
            actual.float()[~touched], expected[~touched], atol=0, rtol=0
        )

    def require_legacy_kernel(self):
        kernel = ops._get_fused_rope_kvcache().decode_fused_rope_kvcache
        if ops._decode_uses_sequence_lengths(kernel):
            self.skipTest("installed kernel already has independent sequence_lengths")

    def test_bf16_cache_uses_physical_slots_with_three_axis_rope(self):
        self.require_legacy_kernel()
        case = self.make_case(host_lengths=True)
        cfg, params, _, qkv, _, _, _, kv_cache = case
        self.check_result(
            ops.FusedRopeKVCacheDecodeOp(cfg).forward(qkv, kv_cache, params), case
        )

    def test_legacy_cuda_graph_replay_reads_updated_lengths_and_positions(self):
        self.require_legacy_kernel()
        case = self.make_case()
        cfg, params, _, qkv, _, _, _, kv_cache = case
        op = ops.FusedRopeKVCacheDecodeOp(cfg)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op.forward(qkv, kv_cache, params)
        torch.cuda.current_stream().wait_stream(stream)
        original_qkv = torch.cat(
            (case[4].flatten(1), case[5].flatten(1), case[6].flatten(1)), dim=-1
        )
        qkv.copy_(original_qkv)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = op.forward(qkv, kv_cache, params)
        kv_cache.kv_cache_base.fill_(0.125)
        qkv.copy_(original_qkv)
        params.sequence_lengths.add_(1)
        params.position_ids.add_(2)
        graph.replay()
        self.check_result(result, case)

    def make_prefill_case(self, lengths, prefixes, explicit_positions=True):
        cfg, _, _, _, _, _, _, _ = self.make_case()
        total = sum(lengths)
        cfg.max_seq_len = max(512, total + max(prefixes) + 1)
        cu_host = [0]
        for length in lengths:
            cu_host.append(cu_host[-1] + length)
        cu = torch.tensor(cu_host, dtype=torch.int32, device="cuda")
        max_len = max(lengths)
        padding = torch.cat(
            [
                torch.full(
                    (length,),
                    batch * max_len - cu_host[batch],
                    dtype=torch.int32,
                    device="cuda",
                )
                for batch, length in enumerate(lengths)
            ]
        )
        physical = torch.cat(
            [
                torch.arange(prefix, prefix + length, dtype=torch.int32, device="cuda")
                for prefix, length in zip(prefixes, lengths)
            ]
        )
        positions = (
            torch.stack((physical // 5, physical % 17, physical % 13), dim=-1)
            if explicit_positions
            else physical[:, None].expand(-1, 3).contiguous()
        )
        block_counts = [
            (prefix + length + 63) // 64 for prefix, length in zip(prefixes, lengths)
        ]
        num_pages = sum(block_counts) + 3
        # Reverse physical page IDs to verify logical->physical page mapping.
        page_ids = list(range(num_pages - 1, 2, -1))
        table = torch.zeros(len(lengths), max(block_counts), dtype=torch.int32)
        cursor = 0
        for batch, count in enumerate(block_counts):
            table[batch, :count] = torch.tensor(page_ids[cursor : cursor + count])
            cursor += count
        table = table.cuda()
        params = ops.FusedRopeAttnParams(
            kv_cache_offset=ops._get_fused_rope_kvcache().convert_offset_to_block_array(
                table
            ),
            kv_cache_offset_h=None,
            padding_offset=padding,
            position_ids=positions if explicit_positions else None,
            cu_seqlens=cu,
            cu_kv_seqlens=cu,
            input_lengths=torch.tensor(lengths, dtype=torch.int32).pin_memory(),
            prefix_lengths=torch.tensor(prefixes, dtype=torch.int32, device="cuda"),
            sequence_lengths=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
            max_seq_len=max_len,
            max_prefix_length=max(prefixes),
            context_total_kv_length=sum(prefixes) + total,
            decode_plan=False,
            attn_type=torch.bfloat16,
        )
        q = torch.randn(total, 4, 256, dtype=torch.bfloat16, device="cuda") * 0.25
        k = torch.randn(total, 2, 256, dtype=torch.bfloat16, device="cuda") * 0.25
        v = torch.randn_like(k) * 0.25
        qkv = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
        return cfg, params, table, num_pages, qkv, q, k, v, positions

    def test_long_image_prefill_qkv_matches_independent_rope(self):
        self.require_legacy_kernel()
        case = self.make_prefill_case([137, 2773], [0, 0])
        cfg, params, _, _, qkv, q, k, v, positions = case
        actual = ops.FusedRopeKVCachePrefillOpQKVOut(cfg).forward(qkv, None, params)
        expected = torch.cat(
            (
                reference_mrope(q, positions).flatten(1),
                reference_mrope(k, positions).flatten(1),
                v.flatten(1),
            ),
            dim=-1,
        )
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
        torch.testing.assert_close(actual[:, -512:], v.flatten(1), atol=0, rtol=0)

    def test_prefill_without_explicit_positions_uses_physical_positions(self):
        self.require_legacy_kernel()
        case = self.make_prefill_case([17, 137], [65, 0], explicit_positions=False)
        cfg, params, _, _, qkv, q, k, v, positions = case
        actual = ops.FusedRopeKVCachePrefillOpQKVOut(cfg).forward(qkv, None, params)
        expected = torch.cat(
            (
                reference_mrope(q, positions).flatten(1),
                reference_mrope(k, positions).flatten(1),
                v.flatten(1),
            ),
            dim=-1,
        )
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)

    def test_new_api_keeps_independent_position_and_length_inputs(self):
        case = self.make_case()
        cfg, params, _, qkv, q, _, _, kv_cache = case
        calls = []

        def decode(qkv_arg, position_ids, sequence_lengths, batch_size, **kwargs):
            calls.append((qkv_arg, position_ids, sequence_lengths, batch_size, kwargs))
            return q

        kernel = SimpleNamespace(decode_fused_rope_kvcache=decode)
        op = ops.FusedRopeKVCacheDecodeOp(cfg)
        with mock.patch.object(
            ops, "_get_fused_rope_kvcache", return_value=kernel
        ), mock.patch.object(
            op,
            "_legacy_mrope_decode",
            side_effect=AssertionError("legacy fallback used for new API"),
        ):
            self.assertIs(op.forward(qkv, kv_cache, params), q)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], qkv)
        self.assertIs(calls[0][1], params.position_ids)
        self.assertIs(calls[0][2], params.sequence_lengths)
        self.assertEqual(calls[0][3], 3)
        self.assertEqual(calls[0][4]["rope_style"], RopeStyle.Mrope)


if __name__ == "__main__":
    unittest.main()
