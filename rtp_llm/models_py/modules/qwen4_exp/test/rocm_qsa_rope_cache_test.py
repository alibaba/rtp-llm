"""Real Qwen4 TP4 writer geometry, with a CPU rotary/cache oracle."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.rocm_rope_cache import RocmQsaRopeCache
from rtp_llm.ops import AttentionConfigs, RopeStyle


@unittest.skipUnless(
    torch.version.hip is not None and torch.cuda.is_available(), "ROCm required"
)
class RocmQsaRopeCacheTest(unittest.TestCase):
    def config(self):
        # Official checkpoint: 24 Q heads / TP4, 2 KV heads replicated over TP4,
        # head_dim256 * partial_rotary_factor0.25, sections11/11/10, theta1e7.
        cfg = AttentionConfigs()
        cfg.head_num, cfg.kv_head_num, cfg.size_per_head = 6, 1, 256
        cfg.kernel_tokens_per_block, cfg.max_seq_len = 128, 8192
        cfg.need_rope_kv_cache, cfg.use_logn_attn = True, False
        r = cfg.rope_config
        r.style, r.dim, r.index_factor = RopeStyle.Mrope, 64, 3
        r.base, r.scale, r.mrope_interleaved = 10000000.0, 1.0, True
        r.mrope_dim1, r.mrope_dim2, r.mrope_dim3 = 11, 11, 10
        return cfg

    def inputs(self, *, prefill, lengths, prefixes, positions, table):
        cpu_lengths = torch.tensor(lengths, dtype=torch.int32)
        cu = torch.cat((torch.zeros(1, dtype=torch.int32), cpu_lengths.cumsum(0).int()))
        return SimpleNamespace(
            is_prefill=prefill,
            is_target_verify=False,
            is_cuda_graph=False,
            context_parallel_info=None,
            is_s_padded=False,
            cache_store_inputs=None,
            is_exact_cuda_graph_batch=True,
            combo_position_ids=positions,
            kv_cache_kernel_block_id_device=table,
            kv_cache_kernel_block_id=table.cpu(),
            input_lengths=cpu_lengths,
            prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
            sequence_lengths=(
                cpu_lengths.pin_memory()
                if not prefill
                else torch.empty(0, dtype=torch.int32)
            ),
            cu_seqlens_device=cu.cuda(),
            cu_kv_seqlens_device=cu.cuda(),
            padding_offset=torch.zeros(int(cu[-1]), device="cuda", dtype=torch.int32),
            context_total_kv_length=int(cu[-1]),
            dtype=torch.bfloat16,
        )

    def oracle(self, qkv, positions, cfg):
        value = qkv.cpu().float().reshape(-1, 8, 256)
        original = value.clone()
        pair = torch.arange(32)
        axis = torch.zeros(32, dtype=torch.long)
        axis[(pair % 3 == 1) & (pair < 33)] = 1
        axis[(pair % 3 == 2) & (pair < 30)] = 2
        pos = positions.cpu().reshape(-1, 3)[:, axis].float()
        angle = pos * torch.pow(torch.tensor(cfg.rope_config.base), -pair.float() / 32)
        cosine, sine = angle.cos()[:, None, :], angle.sin()[:, None, :]
        left, right = original[:, :7, :32], original[:, :7, 32:64]
        value[:, :7, :32] = left * cosine - right * sine
        value[:, :7, 32:64] = right * cosine + left * sine
        return value.to(torch.bfloat16).reshape(-1, 2048)

    def cache(self, pages=12, tail=73):
        # Physical HybridPool stride includes unrelated tail cells.
        return SimpleNamespace(
            kv_cache_base=torch.full(
                (pages, 65536 + tail), -7.0, device="cuda", dtype=torch.bfloat16
            ),
            kv_scale_base=None,
        )

    def check_cache(self, cache, expected_qkv, table, lengths, prefixes, prefill):
        shape = (
            cache.kv_cache_base.shape[0],
            cache.kv_cache_base.numel() // cache.kv_cache_base.shape[0],
        )
        expected = torch.full(shape, -7.0, dtype=torch.bfloat16)
        token = 0
        for row, length in enumerate(lengths):
            for local in range(length if prefill else 1):
                position = prefixes[row] + local if prefill else length
                page = int(table[row, position // 128])
                if page > 0:
                    for kv in range(2):
                        start = kv * 32768 + (position % 128) * 256
                        expected[page, start : start + 256] = expected_qkv[
                            token, (6 + kv) * 256 : (7 + kv) * 256
                        ]
                token += 1
        torch.testing.assert_close(
            cache.kv_cache_base.cpu().reshape(shape), expected, atol=0.03125, rtol=0.01
        )

    def test_prefill_ragged_prefix_positions_and_hybrid_stride(self):
        torch.manual_seed(901)
        cfg = self.config()
        table = torch.tensor([[1, 4, 7], [2, 5, 8]], device="cuda", dtype=torch.int32)
        lengths, prefixes = [129, 3], [127, 128]
        positions = (
            torch.arange(132 * 3, device="cuda", dtype=torch.int32).reshape(132, 3) + 23
        )
        inputs = self.inputs(
            prefill=True,
            lengths=lengths,
            prefixes=prefixes,
            positions=positions,
            table=table,
        )
        op = RocmQsaRopeCache(cfg, is_prefill=True)
        params = op.prepare(inputs)
        qkv = torch.randn(132, 2048, device="cuda", dtype=torch.bfloat16)
        cache = self.cache()
        expected = self.oracle(qkv, positions, cfg)
        actual = op.forward(qkv, cache, params)
        torch.testing.assert_close(actual.cpu(), expected, atol=0.03125, rtol=0.01)
        self.check_cache(cache, expected, table.cpu(), lengths, prefixes, True)

    def test_decode_changed_pages_lengths_and_positions_replay(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import (
            SparseGqaFmhaImpl,
        )

        cfg = self.config()
        for batch in (1, 16):
            torch.manual_seed(902 + batch)
            lengths = ([127, 128, 255, 256] * 4)[:batch]
            table = torch.arange(
                1, batch * 3 + 1, device="cuda", dtype=torch.int32
            ).reshape(batch, 3)
            positions = (
                torch.arange(batch * 3, device="cuda", dtype=torch.int32).reshape(
                    batch, 3
                )
                + 19
            )
            inputs = self.inputs(
                prefill=False,
                lengths=lengths,
                prefixes=[0] * batch,
                positions=positions,
                table=table,
            )
            positions.add_(
                torch.tensor([0, 1024, 8000], device="cuda", dtype=torch.int32)
            )
            inputs.is_cuda_graph = True
            op = RocmQsaRopeCache(cfg, is_prefill=False)
            params = op.prepare(inputs)
            cache = self.cache(batch * 3 + 2, tail=0 if batch == 1 else 73)
            if batch == 1:
                cache.kv_cache_base = cache.kv_cache_base.view(-1, 2, 1, 128, 256)
            qkv = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
            expected = self.oracle(qkv, positions, cfg)
            torch.testing.assert_close(
                op.forward(qkv, cache, params).cpu(),
                expected[:, :1536],
                atol=0.03125,
                rtol=0.01,
            )
            self.check_cache(cache, expected, table.cpu(), lengths, [0] * batch, False)
            for _ in range(3):
                op.forward(qkv, cache, params)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = op.forward(qkv, cache, params)
            old_length_pointer = params.sequence_lengths.data_ptr()
            old_table_pointer = params.kv_cache_offset.data_ptr()
            inputs.sequence_lengths.add_(1)
            inputs.kv_cache_kernel_block_id_device = table.flip(1).contiguous()
            positions.add_(47)
            cache.kv_cache_base.fill_(-7)
            # Exercise the actual owner's captured-offset refresh contract.
            owner = SparseGqaFmhaImpl.__new__(SparseGqaFmhaImpl)
            owner.attn_inputs, owner.is_prefill, owner.is_target_verify = (
                inputs,
                False,
                False,
            )
            owner.rope_kvcache_impl, owner.rope_params = op, params
            owner.prepare_cuda_graph(inputs)
            self.assertEqual(params.sequence_lengths.data_ptr(), old_length_pointer)
            self.assertEqual(params.kv_cache_offset.data_ptr(), old_table_pointer)
            graph.replay()
            expected = self.oracle(qkv, positions, cfg)
            torch.testing.assert_close(
                result.cpu(), expected[:, :1536], atol=0.03125, rtol=0.01
            )
            self.check_cache(
                cache,
                expected,
                inputs.kv_cache_kernel_block_id_device.cpu(),
                inputs.sequence_lengths.tolist(),
                [0] * batch,
                False,
            )

    def test_sparse_owner_prefill_and_decode_real_attention(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import (
            SparseGqaFmhaImpl,
        )

        torch.manual_seed(903)
        cfg = self.config()
        positions = torch.tensor(
            [[23, 47, 19], [24, 48, 20], [25, 49, 21]], device="cuda", dtype=torch.int32
        )
        table = torch.tensor([[1]], device="cuda", dtype=torch.int32)
        inputs = self.inputs(
            prefill=True, lengths=[3], prefixes=[0], positions=positions, table=table
        )
        owner = SparseGqaFmhaImpl(cfg, inputs)
        self.assertIsInstance(owner.rope_kvcache_impl, RocmQsaRopeCache)
        qkv = torch.randn(3, 2048, device="cuda", dtype=torch.bfloat16)
        cache = self.cache()
        selected = torch.tensor(
            [[0, -1, -1], [0, 1, -1], [0, 1, 2]], device="cuda", dtype=torch.int32
        )
        actual = owner.forward(qkv, cache, selected_indices=selected)
        rotated = self.oracle(qkv, positions, cfg).reshape(3, 8, 256).float()
        expected = torch.empty((3, 6, 256))
        for token in range(3):
            for head in range(6):
                score = rotated[token, head] @ rotated[: token + 1, 6].T / 16
                expected[token, head] = score.softmax(-1) @ rotated[: token + 1, 7]
        torch.testing.assert_close(
            actual.cpu(),
            expected.to(torch.bfloat16).reshape(3, 1536),
            atol=0.03125,
            rtol=0.02,
        )
        next_positions = positions[-1:].clone() + 1
        decode_inputs = self.inputs(
            prefill=False,
            lengths=[3],
            prefixes=[0],
            positions=next_positions,
            table=table,
        )
        decode = SparseGqaFmhaImpl(cfg, decode_inputs)
        next_qkv = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
        output = decode.forward(
            next_qkv,
            cache,
            selected_indices=torch.arange(4, device="cuda", dtype=torch.int32).view(
                1, 4
            ),
        )
        all_rotated = torch.cat(
            (
                rotated,
                self.oracle(next_qkv, next_positions, cfg).reshape(1, 8, 256).float(),
            )
        )
        expected_decode = torch.empty((1, 6, 256))
        for head in range(6):
            scores = all_rotated[3, head] @ all_rotated[:, 6].T / 16
            expected_decode[0, head] = scores.softmax(-1) @ all_rotated[:, 7]
        torch.testing.assert_close(
            output.cpu(),
            expected_decode.to(torch.bfloat16).reshape(1, 1536),
            atol=0.03125,
            rtol=0.02,
        )
        self.check_cache(
            cache,
            all_rotated.to(torch.bfloat16).reshape(4, 2048),
            table.cpu(),
            [4],
            [0],
            True,
        )

    def test_base_rope_and_contract_geometry(self):
        cfg = self.config()
        cfg.rope_config.style = RopeStyle.Base
        cfg.rope_config.index_factor = 1
        table = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        positions = torch.tensor([41], device="cuda", dtype=torch.int32)
        inputs = self.inputs(
            prefill=False, lengths=[128], prefixes=[0], positions=positions, table=table
        )
        op = RocmQsaRopeCache(cfg, is_prefill=False)
        params = op.prepare(inputs)
        qkv = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
        cache = self.cache()
        expected = self.oracle(qkv, positions.repeat_interleave(3), cfg)
        torch.testing.assert_close(
            op.forward(qkv, cache, params).cpu(),
            expected[:, :1536],
            atol=0.03125,
            rtol=0.01,
        )
        self.check_cache(cache, expected, table.cpu(), [128], [0], False)
        bad = self.config()
        bad.rope_config.offset = 1
        with self.assertRaisesRegex(NotImplementedError, "canonical"):
            RocmQsaRopeCache(bad, is_prefill=False)
        cache.kv_cache_base = torch.as_strided(cache.kv_cache_base, (2, 65536), (1, 1))
        with self.assertRaisesRegex(ValueError, "geometry"):
            op.forward(qkv, cache, params)
        with self.assertRaisesRegex(ValueError, "rows"):
            op.forward(qkv.repeat(2, 1), self.cache(), params)

    def test_unsupported_contracts_fail_before_cache_write(self):
        cfg = self.config()
        table = torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int32)
        positions = torch.zeros((1, 3), device="cuda", dtype=torch.int32)
        inputs = self.inputs(
            prefill=False, lengths=[0], prefixes=[0], positions=positions, table=table
        )
        op = RocmQsaRopeCache(cfg, is_prefill=False)
        inputs.is_target_verify = True
        with self.assertRaisesRegex(NotImplementedError, "MTP"):
            op.prepare(inputs)
        inputs.is_target_verify = False
        params = op.prepare(inputs)
        cache = self.cache()
        cache.kv_cache_base = cache.kv_cache_base.to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(NotImplementedError, "BF16"):
            op.forward(
                torch.zeros((1, 2048), device="cuda", dtype=torch.bfloat16),
                cache,
                params,
            )


if __name__ == "__main__":
    unittest.main()
