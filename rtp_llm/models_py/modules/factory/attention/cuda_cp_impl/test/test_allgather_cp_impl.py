"""
Unit tests for PCPAllGatherAttnOp (all-gather without overlap).

Tests two scenarios:
  1. Normal context-parallel attention (no prefix cache)
  2. Context-parallel attention with prefix cache
"""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.ops import KvCacheDataType

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.allgather_cp_impl import (
    PCPAllGatherAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    CPAttnTestBase,
    build_cp_attn_inputs,
    compute_rank_positions,
    make_configs,
    make_kv_cache,
    reference_prefill_with_prefix,
)

_AG_MODULE = (
    "rtp_llm.models_py.modules.factory.attention."
    "cuda_cp_impl.prefill_mha.allgather_cp_impl"
)


class TestPCPAllGatherAttnOp(CPAttnTestBase):
    OP_CLASS = PCPAllGatherAttnOp
    AG_MODULE = _AG_MODULE

    # ==================================================================
    # Case 1: Normal CP attention (no prefix cache)
    # ==================================================================

    def test_no_prefix_single_seq_rank0(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[32], cp_size=2, cp_rank=0)

    def test_no_prefix_single_seq_rank1(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[32], cp_size=2, cp_rank=1)

    def test_no_prefix_multi_batch(self):
        self.run_no_prefix(
            batch_size=2, sequence_lengths=[32, 64], cp_size=2, cp_rank=0
        )

    def test_no_prefix_larger(self):
        self.run_no_prefix(
            batch_size=1,
            sequence_lengths=[128],
            cp_size=2,
            cp_rank=0,
            head_num=32,
            kv_head_num=8,
            head_dim=128,
            tokens_per_block=64,
        )

    def test_no_prefix_cp4(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[64], cp_size=4, cp_rank=2)

    def test_no_prefix_gqa4(self):
        self.run_no_prefix(
            batch_size=1,
            sequence_lengths=[32],
            cp_size=2,
            cp_rank=0,
            head_num=16,
            kv_head_num=4,
            head_dim=64,
        )

    def test_no_prefix_multi_batch_cp4(self):
        self.run_no_prefix(
            batch_size=2, sequence_lengths=[64, 64], cp_size=4, cp_rank=1
        )

    # ==================================================================
    # Case 2: CP attention with prefix cache
    # ==================================================================

    def test_prefix_single_seq_rank0(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[32],
            prefix_lengths=[64],
            cp_size=2,
            cp_rank=0,
            tokens_per_block=16,
        )

    def test_prefix_single_seq_rank1(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[32],
            prefix_lengths=[64],
            cp_size=2,
            cp_rank=1,
            tokens_per_block=16,
        )

    def test_prefix_multi_batch(self):
        self.run_with_prefix(
            batch_size=2,
            new_lengths=[32, 64],
            prefix_lengths=[64, 128],
            cp_size=2,
            cp_rank=0,
            tokens_per_block=32,
        )

    def test_prefix_larger(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[64],
            prefix_lengths=[128],
            cp_size=2,
            cp_rank=0,
            head_num=32,
            kv_head_num=8,
            head_dim=128,
            tokens_per_block=64,
        )

    def test_prefix_cp4(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[64],
            prefix_lengths=[64],
            cp_size=4,
            cp_rank=2,
            tokens_per_block=16,
        )

    # ==================================================================
    # Case 3: Irregular seq_len (non-power-of-2, partial pages)
    # ==================================================================

    def test_no_prefix_irregular_seqlen(self):
        self.run_no_prefix(
            batch_size=1,
            sequence_lengths=[20],
            cp_size=2,
            cp_rank=0,
        )

    def test_no_prefix_irregular_multi_batch(self):
        self.run_no_prefix(
            batch_size=2,
            sequence_lengths=[20, 36],
            cp_size=2,
            cp_rank=1,
        )

    def test_prefix_irregular_seqlen(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[20],
            prefix_lengths=[48],
            cp_size=2,
            cp_rank=0,
            tokens_per_block=16,
        )

    def test_prefix_irregular_multi_batch(self):
        self.run_with_prefix(
            batch_size=2,
            new_lengths=[20, 36],
            prefix_lengths=[48, 32],
            cp_size=2,
            cp_rank=0,
            tokens_per_block=16,
        )

    def test_sharded_prefix_and_append_with_split_kernel_pages(self):
        for prefix, dtype in ((128, torch.bfloat16), (83, torch.bfloat16), (83, torch.float16)):
            with self.subTest(prefix=prefix, dtype=dtype):
                self.check_sharded_prefix_and_append(prefix, dtype)

    def test_sharded_allgather_rejects_quantized_kv_cache(self):
        config, parallelism = make_configs()
        parallelism.prefill_cp_config.kv_cache_sharded = True
        config.kv_cache_dtype = KvCacheDataType.FP8
        inputs = build_cp_attn_inputs([32], [16], 2, 16, device=self.device)
        with self.assertRaisesRegex(ValueError, "unquantized KV cache"):
            self.OP_CLASS(config, inputs, parallelism)

    def check_sharded_prefix_and_append(self, prefix, dtype):
        block_size, page_size = 32, 16
        heads, kv_heads, head_dim = 8, 2, 64
        cp_size = 2
        pages_per_block = block_size // page_size
        prefixes = [prefix, 0]
        new_lengths = [68, 36]
        seq_lengths = [prefix + length for prefix, length in zip(prefixes, new_lengths)]

        # Reserve the first physical block. Tables hold each rank's compact pages.
        block_table = torch.tensor(
            [[2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 0, 0, 0, 0, 0, 0]],
            dtype=torch.int32,
        )

        def rand(*shape):
            return torch.randn(shape, dtype=dtype, device=self.device)

        prefix_pages = rand(4, 2, 2, kv_heads, page_size, head_dim)
        if prefix == 83:
            # The last physical block is padding. Its pages must not enter attention.
            prefix_pages[-1].fill_(30)
        prefix_shards = [
            prefix_pages[rank::cp_size].flatten(0, 1) for rank in range(cp_size)
        ]
        prefix_k, prefix_v = (
            prefix_pages.flatten(0, 1)[:, i]
            .transpose(1, 2)
            .reshape(-1, kv_heads, head_dim)
            for i in range(2)
        )
        new_q = rand(sum(new_lengths), heads, head_dim)
        new_k = rand(sum(new_lengths), kv_heads, head_dim)
        new_v = rand(sum(new_lengths), kv_heads, head_dim)
        reference = reference_prefill_with_prefix(
            new_q, prefix_k, prefix_v, new_k, new_v, new_lengths, prefixes
        )
        rank_rows = torch.tensor(
            compute_rank_positions(new_lengths, cp_size), device=self.device
        )
        gathered_rows = rank_rows.flatten()
        gathered_k = new_k[gathered_rows].flatten(1)
        gathered_v = new_v[gathered_rows].flatten(1)
        gathered_prefix = torch.cat(prefix_shards)

        for rank, rows in enumerate(rank_rows):
            with self.subTest(rank=rank):
                config, parallelism = make_configs(
                    heads, kv_heads, head_dim, block_size, cp_size, rank
                )
                config.kernel_tokens_per_block = page_size
                config.dtype = dtype
                parallelism.prefill_cp_config.kv_cache_sharded = True
                inputs = build_cp_attn_inputs(
                    seq_lengths,
                    [length // cp_size for length in new_lengths],
                    cp_size,
                    block_size,
                    prefix_lengths=prefixes,
                    device=self.device,
                )
                inputs.kv_cache_kernel_block_id = block_table
                kv_cache = make_kv_cache(
                    12, kv_heads, page_size, head_dim, device=self.device
                )
                kv_cache.kv_cache_base = kv_cache.kv_cache_base.to(dtype)
                kv_cache.kv_cache_base.fill_(7)
                kv_cache.kv_cache_base[2:6] = prefix_shards[rank]
                expected_cache = kv_cache.kv_cache_base.clone()

                row_start = 0
                for batch, (prefix, length) in enumerate(zip(prefixes, new_lengths)):
                    for index in range(length):
                        position = prefix + index
                        physical_block, offset = divmod(position, block_size)
                        if physical_block % cp_size == rank:
                            local_page = (
                                physical_block // cp_size * pages_per_block
                                + offset // page_size
                            )
                            cache_page = block_table[batch, local_page].item()
                            slot = offset % page_size
                            row = row_start + index
                            expected_cache[cache_page, 0, :, slot] = new_k[row]
                            expected_cache[cache_page, 1, :, slot] = new_v[row]
                    row_start += length

                qkv = torch.cat(
                    [t[rows].flatten(1) for t in (new_q, new_k, new_v)], dim=-1
                )
                op = self.OP_CLASS(config, inputs, parallelism)
                params = op.prepare(inputs)
                with patch(
                    f"{self.AG_MODULE}.all_gather",
                    side_effect=[gathered_k, gathered_v, gathered_prefix],
                ) as gather:
                    output = op.forward(qkv, kv_cache, params)

                self._assert_close(output, reference[rows], rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(
                    gather.call_args.args[0], expected_cache[2:6], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    kv_cache.kv_cache_base, expected_cache, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
