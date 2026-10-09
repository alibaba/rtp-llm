import math
from dataclasses import dataclass
from unittest import TestCase, main

import torch

from rtp_llm.models_py.kernels.cuda.fast_topk import fast_topk_transform_ragged_fused
from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.ops.compute_ops import LayerKVCache, rtp_llm_ops


@dataclass
class _FakeLayerKVCache:
    kv_cache_base: object
    seq_size_per_block: int


def _make_op() -> IndexerOp:
    return IndexerOp(
        index_n_heads=1,
        index_head_dim=128,
        index_topk=1,
        rope_head_dim=0,
        blocksize=64,
        block_size=128,
    )


class IndexerCacheViewTest(TestCase):
    def test_opaque_pool_is_exposed_as_token_addressable_view(self) -> None:
        op = _make_op()
        cache = torch.empty((3, 64 * 132), dtype=torch.uint8)

        view = op._indexer_cache_view(_FakeLayerKVCache(cache, 64))

        self.assertEqual(tuple(view.shape), (3, 64, 132))
        self.assertEqual(view.data_ptr(), cache.data_ptr())

    def test_malformed_opaque_pool_geometry_is_rejected(self) -> None:
        op = _make_op()
        with self.assertRaisesRegex(RuntimeError, "page geometry mismatch"):
            op._indexer_cache_view(
                _FakeLayerKVCache(torch.empty((2, 64 * 132), dtype=torch.uint8), 128)
            )
        with self.assertRaisesRegex(RuntimeError, "kernel-page layout"):
            op._indexer_cache_view(
                _FakeLayerKVCache(torch.empty((2, 64 * 132 - 1), dtype=torch.uint8), 64)
            )


class IndexerCacheKernelTest(TestCase):
    @staticmethod
    def _gather_cache(
        op: IndexerOp, cache: LayerKVCache, token_count: int
    ) -> torch.Tensor:
        page_count = math.ceil(token_count / op.blocksize)
        block_table = torch.arange(
            page_count, dtype=torch.int32, device=cache.kv_cache_base.device
        ).view(1, -1)
        cu_kv_seqlens = torch.tensor(
            [0, token_count], dtype=torch.int32, device=cache.kv_cache_base.device
        )
        gathered_k = torch.empty(
            (token_count, op.index_head_dim),
            dtype=torch.float8_e4m3fn,
            device=cache.kv_cache_base.device,
        )
        gathered_scale_bytes = torch.empty(
            (token_count, 4), dtype=torch.uint8, device=cache.kv_cache_base.device
        )
        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            op._indexer_cache_view(cache),
            gathered_k,
            gathered_scale_bytes,
            block_table,
            cu_kv_seqlens,
        )
        return gathered_k.float() * gathered_scale_bytes.view(torch.float32)

    @staticmethod
    def _reference_sparse_decode(
        query: torch.Tensor,
        indexer_keys: torch.Tensor,
        main_values: torch.Tensor,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(indexer_keys.float(), query.float())
        selected = torch.topk(scores, topk).indices
        weights = torch.softmax(scores[selected], dim=0)
        output = torch.sum(main_values[selected].float() * weights.unsqueeze(-1), dim=0)
        return selected, output

    def test_independent_indexer_prefix_preserves_sparse_decode_computation(self) -> None:
        """Real cache kernels plus reference sparse selection; not a FlashMLA kernel test."""
        device = torch.device("cuda")
        op = IndexerOp(
            index_n_heads=1,
            index_head_dim=128,
            index_topk=2,
            rope_head_dim=0,
            blocksize=64,
            block_size=128,
        )
        history_length = 64
        token_count = history_length + 1
        page_count = 2

        def make_indexer_cache() -> LayerKVCache:
            return LayerKVCache(
                torch.zeros(
                    (page_count, op.blocksize * 132),
                    dtype=torch.uint8,
                    device=device,
                ),
                op.blocksize,
                tag="indexer_kv",
            )

        history_keys = torch.zeros(
            (history_length, op.index_head_dim), dtype=torch.bfloat16, device=device
        )
        history_keys[:, 0] = -8
        history_keys[0, 0] = 16
        history_keys[1, 0] = 8
        current_key = torch.zeros((1, op.index_head_dim), dtype=torch.bfloat16, device=device)
        current_key[0, 0] = -16
        query = torch.zeros((1, 1, op.index_head_dim), dtype=torch.bfloat16, device=device)
        query[0, 0, 0] = 1
        history_slots = torch.arange(history_length, dtype=torch.int64, device=device)
        current_slot = torch.tensor([history_length], dtype=torch.int64, device=device)
        main_values = torch.stack(
            (
                torch.arange(token_count, dtype=torch.float32, device=device),
                torch.arange(token_count, dtype=torch.float32, device=device).square(),
            ),
            dim=-1,
        )

        no_reuse_cache = make_indexer_cache()
        op.quant_k_only(history_keys, no_reuse_cache, history_slots)
        q_fp8, q_scale = op.quant_q_k(query, current_key, no_reuse_cache, current_slot)
        query_dequant = (q_fp8.float() * q_scale).reshape(-1)
        no_reuse_keys = self._gather_cache(op, no_reuse_cache, token_count)
        no_reuse_topk, no_reuse_output = self._reference_sparse_decode(
            query_dequant, no_reuse_keys, main_values, op.index_topk
        )
        self.assertEqual(no_reuse_topk.cpu().tolist(), [0, 1])

        restored_cache = make_indexer_cache()
        restored_cache.kv_cache_base[0].copy_(no_reuse_cache.kv_cache_base[0])
        restored_main_values = torch.zeros_like(main_values)
        restored_main_values[:history_length].copy_(main_values[:history_length])
        restored_main_values[history_length].copy_(main_values[history_length])
        op.quant_q_k(query, current_key, restored_cache, current_slot)
        restored_keys = self._gather_cache(op, restored_cache, token_count)
        restored_topk, restored_output = self._reference_sparse_decode(
            query_dequant, restored_keys, restored_main_values, op.index_topk
        )
        self.assertTrue(torch.equal(restored_topk, no_reuse_topk))
        torch.testing.assert_close(restored_output, no_reuse_output, rtol=0, atol=0)

        perturbed_cache = make_indexer_cache()
        perturbed_cache.kv_cache_base[0].copy_(no_reuse_cache.kv_cache_base[0])
        perturbed_history = history_keys.clone()
        perturbed_history[0, 0] = -16
        perturbed_history[1, 0] = -8
        perturbed_history[2, 0] = 16
        perturbed_history[3, 0] = 8
        op.quant_k_only(perturbed_history, perturbed_cache, history_slots)
        op.quant_q_k(query, current_key, perturbed_cache, current_slot)
        perturbed_keys = self._gather_cache(op, perturbed_cache, token_count)
        perturbed_topk, perturbed_output = self._reference_sparse_decode(
            query_dequant, perturbed_keys, main_values, op.index_topk
        )
        self.assertEqual(perturbed_topk.cpu().tolist(), [2, 3])
        self.assertFalse(torch.equal(perturbed_topk, no_reuse_topk))
        self.assertFalse(torch.equal(perturbed_output, no_reuse_output))

    def test_short_prefill_indexer_write_is_read_by_decode_selection(self) -> None:
        """Exercise real indexer cache write/read kernels around reference decode math."""
        device = torch.device("cuda")
        op = _make_op()
        history_length = 8
        cache = LayerKVCache(
            torch.zeros((1, op.blocksize * 132), dtype=torch.uint8, device=device),
            op.blocksize,
            tag="indexer_kv",
        )
        history_keys = torch.zeros(
            (history_length, op.index_head_dim), dtype=torch.bfloat16, device=device
        )
        history_keys[3, 0] = 16
        op.quant_k_only(
            history_keys,
            cache,
            torch.arange(history_length, dtype=torch.int64, device=device),
        )
        query = torch.zeros((1, 1, op.index_head_dim), dtype=torch.bfloat16, device=device)
        query[0, 0, 0] = 1
        current_key = torch.zeros((1, op.index_head_dim), dtype=torch.bfloat16, device=device)
        q_fp8, q_scale = op.quant_q_k(
            query,
            current_key,
            cache,
            torch.tensor([history_length], dtype=torch.int64, device=device),
        )
        gathered = self._gather_cache(op, cache, history_length + 1)
        main_values = torch.arange(history_length + 1, dtype=torch.float32, device=device).unsqueeze(-1)
        selected, output = self._reference_sparse_decode(
            (q_fp8.float() * q_scale).reshape(-1), gathered, main_values, topk=1
        )
        self.assertEqual(selected.item(), 3)
        self.assertEqual(output.item(), 3)

    def test_real_cache_quant_gather_and_topk_cross_page(self) -> None:
        device = torch.device("cuda")
        op = IndexerOp(
            index_n_heads=1,
            index_head_dim=128,
            index_topk=2048,
            rope_head_dim=0,
            blocksize=64,
            block_size=128,
        )
        token_count = op.index_topk + 1
        page_count = (token_count + op.blocksize - 1) // op.blocksize
        sentinel = 0xA5
        opaque_pool = torch.full(
            (page_count, op.blocksize * 132),
            sentinel,
            dtype=torch.uint8,
            device=device,
        )
        cache = LayerKVCache(
            opaque_pool,
            op.blocksize,
            layer_id=3,
            tag="indexer_kv",
        )
        self.assertEqual(cache.layer_id, 3)
        self.assertEqual(cache.tag, "indexer_kv")

        keys = torch.ones(
            (token_count, op.index_head_dim), dtype=torch.bfloat16, device=device
        )
        keys[:, 1:] = 0
        keys[0, 0] = -16
        keys[op.blocksize - 1, 0] = 8
        keys[op.blocksize, 0] = 12
        slots = torch.arange(token_count, dtype=torch.int64, device=device)
        op.quant_k_only(keys, cache, slots)

        cache_view = op._indexer_cache_view(cache)
        block_table = torch.arange(page_count, dtype=torch.int32, device=device).view(
            1, -1
        )
        cu_kv_seqlens = torch.tensor([0, token_count], dtype=torch.int32, device=device)
        gathered_k = torch.empty(
            (token_count, op.index_head_dim),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        gathered_scale_bytes = torch.empty(
            (token_count, 4), dtype=torch.uint8, device=device
        )
        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            cache_view,
            gathered_k,
            gathered_scale_bytes,
            block_table,
            cu_kv_seqlens,
        )
        gathered_scales = gathered_scale_bytes.view(torch.float32)

        key_fp32 = keys.float()
        expected_scales = torch.pow(
            2.0,
            torch.ceil(
                torch.log2(
                    torch.clamp(key_fp32.abs().amax(dim=1, keepdim=True), min=1e-4)
                    / 448.0
                )
            ),
        )
        expected_quant = (key_fp32 / expected_scales).to(torch.float8_e4m3fn)
        expected_dequant = expected_quant.float() * expected_scales
        gathered_dequant = gathered_k.float() * gathered_scales
        torch.testing.assert_close(gathered_scales, expected_scales, rtol=0, atol=0)
        torch.testing.assert_close(gathered_dequant, expected_dequant, rtol=0, atol=0)

        boundary = (op.blocksize - 1, op.blocksize)
        torch.testing.assert_close(
            gathered_dequant[list(boundary)],
            key_fp32[list(boundary)],
            rtol=0,
            atol=0,
        )
        last_block = opaque_pool[-1]
        written_in_last_block = token_count % op.blocksize
        unwritten_key_offset = written_in_last_block * op.index_head_dim
        scale_region_offset = op.blocksize * op.index_head_dim
        unwritten_scale_offset = scale_region_offset + written_in_last_block * 4
        self.assertTrue(
            (
                last_block[
                    unwritten_key_offset : unwritten_key_offset + op.index_head_dim
                ]
                == sentinel
            )
            .all()
            .item()
        )
        self.assertTrue(
            (
                last_block[unwritten_scale_offset : unwritten_scale_offset + 4]
                == sentinel
            )
            .all()
            .item()
        )

        scores = gathered_dequant[:, 0].view(1, -1).contiguous()
        lengths = torch.tensor([token_count], dtype=torch.int32, device=device)
        offsets = torch.zeros(1, dtype=torch.int32, device=device)
        topk = fast_topk_transform_ragged_fused(
            scores,
            lengths,
            offsets,
            op.index_topk,
        )
        expected_topk = torch.topk(scores, op.index_topk, dim=-1).indices.to(
            torch.int32
        )
        self.assertTrue(
            torch.equal(
                torch.sort(topk, dim=-1).values,
                torch.sort(expected_topk, dim=-1).values,
            )
        )
        self.assertNotIn(0, topk.cpu().tolist()[0])


if __name__ == "__main__":
    main()
