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
