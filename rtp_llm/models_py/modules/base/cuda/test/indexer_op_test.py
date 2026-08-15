"""Negative-path tests for IndexerOp._indexer_cache_view geometry guards.

These guards protect indexer kernels from silently reading out of bounds when the
KV-cache layout does not match the kernel page geometry. They run before any CUDA
kernel, so CPU tensors and a duck-typed cache are sufficient to exercise them.
"""

from dataclasses import dataclass
from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp


@dataclass
class _FakeKVCache:
    kv_cache_base: object
    seq_size_per_block: int


def _make_op(
    blocksize: int = 64, block_size: int = 128, index_head_dim: int = 128
) -> IndexerOp:
    return IndexerOp(
        index_n_heads=1,
        index_head_dim=index_head_dim,
        index_topk=1,
        rope_head_dim=0,
        blocksize=blocksize,
        block_size=block_size,
    )


class IndexerCacheViewGuardTest(TestCase):
    def test_valid_layout_returns_paged_view(self):
        op = _make_op()
        cache = torch.zeros((2, op.page_elems), dtype=torch.uint8)
        view = op._indexer_cache_view(_FakeKVCache(cache, op.blocksize))
        self.assertEqual(tuple(view.shape), (2, op.blocksize, op.entry_elems))

    def test_non_contiguous_cache_raises(self):
        op = _make_op()
        cache = torch.zeros(
            (op.page_elems, 2), dtype=torch.uint8
        ).t()  # shape (2, page_elems), non-contiguous
        self.assertFalse(cache.is_contiguous())
        with self.assertRaisesRegex(RuntimeError, "must be a contiguous tensor"):
            op._indexer_cache_view(_FakeKVCache(cache, op.blocksize))

    def test_page_geometry_mismatch_raises(self):
        op = _make_op()
        cache = torch.zeros((2, op.page_elems), dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "page geometry mismatch"):
            op._indexer_cache_view(_FakeKVCache(cache, op.blocksize + 1))

    def test_row_width_mismatch_raises(self):
        op = _make_op()
        cache = torch.zeros((2, op.page_elems - 1), dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "exact 2D kernel-page layout"):
            op._indexer_cache_view(_FakeKVCache(cache, op.blocksize))


if __name__ == "__main__":
    main()
