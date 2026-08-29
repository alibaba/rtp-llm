from dataclasses import dataclass
from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp


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


if __name__ == "__main__":
    main()
