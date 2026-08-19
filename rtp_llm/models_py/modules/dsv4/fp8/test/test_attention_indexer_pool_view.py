from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import INDEXER_KV


def test_indexer_pool_view_excludes_shared_stride_padding() -> None:
    entry_bytes = 132
    useful_entries = 64
    stride_entries = 128
    base = torch.arange(3 * stride_entries * entry_bytes, dtype=torch.int64)
    base = base.remainder(251).to(torch.uint8).view(3, -1)

    layer_cache = SimpleNamespace(
        kv_cache_base=base,
        seq_size_per_block=256,
    )

    class FakeKVCache:
        def has_layer_cache(self, layer_id: int, tag: str) -> bool:
            assert layer_id == 0
            return tag == INDEXER_KV

        def get_layer_cache(self, layer_id: int, tag: str):
            assert layer_id == 0
            assert tag == INDEXER_KV
            return layer_cache

    layer = AttentionFP8.__new__(AttentionFP8)
    torch.nn.Module.__init__(layer)
    layer.layer_id = 0
    layer._kv_cache = FakeKVCache()
    layer._pool_spec = {INDEXER_KV: (torch.uint8, entry_bytes)}
    layer.indexer = SimpleNamespace(compress_ratio=4)

    assert layer._pool_entries_per_block(INDEXER_KV) == useful_entries
    view = layer._pool_view_3d_fp8(INDEXER_KV)
    assert view is not None
    assert view.shape == (3, useful_entries, entry_bytes)
    assert view.stride() == (stride_entries * entry_bytes, entry_bytes, 1)
    assert torch.equal(view[1, 0], base[1, :entry_bytes])


def test_pool_probe_skips_unowned_region_without_get() -> None:
    class FakeKVCache:
        def has_layer_cache(self, layer_id: int, tag: str) -> bool:
            return False

        def get_layer_cache(self, layer_id: int, tag: str):
            raise AssertionError("unowned cache must not be fetched")

    layer = AttentionFP8.__new__(AttentionFP8)
    torch.nn.Module.__init__(layer)
    layer.layer_id = 0
    layer._kv_cache = FakeKVCache()
    layer._pool_spec = {INDEXER_KV: (torch.uint8, 132)}
    layer.indexer = SimpleNamespace(compress_ratio=4)

    assert layer._pool_entries_per_block(INDEXER_KV) == 0
    assert layer._pool_view_3d_fp8(INDEXER_KV) is None


if __name__ == "__main__":
    test_indexer_pool_view_excludes_shared_stride_padding()
    test_pool_probe_skips_unowned_region_without_get()
    print("OK")
