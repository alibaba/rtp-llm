from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import INDEXER_KV, TAG_BY_ATTN_TYPE
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8


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
        def get_layer_cache(self, layer_id: int, tag: str):
            assert layer_id == 0
            assert tag == TAG_BY_ATTN_TYPE[INDEXER_KV]
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


if __name__ == "__main__":
    test_indexer_pool_view_excludes_shared_stride_padding()
    print("OK")
