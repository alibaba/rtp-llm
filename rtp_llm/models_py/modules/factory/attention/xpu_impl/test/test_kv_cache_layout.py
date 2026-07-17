"""XPU KV cache layout consistency test.

Verifies that the KV cache layout produced by KVCache::getLayerCache() (XPU
branch in rtp_llm/models_py/bindings/OpDefs.h) matches what the XPU paged flash
attention consumer in vllm_flash_attn.py reads and writes.

The XPU layout is NSHD: [num_blocks, 2, seq_size_per_block, num_kv_heads,
head_dim] (seq before head), which differs from the CUDA HND layout
[num_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]. This test guards
against silent producer/consumer drift. It is CPU-runnable (no XPU device
required); it only needs the compiled rtp_llm.ops bindings for KVCache.
"""

from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.modules.factory.attention.xpu_impl.test._import_guard import (
    skip_or_fail_on_missing_import,
)

try:
    from rtp_llm.ops.compute_ops import CacheGroupType, KVCache
    from rtp_llm.models_py.modules.factory.attention.xpu_impl.vllm_flash_attn import (
        _assert_nshd_cache,
        _read_from_paged_cache,
        _write_to_paged_cache,
    )
    _IMPORT_OK = True
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover - import-time environment guard
    _IMPORT_OK = False
    _IMPORT_ERR = _e

# Distinct seq/head so the NSHD vs HND axis order is unambiguous.
NUM_BLOCKS = 4
TPB = 8          # seq_size_per_block
NUM_KV_HEADS = 3
HEAD_DIM = 16


class XpuKVCacheLayoutTest(TestCase):
    def setUp(self):
        skip_or_fail_on_missing_import(
            self, _IMPORT_OK, _IMPORT_ERR, "rtp_llm.ops / xpu consumer"
        )

    def _make_layer_cache(self):
        # Physical 2-D block buffer: [num_blocks, 2*tpb*heads*dim].
        base = torch.zeros(NUM_BLOCKS, 2 * TPB * NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        kv = KVCache()
        kv.seq_size_per_block = TPB
        kv.kernel_seq_size_per_block = TPB
        kv.num_kv_heads = NUM_KV_HEADS
        kv.head_dim = HEAD_DIM
        kv.use_mla = False
        kv.kv_cache_base_by_layer = [base]
        kv.layer_attn_types = [CacheGroupType.FULL]
        return kv.get_layer_cache(0)

    def test_producer_layout_is_nshd(self):
        lc = self._make_layer_cache()
        shape = tuple(lc.kv_cache_base.shape)
        expected = (NUM_BLOCKS, 2, TPB, NUM_KV_HEADS, HEAD_DIM)
        if shape != expected:
            # Non-XPU build: getLayerCache emits HND; the XPU consumer is not
            # used there, so this linkage assertion does not apply.
            raise SkipTest(f"non-XPU getLayerCache layout {shape}; expected NSHD {expected}")
        self.assertEqual(lc.seq_size_per_block, TPB)

    def test_write_read_roundtrip_matches_producer(self):
        lc = self._make_layer_cache()
        if tuple(lc.kv_cache_base.shape) != (NUM_BLOCKS, 2, TPB, NUM_KV_HEADS, HEAD_DIM):
            raise SkipTest("non-XPU getLayerCache layout; consumer linkage N/A")
        total = NUM_BLOCKS * TPB
        torch.manual_seed(0)
        k_in = torch.randn(total, NUM_KV_HEADS, HEAD_DIM)
        v_in = torch.randn(total, NUM_KV_HEADS, HEAD_DIM)
        bids = torch.arange(NUM_BLOCKS, dtype=torch.long)

        _write_to_paged_cache(k_in, v_in, lc, bids, 0, NUM_KV_HEADS, HEAD_DIM)
        k_out, v_out = _read_from_paged_cache(lc, bids, total, NUM_KV_HEADS, HEAD_DIM)

        self.assertTrue(torch.equal(k_in, k_out))
        self.assertTrue(torch.equal(v_in, v_out))

        # Token 0 must land at NSHD position cache[block0, k=0, off0, :, :].
        cache = lc.kv_cache_base
        self.assertTrue(torch.equal(k_in[0], cache[0, 0, 0, :, :]))
        self.assertTrue(torch.equal(v_in[0], cache[0, 1, 0, :, :]))

    def test_guard_rejects_hnd_layout(self):
        skip_or_fail_on_missing_import(
            self, _IMPORT_OK, _IMPORT_ERR, "rtp_llm.ops / xpu consumer"
        )
        # HND tensor [blocks, 2, heads, seq, dim] must be rejected.
        hnd = torch.zeros(NUM_BLOCKS, 2, NUM_KV_HEADS, TPB, HEAD_DIM)
        with self.assertRaises(RuntimeError):
            _assert_nshd_cache(hnd, TPB, NUM_KV_HEADS, HEAD_DIM)

    def test_guard_accepts_nshd_layout(self):
        skip_or_fail_on_missing_import(
            self, _IMPORT_OK, _IMPORT_ERR, "rtp_llm.ops / xpu consumer"
        )
        nshd = torch.zeros(NUM_BLOCKS, 2, TPB, NUM_KV_HEADS, HEAD_DIM)
        _assert_nshd_cache(nshd, TPB, NUM_KV_HEADS, HEAD_DIM)  # must not raise


if __name__ == "__main__":
    main()
