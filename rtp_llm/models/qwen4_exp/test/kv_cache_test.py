import unittest

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
    PLE_NGRAM_CTX_TAG,
    PLE_STATE_TAG,
    build_qwen4_exp_kv_cache_spec_descs,
    indexer_kv_desc,
    indexer_state_desc,
    ple_short_conv_state_len,
    ple_state_desc,
)
from rtp_llm.ops import (
    DataType,
    HybridAttentionType,
    KVCacheSpecType,
    OpaqueBlockEntryCountMode,
)

_HC = 4
_HIDDEN = 2560
_NGRAM = 3
_PLE_KERNEL = 4


def _config(num_layers=8, interval=4):
    config = ModelConfig()
    config.num_layers = num_layers
    config.hidden_size = _HIDDEN
    config.hc_mult = _HC
    types = [
        (
            HybridAttentionType.NONE
            if (i + 1) % interval == 0
            else HybridAttentionType.LINEAR
        )
        for i in range(num_layers)
    ]
    config.hybrid_attention_config.enable_hybrid_attention = True
    config.hybrid_attention_config.hybrid_attention_types = types
    return config


class PleStateDescTest(unittest.TestCase):
    def test_state_length_reaches_back_over_whole_ngrams(self):
        # dilation == ngram_size, so the kernel steps over n-grams not tokens.
        self.assertEqual(ple_short_conv_state_len(_PLE_KERNEL, _NGRAM), 9)

    def test_desc_is_a_fixed_bf16_state_region(self):
        desc = ple_state_desc(_HC * _HIDDEN, _PLE_KERNEL, _NGRAM)

        self.assertEqual(desc.tag, PLE_STATE_TAG)
        self.assertEqual(desc.cache_type, KVCacheSpecType.OPAQUE_STATE)
        self.assertTrue(desc.is_state_cache)
        self.assertEqual(desc.entry_dtype, DataType.TYPE_BF16)
        self.assertEqual(desc.entry_elems, _HC * _HIDDEN)
        self.assertEqual(desc.entry_count_mode, OpaqueBlockEntryCountMode.EXPLICIT)
        self.assertEqual(desc.explicit_entry_count, 9)
        self.assertIsNotNone(desc.reuse)
        self.assertFalse(desc.reuse.enable_prefix_reuse)

    def test_rejects_degenerate_state_length(self):
        with self.assertRaisesRegex(ValueError, "state length must be positive"):
            ple_state_desc(_HC * _HIDDEN, ple_conv_kernel_size=1, ngram_size=3)


class BuildDescsTest(unittest.TestCase):
    def _build(self):
        return build_qwen4_exp_kv_cache_spec_descs(
            _config(),
            ple_layer_indices=[1],
            ple_conv_kernel_size=_PLE_KERNEL,
            ngram_size=_NGRAM,
        )

    def test_only_the_ple_layer_gets_extra_regions(self):
        descs = self._build()
        tags = [[d.tag for d in layer] for layer in descs]

        self.assertEqual(len(descs), 8)
        self.assertIn(PLE_STATE_TAG, tags[1])
        self.assertIn(PLE_NGRAM_CTX_TAG, tags[1])
        # linear region + conv state + ngram context
        self.assertEqual(len(descs[1]), 3)
        for layer_idx, layer_tags in enumerate(tags):
            if layer_idx != 1:
                self.assertNotIn(PLE_STATE_TAG, layer_tags)
                self.assertNotIn(PLE_NGRAM_CTX_TAG, layer_tags)
                self.assertEqual(len(descs[layer_idx]), 1)

    def test_ngram_ctx_region_is_int64_of_context_len(self):
        descs = self._build()

        ctx = next(d for d in descs[1] if d.tag == PLE_NGRAM_CTX_TAG)
        self.assertEqual(ctx.cache_type, KVCacheSpecType.OPAQUE_STATE)
        self.assertEqual(ctx.entry_dtype, DataType.TYPE_INT64)
        self.assertEqual(ctx.entry_elems, 1)
        self.assertEqual(ctx.explicit_entry_count, _NGRAM - 1)
        self.assertIsNotNone(ctx.reuse)
        self.assertFalse(ctx.reuse.enable_prefix_reuse)

    def test_ple_layer_keeps_its_own_attention_region(self):
        descs = self._build()

        non_ple = [
            d for d in descs[1] if d.tag not in (PLE_STATE_TAG, PLE_NGRAM_CTX_TAG)
        ]
        self.assertEqual(len(non_ple), 1)
        self.assertEqual(non_ple[0].cache_type, KVCacheSpecType.LINEAR)

    def test_state_width_follows_hc_mult_times_hidden(self):
        descs = self._build()

        ple = next(d for d in descs[1] if d.tag == PLE_STATE_TAG)
        self.assertEqual(ple.entry_elems, _HC * _HIDDEN)

    def test_rejects_out_of_range_ple_layer(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            build_qwen4_exp_kv_cache_spec_descs(
                _config(num_layers=4),
                ple_layer_indices=[9],
                ple_conv_kernel_size=_PLE_KERNEL,
                ngram_size=_NGRAM,
            )


class IndexerDescTest(unittest.TestCase):
    HEAD_DIM = 128
    RATIO = 4

    def test_kv_desc_is_a_compressed_byte_addressed_pool(self):
        desc = indexer_kv_desc(self.HEAD_DIM, self.RATIO)

        self.assertEqual(desc.tag, INDEXER_KV_TAG)
        self.assertEqual(desc.cache_type, KVCacheSpecType.OPAQUE_KV)
        self.assertFalse(desc.is_state_cache)
        # UINT8 so entry_elems is a byte count, like DSv4's indexer pool.
        self.assertEqual(desc.entry_dtype, DataType.TYPE_UINT8)
        self.assertEqual(desc.entry_elems, self.HEAD_DIM * 2)
        self.assertEqual(
            desc.entry_count_mode,
            OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED,
        )
        self.assertEqual(desc.compression_ratio, self.RATIO)
        self.assertFalse(desc.reuse.enable_prefix_reuse)

    def test_kv_desc_is_fixed_to_the_bf16_production_abi(self):
        """The 128-wide key is 256 B and never carries an inline FP8 scale."""
        desc = indexer_kv_desc(self.HEAD_DIM, self.RATIO)

        self.assertEqual(desc.entry_elems, 256)

    def test_kv_desc_rejects_invalid_geometry(self):
        with self.assertRaisesRegex(ValueError, "indexer_head_dim"):
            indexer_kv_desc(0, self.RATIO)
        with self.assertRaisesRegex(ValueError, "compress_ratio"):
            indexer_kv_desc(self.HEAD_DIM, 0)

    def test_state_desc_holds_one_fp32_slot_per_block_position(self):
        desc = indexer_state_desc(self.HEAD_DIM, self.RATIO)

        self.assertEqual(desc.tag, INDEXER_STATE_TAG)
        self.assertEqual(desc.cache_type, KVCacheSpecType.OPAQUE_STATE)
        self.assertTrue(desc.is_state_cache)
        self.assertEqual(desc.entry_dtype, DataType.TYPE_FP32)
        self.assertEqual(desc.entry_elems, self.HEAD_DIM)
        self.assertEqual(desc.entry_count_mode, OpaqueBlockEntryCountMode.STATE_RING)
        self.assertEqual(desc.state_ring_overlap, 1)
        self.assertFalse(desc.reuse.enable_prefix_reuse)


class IndexerLayerPlacementTest(unittest.TestCase):
    HEAD_DIM = 128
    RATIO = 4
    SIDE = frozenset(
        {INDEXER_KV_TAG, INDEXER_STATE_TAG, PLE_STATE_TAG, PLE_NGRAM_CTX_TAG}
    )

    def _build(self, **kwargs):
        return build_qwen4_exp_kv_cache_spec_descs(
            _config(),
            ple_layer_indices=[1],
            ple_conv_kernel_size=_PLE_KERNEL,
            ngram_size=_NGRAM,
            **kwargs,
        )

    def _with_indexer(self):
        return self._build(
            indexer_head_dim=self.HEAD_DIM, indexer_compress_ratio=self.RATIO
        )

    def test_indexer_regions_land_only_on_full_attention_layers(self):
        for layer_idx, layer in enumerate(self._with_indexer()):
            tags = {d.tag for d in layer}
            has_indexer = {INDEXER_KV_TAG, INDEXER_STATE_TAG} <= tags
            self.assertEqual(has_indexer, (layer_idx + 1) % 4 == 0, msg=str(layer_idx))

    def test_every_layer_keeps_exactly_one_attention_region(self):
        """What ``Qwen4ExpModel._attention_tag`` relies on to route attention."""
        for layer in self._with_indexer():
            self.assertEqual(len([d for d in layer if d.tag not in self.SIDE]), 1)

    def test_indexer_is_skipped_when_head_dim_is_zero(self):
        """The dense-fallback configuration must not allocate indexer pools."""
        for layer in self._build():
            tags = {d.tag for d in layer}
            self.assertNotIn(INDEXER_KV_TAG, tags)
            self.assertNotIn(INDEXER_STATE_TAG, tags)

    def test_rejects_missing_compress_ratio(self):
        with self.assertRaisesRegex(ValueError, "compress_ratio must be positive"):
            self._build(indexer_head_dim=self.HEAD_DIM, indexer_compress_ratio=0)


if __name__ == "__main__":
    unittest.main()
