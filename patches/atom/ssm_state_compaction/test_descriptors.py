"""Verify the ATOM model's descriptors against the installed RTP extension."""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atom.plugin.rtpllm.models.qwen4_exp import ATOMQwen4Exp
from rtp_llm.ops import DataType, HybridAttentionType, KVCacheSpecDesc, LinearStateLayout


class DescriptorTests(unittest.TestCase):
    def descriptors(self, *, role="DECODE", hip="7.2", draft=False, vision=False):
        linear = HybridAttentionType.LINEAR
        full = HybridAttentionType.NONE
        with tempfile.TemporaryDirectory() as ckpt:
            text = {"indexer_compress_ratio": 4, "indexer_head_dim": 128,
                    "ple_layer_ids": [1], "ple_conv_kernel_size": 3, "ngram_size": 4}
            Path(ckpt, "config.json").write_text(json.dumps({"text_config": text}))
            config = SimpleNamespace(
                ckpt_path=ckpt, hidden_size=256, hc_mult=2, is_mtp=draft,
                hybrid_attention_config=SimpleNamespace(hybrid_attention_types=[full] if draft else [linear, full]),
                attn_config=SimpleNamespace(tokens_per_block=64),
                mm_model_config=SimpleNamespace(is_multimodal=vision),
            )
            with patch.dict(os.environ, {"ROLE_TYPE": role}), patch("torch.version.hip", hip):
                ATOMQwen4Exp._post_build_model_config(config)
        return [{desc.tag: desc for desc in layer} for layer in config.kv_cache_spec_descs]

    def test_rocm_target_roles_request_split_export(self):
        for role in ("PREFILL", "DECODE", "PDFUSION"):
            with self.subTest(role=role):
                linear, full = self.descriptors(role=role)
                self.assertEqual(linear["linear"].linear_state_layout, LinearStateLayout.SPLIT_EXPORT)
                self.assertEqual(linear["linear"].dtype, DataType.TYPE_BF16)
                self.assertTrue(linear["linear"].capacity.charge_to_paged_budget)
                self.assertEqual(set(linear), {"linear", "ple_conv_state", "ple_ngram_ctx"})
                self.assertEqual(set(full), {"full", "indexer_kv", "indexer_state"})

    def test_position_pool_belongs_to_multimodal_target_only(self):
        _, full = self.descriptors(vision=True)
        pos = full["indexer_rope_positions"]
        self.assertEqual((pos.entry_elems, pos.entry_dtype), (3, DataType.TYPE_INT32))
        self.assertEqual(pos.compression_ratio, full["indexer_state"].compression_ratio)
        draft, = self.descriptors(vision=True, draft=True)
        self.assertEqual(set(draft), {"full", "indexer_kv", "indexer_state"})

    def test_non_rocm_retains_default_layout(self):
        linear, _ = self.descriptors(hip=None)
        self.assertEqual(linear["linear"].linear_state_layout, LinearStateLayout.LEGACY)

    def test_descriptor_pickle_preserves_layout_and_legacy_component(self):
        desc = KVCacheSpecDesc()
        desc.linear_state_layout = LinearStateLayout.SINGLE_SSM
        desc.linear_state_component = 1
        restored = pickle.loads(pickle.dumps(desc))
        self.assertEqual(restored.linear_state_layout, desc.linear_state_layout)
        self.assertEqual(restored.linear_state_component, 1)
        for size in (20, 21):
            legacy = KVCacheSpecDesc()
            legacy.__setstate__(desc.__getstate__()[:size])
            self.assertEqual(legacy.linear_state_component, 0)


if __name__ == "__main__":
    unittest.main()
