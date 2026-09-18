"""Verify the weight mapping against a real Qwen3.8-Flash-Next weight index.

Needs only `config.json` + `model.safetensors.index.json` (~170KB), not the
360GB of tensors. Point QWEN4_EXP_REF_DIR at a directory holding those two files;
the test skips when it is absent so CI stays green without them.

    mkdir -p ref && cd ref
    base=https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master
    curl -sSLO $base/config.json
    curl -sSLO $base/model.safetensors.index.json
"""

import json
import os
import re
import unittest

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp
from rtp_llm.models.qwen4_exp.qwen4_exp_weight import Qwen4ExpWeight
from rtp_llm.ops import HWKernelConfig, ParallelismConfig

_REF_DIR = os.environ.get("QWEN4_EXP_REF_DIR", "")
_INDEX = "model.safetensors.index.json"

# Checkpoint tensors we knowingly do not consume yet, as ckpt-key patterns.
# Anything outside this set showing up as unmapped is a real gap.
# (The PLE / indexer tensors used to live here; they are mapped now.)
_EXPECTED_UNMAPPED = {
    "model.visual.",
    "mtp.",
}


def _available():
    return bool(_REF_DIR) and os.path.exists(os.path.join(_REF_DIR, _INDEX))


def _collapse(key):
    key = re.sub(r"\blayers\.\d+\.", "layers.{i}.", key)
    key = re.sub(r"\bblocks\.\d+\.", "blocks.{i}.", key)
    return re.sub(r"\bshard_\d+\b", "shard_{s}", key)


def _ckpt_keys(modules):
    out = set()
    for module in modules:
        sub_weights = getattr(module, "sub_weights", None)
        if sub_weights:
            out |= _ckpt_keys(sub_weights.values())
        else:
            out |= {info.name for info in module.weights}
    return out


@unittest.skipUnless(_available(), "QWEN4_EXP_REF_DIR with the weight index not set")
class RealCkptMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(_REF_DIR, _INDEX)) as reader:
            cls.real_keys = set(json.load(reader)["weight_map"])
        cls.config = Qwen4Exp.create_config(_REF_DIR)
        # Manifest coverage is intentionally broader than the safe serving
        # default. Enabling the descriptors here does not load tensor data; it
        # verifies that both opt-in paths claim every real checkpoint key.
        cls.config.enable_qwen4_ple = True
        cls.config.enable_qwen4_qsa = True
        weight = Qwen4ExpWeight(
            model_config=cls.config,
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )
        weight._process_meta({}, cls.real_keys)
        cls.weight = weight
        info = weight._get_weight_info()
        requested = _ckpt_keys(info.weights)
        for layer_id, layer in enumerate(info.layer_weights):
            requested |= {
                key.replace("{i}", str(layer_id)) for key in _ckpt_keys(layer)
            }
        cls.requested = requested

    def test_config_parses_the_real_topology(self):
        self.assertEqual(self.config.num_layers, 48)
        self.assertEqual(self.config.hc_mult, 4)
        self.assertEqual(self.config.expert_num, 512)
        self.assertEqual(self.config.attn_config.kv_head_num, 2)
        self.assertEqual(self.config.attn_config.size_per_head, 256)

    def test_prefix_and_stacked_layout_detected_from_real_keys(self):
        self.assertEqual(self.weight.prefix, "model.language_model.")
        self.assertTrue(self.weight._has_stacked_ckpt)

    def test_every_requested_key_exists_in_the_checkpoint(self):
        missing = sorted(self.requested - self.real_keys)

        self.assertEqual(
            missing, [], f"mapping asks for absent tensors: {missing[:10]}"
        )

    def test_unmapped_checkpoint_tensors_are_only_the_known_gaps(self):
        unmapped = self.real_keys - self.requested
        unexpected = sorted(
            key
            for key in unmapped
            if not any(_collapse(key).startswith(p) for p in _EXPECTED_UNMAPPED)
        )

        self.assertEqual(
            unexpected, [], f"checkpoint tensors nothing claims: {unexpected[:10]}"
        )

    def test_all_forty_eight_layers_are_covered(self):
        for layer_id in range(48):
            prefix = f"model.language_model.layers.{layer_id}."
            self.assertTrue(
                any(key.startswith(prefix) for key in self.requested),
                f"layer {layer_id} has no mapped tensors",
            )

    def test_ple_and_indexer_tensors_are_claimed(self):
        """The opt-in PLE descriptor and QSA indexer claim all their ckpt keys."""
        ple_suffixes = (
            "ple.conv1d.weight",
            "ple.key_proj.weight",
            "ple.value_proj.weight",
            "ple.norm_conv.weight",
            "ple.norm_key.weight",
            "ple.norm_query.weight",
            "ple.ple_embedding.layer_multipliers",
            "ple.ple_embedding.ngram_heads_offsets",
            "ple.ple_embedding.ngram_heads_vocab_sizes",
            "ple.ple_embedding.ngram_embedding.shard_0.weight",
            "ple.ple_embedding.ngram_embedding.shard_127.weight",
        )
        for suffix in ple_suffixes:
            key = f"model.language_model.layers.1.{suffix}"
            self.assertIn(key, self.requested, f"{key} not mapped")

        for layer_id in (3, 47):  # first and last full-attention layers
            for suffix in (
                "self_attn.indexer.index_qk_proj.weight",
                "self_attn.indexer.q_layernorm.weight",
                "self_attn.indexer.k_layernorm.weight",
            ):
                key = f"model.language_model.layers.{layer_id}.{suffix}"
                self.assertIn(key, self.requested, f"{key} not mapped")
        # Linear-attention layers have no indexer.
        linear_key = (
            "model.language_model.layers.2.self_attn.indexer.index_qk_proj.weight"
        )
        self.assertNotIn(linear_key, self.requested)


if __name__ == "__main__":
    unittest.main()
