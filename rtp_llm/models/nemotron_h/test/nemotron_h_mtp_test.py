import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.models.nemotron_h.nemotron_h import NemotronHBase
from rtp_llm.models.nemotron_h.nemotron_h_mtp import NemotronHMTP, NemotronHMTPWeight
from rtp_llm.model_factory_register import ModelDict
from rtp_llm.ops import HybridAttentionType, KVCacheSpecType

from nemotron_h_test import _config_json


class NemotronHMTPConfigTest(unittest.TestCase):
    def _make_mtp_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_text(json.dumps(_config_json()))
            config = NemotronHMTP.create_config(temp_dir)
        return config

    def test_architecture_resolution(self):
        self.assertEqual(
            ModelDict.get_ft_model_type_by_config(
                {"architectures": ["NemotronHForCausalLMNextN"]}
            ),
            "nemotron_h_mtp",
        )

    def test_mtp_config(self):
        config = self._make_mtp_config()
        self.assertEqual(config.num_layers, 2)
        self.assertTrue(config.is_mtp)
        self.assertEqual(
            list(config.hybrid_attention_config.hybrid_attention_types),
            [HybridAttentionType.NONE, HybridAttentionType.NONE],
        )
        self.assertEqual(config.moe_layer_index, [1])
        self.assertEqual(config.moe_w1_layout, "up_only")

    def test_mtp_kv_cache_spec_descs(self):
        config = self._make_mtp_config()
        descs = config.kv_cache_spec_descs
        self.assertEqual(len(descs), 2)
        self.assertEqual(descs[0][0].cache_type, KVCacheSpecType.MHA)
        self.assertEqual(descs[0][0].tag, "full")
        self.assertEqual(len(descs[1]), 0)

    def test_weight_cls(self):
        self.assertIs(NemotronHMTP.get_weight_cls(), NemotronHMTPWeight)

    def test_mtp_block_types_follow_moe_index(self):
        config = self._make_mtp_config()
        # Layer 0 is attention (not in moe_layer_index), layer 1 is moe.
        from rtp_llm.models_py.model_desc.nemotron_h import _block_type

        self.assertEqual(_block_type(config, 0), "attention")
        self.assertEqual(_block_type(config, 1), "moe")


if __name__ == "__main__":
    unittest.main()
