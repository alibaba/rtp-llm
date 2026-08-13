import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.model_factory_register import ModelDict, ensure_model_registered
from rtp_llm.models.qwen_3_dspark import Qwen3DSpark
from rtp_llm.models_py.model_desc.qwen3_dspark_model import Qwen3DSparkModel


class Qwen3DSparkRegistrationTest(unittest.TestCase):
    def test_commit_uses_kernel_block_ids_not_physical_cache_ids(self) -> None:
        kernel_ids = torch.tensor([[3, 4]], dtype=torch.int32)
        physical_ids = torch.tensor([[3003, 3004]], dtype=torch.int32)
        attention = SimpleNamespace(
            kv_cache_kernel_block_id_device=kernel_ids,
            kv_cache_kernel_block_id=torch.tensor([[5, 6]], dtype=torch.int32),
            kv_cache_block_id_device=physical_ids,
            kv_cache_block_id=torch.tensor([[5005, 5006]], dtype=torch.int32),
        )
        model = SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=torch.empty(0)), kv_cache=None
        )
        model._draft_attention_inputs = lambda inputs: inputs.attention_inputs

        selected = Qwen3DSparkModel._block_table(
            model, SimpleNamespace(attention_inputs=attention)
        )

        self.assertIs(selected, kernel_ids)

    def test_config_wires_shared_dspark_contract(self) -> None:
        raw = {
            "architectures": ["Qwen3DSparkForCausalLM"],
            "transformer_layer_config": {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "vocab_size": 1024,
            },
            "draft_vocab_size": 1001,
            "aux_hidden_state_layer_ids": [0, 1],
            "mask_token_id": 1000,
            "block_size": 8,
            "markov_rank": 32,
        }
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            config = Qwen3DSpark._create_config(path)

        self.assertFalse(config.attn_config.is_causal)
        self.assertTrue(config.dspark_sample_from_anchor)
        self.assertEqual(config.dspark_noise_token_id, 1000)
        self.assertEqual(config.dspark_target_layer_ids, [0, 1])
        self.assertEqual(config.dspark_markov_rank, 32)
        self.assertEqual(config.input_vocab_size, 1024)
        self.assertEqual(config.vocab_size, 1001)
        self.assertTrue(config.qk_norm)
        self.assertTrue(ensure_model_registered("qwen_3_dspark"))
        self.assertEqual(
            ModelDict.get_ft_model_type_by_config(raw), "qwen_3_dspark"
        )

    def test_speculators_checkpoint_uses_bonus_anchor_layout(self) -> None:
        raw = {
            "architectures": ["DSparkDraftModel"],
            "speculators_model_type": "dspark",
            "transformer_layer_config": {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "vocab_size": 1024,
            },
            "draft_vocab_size": 1001,
            "aux_hidden_state_layer_ids": [1, 2],
            "mask_token_id": 1000,
            "markov_rank": 32,
        }
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            config = Qwen3DSpark._create_config(path)

        self.assertFalse(config.dspark_sample_from_anchor)
        self.assertEqual(config.dspark_target_layer_ids, [0, 1])


if __name__ == "__main__":
    unittest.main()
