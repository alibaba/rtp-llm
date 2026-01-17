"""Unit tests for model configuration creators.

These tests verify that configuration creators produce the same results
as the original model class _create_config methods.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.bert import create_bert_config, create_roberta_config
from rtp_llm.model_config_creators.registry import (
    get_config_creator,
    list_registered_creators,
)
from rtp_llm.model_config_creators.tbstars2_5 import create_tbstars2_5_config


class TestConfigCreators(unittest.TestCase):
    """Test cases for configuration creators."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_config_json(self, config_data: dict) -> str:
        """Create a temporary config.json file with test data.

        Args:
            config_data: Dictionary containing config.json content

        Returns:
            Path to the created checkpoint directory
        """
        ckpt_path = os.path.join(self.temp_dir, "test_model")
        os.makedirs(ckpt_path, exist_ok=True)
        config_path = os.path.join(ckpt_path, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f)
        return ckpt_path

    def test_bert_config_creator(self):
        """Test BERT configuration creator."""
        config_data = {
            "num_attention_heads": 12,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "max_position_embeddings": 512,
            "vocab_size": 30522,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12,
            "intermediate_size": 3072,
            "torch_dtype": "float32",
        }
        ckpt_path = self._create_test_config_json(config_data)

        config = create_bert_config(ckpt_path)

        self.assertIsInstance(config, ModelConfig)
        self.assertEqual(config.ckpt_path, ckpt_path)
        self.assertEqual(config.activation_type, "gelu")
        self.assertEqual(config.norm_type, "layernorm")
        self.assertEqual(config.attn_config.head_num, 12)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_layers, 12)
        self.assertEqual(config.max_seq_len, 512)
        self.assertEqual(config.vocab_size, 30522)
        self.assertEqual(config.type_vocab_size, 2)
        self.assertEqual(config.inter_size, 3072)
        self.assertTrue(config.has_positional_encoding)
        self.assertTrue(config.has_pre_decoder_layernorm)
        self.assertFalse(config.attn_config.is_causal)

    def test_roberta_config_creator(self):
        """Test RoBERTa configuration creator."""
        config_data = {
            "num_attention_heads": 12,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "max_position_embeddings": 512,
            "vocab_size": 50265,
            "type_vocab_size": 1,
            "layer_norm_eps": 1e-5,
            "intermediate_size": 3072,
            "pad_token_id": 1,
            "torch_dtype": "float32",
        }
        ckpt_path = self._create_test_config_json(config_data)

        config = create_roberta_config(ckpt_path)

        self.assertIsInstance(config, ModelConfig)
        self.assertEqual(config.ckpt_path, ckpt_path)
        self.assertEqual(config.position_ids_style, 1)
        self.assertEqual(config.special_tokens.pad_token_id, 1)
        # Should inherit BERT settings
        self.assertEqual(config.activation_type, "gelu")
        self.assertEqual(config.norm_type, "layernorm")

    def test_tbstars2_5_config_creator(self):
        """Test TBStars2_5 configuration creator."""
        config_data = {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_hidden_layers": 32,
            "max_position_embeddings": 8192,
            "vocab_size": 100000,
            "num_routed_experts": 160,
            "num_shared_experts": 2,
            "num_experts_per_tok": 8,
            "first_k_dense_layers": 0,
            "intermediate_size": 5632,
            "q_lora_rank": 1536,
            "kv_lora_rank": 384,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "qk_layernorm": True,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "tie_word_embeddings": False,
        }
        ckpt_path = self._create_test_config_json(config_data)

        config = create_tbstars2_5_config(ckpt_path)

        self.assertIsInstance(config, ModelConfig)
        self.assertEqual(config.ckpt_path, ckpt_path)
        self.assertEqual(config.activation_type, "SiGLU")
        self.assertEqual(config.norm_type, "rmsnorm")
        self.assertEqual(config.attn_config.head_num, 16)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.num_layers, 32)
        self.assertEqual(config.max_seq_len, 8192)
        self.assertEqual(config.vocab_size, 100000)
        self.assertEqual(config.expert_num, 160)
        self.assertEqual(config.moe_k, 8)
        self.assertEqual(config.moe_style, 2)  # Has shared experts
        self.assertTrue(config.attn_config.use_mla)
        self.assertEqual(config.attn_config.q_lora_rank, 1536)
        self.assertEqual(config.attn_config.kv_lora_rank, 384)
        self.assertEqual(config.qk_norm, True)

    def test_registry_functionality(self):
        """Test configuration creator registry."""
        # Test that creators are registered
        bert_creator = get_config_creator("bert")
        self.assertIsNotNone(bert_creator)
        self.assertEqual(bert_creator, create_bert_config)

        roberta_creator = get_config_creator("roberta")
        self.assertIsNotNone(roberta_creator)
        self.assertEqual(roberta_creator, create_roberta_config)

        tbstars_creator = get_config_creator("tbstars2_5")
        self.assertIsNotNone(tbstars_creator)
        self.assertEqual(tbstars_creator, create_tbstars2_5_config)

        # Test listing registered creators
        registered = list_registered_creators()
        self.assertIn("bert", registered)
        self.assertIn("roberta", registered)
        self.assertIn("tbstars2_5", registered)

        # Test non-existent model type
        nonexistent = get_config_creator("nonexistent_model")
        self.assertIsNone(nonexistent)

    def test_config_creator_missing_config_json(self):
        """Test that config creators raise appropriate errors for missing config.json."""
        ckpt_path = os.path.join(self.temp_dir, "empty_model")
        os.makedirs(ckpt_path, exist_ok=True)

        with self.assertRaises(FileNotFoundError):
            create_bert_config(ckpt_path)


if __name__ == "__main__":
    unittest.main()
