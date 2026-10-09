import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.gemma4_assistant import Gemma4Assistant

CONFIG = {
    "architectures": ["Gemma4AssistantForCausalLM"],
    "audio_token_id": 258881,
    "backbone_hidden_size": 32,
    "boa_token_id": 256000,
    "boi_token_id": 255999,
    "centroid_intermediate_top_k": 2,
    "dtype": "bfloat16",
    "eoa_token_id": 258883,
    "eoi_token_id": 258882,
    "image_token_id": 258880,
    "model_type": "gemma4_assistant",
    "num_centroids": 8,
    "text_config": {
        "_name_or_path": "",
        "architectures": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_k_eq_v": True,
        "bos_token_id": 2,
        "chunk_size_feed_forward": 0,
        "dtype": "bfloat16",
        "enable_moe_block": False,
        "eos_token_id": 1,
        "final_logit_softcapping": None,
        "global_head_dim": 16,
        "head_dim": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 16,
        "hidden_size_per_layer_input": 0,
        "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
        "initializer_range": 0.02,
        "intermediate_size": 32,
        "is_encoder_decoder": False,
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "max_position_embeddings": 256,
        "model_type": "gemma4_text",
        "moe_intermediate_size": None,
        "num_attention_heads": 4,
        "num_experts": None,
        "num_global_key_value_heads": 1,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "num_kv_shared_layers": 4,
        "output_attentions": False,
        "output_hidden_states": False,
        "pad_token_id": 0,
        "problem_type": None,
        "return_dict": True,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
        "sliding_window": 32,
        "tie_word_embeddings": True,
        "top_k_experts": None,
        "use_bidirectional_attention": None,
        "use_cache": True,
        "use_double_wide_mlp": False,
        "vocab_size": 64,
        "vocab_size_per_layer_input": 0,
    },
    "tie_word_embeddings": True,
    "transformers_version": "5.7.0.dev0",
    "use_ordered_embeddings": False,
}


class Gemma4AssistantConfigTest(unittest.TestCase):
    def test_factory_and_complete_four_layer_plan(self):
        self.assertIs(ModelFactory.get_model_cls("gemma4_assistant"), Gemma4Assistant)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(CONFIG))
            Path(directory, "generation_config.json").write_text(
                json.dumps({"num_assistant_tokens": 6})
            )
            config = Gemma4Assistant.create_config(directory)
            Gemma4Assistant._post_build_model_config(config)
        self.assertEqual(config.num_layers, 4)
        self.assertTrue(config.shares_target_kv)
        self.assertEqual(config.gen_num_per_cycle, 6)
        self.assertEqual(
            [row[0].tag for row in config.kv_cache_spec_descs],
            ["swa", "swa", "swa", "full"],
        )
        self.assertEqual(
            config.mm_related_params.config["assistant_backbone_hidden_size"], 32
        )


if __name__ == "__main__":
    unittest.main()
