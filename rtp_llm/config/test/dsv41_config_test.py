import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models.multimodal.deepseek_v41_processor import V41ImageProcessorConfig


class V41ConfigTest(TestCase):
    def setUp(self):
        self.source = {
            "text_config": {
                "hidden_size": 64,
                "vocab_size": 256,
                "max_position_embeddings": 512,
                "num_hidden_layers": 2,
                "num_nextn_predict_layers": 1,
                "hc_mult": 2,
                "compress_ratios": [0, 1, 0],
                "kv_source_layer_ids": [1],
                "index_source_layer_ids": [1],
                "engram_layer_ids": [0],
                "dspark_target_layer_ids": [1],
                "dspark_block_size": 2,
            },
            "vision_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "patch_size": 8,
                "downsample_ratio": 2,
                "rope_theta": 10000.0,
                "max_image_tokens": 64,
                "min_pixels": 256,
            },
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [32, 32],
            },
            "dtype": "bfloat16",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "image_token_id": 255,
        }

    def test_parses_checkpoint_metadata_without_fixed_release_constraints(self):
        config = V41Config.from_dict(self.source)
        self.assertEqual(config.text, self.source["text_config"])
        self.assertEqual(config.vision, self.source["vision_config"])
        self.assertEqual(config.quantization, self.source["quantization_config"])
        self.assertEqual(config.dtype, "bfloat16")
        self.assertEqual(
            (
                config.bos_token_id,
                config.eos_token_id,
                config.pad_token_id,
                config.image_token_id,
            ),
            (0, 1, 2, 255),
        )
        self.assertEqual(
            config.vision_parameters(),
            {
                "vision_dim": 32,
                "vision_inter_dim": 64,
                "vision_n_heads": 4,
                "vision_n_layers": 1,
                "vision_patch_size": 8,
                "vision_downsample_ratio": 2,
                "vision_rope_theta": 10000.0,
                "hidden_size": 64,
            },
        )
        self.assertEqual(
            V41ImageProcessorConfig.from_model_config(config),
            V41ImageProcessorConfig(
                vision_patch_size=8,
                vision_downsample_ratio=2,
                vision_max_n_token=64,
                vision_min_pixels=256,
                image_token_id=255,
                vocab_size=256,
                max_seq_len=512,
            ),
        )

    def test_metadata_is_copied_independently_of_source(self):
        config = V41Config.from_dict(self.source)
        self.source["text_config"]["kv_source_layer_ids"].append(2)
        self.source["vision_config"]["patch_size"] = 16
        self.source["quantization_config"]["weight_block_size"][0] = 64
        self.assertEqual(config.text["kv_source_layer_ids"], [1])
        self.assertEqual(config.vision["patch_size"], 8)
        self.assertEqual(config.quantization["weight_block_size"], [32, 32])

    def test_from_path_reads_checkpoint_json(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(
                json.dumps(self.source), encoding="utf-8"
            )
            self.assertEqual(
                V41Config.from_path(directory), V41Config.from_dict(self.source)
            )


if __name__ == "__main__":
    main()
