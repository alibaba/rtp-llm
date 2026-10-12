import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.dsv41_config import V41Config, dsv41_pp_partition
from rtp_llm.model_factory_register import ModelDict, get_lazy_model_module_path


class V41ConfigTest(unittest.TestCase):
    def setUp(self):
        self.source = json.loads(
            (Path(__file__).parent / "testdata/deepseek_v41_flash.json").read_text()
        )

    def test_nested_metadata_and_copy_isolation(self):
        config = V41Config.from_dict(self.source)
        self.source["text_config"]["kv_source_layer_ids"].append(24)
        self.source["quantization_config"]["weight_block_size"][0] = 128
        self.assertEqual(config.text["kv_source_layer_ids"], [2, 8, 14, 20])
        self.assertEqual(config.quantization["weight_block_size"], [32, 32])
        text = config.runtime_text_config("/checkpoint")
        self.assertEqual(text["checkpoint_path"], "/checkpoint")
        self.assertEqual(text["dtype"], "bfloat16")
        text["kv_source_layer_ids"].clear()
        self.assertEqual(config.text["kv_source_layer_ids"], [2, 8, 14, 20])
        self.assertEqual(config.engram_host_bytes, 202758032400)

    def test_global_sources_survive_pp_boundary(self):
        bindings = V41Config.from_dict(self.source).layer_sources()
        self.assertEqual(len(bindings), 40)
        self.assertIsNone(bindings[1].kv_source_layer_id)
        self.assertTrue(bindings[20].owns_kv)
        self.assertEqual(bindings[21].kv_source_layer_id, 20)
        self.assertEqual(bindings[21].index_source_layer_id, 20)
        self.assertEqual(bindings[21].candidate_source_layer_id, 20)
        self.assertEqual(bindings[39].kv_source_layer_id, 20)
        self.assertEqual(bindings[39].index_source_layer_id, 36)
        self.assertEqual([b.layer_id for b in bindings if b.owns_kv], [2, 8, 14, 20])
        self.assertEqual(
            [b.layer_id for b in bindings if b.owns_index],
            [2, 8, 14, 20, 24, 28, 32, 36],
        )

    def test_invalid_source_schedules(self):
        cases = (
            ("kv_source_layer_ids", [8, 14, 20]),
            ("kv_source_layer_ids", [2, 8, 8, 20]),
            ("kv_source_layer_ids", [2, 8, 20, 14]),
            ("kv_source_layer_ids", [2, 8, 14, 40]),
            ("kv_source_layer_ids", [0, 2, 8, 14, 20]),
            ("kv_source_layer_ids", [2, 8, 14]),
            ("index_source_layer_ids", [2, 8, 14, 24, 28, 32, 36]),
            ("candidate_source_layer_id", 14),
            ("candidate_source_layer_id", 40),
            ("candidate_source_layer_id", -2),
            ("candidate_block_size", 0),
            ("compress_ratios", [0, 0] + [2] * 37 + [1] * 4),
            ("compress_ratios", [0] * 39),
            ("compress_ratios", [4] * 43),
            ("compress_ratios", [0, 0] + [2] * 17 + [1] * 21 + [0] * 3),
            ("engram_num_embeddings", [100]),
            ("num_experts_per_tok", 385),
            ("num_hash_layers", 1),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                source = json.loads(json.dumps(self.source))
                source["text_config"][key] = value
                with self.assertRaises(ValueError):
                    V41Config.from_dict(source)

    def test_quantization_recipe_validation(self):
        for key, value in (
            ("weight_block_size", [128, 128]),
            ("scale_fmt", "float32"),
            ("expert_dtype", "fp8"),
            ("activation_scheme", "static"),
        ):
            with self.subTest(key=key):
                source = json.loads(json.dumps(self.source))
                source["quantization_config"][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    V41Config.from_dict(source)

    def test_expert_ownership(self):
        config = V41Config.from_dict(self.source)
        for ep_size in (4, 8):
            owners = [
                list(config.local_expert_ids(ep_size, rank)) for rank in range(ep_size)
            ]
            self.assertEqual([e for owner in owners for e in owner], list(range(384)))
            self.assertTrue(all(len(owner) == 384 // ep_size for owner in owners))
        for size, rank in ((0, 0), (7, 0), (8, -1), (8, 8)):
            with self.assertRaises(ValueError):
                config.local_expert_ids(size, rank)

    def test_pp_partition_ignores_v4_override(self):
        with patch.dict(os.environ, {"DSV4_PP_STAGE_LAYER_COUNTS": "20,20"}):
            self.assertEqual(dsv41_pp_partition(40, 2), [21, 19])
            self.assertEqual(dsv41_pp_partition(40, 1), [40])
        for layers, stages in ((40, 4), (39, 2), (0, 1)):
            with self.assertRaises(ValueError):
                dsv41_pp_partition(layers, stages)

    def test_pp_boundary_rejects_incompatible_producers(self):
        model = SimpleNamespace(deepseek_v41_config=self.source["text_config"])
        self.assertEqual(dsv41_pp_partition(40, 2, model), [21, 19])
        model.deepseek_v41_config["engram_layer_ids"] = [1, 21]
        with self.assertRaisesRegex(ValueError, "all Engram layers"):
            dsv41_pp_partition(40, 2, model)
        model.deepseek_v41_config["engram_layer_ids"] = [1, 14]
        model.deepseek_v41_config["candidate_source_layer_id"] = 24
        with self.assertRaisesRegex(ValueError, "L20"):
            dsv41_pp_partition(40, 2, model)

    def test_from_path_and_lazy_model_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(self.source))
            self.assertEqual(
                V41Config.from_path(directory), V41Config.from_dict(self.source)
            )
        self.assertEqual(
            ModelDict.get_ft_model_type_by_config(self.source), "deepseek_v41"
        )
        self.assertEqual(
            get_lazy_model_module_path("deepseek_v41"), "rtp_llm.models.deepseek_v41"
        )
        self.assertEqual(
            ModelDict.get_ft_model_type_by_hf_architectures("DeepseekV4ForCausalLM"),
            "deepseek_v4",
        )


if __name__ == "__main__":
    unittest.main()
