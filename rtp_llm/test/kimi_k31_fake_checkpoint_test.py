"""Check fake fixture provenance and coverage before writing large checkpoints."""

import copy
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.kimi_k31_config_test import text_config
from rtp_llm.utils.kimi_k31_fake_checkpoint import fixture_config, target_tensors, _write_shards
from rtp_llm.utils.kimi_k3_mtp_checkpoint import expected_tensors


class KimiK31FakeCheckpointTest(unittest.TestCase):
    def test_parallel_generation_preserves_serial_checkpoint_bytes(self):
        groups = [
            [("layer.norm.weight", {"shape": [128], "dtype": "BF16"}, 256)],
            [("layer.w.weight", {"shape": [32, 128], "dtype": "BF16"}, 8192)],
            [("layer.w.weight_packed", {"shape": [32, 64], "dtype": "U8"}, 2048),
             ("layer.w.weight_scale", {"shape": [32, 4], "dtype": "U8"}, 128)],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            serial, parallel = root / "serial", root / "parallel"
            serial.mkdir()
            parallel.mkdir()
            a = sorted(_write_shards(serial, "fake_target", groups, 3101, 1))
            b = sorted(_write_shards(parallel, "fake_target", groups, 3101, 3))
            self.assertEqual(a, b)
            for name, _, _ in a:
                self.assertEqual((serial / name).read_bytes(), (parallel / name).read_bytes())
            before = {p.name: p.read_bytes() for p in parallel.iterdir()}
            with self.assertRaises(FileExistsError):
                list(_write_shards(parallel, "fake_target", groups, 3101, 3))
            self.assertEqual(before, {p.name: p.read_bytes() for p in parallel.iterdir()})

    def source(self):
        text = text_config()
        text["linear_attn_config"]["kda_layers"] = [
            i for i in range(1, 94) if i not in text["dsa_config"]["dsa_layers"]
        ]
        text.update(hidden_size=7168, vocab_size=163840, num_attention_heads=96,
                    q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128,
                    qk_rope_head_dim=64, v_head_dim=128,
                    routed_expert_hidden_size=3584, moe_intermediate_size=3072,
                    num_shared_experts=2, num_experts=896, first_k_dense_replace=1)
        return {"model_type": "kimi_k31", "text_config": text}

    def test_four_layer_and_synthetic_draft_are_distinct(self):
        source = self.source()
        original = copy.deepcopy(source)
        target = fixture_config(source, 4, 3101)
        draft = fixture_config(source, 4, 3101, mtp=True)
        self.assertEqual(source, original)
        self.assertEqual(target["text_config"]["dsa_config"]["dsa_layers"], [4])
        self.assertEqual(target["text_config"]["num_nextn_predict_layers"], 0)
        self.assertEqual(draft["text_config"]["dsa_config"]["dsa_layers"], [4, 5])
        self.assertEqual(draft["rtp_llm_fixture"]["kind"], "synthetic_dsa_mtp")
        self.assertFalse(draft["rtp_llm_fixture"]["native_k31_mtp"])
        tensors = expected_tensors(draft["text_config"])
        prefix = "language_model.model.layers.4."
        self.assertTrue(all(name.startswith(prefix) for name in tensors))
        self.assertEqual(tensors[prefix + "self_attn.indexer.wq_b.weight"],
                         ((4096, 1536), "BF16"))
        self.assertEqual(tensors[prefix + "block_sparse_moe.experts.895.w1.weight_packed"],
                         ((3072, 1792), "U8"))

    def test_full_target_preserves_consecutive_final_dsa_layers(self):
        target = fixture_config(self.source(), 93, 3101)
        linear = target["text_config"]["linear_attn_config"]
        self.assertEqual(len(linear["kda_layers"]), 69)
        self.assertEqual(len(linear["full_attn_layers"]), 24)
        self.assertEqual(linear["full_attn_layers"][-2:], [92, 93])

    def test_templates_expand_all_experts_without_duplicate_names(self):
        config = fixture_config(self.source(), 4, 3101)
        tensor = {"shape": [2, 2], "dtype": "BF16"}
        suffixes = {
            0: ["self_attn.q_proj.weight", "input_layernorm.weight", "mlp.gate_proj.weight"],
            1: ["block_sparse_moe.gate.weight", "block_sparse_moe.experts.0.w1.weight_packed"],
            3: ["self_attn.q_a_proj.weight", "self_attn.indexer.wk.weight", "input_layernorm.weight"],
        }
        templates = {"tensors": {"language_model.model.norm.weight": tensor}}
        for layer, names in suffixes.items():
            for suffix in names:
                templates["tensors"][f"language_model.model.layers.{layer}.{suffix}"] = tensor
        entries = list(target_tensors(config, templates))
        names = [name for name, _ in entries]
        self.assertEqual(len(names), len(set(names)))
        for layer in (1, 2, 3):
            self.assertIn(f"language_model.model.layers.{layer}.block_sparse_moe.experts.895.w1.weight_packed", names)
        self.assertNotIn("language_model.model.layers.0.block_sparse_moe.gate.weight", names)
        self.assertIn("language_model.model.layers.3.self_attn.indexer.wk.weight", names)
        self.assertNotIn("language_model.model.layers.2.self_attn.indexer.wk.weight", names)


if __name__ == "__main__":
    unittest.main()
