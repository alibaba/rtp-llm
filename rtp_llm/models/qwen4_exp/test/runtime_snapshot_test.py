"""Configuration ownership across final overrides and target/draft spawn."""

import json
import pickle
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from rtp_llm.config.qwen4_exp_config import Qwen4ModelConfig as ModelConfig
from rtp_llm.config.qwen4_exp_config import freeze_runtime_config
from rtp_llm.ops import HiddenStateCaptureDtype


class RuntimeSnapshotTest(unittest.TestCase):
    def config(self):
        config = ModelConfig()
        config.hidden_size = 2560
        config.hc_mult = 4
        config.is_mtp = False
        config.quant_config = None
        config._qwen4_ple_layer_ids = [1, 5]
        config._qwen4_split_ngram_parts = 128
        config._qwen4_indexer_head_dim = 128
        config._qwen4_indexer_compress_ratio = 4
        config._qwen4_indexer_budget = 1024
        config._qwen4_indexer_kv_heads = 1
        return config

    def test_final_override_snapshot_survives_pickle_and_is_immutable(self):
        config = self.config()
        config.qwen4_ple_prefetch_mb = 512
        snapshot = freeze_runtime_config(
            config, SimpleNamespace(get_attn_tp_size=lambda: 2)
        )
        spawned = pickle.loads(pickle.dumps(snapshot))
        self.assertEqual(spawned, snapshot)
        self.assertEqual(spawned.options.ple_prefetch_mb, 512)
        with self.assertRaises(FrozenInstanceError):
            spawned.options.ple_prefetch_mb = 128
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            config.qwen4_ple_prefetch_mb = 128

    def test_frozen_budget_override_leaves_capture_settings_unchanged(self):
        config = self.config()
        config.num_layers = 8
        config.hidden_state_capture_layer_ids = [7]
        freeze_runtime_config(config, SimpleNamespace(get_attn_tp_size=lambda: 2))
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            config.apply_override_args(
                '{"hidden_state_capture_layer_ids":[1,3],'
                '"hidden_state_capture_dtype":"fp8_e4m3",'
                '"qwen4_ple_prefetch_mb":64}'
            )
        self.assertEqual(config.hidden_state_capture_layer_ids, [7])
        self.assertEqual(
            config.hidden_state_capture_dtype, HiddenStateCaptureDtype.BF16
        )
        self.assertEqual(config.qwen4_ple_prefetch_mb, 256)

    def test_invalid_ple_budget_leaves_other_overrides_unchanged(self):
        config = self.config()
        config.num_layers = 8
        config.hidden_state_capture_layer_ids = [7]
        original_rope_style = config.attn_config.rope_config.style
        for budget in (None, True, 0, -1, 1.5, "256"):
            with (
                self.subTest(budget=budget),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                config.apply_override_args(
                    json.dumps(
                        {
                            "hidden_state_capture_layer_ids": [1, 3],
                            "hidden_state_capture_dtype": "fp8_e4m3",
                            "rope_scaling": {"type": "base"},
                            "qwen4_ple_prefetch_mb": budget,
                        }
                    )
                )
            self.assertEqual(config.hidden_state_capture_layer_ids, [7])
            self.assertEqual(
                config.hidden_state_capture_dtype, HiddenStateCaptureDtype.BF16
            )
            self.assertEqual(config.attn_config.rope_config.style, original_rope_style)
            self.assertEqual(config.qwen4_ple_prefetch_mb, 256)

    def test_invalid_capture_override_leaves_ple_budget_unchanged(self):
        config = self.config()
        config.num_layers = 8
        with self.assertRaisesRegex(ValueError, "hidden_state_capture_dtype"):
            config.apply_override_args(
                '{"hidden_state_capture_dtype":"float16","qwen4_ple_prefetch_mb":64}'
            )
        self.assertEqual(config.qwen4_ple_prefetch_mb, 256)

    def test_draft_snapshot_has_independent_geometry_and_role(self):
        target = self.config()
        draft = self.config()
        draft.is_mtp = True
        draft._qwen4_ple_layer_ids = []
        parallelism = SimpleNamespace(get_attn_tp_size=lambda: 2)
        first = freeze_runtime_config(target, parallelism)
        second = freeze_runtime_config(draft, parallelism)
        self.assertEqual(first.geometry.ple_layer_ids, (1, 5))
        self.assertEqual(second.geometry.ple_layer_ids, ())
        self.assertFalse(first.is_mtp_draft)
        self.assertTrue(second.is_mtp_draft)

    def test_changed_effective_topology_cannot_reuse_frozen_snapshot(self):
        config = self.config()
        freeze_runtime_config(config, SimpleNamespace(get_attn_tp_size=lambda: 2))
        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
            freeze_runtime_config(config, SimpleNamespace(get_attn_tp_size=lambda: 4))

    def test_target_and_draft_precision_survives_spawn_and_cannot_change_after_freeze(
        self,
    ):
        parallelism = SimpleNamespace(get_attn_tp_size=lambda: 2)
        for field, value in (
            ("_qwen4_dense_storage", "source_bf16"),
            ("_qwen4_vocab_storage", "source_bf16"),
            ("_qwen4_mtp_dense_storage_mode", "source_bf16_shared_vocab"),
        ):
            with self.subTest(field=field):
                config = self.config()
                setattr(config, field, value)
                snapshot = freeze_runtime_config(config, parallelism)
                self.assertEqual(snapshot, pickle.loads(pickle.dumps(snapshot)))
                setattr(
                    config,
                    field,
                    "fp8" if field != "_qwen4_mtp_dense_storage_mode" else "inherit",
                )
                with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                    freeze_runtime_config(config, parallelism)

    def test_budget_setter_rejects_boolean_values(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.config().qwen4_ple_prefetch_mb = True


if __name__ == "__main__":
    unittest.main()
