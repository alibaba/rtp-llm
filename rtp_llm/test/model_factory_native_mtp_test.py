import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.py_config_modules import LoraConfig, PyEnvConfigs
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import ModelDict
from rtp_llm.ops import SpeculativeType


def _checkpoint_config():
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "language_model_only": True,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_hidden_layers": 4,
            "hidden_size": 2560,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "hc_count": 4,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4,
            "ngram_size": 3,
            "ngram_vocab_size_base": 20000000,
            "ple_conv_kernel_size": 4,
            "ple_embed_dim": 2560,
            "ple_layer_ids": [2],
            "split_ngram_parts": 128,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "mtp": {
                "hybrid": True,
                "layer_types": ["full_attention"],
                "mtp_use_hidden_state_from_layer": None,
                "num_hidden_layers": 1,
                "rope_theta": 10000000,
            },
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        },
    }


class ModelFactoryNativeMTPTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base_path = self._write_checkpoint("target")
        self.py_config = PyEnvConfigs()
        self.py_config.kv_cache_config.seq_size_per_block = 64
        self.engine_config = SimpleNamespace(
            sp_config=self.py_config.sp_config,
            kv_cache_config=self.py_config.kv_cache_config,
            profiling_debug_logging_config=self.py_config.profiling_debug_logging_config,
            runtime_config=self.py_config.runtime_config,
        )
        self.sp_config = self.engine_config.sp_config
        self.sp_config.type = SpeculativeType.MTP
        self.sp_config.model_type = "qwen4_exp_mtp"
        self.model_args = ModelArgs()
        self.model_args.ckpt_path = self.base_path
        self.model_args.tokenizer_path = self.base_path
        self.model_args.model_type = "qwen4_exp"
        self.model_args.act_type = "bf16"
        self.model_args.max_seq_len = 4096
        self.model_config = self._main_config()

    def _write_checkpoint(self, name, config=None):
        path = Path(self.directory.name, name)
        path.mkdir()
        (path / "config.json").write_text(
            json.dumps(_checkpoint_config() if config is None else config)
        )
        return str(path)

    def _main_config(self):
        return ModelFactory.create_model_config(
            model_args=self.model_args,
            lora_config=LoraConfig(),
            kv_cache_config=self.engine_config.kv_cache_config,
            profiling_debug_logging_config=self.engine_config.profiling_debug_logging_config,
        )

    def _draft_config(self):
        return ModelFactory.create_propose_model_config(
            self.engine_config, self.model_config, self.model_args
        )

    def _assert_independent_draft(self, draft, path):
        self.assertIsNot(draft, self.model_config)
        self.assertIsNot(draft.attn_config, self.model_config.attn_config)
        self.assertEqual(draft.ckpt_path, path)
        self.assertEqual(draft.model_type, "qwen4_exp_mtp")
        self.assertEqual(draft.num_layers, 1)
        self.assertTrue(draft.is_mtp)
        self.assertFalse(draft.enable_qwen4_ple)
        self.assertTrue(draft.enable_qwen4_qsa)
        self.assertFalse(draft.mm_model_config.is_multimodal)
        self.assertEqual(draft.max_seq_len, self.model_config.max_seq_len)
        self.assertEqual(len(draft.kv_cache_spec_descs), 1)

    def test_default_path_uses_resolved_target_without_mutation(self):
        self.assertEqual(self.sp_config.checkpoint_path, "")
        before = self.model_config.to_string()
        sp_before = self.sp_config.to_string()
        # The resolved ModelConfig, not an earlier ModelArgs path, owns the default.
        self.model_args.ckpt_path = "/unresolved-checkpoint"
        draft = self._draft_config()
        self._assert_independent_draft(draft, self.base_path)
        self.assertEqual(self.model_config.to_string(), before)
        self.assertEqual(self.sp_config.to_string(), sp_before)
        self.assertEqual(self.model_config.num_layers, 4)
        self.assertFalse(self.model_config.is_mtp)
        self.assertEqual(self.model_config._qwen4_ple_layer_ids, [2])

    def test_explicit_empty_path_and_target_alias(self):
        self.assertIs(
            ModelFactory.get_model_cls("qwen_exp"),
            ModelFactory.get_model_cls("qwen4_exp"),
        )
        self.assertEqual(
            ModelDict.get_ft_model_type_by_config(
                {"architectures": ["Qwen4ExpForConditionalGeneration"]}
            ),
            "qwen4_exp",
        )
        self.model_args.model_type = "qwen_exp"
        self.model_config = self._main_config()
        self.sp_config.checkpoint_path = ""
        self.assertEqual(self.model_config.model_type, "qwen_exp")
        before = self.model_config.to_string()
        self._assert_independent_draft(self._draft_config(), self.base_path)
        self.assertEqual(self.model_config.to_string(), before)

    def test_explicit_draft_path_has_priority_even_without_target_path(self):
        config = _checkpoint_config()
        config["text_config"]["mtp"]["rope_theta"] = 20000000
        explicit = self._write_checkpoint("explicit", config)
        self.sp_config.checkpoint_path = explicit
        for target_path in (self.base_path, ""):
            with self.subTest(target_path_present=bool(target_path)):
                self.model_config.ckpt_path = target_path
                before = self.model_config.to_string()
                draft = self._draft_config()
                self._assert_independent_draft(draft, explicit)
                self.assertEqual(draft.attn_config.rope_config.base, 20000000)
                self.assertEqual(self.model_config.to_string(), before)
                self.assertEqual(self.sp_config.checkpoint_path, explicit)

    def test_disabled_and_deterministic_need_no_draft_checkpoint(self):
        for sp_type in (SpeculativeType.NONE, SpeculativeType.DETERMINISTIC):
            with self.subTest(sp_type=sp_type):
                self.sp_config.type = sp_type
                self.sp_config.model_type = ""
                self.model_config.ckpt_path = ""
                self.assertIsNone(self._draft_config())
                with mock.patch("rtp_llm.model_factory.configure_warmup"):
                    proposal = ModelFactory.get_sp_model(
                        self.model_config, None, self.engine_config
                    )
                if sp_type == SpeculativeType.NONE:
                    self.assertIsNone(proposal)
                else:
                    self.assertEqual(proposal.sp_type, sp_type)
                    self.assertIsNone(proposal.model)

    def test_other_target_draft_or_speculation_type_does_not_inherit(self):
        cases = (
            (SpeculativeType.MTP, "qwen3_next", "qwen4_exp_mtp"),
            (SpeculativeType.MTP, "qwen4_exp", "qwen3_next_mtp"),
            (SpeculativeType.EAGLE, "qwen4_exp", "qwen4_exp_mtp"),
            (SpeculativeType.DSPARK, "qwen4_exp", "qwen4_exp_mtp"),
        )
        for sp_type, target_type, draft_type in cases:
            with self.subTest(sp_type=sp_type, target=target_type, draft=draft_type):
                self.sp_config.type = sp_type
                self.sp_config.model_type = draft_type
                self.model_config.model_type = target_type
                with self.assertRaisesRegex(ValueError, "SP_CHECKPOINT_PATH"):
                    self._draft_config()

    def test_missing_draft_type_is_not_inferred(self):
        self.sp_config.model_type = ""
        for path in ("", self.base_path):
            with self.subTest(path_present=bool(path)):
                self.sp_config.checkpoint_path = path
                with self.assertRaisesRegex(ValueError, "SP_MODEL_TYPE") as error:
                    self._draft_config()
                self.assertNotIn(self.base_path, str(error.exception))

    def test_missing_resolved_target_path_fails(self):
        self.model_config.ckpt_path = ""
        with self.assertRaisesRegex(ValueError, "SP_CHECKPOINT_PATH"):
            self._draft_config()

    def test_checkpoint_must_have_valid_native_mtp_config(self):
        for name in ("missing", "invalid"):
            with self.subTest(config=name):
                config = _checkpoint_config()
                if name == "missing":
                    del config["text_config"]["mtp"]
                else:
                    config["text_config"]["mtp"]["num_hidden_layers"] = 2
                self.model_config.ckpt_path = self._write_checkpoint(name, config)
                with self.assertRaisesRegex(ValueError, "MTP"):
                    self._draft_config()

    def test_missing_proposal_fails_before_main_weight_loading(self):
        for sp_type in (
            SpeculativeType.VANILLA,
            SpeculativeType.MTP,
            SpeculativeType.EAGLE3,
            SpeculativeType.EAGLE,
            SpeculativeType.DSPARK,
        ):
            with self.subTest(sp_type=sp_type):
                self.sp_config.type = sp_type
                before = self.model_config.to_string()
                with mock.patch.object(ModelFactory, "_create_model") as load:
                    with self.assertRaisesRegex(ValueError, "propose model config"):
                        ModelFactory.from_model_configs(
                            self.model_config, self.engine_config, world_info=None
                        )
                load.assert_not_called()
                self.assertEqual(self.model_config.to_string(), before)

    def test_get_sp_model_rejects_missing_learned_draft(self):
        with self.assertRaisesRegex(ValueError, "propose model config"):
            ModelFactory.get_sp_model(self.model_config, None, self.engine_config)


if __name__ == "__main__":
    unittest.main()
