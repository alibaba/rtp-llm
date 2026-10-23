import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops import SpeculativeType


class DSparkRuntimeConfigTest(unittest.TestCase):

    def test_confidence_defaults_off_and_is_v41_only(self):
        sp, target, draft = self._configs(3)
        ModelFactory._setup_dspark_configs(sp, target, draft)
        self.assertFalse(draft.dspark_confidence_enabled)
        sp.sp_dspark_confidence_mode = "observe"
        with self.assertRaisesRegex(ValueError, "only for DeepSeek V4.1"):
            ModelFactory._setup_dspark_configs(sp, target, draft)
        draft.is_deepseek_v41 = True
        with self.assertRaisesRegex(ValueError, "only for DeepSeek V4.1"):
            ModelFactory._setup_dspark_configs(sp, target, draft)
        target.is_deepseek_v41 = True
        ModelFactory._setup_dspark_configs(sp, target, draft)
        self.assertTrue(draft.dspark_confidence_enabled)
        draft.dspark_sample_from_anchor = False
        with self.assertRaisesRegex(ValueError, "requires sampling from anchor"):
            ModelFactory._setup_dspark_configs(sp, target, draft)

    def test_confidence_rejects_invalid_temperature_and_budget(self):
        sp, target, draft = self._configs(3)
        draft.is_deepseek_v41 = True
        target.is_deepseek_v41 = True
        sp.sp_dspark_confidence_mode = "observe"
        for temperature in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(temperature=temperature):
                sp.sp_dspark_confidence_temperature = temperature
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    ModelFactory._setup_dspark_configs(sp, target, draft)
        sp.sp_dspark_confidence_temperature = 0.7
        sp.sp_dspark_verify_budget = 1
        with self.assertRaisesRegex(ValueError, "requires bounded"):
            ModelFactory._setup_dspark_configs(sp, target, draft)
        sp.sp_dspark_confidence_mode = "bounded"
        for budget in (-1, 0, 1, 3):
            sp.sp_dspark_verify_budget = budget
            ModelFactory._setup_dspark_configs(sp, target, draft)
        for budget in (-2, 4):
            sp.sp_dspark_verify_budget = budget
            with self.assertRaisesRegex(ValueError, "between 0 and gamma"):
                ModelFactory._setup_dspark_configs(sp, target, draft)

    def test_confidence_config_pickle_roundtrip(self):
        import pickle

        from rtp_llm.ops import SpeculativeExecutionConfig

        config = SpeculativeExecutionConfig()
        config.sp_dspark_confidence_mode = "bounded"
        config.sp_dspark_confidence_temperature = 0.75
        config.sp_dspark_verify_budget = 2
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.sp_dspark_confidence_mode, "bounded")
        self.assertEqual(restored.sp_dspark_confidence_temperature, 0.75)
        self.assertEqual(restored.sp_dspark_verify_budget, 2)

    def test_invalid_verify_mode_is_rejected_before_model_loading(self):
        sp, target, draft = self._configs(3)
        sp.sp_dspark_verify_mode = "ragged"
        with self.assertRaisesRegex(ValueError, "invalid DSpark verify mode"):
            ModelFactory._setup_dspark_configs(sp, target, draft)

    def test_packed_requires_bounded_mode_and_explicit_budget(self):
        for mode, budget in (
            ("off", -1),
            ("off", 0),
            ("observe", -1),
            ("observe", 0),
            ("bounded", -1),
        ):
            with self.subTest(mode=mode, budget=budget):
                sp, target, draft = self._configs(3)
                target.is_deepseek_v41 = draft.is_deepseek_v41 = True
                sp.sp_dspark_verify_mode = "packed"
                sp.sp_dspark_confidence_mode = mode
                sp.sp_dspark_verify_budget = budget
                with self.assertRaisesRegex(ValueError, "requires bounded"):
                    ModelFactory._setup_dspark_configs(sp, target, draft)

    def test_packed_requires_v41_target_and_draft(self):
        for target_v41, draft_v41 in ((False, False), (False, True), (True, False)):
            with self.subTest(target_v41=target_v41, draft_v41=draft_v41):
                sp, target, draft = self._configs(3)
                target.is_deepseek_v41 = target_v41
                draft.is_deepseek_v41 = draft_v41
                sp.sp_dspark_verify_mode = "packed"
                sp.sp_dspark_confidence_mode = "bounded"
                sp.sp_dspark_verify_budget = 1
                with self.assertRaisesRegex(ValueError, "only for DeepSeek V4.1"):
                    ModelFactory._setup_dspark_configs(sp, target, draft)

    def test_packed_budget_zero_partial_and_full_width_are_valid(self):
        for budget in (0, 1, 3):
            with self.subTest(budget=budget):
                sp, target, draft = self._configs(3)
                target.is_deepseek_v41 = draft.is_deepseek_v41 = True
                sp.sp_dspark_verify_mode = "packed"
                sp.sp_dspark_confidence_mode = "bounded"
                sp.sp_dspark_verify_budget = budget
                ModelFactory._setup_dspark_configs(sp, target, draft)
                self.assertTrue(draft.dspark_confidence_enabled)
                self.assertEqual(sp.sp_dspark_verify_budget, budget)
                self.assertEqual(sp.gen_num_per_cycle, 3)
                self.assertEqual(target.capture_aux_hidden_layer_ids, [40, 41, 42])

    @staticmethod
    def _configs(gamma: int):
        sp_config = SimpleNamespace(
            gen_num_per_cycle=gamma,
            sp_dspark_mask_token_id=-1,
            sp_dspark_sample_from_anchor=False,
        )
        target_config = SimpleNamespace(
            num_layers=43,
            capture_aux_hidden_layer_ids=None,
        )
        draft_config = SimpleNamespace(
            dspark_noise_token_id=128799,
            dspark_target_layer_ids=[40, 41, 42],
            dspark_markov_rank=256,
            vocab_size=129280,
            input_vocab_size=129280,
        )
        return sp_config, target_config, draft_config

    def test_gen_num_per_cycle_three_is_accepted_without_checkpoint_width(self):
        sp_config, target_config, draft_config = self._configs(gamma=3)

        ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

        self.assertEqual(sp_config.sp_dspark_mask_token_id, 128799)
        self.assertTrue(sp_config.sp_dspark_sample_from_anchor)
        self.assertEqual(target_config.capture_aux_hidden_layer_ids, [40, 41, 42])

    def test_noise_token_uses_input_vocabulary_for_reduced_draft_vocab(self):
        sp_config, target_config, draft_config = self._configs(gamma=7)
        draft_config.vocab_size = 20_000

        ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

        self.assertEqual(sp_config.sp_dspark_mask_token_id, 128799)

    def test_zero_input_vocabulary_falls_back_to_model_vocabulary(self):
        sp_config, target_config, draft_config = self._configs(gamma=3)
        draft_config.input_vocab_size = 0

        ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

        self.assertEqual(sp_config.sp_dspark_mask_token_id, 128799)

    def test_noise_token_outside_input_vocabulary_is_rejected(self):
        sp_config, target_config, draft_config = self._configs(gamma=7)
        draft_config.dspark_noise_token_id = draft_config.input_vocab_size

        with self.assertRaisesRegex(ValueError, "input_vocab_size"):
            ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

    def test_gen_num_per_cycle_must_be_positive(self):
        sp_config, target_config, draft_config = self._configs(gamma=0)

        with self.assertRaisesRegex(ValueError, "positive gen_num_per_cycle"):
            ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

    def test_only_explicit_dspark_wires_dspark_runtime_config(self):
        for sp_type, model_type, should_setup_dspark in (
            (SpeculativeType.MTP, "deepseek_v4_mtp", False),
            (SpeculativeType.DSPARK, "deepseek_v4_dspark", True),
        ):
            with self.subTest(sp_type=sp_type, model_type=model_type):
                sp_config, target_config, draft_config = self._configs(gamma=3)
                sp_config.type = sp_type
                sp_config.checkpoint_path = "/unused/checkpoint"
                sp_config.model_type = model_type
                sp_config.quantization = None
                target_config.max_seq_len = 4096
                target_config.capture_aux_hidden_layer_ids = None
                draft_config.tokenizer_path = ""
                draft_config.ckpt_path = ""
                draft_config.max_seq_len = 0

                model_cls = Mock()
                model_cls._create_config.return_value = draft_config
                engine_config = SimpleNamespace(
                    sp_config=sp_config,
                    kv_cache_config=SimpleNamespace(),
                    profiling_debug_logging_config=SimpleNamespace(),
                )
                model_args = SimpleNamespace(
                    tokenizer_path="/unused/tokenizer",
                    act_type=None,
                    mla_ops_type=None,
                    enable_fp32_lm_head=False,
                )

                with patch.object(
                    ModelFactory, "get_model_cls", return_value=model_cls
                ), patch("rtp_llm.model_factory.build_model_config"), patch.object(
                    ModelFactory, "_setup_dspark_configs"
                ) as setup_dspark:
                    result = ModelFactory.create_propose_model_config(
                        engine_config, target_config, model_args
                    )

                self.assertIs(result, draft_config)
                self.assertEqual(result.max_seq_len, target_config.max_seq_len)
                if should_setup_dspark:
                    setup_dspark.assert_called_once_with(
                        sp_config, target_config, draft_config
                    )
                else:
                    setup_dspark.assert_not_called()
                    self.assertIsNone(target_config.capture_aux_hidden_layer_ids)


if __name__ == "__main__":
    unittest.main()
