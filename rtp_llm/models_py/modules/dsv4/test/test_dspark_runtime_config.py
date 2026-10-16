import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.config.maxpander_config import MaxPanderConfig, MaxPanderRuntimeConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops import EPLBConfig, SpeculativeType


class DSparkRuntimeConfigTest(unittest.TestCase):
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
            maxpander_config=MaxPanderRuntimeConfig(),
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

    def test_maxpander_draft_uses_prefill_snapshot_and_ratio_only(self):
        for role in ("PREFILL", "DECODE"):
            sp_config, target_config, draft_config = self._configs(gamma=3)
            sp_config.type = SpeculativeType.DSPARK
            sp_config.checkpoint_path = "/unused/checkpoint"
            sp_config.model_type = "deepseek_v4_dspark"
            sp_config.quantization = None
            target_config.max_seq_len = 4096
            target_config.maxpander_config = MaxPanderConfig(
                enable=True, seed=17
            ).resolve(role)
            target_config.expert_num = 256
            target_config.eplb_config = EPLBConfig()
            target_config.eplb_config.redundant_expert = 64 if role == "PREFILL" else 0
            draft_config.expert_num = 128
            draft_config.eplb_config = EPLBConfig()
            draft_config.phy2log_path = ""
            draft_config.tokenizer_path = ""
            draft_config.ckpt_path = ""
            model_cls = Mock()
            model_cls._create_config.return_value = draft_config
            engine_config = SimpleNamespace(
                sp_config=sp_config,
                kv_cache_config=SimpleNamespace(),
                profiling_debug_logging_config=SimpleNamespace(),
                parallelism_config=SimpleNamespace(ep_size=2),
            )
            model_args = SimpleNamespace(
                tokenizer_path="/unused/tokenizer",
                act_type=None,
                mla_ops_type=None,
                enable_fp32_lm_head=False,
            )
            with patch.dict(
                os.environ, {"MAXPANDER_ENABLE": "0" if role == "PREFILL" else "1"}
            ), patch.object(
                ModelFactory, "get_model_cls", return_value=model_cls
            ), patch(
                "rtp_llm.model_factory.build_model_config"
            ), patch.object(
                ModelFactory, "_setup_dspark_configs"
            ):
                result = ModelFactory.create_propose_model_config(
                    engine_config, target_config, model_args
                )
            self.assertIs(result.maxpander_config, target_config.maxpander_config)
            self.assertEqual(result.maxpander_config.enabled, role == "PREFILL")
            self.assertEqual(
                result.eplb_config.redundant_expert, 32 if role == "PREFILL" else 0
            )
            self.assertEqual(
                target_config.eplb_config.redundant_expert,
                64 if role == "PREFILL" else 0,
            )


if __name__ == "__main__":
    unittest.main()
