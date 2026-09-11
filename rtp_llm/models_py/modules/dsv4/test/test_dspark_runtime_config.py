import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops import SpeculativeType


class DSparkRuntimeConfigTest(unittest.TestCase):
    @staticmethod
    def _configs(gamma: int):
        sp_config = SimpleNamespace(
            gen_num_per_cycle=gamma,
            sp_dspark_mask_token_id=-1,
        )
        target_config = SimpleNamespace(
            num_layers=43,
            capture_aux_hidden_layer_ids=None,
        )
        # Deliberately has no dspark_block_size: proposal width must come only
        # from sp_config.gen_num_per_cycle.
        draft_config = SimpleNamespace(
            dspark_noise_token_id=128799,
            dspark_target_layer_ids=[40, 41, 42],
            dspark_markov_rank=256,
            vocab_size=129280,
        )
        return sp_config, target_config, draft_config

    def test_gen_num_per_cycle_three_is_accepted_without_checkpoint_width(self):
        sp_config, target_config, draft_config = self._configs(gamma=3)

        ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

        self.assertEqual(sp_config.sp_dspark_mask_token_id, 128799)
        self.assertEqual(target_config.capture_aux_hidden_layer_ids, [40, 41, 42])

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
