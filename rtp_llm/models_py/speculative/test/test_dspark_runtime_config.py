import pickle
import unittest
from types import SimpleNamespace

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops import SpeculativeExecutionConfig, SpeculativeType


class DSparkRuntimeConfigTest(unittest.TestCase):
    def test_cpp_config_string_and_pickle_round_trip(self):
        config = SpeculativeExecutionConfig()
        config.type = "dspark"
        config.sp_dspark_mask_token_id = 17
        config.sp_dspark_sample_from_anchor = False

        restored = pickle.loads(pickle.dumps(config))

        self.assertEqual(restored.type, SpeculativeType.DSPARK)
        self.assertEqual(restored.sp_dspark_mask_token_id, 17)
        self.assertFalse(restored.sp_dspark_sample_from_anchor)
        self.assertIn("type: dspark", restored.to_string())

    def test_python_model_config_defaults_to_anchor_sampling(self):
        config = ModelConfig()
        self.assertTrue(config.dspark_sample_from_anchor)
        self.assertIsNone(config.dspark_noise_token_id)
        self.assertIsNone(config.dspark_target_layer_ids)
        self.assertIsNone(config.dspark_markov_rank)

    def test_factory_uses_input_vocab_and_sets_minimax_capture(self):
        sp_config = SpeculativeExecutionConfig()
        sp_config.type = SpeculativeType.DSPARK
        sp_config.gen_num_per_cycle = 3
        target = SimpleNamespace(num_layers=4, model_type="minimax_m3")
        draft = SimpleNamespace(
            dspark_noise_token_id=7,
            dspark_target_layer_ids=[1, 3],
            dspark_markov_rank=2,
            dspark_sample_from_anchor=False,
            input_vocab_size=10,
            vocab_size=4,
        )

        ModelFactory._setup_dspark_configs(sp_config, target, draft)

        self.assertEqual(sp_config.sp_dspark_mask_token_id, 7)
        self.assertFalse(sp_config.sp_dspark_sample_from_anchor)
        self.assertEqual(target.capture_aux_hidden_layer_ids, [1, 3])
        self.assertEqual(target._minimax_m3_target_hidden_state_layer_ids, (1, 3))
        self.assertEqual(target.hc_mult, 2)

    def test_factory_rejects_unordered_or_duplicate_target_layers(self):
        sp_config = SpeculativeExecutionConfig()
        sp_config.gen_num_per_cycle = 3
        target = SimpleNamespace(num_layers=4, model_type="minimax_m3")
        draft = SimpleNamespace(
            dspark_noise_token_id=1,
            dspark_target_layer_ids=[2, 1, 2],
            dspark_markov_rank=2,
            dspark_sample_from_anchor=True,
            input_vocab_size=4,
            vocab_size=4,
        )
        with self.assertRaisesRegex(ValueError, "unique and ordered"):
            ModelFactory._setup_dspark_configs(sp_config, target, draft)


if __name__ == "__main__":
    unittest.main()
