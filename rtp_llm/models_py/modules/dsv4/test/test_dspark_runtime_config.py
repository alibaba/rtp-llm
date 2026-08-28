import unittest
from types import SimpleNamespace

from rtp_llm.model_factory import ModelFactory


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

        ModelFactory._setup_dspark_configs(
            sp_config, target_config, draft_config
        )

        self.assertEqual(sp_config.sp_dspark_mask_token_id, 128799)
        self.assertTrue(sp_config.sp_dspark_sample_from_anchor)
        self.assertEqual(target_config.capture_aux_hidden_layer_ids, [40, 41, 42])

    def test_noise_token_uses_input_vocabulary_for_reduced_draft_vocab(self):
        sp_config, target_config, draft_config = self._configs(gamma=7)
        draft_config.vocab_size = 20_000

        ModelFactory._setup_dspark_configs(
            sp_config, target_config, draft_config
        )

        self.assertEqual(sp_config.sp_dspark_mask_token_id, 128799)

    def test_noise_token_outside_input_vocabulary_is_rejected(self):
        sp_config, target_config, draft_config = self._configs(gamma=7)
        draft_config.dspark_noise_token_id = draft_config.input_vocab_size

        with self.assertRaisesRegex(ValueError, "input_vocab_size"):
            ModelFactory._setup_dspark_configs(
                sp_config, target_config, draft_config
            )

    def test_gen_num_per_cycle_must_be_positive(self):
        sp_config, target_config, draft_config = self._configs(gamma=0)

        with self.assertRaisesRegex(ValueError, "positive gen_num_per_cycle"):
            ModelFactory._setup_dspark_configs(
                sp_config, target_config, draft_config
            )


if __name__ == "__main__":
    unittest.main()
