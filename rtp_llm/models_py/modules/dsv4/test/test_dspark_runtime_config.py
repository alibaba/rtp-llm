import unittest
from types import SimpleNamespace

from rtp_llm.model_factory import ModelFactory


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

        ModelFactory._setup_dspark_configs(
            sp_config, target_config, draft_config
        )

        self.assertEqual(sp_config.sp_dspark_mask_token_id, 128799)
        self.assertEqual(sp_config.draft_sample_method, "greedy")
        self.assertEqual(draft_config.dspark_draft_sample_method, "greedy")
        self.assertEqual(target_config.capture_aux_hidden_layer_ids, [40, 41, 42])

    def test_probabilistic_draft_sample_method_is_wired_to_draft_model(self):
        sp_config, target_config, draft_config = self._configs(gamma=3)
        sp_config.draft_sample_method = "Probabilistic"

        ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

        self.assertEqual(sp_config.draft_sample_method, "probabilistic")
        self.assertEqual(draft_config.dspark_draft_sample_method, "probabilistic")

    def test_rejects_unknown_draft_sample_method(self):
        sp_config, target_config, draft_config = self._configs(gamma=3)
        sp_config.draft_sample_method = "invalid"

        with self.assertRaisesRegex(ValueError, "draft_sample_method"):
            ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

    def test_gen_num_per_cycle_must_be_positive(self):
        sp_config, target_config, draft_config = self._configs(gamma=0)

        with self.assertRaisesRegex(ValueError, "positive gen_num_per_cycle"):
            ModelFactory._setup_dspark_configs(
                sp_config, target_config, draft_config
            )


if __name__ == "__main__":
    unittest.main()
