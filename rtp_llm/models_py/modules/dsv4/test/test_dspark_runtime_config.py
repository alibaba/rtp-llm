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

        ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

        self.assertEqual(sp_config.sp_dspark_mask_token_id, 128799)

    def test_gen_num_per_cycle_must_be_positive(self):
        sp_config, target_config, draft_config = self._configs(gamma=0)

        with self.assertRaisesRegex(ValueError, "positive gen_num_per_cycle"):
            ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)


class TargetAuxHiddenCaptureConfigTest(unittest.TestCase):
    @staticmethod
    def _configs():
        return (
            SimpleNamespace(num_layers=43, capture_aux_hidden_layer_ids=None),
            SimpleNamespace(capture_aux_hidden_layer_ids=None),
        )

    def test_capture_layers_are_wired_to_target_and_draft(self):
        target_config, draft_config = self._configs()

        ModelFactory._configure_target_aux_hidden_capture(
            target_config, draft_config, (40, 41, 42)
        )

        self.assertEqual(target_config.capture_aux_hidden_layer_ids, [40, 41, 42])
        self.assertEqual(draft_config.capture_aux_hidden_layer_ids, [40, 41, 42])

    def test_capture_layers_must_be_distinct(self):
        target_config, draft_config = self._configs()

        with self.assertRaisesRegex(ValueError, "must be distinct"):
            ModelFactory._configure_target_aux_hidden_capture(
                target_config, draft_config, (40, 40, 42)
            )

    def test_capture_layers_must_be_in_target_range(self):
        target_config, draft_config = self._configs()

        with self.assertRaisesRegex(ValueError, "out of range"):
            ModelFactory._configure_target_aux_hidden_capture(
                target_config, draft_config, (40, 41, 43)
            )


if __name__ == "__main__":
    unittest.main()
