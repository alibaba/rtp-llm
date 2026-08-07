import unittest
from types import SimpleNamespace

from rtp_llm.model_factory import ModelFactory


class DSparkRuntimeConfigTest(unittest.TestCase):
    @staticmethod
    def _configs(configured_gamma: int, checkpoint_gamma: int = 5):
        sp_config = SimpleNamespace(
            gen_num_per_cycle=configured_gamma,
            sp_dspark_mask_token_id=-1,
        )
        target_config = SimpleNamespace(
            num_layers=43,
            gen_num_per_cycle=configured_gamma,
            capture_aux_hidden_layer_ids=None,
        )
        draft_config = SimpleNamespace(
            dspark_block_size=checkpoint_gamma,
            dspark_noise_token_id=128799,
            dspark_target_layer_ids=[40, 41, 42],
            dspark_markov_rank=256,
            gen_num_per_cycle=configured_gamma,
            vocab_size=129280,
        )
        return sp_config, target_config, draft_config

    def test_checkpoint_gamma_overrides_user_configuration(self):
        sp_config, target_config, draft_config = self._configs(configured_gamma=3)

        ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

        self.assertEqual(sp_config.gen_num_per_cycle, 5)
        self.assertEqual(target_config.gen_num_per_cycle, 5)
        self.assertEqual(draft_config.gen_num_per_cycle, 5)
        self.assertEqual(sp_config.sp_dspark_mask_token_id, 128799)
        self.assertEqual(target_config.capture_aux_hidden_layer_ids, [40, 41, 42])

    def test_checkpoint_gamma_must_be_positive(self):
        sp_config, target_config, draft_config = self._configs(
            configured_gamma=3, checkpoint_gamma=0
        )

        with self.assertRaisesRegex(ValueError, "positive dspark_block_size"):
            ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

    def test_checkpoint_gamma_is_required(self):
        sp_config, target_config, draft_config = self._configs(configured_gamma=3)
        draft_config.dspark_block_size = None

        with self.assertRaisesRegex(ValueError, "dspark_block_size"):
            ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

    def test_checkpoint_gamma_refreshes_deepep_capacity(self):
        engine_config = SimpleNamespace(
            sp_config=SimpleNamespace(gen_num_per_cycle=5),
            concurrency_config=SimpleNamespace(concurrency_limit=32),
            moe_config=SimpleNamespace(ll_num_max_token=32 * (3 + 1)),
        )

        ModelFactory._sync_dspark_deepep_capacity(engine_config)

        self.assertEqual(engine_config.moe_config.ll_num_max_token, 32 * (5 + 1))


if __name__ == "__main__":
    unittest.main()
