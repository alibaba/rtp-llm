import unittest
from unittest.mock import patch

from rtp_llm.config.maxpander_config import MaxPanderConfig, MaxPanderRuntimeConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc.deepseek_v41_base_model import (
    _args_from_model_config,
)
from rtp_llm.models_py.modules.dsv41.transformer import (
    V4Args,
    _block_kwargs,
    _build_block,
)


class DeepSeekV41MaxPanderTest(unittest.TestCase):
    def test_default_block_arguments_keep_maxpander_disabled(self):
        arguments = V4Args()
        for commit_only in (False, True):
            with self.subTest(commit_only=commit_only):
                block_args = _block_kwargs(0, arguments, None, commit_only)
                self.assertEqual(
                    block_args["maxpander_config"], MaxPanderRuntimeConfig()
                )
                self.assertFalse(block_args["maxpander_config"].enabled)
                self.assertEqual(block_args["commit_only"], commit_only)

    def test_target_and_draft_configuration_reaches_block_unchanged(self):
        for role in ("PREFILL", "DECODE", "PDFUSION"):
            for requested in (False, True):
                if requested and role == "PDFUSION":
                    continue
                for expert_num in (128, 256, 384):
                    for commit_only in (False, True):
                        with self.subTest(
                            role=role,
                            requested=requested,
                            expert_num=expert_num,
                            commit_only=commit_only,
                        ):
                            config = ModelConfig()
                            config.num_layers = 1
                            config.expert_num = expert_num
                            config.scoring_func = 2
                            config.attn_config.layer_compress_ratios = [0]
                            config.maxpander_config = MaxPanderConfig(
                                enable=requested, seed=17, wf_passes=11
                            ).resolve(role)
                            arguments = _args_from_model_config(config)
                            arguments.is_decode_role = role == "DECODE"
                            self.assertIs(
                                arguments.maxpander_config, config.maxpander_config
                            )
                            self.assertEqual(arguments.n_routed_experts, expert_num)
                            with patch(
                                "rtp_llm.models_py.modules.dsv41.transformer.Block"
                            ) as block:
                                _build_block(0, arguments, commit_only=commit_only)
                            actual = block.call_args.kwargs["maxpander_config"]
                            self.assertIs(actual, config.maxpander_config)
                            self.assertEqual(
                                actual.enabled, requested and role == "PREFILL"
                            )
                            self.assertEqual(actual.seed, 17)
                            self.assertEqual(actual.wf_passes, 11)


if __name__ == "__main__":
    unittest.main()
