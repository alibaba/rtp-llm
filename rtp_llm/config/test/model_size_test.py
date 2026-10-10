import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig, build_model_config
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.ops import ProfilingDebugLoggingConfig, VitSeparation


class ModelSizeTest(unittest.TestCase):
    def test_visual_estimation_matches_deployment(self):
        modes = (
            (None, True),
            (VitSeparation.VIT_SEPARATION_LOCAL, True),
            (VitSeparation.VIT_SEPARATION_ROLE, True),
            (VitSeparation.VIT_SEPARATION_REMOTE, False),
        )
        with TemporaryDirectory() as ckpt_path:
            args = ModelArgs()
            args.ckpt_path = args.tokenizer_path = ckpt_path
            args.model_type = "unused_visual_model"
            args.max_seq_len = 16
            for mode, owns_vit in modes:
                for multimodal, mtp in ((True, False), (True, True), (False, False)):
                    with self.subTest(mode=mode, multimodal=multimodal, mtp=mtp):
                        config = ModelConfig()
                        config.hidden_size = 8
                        config.vocab_size = 16
                        config.num_layers = 1
                        config.inter_size = 16
                        config.attn_config.head_num = 2
                        config.attn_config.kv_head_num = 1
                        config.attn_config.size_per_head = 4
                        vit = None
                        if mode is not None:
                            vit = VitConfig()
                            vit.vit_separation = mode
                        build_model_config(
                            config,
                            args,
                            KVCacheConfig(),
                            ProfilingDebugLoggingConfig(),
                            vit_config=vit,
                        )
                        baseline = (
                            config.eval_model_weight_size(),
                            config.eval_model_size(),
                            config.model_param_count(),
                        )
                        config.mm_model_config.is_multimodal = multimodal
                        config.mm_model_config.mm_sep_tokens = [[200025], [200026]]
                        config.is_mtp = mtp
                        include_vit = owns_vit and multimodal and not mtp
                        with patch(
                            "rtp_llm.config.model_config.get_multimodal_mixin_cls"
                        ) as factory:
                            factory.return_value.eval_mm_model_size.return_value = 1024
                            factory.return_value.eval_mm_model_param_count.return_value = (
                                512
                            )
                            if not include_vit:
                                factory.side_effect = AssertionError(
                                    "must not load ViT"
                                )
                            actual = (
                                config.eval_model_weight_size(),
                                config.eval_model_size(),
                                config.model_param_count(),
                            )
                            self.assertEqual(
                                actual,
                                tuple(
                                    base + (extra if include_vit else 0)
                                    for base, extra in zip(baseline, (1024, 1024, 512))
                                ),
                            )
                        self.assertEqual(config.is_multimodal(), multimodal)
                        self.assertEqual(
                            config.mm_model_config.mm_sep_tokens, [[200025], [200026]]
                        )


if __name__ == "__main__":
    unittest.main()
