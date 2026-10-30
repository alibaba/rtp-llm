import unittest

from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3, KimiK3ModelConfig, KimiK3Mtp


class KimiK3CacheConfigTest(unittest.TestCase):
    def test_target_and_draft_select_compact_from_the_same_config(self):
        for model_type in ("kimi_k3", "kimi_k3_mtp"):
            with self.subTest(model_type=model_type):
                config = KimiK3ModelConfig()
                config.model_type = model_type
                self.assertTrue(config.use_page_rr_kv_cache)

    def test_compact_only_model_rejects_disabling_page_rr(self):
        for owner in (KimiK3, KimiK3Mtp):
            with self.subTest(owner=owner.__name__):
                config = KimiK3ModelConfig()
                config.use_page_rr_kv_cache = False
                with self.assertRaisesRegex(
                    ValueError, "requires compact local PageRR"
                ):
                    owner._post_build_model_config(config)


if __name__ == "__main__":
    unittest.main()
