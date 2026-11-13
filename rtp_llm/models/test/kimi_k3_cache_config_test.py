import unittest

from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3, KimiK3ModelConfig, KimiK3Mtp
from rtp_llm.ops import LocalKVCacheLayoutPolicy


class KimiK3CacheConfigTest(unittest.TestCase):
    def test_target_and_draft_select_compact_from_the_same_config(self):
        for model_type in ("kimi_k3", "kimi_k3_mtp"):
            with self.subTest(model_type=model_type):
                config = KimiK3ModelConfig()
                config.model_type = model_type
                self.assertEqual(
                    config.local_kv_cache_layout, LocalKVCacheLayoutPolicy.PAGE_RR
                )

    def test_compact_only_model_rejects_other_layout_policies(self):
        for owner in (KimiK3, KimiK3Mtp):
            for policy in (
                LocalKVCacheLayoutPolicy.FOLLOW_QUERY_CP,
                LocalKVCacheLayoutPolicy.FULL_TABLE,
            ):
                with self.subTest(owner=owner.__name__, policy=policy):
                    config = KimiK3ModelConfig()
                    config.local_kv_cache_layout = policy
                    with self.assertRaisesRegex(
                        ValueError, "requires compact local PageRR"
                    ):
                        owner._post_build_model_config(config)


if __name__ == "__main__":
    unittest.main()
