"""Decode's bounded KDA pool must leave the remaining budget for paged KV."""

import unittest
from types import SimpleNamespace

from rtp_llm.models.glm53_cache_policy import configure_decode_request_state_pool
from rtp_llm.ops import RoleType


class GlmDecodeCachePolicyTest(unittest.TestCase):
    def test_request_and_mtp_capacity_matches_feature_branch(self):
        for batch, draft, expected in ((64, 4, 449), (64, 0, 193), (1, 4, 8)):
            config = SimpleNamespace(linear_pool_blocks=0, shared_hybrid_cache=0)
            result = configure_decode_request_state_pool(
                config, RoleType.DECODE, batch, draft, enabled=True
            )
            self.assertEqual(result, expected)
            self.assertEqual(config.linear_pool_blocks, expected)

    def test_disabled_feature_other_roles_and_explicit_policies_are_preserved(self):
        for role, enabled, blocks, shared in (
            (RoleType.DECODE, False, 0, 0),
            (RoleType.PREFILL, True, 0, 0),
            (RoleType.PDFUSION, True, 0, 0),
            (RoleType.DECODE, True, 777, 0),
            (RoleType.DECODE, True, 0, 1),
        ):
            with self.subTest(role=role, enabled=enabled, blocks=blocks, shared=shared):
                config = SimpleNamespace(
                    linear_pool_blocks=blocks, shared_hybrid_cache=shared
                )
                self.assertIsNone(
                    configure_decode_request_state_pool(
                        config, role, 64, 4, enabled=enabled
                    )
                )
                self.assertEqual(config.linear_pool_blocks, blocks)

    def test_auto_arena_policy_uses_independent_decode_request_pool(self):
        config = SimpleNamespace(linear_pool_blocks=0, shared_hybrid_cache=-1)
        configure_decode_request_state_pool(
            config, RoleType.DECODE, 64, 4, enabled=True
        )
        self.assertEqual(config.linear_pool_blocks, 449)

    def test_invalid_capacity_fails_before_native_allocation(self):
        for batch in (0, -1, 2147483647):
            with self.subTest(batch=batch):
                config = SimpleNamespace(linear_pool_blocks=0, shared_hybrid_cache=0)
                with self.assertRaisesRegex(ValueError, "request-state pool"):
                    configure_decode_request_state_pool(
                        config, RoleType.DECODE, batch, 4, enabled=True
                    )


if __name__ == "__main__":
    unittest.main()
