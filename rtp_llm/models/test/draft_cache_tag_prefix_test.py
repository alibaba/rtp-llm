"""Draft modules declare module-scoped cache tags in the Python model description.

Every draft model class (MTP / DSpark / Eagle) calls
``rtp_llm.models_py.utils.kvcache.declare_draft_cache_tags`` from its own
``_post_build_model_config``.  The published tags are therefore part of the
model description: the main model keeps the canonical names, a draft module
publishes ``mtp_<tag>``.  Nothing on the C++ side infers or rewrites them; the
cache config only validates that two modules never publish the same tag.
"""

import unittest
from types import SimpleNamespace

from rtp_llm.models_py.utils.kvcache import (
    DRAFT_CACHE_TAG_PREFIX,
    declare_draft_cache_tags,
)


class _Desc:
    def __init__(self, tag):
        self.tag = tag


def _config(desc_tags):
    return SimpleNamespace(
        kv_cache_spec_descs=[[_Desc(tag) for tag in layer] for layer in desc_tags]
    )


class DeclareDraftCacheTagsTest(unittest.TestCase):
    def test_prefix_is_mtp(self):
        self.assertEqual(DRAFT_CACHE_TAG_PREFIX, "mtp_")

    def test_prefixes_every_desc(self):
        config = _config([["swa_kv", "csa_kv"], ["swa_kv"]])
        declare_draft_cache_tags(config)
        self.assertEqual(
            [desc.tag for layer in config.kv_cache_spec_descs for desc in layer],
            ["mtp_swa_kv", "mtp_csa_kv", "mtp_swa_kv"],
        )
        self.assertEqual(config.cache_tag_prefix, "mtp_")

    def test_is_idempotent(self):
        config = _config([["mtp_swa_kv"]])
        declare_draft_cache_tags(config)
        declare_draft_cache_tags(config)
        self.assertEqual(config.kv_cache_spec_descs[0][0].tag, "mtp_swa_kv")

    def test_empty_descs_declares_nothing(self):
        config = _config([])
        declare_draft_cache_tags(config)
        self.assertFalse(hasattr(config, "cache_tag_prefix"))

    def test_main_style_canonical_tag_is_never_left_unprefixed(self):
        config = _config([["default"]])
        declare_draft_cache_tags(config)
        self.assertEqual(config.kv_cache_spec_descs[0][0].tag, "mtp_default")


if __name__ == "__main__":
    unittest.main()
