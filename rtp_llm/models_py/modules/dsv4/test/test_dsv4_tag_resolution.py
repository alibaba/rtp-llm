"""``resolve_dsv4_tag`` — canonical DSV4 tag -> published cache-group tag.

MTP modules publish their cache groups as ``mtp_<tag>`` (``renameMTPGroupTags``
in ``CacheConfigCreator``) while the kernels keep addressing pools by canonical
name, so the resolver must be the identity whenever the canonical tag is
already present. These are pure-CPU contract tests for that mapping.
"""

import unittest

from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import _semantic_tag
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    CSA_KV,
    SWA_KV,
    group_tags,
    resolve_dsv4_tag,
)


class _FakeKVCache:
    def __init__(self, tags):
        self.group_tags = tags


class ResolveDsv4TagTest(unittest.TestCase):
    def test_identity_when_tag_already_published(self):
        cache = _FakeKVCache([SWA_KV, CSA_KV])
        self.assertEqual(resolve_dsv4_tag(cache, SWA_KV), SWA_KV)
        self.assertEqual(resolve_dsv4_tag(cache, CSA_KV), CSA_KV)

    def test_mtp_prefixed_tag(self):
        cache = _FakeKVCache(["mtp_" + SWA_KV, "mtp_" + CSA_KV])
        self.assertEqual(resolve_dsv4_tag(cache, SWA_KV), "mtp_" + SWA_KV)
        self.assertEqual(resolve_dsv4_tag(cache, CSA_KV), "mtp_" + CSA_KV)

    def test_canonical_wins_over_prefixed(self):
        # Both present: the canonical name is authoritative, never the mtp one.
        cache = _FakeKVCache([SWA_KV, "mtp_" + SWA_KV])
        self.assertEqual(resolve_dsv4_tag(cache, SWA_KV), SWA_KV)

    def test_single_group_fallback(self):
        cache = _FakeKVCache(["only_group"])
        self.assertEqual(resolve_dsv4_tag(cache, SWA_KV), "only_group")

    def test_multi_group_without_match_is_identity(self):
        # No match and ambiguous -> return unchanged so the caller reports the
        # original error instead of silently using an unrelated pool.
        cache = _FakeKVCache([CSA_KV, "indexer_kv"])
        self.assertEqual(resolve_dsv4_tag(cache, SWA_KV), SWA_KV)

    def test_missing_or_empty_cache_is_identity(self):
        self.assertEqual(resolve_dsv4_tag(None, SWA_KV), SWA_KV)
        self.assertEqual(resolve_dsv4_tag(_FakeKVCache([]), SWA_KV), SWA_KV)
        self.assertEqual(group_tags(_FakeKVCache([])), [])


class SemanticTagTest(unittest.TestCase):
    def test_strips_mtp_prefix(self):
        self.assertEqual(_semantic_tag("mtp_" + SWA_KV), SWA_KV)
        self.assertEqual(_semantic_tag(SWA_KV), SWA_KV)
        self.assertIsNone(_semantic_tag(None))


if __name__ == "__main__":
    unittest.main()
