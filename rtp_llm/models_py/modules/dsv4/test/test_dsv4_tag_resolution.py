"""``resolve_dsv4_tag`` — canonical DSV4 tag -> declared module-scoped tag.

Draft modules declare their cache tags in the Python model description
(``declare_draft_cache_tags``, ``<prefix>`` = ``mtp_``), so the published tag is
``<prefix><tag>``.  The resolver is the identity whenever the canonical tag is
already present and never guesses: an unmatched tag is returned unchanged so
the caller reports the miss.  These are pure-CPU contract tests.
"""

import unittest

from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import _semantic_tag
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    CSA_KV,
    DRAFT_CACHE_TAG_PREFIX,
    SWA_KV,
    as_attention_inputs_by_tag,
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

    def test_declared_prefix_tag(self):
        cache = _FakeKVCache(
            [DRAFT_CACHE_TAG_PREFIX + SWA_KV, DRAFT_CACHE_TAG_PREFIX + CSA_KV]
        )
        self.assertEqual(
            resolve_dsv4_tag(cache, SWA_KV), DRAFT_CACHE_TAG_PREFIX + SWA_KV
        )
        self.assertEqual(
            resolve_dsv4_tag(cache, CSA_KV), DRAFT_CACHE_TAG_PREFIX + CSA_KV
        )

    def test_canonical_wins_over_prefixed(self):
        # Both present: the canonical name is authoritative, never the mtp one.
        cache = _FakeKVCache([SWA_KV, "mtp_" + SWA_KV])
        self.assertEqual(resolve_dsv4_tag(cache, SWA_KV), SWA_KV)

    def test_single_group_is_not_guessed(self):
        # No heuristics: one unrelated group must not be substituted silently.
        cache = _FakeKVCache(["only_group"])
        self.assertEqual(resolve_dsv4_tag(cache, SWA_KV), SWA_KV)

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


class AttentionInputMappingTest(unittest.TestCase):
    def test_published_keys_are_normalized_to_canonical_tags(self):
        # The framework hands one PyAttentionInputs per cache group keyed by
        # the module's *published* tag; callers index the mapping by the
        # canonical tag.
        csa, swa = object(), object()
        mapping = {
            DRAFT_CACHE_TAG_PREFIX + CSA_KV: csa,
            DRAFT_CACHE_TAG_PREFIX + SWA_KV: swa,
        }
        self.assertEqual(
            as_attention_inputs_by_tag(mapping),
            {CSA_KV: csa, SWA_KV: swa},
        )
        self.assertIs(as_attention_inputs_by_tag(mapping)[SWA_KV], swa)


if __name__ == "__main__":
    unittest.main()
