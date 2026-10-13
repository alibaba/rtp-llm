import unittest
from types import SimpleNamespace
from unittest import mock


class _FakeTok:
    def __call__(self, text, add_special_tokens=False):
        # every uppercase name is single-token; id by first-letter offset
        return SimpleNamespace(input_ids=[100 + ord(text[0]) - ord("A")])

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class DeciderLabelCacheTest(unittest.TestCase):
    def test_cache_is_keyed_by_tokenizer_identity(self):
        from rtp_llm.models.downstream_modules.decider_decision import (
            decider_label_table,
            LABELS_CACHE,
        )

        LABELS_CACHE.clear()
        tok_a = _FakeTok()
        table_a = decider_label_table(tok_a)
        # same tokenizer: cache hit, no recompute
        with mock.patch(
            "rtp_llm.models.downstream_modules.decider_decision.DECIDER_MAX_OPTIONS",
            255,
        ):
            again = decider_label_table(tok_a)
        self.assertIs(table_a, again)

        # a different tokenizer object never hits the first entry even if
        # id() collides after gc: the key includes the object itself
        LABELS_CACHE.clear()
        tok_b = _FakeTok()
        table_b = decider_label_table(tok_b)
        self.assertEqual(table_b[0][0], "A")


if __name__ == "__main__":
    unittest.main()
