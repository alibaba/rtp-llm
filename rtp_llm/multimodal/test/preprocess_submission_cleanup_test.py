from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock

from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCache
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine


class PreprocessSubmissionCleanupTest(TestCase):
    @staticmethod
    def engine():
        engine = MMProcessEngine.__new__(MMProcessEngine)
        engine.mm_preprocess_batch_size = 1
        engine.vit_config = SimpleNamespace(mm_timeout_ms=120000)
        engine._embedding_cache = MMEmbeddingCache()
        engine.preprocess_executor = Mock()
        return engine

    @staticmethod
    def input(url):
        return SimpleNamespace(
            url=url,
            cache_key=lambda: url,
            mm_type=0,
            mm_preprocess_config=SimpleNamespace(mm_timeout_ms=0),
        )

    def test_failed_first_submission_resolves_claim(self):
        engine = self.engine()
        observed = []

        def submit(work_item):
            observed.append(work_item.cache_entry)
            raise OSError("submission failed")

        engine.preprocess_executor.submit.side_effect = submit
        with self.assertRaisesRegex(OSError, "submission failed"):
            engine._create_work_items([self.input("first")])

        with self.assertRaisesRegex(OSError, "submission failed"):
            observed[0].wait(timeout=0)
        state, _ = engine._embedding_cache.try_acquire("first")
        self.assertEqual(state, "miss")

    def test_partial_submission_resolves_all_owned_claims(self):
        engine = self.engine()
        observed = []

        def submit(work_item):
            observed.append(work_item.cache_entry)
            if len(observed) == 2:
                raise OSError("second submission failed")

        engine.preprocess_executor.submit.side_effect = submit
        with self.assertRaisesRegex(OSError, "second submission failed"):
            engine._create_work_items([self.input("first"), self.input("second")])

        self.assertEqual(len(observed), 2)
        for key, entry in zip(("first", "second"), observed):
            with self.assertRaisesRegex(OSError, "second submission failed"):
                entry.wait(timeout=0)
            state, _ = engine._embedding_cache.try_acquire(key)
            self.assertEqual(state, "miss")

    def test_submission_failure_preserves_another_requests_claim(self):
        engine = self.engine()
        _, other_entry = engine._embedding_cache.try_acquire("other-request")

        def submit(work_item):
            if work_item.should_preprocess:
                raise OSError("new submission failed")

        engine.preprocess_executor.submit.side_effect = submit
        with self.assertRaisesRegex(OSError, "new submission failed"):
            engine._create_work_items(
                [self.input("other-request"), self.input("new-request")]
            )

        self.assertFalse(other_entry.is_done)
        self.assertIs(engine._embedding_cache.peek("other-request"), other_entry)
        engine._embedding_cache.fail(
            "other-request", other_entry, ValueError("owner's result")
        )
        with self.assertRaisesRegex(ValueError, "owner's result"):
            other_entry.wait(timeout=0)
        state, _ = engine._embedding_cache.try_acquire("new-request")
        self.assertEqual(state, "miss")


if __name__ == "__main__":
    main()
