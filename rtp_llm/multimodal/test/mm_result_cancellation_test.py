import threading
import time
from concurrent.futures import CancelledError
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCacheEntry
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine


class ViTResultCancellationTest(TestCase):
    def test_cancelled_result_wait_does_not_wait_for_full_timeout(self):
        entry = MMEmbeddingCacheEntry()
        cancellation = threading.Event()
        cancellation.set()
        start = time.monotonic()
        with self.assertRaises(CancelledError):
            entry.wait(timeout=60, cancellation_event=cancellation)
        self.assertLess(time.monotonic() - start, 1)

    def test_one_cancelled_waiter_does_not_invalidate_shared_entry(self):
        entry = MMEmbeddingCacheEntry()
        cancellation = threading.Event()
        cancellation.set()
        with self.assertRaises(CancelledError):
            entry.wait(timeout=60, cancellation_event=cancellation)
        payload = object()
        self.assertTrue(entry.complete(payload))
        self.assertIs(entry.wait(timeout=0), payload)

    def test_successful_inspection_preserves_pending_embedding(self):
        entry = MMEmbeddingCacheEntry()
        entry.set_greennet_verdict(GreenNetVerdict(passed=True))
        cancellation = threading.Event()
        self.assertTrue(
            entry.wait_greennet(timeout=0, cancellation_event=cancellation).passed
        )
        self.assertFalse(entry.is_done)
        payload = object()
        entry.complete(payload)
        self.assertIs(entry.wait(timeout=0), payload)

    @staticmethod
    def engine(entry):
        engine = MMProcessEngine.__new__(MMProcessEngine)
        engine.mm_part = SimpleNamespace(validate_inputs=lambda inputs: None)
        engine._claim_and_submit_async = MagicMock(return_value=[("key", entry)])
        engine.cancel_queued_request = MagicMock()
        return engine

    def test_expired_rpc_releases_request_ownership(self):
        engine = self.engine(MMEmbeddingCacheEntry())
        with self.assertRaises(TimeoutError):
            engine.get_embedding_result([], timeout_ms=0, request_id=75)
        engine._claim_and_submit_async.assert_not_called()
        engine.cancel_queued_request.assert_called_once_with(75)

    def test_cancel_during_result_wait_releases_request_ownership(self):
        cancellation = threading.Event()
        entry = MMEmbeddingCacheEntry()
        engine = self.engine(entry)

        def submit(*args, **kwargs):
            cancellation.set()
            return [("key", entry)]

        engine._claim_and_submit_async.side_effect = submit
        with self.assertRaises(FtRuntimeException) as caught:
            engine.get_embedding_result(
                [], request_id=75, cancellation_event=cancellation
            )
        self.assertEqual(caught.exception.exception_type, ExceptionType.CANCELLED_ERROR)
        engine.cancel_queued_request.assert_called_once_with(75)


if __name__ == "__main__":
    main()
