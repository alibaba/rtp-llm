import concurrent.futures
import threading
import time
from unittest import TestCase, main, mock

from rtp_llm.multimodal.download_singleflight import (
    DownloadSingleFlight,
    check_download_waiter,
    wait_for_download,
)


class DownloadSingleFlightTest(TestCase):
    def setUp(self):
        self.producers = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.waiters = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.flight = DownloadSingleFlight(self.producers)
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.started = threading.Event()
        self.release = threading.Event()
        self.joined = threading.Event()
        self.calls = []
        self.addCleanup(self.producers.shutdown, wait=True)
        self.addCleanup(self.waiters.shutdown, wait=True)
        self.addCleanup(self.release.set)

    def get_cached(self, key):
        with self.cache_lock:
            return self.cache.get(key)

    def cache_bytes(self, key, content):
        self.assertIsInstance(content, bytes)
        with self.cache_lock:
            self.cache[key] = content

    def produce(self):
        self.calls.append("download")
        self.started.set()
        if not self.release.wait(3):
            raise TimeoutError("test producer was not released")
        return b"payload"

    def download(self, **kwargs):
        return self.flight.download(
            "url", self.produce, self.get_cached, self.cache_bytes, **kwargs
        )

    def observe_join(self):
        original = self.flight.submit

        def submit(*args):
            future, owns = original(*args)
            if not owns:
                self.joined.set()
            return future, owns

        return mock.patch.object(self.flight, "submit", side_effect=submit)

    def test_no_controls_share_one_producer_and_immutable_result(self):
        with self.observe_join():
            first = self.waiters.submit(self.download)
            self.assertTrue(self.started.wait(1))
            second = self.waiters.submit(self.download)
            self.assertTrue(self.joined.wait(1))
            self.release.set()
            self.assertEqual(first.result(1), b"payload")
            self.assertEqual(second.result(1), b"payload")
        self.assertEqual(self.calls, ["download"])
        self.assertIsInstance(self.get_cached("url"), bytes)

    def test_first_waiter_cancellation_does_not_cancel_shared_producer(self):
        cancelled = threading.Event()
        with self.observe_join():
            first = self.waiters.submit(self.download, cancellation_event=cancelled)
            self.assertTrue(self.started.wait(1))
            second = self.waiters.submit(self.download)
            self.assertTrue(self.joined.wait(1))
            cancelled.set()
            with self.assertRaises(concurrent.futures.CancelledError):
                first.result(1)
            self.assertFalse(second.done())
            self.release.set()
            self.assertEqual(second.result(1), b"payload")
        self.assertEqual(self.calls, ["download"])

    def test_follower_cancellation_does_not_cancel_first_waiter(self):
        cancelled = threading.Event()
        with self.observe_join():
            first = self.waiters.submit(self.download)
            self.assertTrue(self.started.wait(1))
            second = self.waiters.submit(self.download, cancellation_event=cancelled)
            self.assertTrue(self.joined.wait(1))
            cancelled.set()
            with self.assertRaises(concurrent.futures.CancelledError):
                second.result(1)
            self.assertFalse(first.done())
            self.release.set()
            self.assertEqual(first.result(1), b"payload")
        self.assertEqual(self.calls, ["download"])

    def test_first_deadline_leaves_other_waiter_and_producer_alive(self):
        with self.observe_join():
            first = self.waiters.submit(self.download, deadline=time.monotonic() + 0.2)
            self.assertTrue(self.started.wait(1))
            second = self.waiters.submit(self.download)
            self.assertTrue(self.joined.wait(1))
            with self.assertRaisesRegex(TimeoutError, "deadline expired"):
                first.result(1)
            self.assertFalse(second.done())
            self.release.set()
            self.assertEqual(second.result(1), b"payload")
        self.assertEqual(self.calls, ["download"])

    def test_follower_deadline_leaves_first_waiter_alive(self):
        with self.observe_join():
            first = self.waiters.submit(self.download)
            self.assertTrue(self.started.wait(1))
            second = self.waiters.submit(self.download, deadline=time.monotonic() + 0.2)
            self.assertTrue(self.joined.wait(1))
            with self.assertRaisesRegex(TimeoutError, "deadline expired"):
                second.result(1)
            self.release.set()
            self.assertEqual(first.result(1), b"payload")

    def test_cancelled_only_waiter_still_allows_late_cache_fill(self):
        cancelled = threading.Event()
        completed = threading.Event()

        def insert(key, content):
            self.cache_bytes(key, content)
            completed.set()

        first = self.waiters.submit(
            self.flight.download, "url", self.produce, self.get_cached, insert,
            cancellation_event=cancelled,
        )
        self.assertTrue(self.started.wait(1))
        cancelled.set()
        with self.assertRaises(concurrent.futures.CancelledError):
            first.result(1)
        self.release.set()
        self.assertTrue(completed.wait(1))
        self.assertEqual(self.download(), b"payload")
        self.assertEqual(self.calls, ["download"])

    def test_cache_hit_does_not_start_producer(self):
        self.cache_bytes("url", b"cached")
        self.assertEqual(self.download(), b"cached")
        self.assertEqual(self.calls, [])

    def test_expired_or_cancelled_cache_hit_is_rejected_without_producer(self):
        self.cache_bytes("url", b"cached")
        cancelled = threading.Event()
        cancelled.set()
        for kwargs, error in (
            ({"deadline": time.monotonic() - 1}, TimeoutError),
            ({"cancellation_event": cancelled}, concurrent.futures.CancelledError),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                self.download(**kwargs)
        self.assertEqual(self.calls, [])

    def test_cache_disabled_preserves_native_follower_fallback(self):
        def download():
            return self.flight.download("url", self.produce, lambda key: None,
                                        lambda key, content: None)

        with self.observe_join():
            first = self.waiters.submit(download)
            self.assertTrue(self.started.wait(1))
            second = self.waiters.submit(download)
            self.assertTrue(self.joined.wait(1))
            self.release.set()
            self.assertEqual(first.result(1), b"payload")
            self.assertEqual(second.result(1), b"payload")
        self.assertEqual(self.calls, ["download", "download"])
        self.assertIsNone(self.get_cached("url"))

    def test_producer_failure_preserves_owner_error_and_follower_retry(self):
        error = RuntimeError("network failed")
        count = []

        def produce():
            count.append(1)
            if len(count) == 1:
                self.started.set()
                if not self.release.wait(3):
                    raise TimeoutError("test producer was not released")
                raise error
            return b"retried"

        def download():
            return self.flight.download("url", produce, self.get_cached, self.cache_bytes)

        with self.observe_join():
            first = self.waiters.submit(download)
            self.assertTrue(self.started.wait(1))
            second = self.waiters.submit(download)
            self.assertTrue(self.joined.wait(1))
            self.release.set()
            with self.assertRaises(RuntimeError) as raised:
                first.result(1)
            self.assertIs(raised.exception, error)
            self.assertEqual(second.result(1), b"retried")
        self.assertEqual(len(count), 2)

    def test_producer_timeout_is_not_a_waiter_deadline(self):
        error = TimeoutError("producer timeout")
        future = concurrent.futures.Future()
        future.set_exception(error)
        wait_for_download(future, deadline=time.monotonic() + 1)
        with self.assertRaises(TimeoutError) as raised:
            future.result()
        self.assertIs(raised.exception, error)

    def test_submit_failure_removes_claim_for_later_attempt(self):
        with mock.patch.object(self.producers, "submit", side_effect=RuntimeError("closed")):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                self.flight.submit("url", lambda: b"unused")
        future, owns = self.flight.submit("url", lambda: b"retry")
        self.assertTrue(owns)
        self.assertEqual(future.result(1), b"retry")

    def test_completed_failure_releases_key(self):
        error = RuntimeError("failed")

        def fail():
            raise error

        future, owns = self.flight.submit("url", fail)
        self.assertTrue(owns)
        with self.assertRaises(RuntimeError):
            future.result(1)
        # synchronize with producer cleanup without sleeping or timing guesses
        self.producers.shutdown(wait=True)
        replacement = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.addCleanup(replacement.shutdown, wait=True)
        self.flight._executor = replacement
        future, owns = self.flight.submit("url", lambda: b"next")
        self.assertTrue(owns)
        self.assertEqual(future.result(1), b"next")

    def test_cancel_precedes_deadline_and_never_mutates_future(self):
        cancelled = threading.Event()
        cancelled.set()
        future = concurrent.futures.Future()
        with self.assertRaises(concurrent.futures.CancelledError):
            wait_for_download(future, deadline=time.monotonic() - 1,
                              cancellation_event=cancelled)
        self.assertFalse(future.cancelled())
        self.assertFalse(future.done())
        with self.assertRaises(concurrent.futures.CancelledError):
            check_download_waiter(time.monotonic() - 1, cancelled)


if __name__ == "__main__":
    main()
