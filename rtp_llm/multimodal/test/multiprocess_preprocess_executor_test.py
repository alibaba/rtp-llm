import io
import multiprocessing
import threading
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock

from PIL import Image, UnidentifiedImageError

from rtp_llm.multimodal.mm_process_engine import MultiprocessPreprocessExecutor


def preprocess_image(inputs, vit_config):
    if inputs == ["bad-image"]:
        Image.open(io.BytesIO(b"not an image"))
    if inputs == ["io-error"]:
        raise OSError("failed to read image")
    if inputs == ["eof-error"]:
        raise EOFError("truncated image stream")
    return inputs


class _NotifyingLock:
    def __init__(self, submit_attempted):
        self.lock = threading.Lock()
        self.submit_attempted = submit_attempted

    def __enter__(self):
        if threading.current_thread().name == "submit-during-rebuild":
            self.submit_attempted.set()
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()


class MultiprocessPreprocessExecutorTest(TestCase):
    @staticmethod
    def work_item(value="image"):
        return SimpleNamespace(
            should_preprocess=True,
            mm_inputs=[value],
            future=None,
            embedding_result=None,
            mm_timeout_ms=60000,
        )

    @staticmethod
    def executor(old_pool, new_pool):
        executor = MultiprocessPreprocessExecutor.__new__(
            MultiprocessPreprocessExecutor
        )
        executor.pool = old_pool
        executor.mp_context = SimpleNamespace(Pool=Mock(return_value=new_pool))
        executor.vit_config = SimpleNamespace(mm_preprocess_max_workers=1)
        executor.preprocess_params = {}
        executor.preprocess_func = preprocess_image
        executor._pool_lock = threading.Lock()
        executor._consecutive_timeouts = 0
        executor._max_consecutive_timeouts = 1
        return executor

    def test_submission_waits_for_concurrent_rebuild(self):
        old_pool, new_pool = Mock(), Mock()
        executor = self.executor(old_pool, new_pool)
        rebuilding, release, attempted = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        executor._pool_lock = _NotifyingLock(attempted)
        errors = []
        work = self.work_item()

        def join_old_pool():
            rebuilding.set()
            if not release.wait(5):
                raise RuntimeError("test did not release the old pool")

        old_pool.join.side_effect = join_old_pool

        def rebuild():
            with executor._pool_lock:
                executor._rebuild_pool()

        def submit():
            try:
                executor.submit(work)
            except Exception as error:
                errors.append(error)
            finally:
                attempted.set()

        rebuilder = threading.Thread(target=rebuild)
        submitter = threading.Thread(target=submit, name="submit-during-rebuild")
        rebuilder.start()
        try:
            self.assertTrue(rebuilding.wait(5))
            submitter.start()
            self.assertTrue(attempted.wait(5))
            self.assertEqual(errors, [])
            new_pool.apply_async.assert_not_called()
        finally:
            release.set()
            rebuilder.join(5)
            if submitter.ident is not None:
                submitter.join(5)
        self.assertFalse(rebuilder.is_alive())
        self.assertFalse(submitter.is_alive())
        self.assertEqual(errors, [])
        self.assertIs(work.future, new_pool.apply_async.return_value)

    def test_broken_submission_rebuilds_and_retries_once(self):
        old_pool, new_pool = Mock(), Mock()
        old_pool.apply_async.side_effect = BrokenPipeError("broken pool pipe")
        executor = self.executor(old_pool, new_pool)
        work = self.work_item()
        executor.submit(work)
        old_pool.terminate.assert_called_once()
        old_pool.join.assert_called_once()
        executor.mp_context.Pool.assert_called_once()
        new_pool.apply_async.assert_called_once()
        self.assertIs(work.future, new_pool.apply_async.return_value)

    def test_submission_recovers_after_replacement_creation_failed(self):
        old_pool, new_pool = Mock(), Mock()
        old_pool.apply_async.side_effect = BrokenPipeError("broken pool pipe")
        executor = self.executor(old_pool, new_pool)
        executor.mp_context.Pool.side_effect = [
            OSError("worker resources unavailable"),
            new_pool,
        ]
        with self.assertRaisesRegex(OSError, "worker resources unavailable"):
            executor.submit(self.work_item())
        work = self.work_item()
        executor.submit(work)
        self.assertIs(executor.pool, new_pool)
        self.assertIs(work.future, new_pool.apply_async.return_value)

    def test_invalid_image_keeps_pool_and_other_queued_request(self):
        executor = MultiprocessPreprocessExecutor(
            multiprocessing.get_context("spawn"),
            SimpleNamespace(mm_preprocess_max_workers=1),
            {},
            preprocess_image,
        )
        original_pool = executor.pool
        try:
            for value, error_type in (
                ("bad-image", UnidentifiedImageError),
                ("io-error", OSError),
                ("eof-error", EOFError),
            ):
                with self.subTest(value=value):
                    bad, good = self.work_item(value), self.work_item("valid-image")
                    executor.submit(bad)
                    executor.submit(good)
                    with self.assertRaises(error_type):
                        executor.get_result(bad)
                    self.assertIs(executor.pool, original_pool)
                    executor.get_result(good)
                    self.assertEqual(good.preprocess_result, ["valid-image"])
        finally:
            executor.shutdown()


if __name__ == "__main__":
    main()
