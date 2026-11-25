import io
import multiprocessing
import os
import sys
import threading
import time
from multiprocessing.connection import wait
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from PIL import Image, UnidentifiedImageError

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.multimodal.mm_process_engine import (
    LocalPreprocessExecutor,
    MultiprocessPreprocessExecutor,
    _PreprocessPool,
)


def preprocess_image(inputs, vit_config):
    if inputs == ["bad-image"]:
        Image.open(io.BytesIO(b"not an image"))
    if inputs == ["io-error"]:
        raise OSError("failed to read image")
    if inputs == ["eof-error"]:
        raise EOFError("truncated image stream")
    if inputs == ["crash-worker"]:
        os._exit(7)
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
    def work_item(value="image", deadline=None):
        return SimpleNamespace(
            should_preprocess=True,
            mm_inputs=[value],
            future=None,
            embedding_result=None,
            deadline=deadline,
            mm_timeout_ms=60000,
            remaining_timeout_s=lambda: 60,
        )

    def executor(self, old_pool, new_pool):
        patcher = patch(
            "rtp_llm.multimodal.mm_process_engine._PreprocessPool",
            return_value=old_pool,
        )
        factory = patcher.start()
        self.addCleanup(patcher.stop)
        executor = MultiprocessPreprocessExecutor(
            SimpleNamespace(),
            SimpleNamespace(mm_preprocess_max_workers=1),
            {},
            preprocess_image,
        )
        factory.reset_mock()
        factory.return_value = new_pool
        return executor, factory

    def test_submission_waits_for_concurrent_rebuild(self):
        old_pool, new_pool = Mock(), Mock()
        executor, _ = self.executor(old_pool, new_pool)
        rebuilding, release, attempted = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        executor._pool_lock = _NotifyingLock(attempted)
        errors = []
        work = self.work_item()

        def stop_old_pool(timeout):
            rebuilding.set()
            if not release.wait(5):
                raise RuntimeError("test did not release the old pool")

        old_pool.terminate.side_effect = stop_old_pool

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
        executor, factory = self.executor(old_pool, new_pool)
        work = self.work_item()
        executor.submit(work)
        old_pool.terminate.assert_called_once()
        factory.assert_called_once()
        new_pool.apply_async.assert_called_once()
        self.assertIs(work.future, new_pool.apply_async.return_value)

    def test_submission_recovers_after_replacement_creation_failed(self):
        old_pool, new_pool = Mock(), Mock()
        old_pool.apply_async.side_effect = BrokenPipeError("broken pool pipe")
        executor, factory = self.executor(old_pool, new_pool)
        factory.side_effect = [
            OSError("worker resources unavailable"),
            new_pool,
        ]
        with self.assertRaisesRegex(OSError, "worker resources unavailable"):
            executor.submit(self.work_item())
        work = self.work_item()
        self.assertEqual(executor._pending_preprocess_tasks, set())
        executor.submit(work)
        self.assertEqual(len(executor._pending_preprocess_tasks), 1)
        self.assertIs(executor.pool, new_pool)
        self.assertIs(work.future, new_pool.apply_async.return_value)

    def test_broken_result_rebuilds_pool(self):
        old_pool, new_pool = Mock(), Mock()
        executor, _ = self.executor(old_pool, new_pool)
        work = self.work_item()
        executor.submit(work)
        work.future.get.side_effect = EOFError("broken result pipe")
        with self.assertRaisesRegex(EOFError, "broken result pipe"):
            executor.get_result(work)
        old_pool.terminate.assert_called_once()
        self.assertIs(executor.pool, new_pool)
        self.assertEqual(executor._pending_preprocess_tasks, set())

    def test_retry_failure_clears_queue_accounting(self):
        old_pool, new_pool = Mock(), Mock()
        old_pool.apply_async.side_effect = BrokenPipeError("broken pool pipe")
        new_pool.apply_async.side_effect = OSError("retry failed")
        executor, _ = self.executor(old_pool, new_pool)
        with self.assertRaisesRegex(OSError, "retry failed"):
            executor.submit(self.work_item())
        old_pool.apply_async.assert_called_once()
        new_pool.apply_async.assert_called_once()
        self.assertEqual(executor._pending_preprocess_tasks, set())

    def test_progress_resets_watchdog_and_stale_timeouts_keep_new_pool(self):
        old_pool, new_pool = Mock(), Mock()
        executor, factory = self.executor(old_pool, new_pool)
        executor._max_consecutive_timeouts = 2
        old_pool.apply_async.return_value.get.side_effect = multiprocessing.TimeoutError
        slow, completed, stalled, stale = (self.work_item() for _ in range(4))
        for work in (slow, completed, stalled, stale):
            executor.submit(work)

        with self.assertRaisesRegex(TimeoutError, "Preprocessing timeout"):
            executor.get_result(slow)
        old_pool.apply_async.call_args_list[1].kwargs["error_callback"](
            ValueError("invalid input, healthy worker")
        )
        with self.assertRaises(TimeoutError):
            executor.get_result(stalled)
        self.assertIs(executor.pool, old_pool)
        self.assertEqual(executor._consecutive_timeouts, 0)

        for work in (stalled, stale, slow, stale):
            with self.assertRaises(TimeoutError):
                executor.get_result(work)
        self.assertIs(executor.pool, new_pool)
        factory.assert_called_once()
        self.assertEqual(executor._consecutive_timeouts, 0)
        self.assertEqual(executor._pending_preprocess_tasks, set())
        old_pool.apply_async.call_args.kwargs["callback"](None)
        self.assertEqual(executor._completed_tasks, 1)

    def test_replacement_failure_does_not_mask_wait_timeout(self):
        old_pool, new_pool = Mock(), Mock()
        executor, factory = self.executor(old_pool, new_pool)
        factory.side_effect = [OSError("workers unavailable"), new_pool]
        work = self.work_item()
        executor.submit(work)
        work.future.get.side_effect = multiprocessing.TimeoutError
        with self.assertRaisesRegex(TimeoutError, "Preprocessing timeout"):
            executor.get_result(work)
        self.assertIsNone(executor.pool)
        executor.submit(self.work_item())
        self.assertIs(executor.pool, new_pool)

    def test_get_result_waits_only_for_remaining_deadline(self):
        executor, _ = self.executor(Mock(), Mock())
        work = self.work_item(deadline=time.monotonic() + 1)
        work.remaining_timeout_s = lambda: 0.125
        executor.submit(work)
        work.future.get.return_value = (["image"], 1, 0, [])
        executor.get_result(work)
        work.future.get.assert_called_once_with(timeout=0.125)
        self.assertEqual(work.preprocess_result, ["image"])

    def test_unreaped_pool_blocks_replacement_until_teardown_succeeds(self):
        old_pool, new_pool = Mock(), Mock()
        executor, factory = self.executor(old_pool, new_pool)
        old_pool.terminate.side_effect = [
            TimeoutError("worker did not exit"),
            TimeoutError("worker did not exit"),
            None,
        ]
        work = self.work_item()
        executor.submit(work)
        work.future.get.side_effect = multiprocessing.TimeoutError
        with self.assertRaisesRegex(TimeoutError, "Preprocessing timeout"):
            executor.get_result(work)
        with self.assertRaisesRegex(TimeoutError, "worker did not exit"):
            executor.submit(self.work_item())
        factory.assert_not_called()
        self.assertIs(executor._retiring_pool, old_pool)
        executor.submit(self.work_item())
        factory.assert_called_once()
        self.assertIs(executor.pool, new_pool)

    def test_shutdown_drains_and_terminates_from_same_thread(self):
        pool = Mock()
        executor, factory = self.executor(pool, Mock())
        executor._POOL_EXIT_TIMEOUT_S = 0.05
        pool._pool = []
        worker = Mock()
        caller = threading.current_thread()
        steps = []

        def record(step, timeout):
            self.assertIs(threading.current_thread(), caller)
            self.assertGreaterEqual(timeout, 0)
            self.assertLessEqual(timeout, executor._POOL_EXIT_TIMEOUT_S)
            steps.append(step)

        def finish_worker_handler(timeout):
            record("worker handler", timeout)
            pool._pool.append(worker)

        pool._worker_handler.join.side_effect = finish_worker_handler
        pool._worker_handler.is_alive.return_value = False
        pool._task_handler.is_alive.return_value = False
        pool._result_handler.is_alive.return_value = False
        worker.join.side_effect = lambda timeout: record("worker", timeout)
        worker.is_alive.return_value = True
        pool.drain.side_effect = lambda timeout: _PreprocessPool.drain(pool, timeout)
        pool.terminate.side_effect = lambda timeout: record("terminate", timeout)

        executor.shutdown()
        self.assertIsNone(executor.pool)
        with self.assertRaisesRegex(RuntimeError, "pool is closed"):
            executor.submit(self.work_item())
        executor.shutdown()
        factory.assert_not_called()
        self.assertEqual(steps, ["worker handler", "worker", "terminate"])
        pool.close.assert_called_once()
        pool.join.assert_not_called()
        pool.terminate.assert_called_once_with(timeout=0.05)

    def real_executor(self):
        executor = MultiprocessPreprocessExecutor(
            multiprocessing.get_context("spawn"),
            SimpleNamespace(mm_preprocess_max_workers=1),
            {},
            preprocess_image,
        )
        self.addCleanup(executor.shutdown)
        return executor

    def test_expired_worker_task_keeps_pool_usable(self):
        executor = self.real_executor()
        pool = executor.pool
        expired = self.work_item("eof-error", deadline=time.monotonic() - 1)
        executor.submit(expired)
        with self.assertRaises(FtRuntimeException) as raised:
            executor.get_result(expired)
        self.assertEqual(
            raised.exception.exception_type, ExceptionType.GENERATE_TIMEOUT
        )
        self.assertIs(executor.pool, pool)
        valid = self.work_item()
        executor.submit(valid)
        executor.get_result(valid)
        self.assertEqual(valid.preprocess_result, ["image"])

    def test_crashed_worker_is_replaced_after_timeout(self):
        executor = self.real_executor()
        old_pool = executor.pool
        worker = old_pool._pool[0]
        crashed = self.work_item("crash-worker")
        executor.submit(crashed)
        self.assertEqual(wait([worker.sentinel], timeout=20), [worker.sentinel])
        crashed.remaining_timeout_s = lambda: 0.05
        with self.assertRaises(TimeoutError):
            executor.get_result(crashed)
        self.assertIsNot(executor.pool, old_pool)
        valid = self.work_item()
        executor.submit(valid)
        executor.get_result(valid)
        self.assertEqual(valid.preprocess_result, ["image"])

    def test_rebuild_reaps_workers_when_writer_blocks_on_orphaned_input_lock(self):
        executor = self.real_executor()
        old_pool = executor.pool
        victim = old_pool._pool[0]

        def reader_lock_is_held():
            lock = old_pool._inqueue._rlock
            if lock.acquire(block=False):
                lock.release()
                return False
            return True

        self.wait_until(reader_lock_is_held, "worker never took the input read lock")
        victim.kill()
        self.assertEqual(wait([victim.sentinel], timeout=5), [victim.sentinel])
        self.wait_until(
            lambda: any(worker is not victim for worker in old_pool._pool),
            "pool did not replace the stopped worker",
        )
        self.assertIsNotNone(victim.exitcode)
        blocked_workers = list(old_pool._pool)
        old_pids = [worker.pid for worker in blocked_workers]
        work = self.work_item("x" * (4 * 1024 * 1024))
        executor.submit(work)

        def task_writer_is_blocked():
            frame = sys._current_frames().get(old_pool._task_handler.ident)
            while frame is not None:
                if frame.f_code.co_name == "_send":
                    return True
                frame = frame.f_back
            return False

        self.wait_until(task_writer_is_blocked, "task writer did not fill the pipe")
        work.remaining_timeout_s = lambda: 0.01
        with self.assertRaisesRegex(TimeoutError, "Preprocessing timeout"):
            executor.get_result(work)

        self.assertIsNot(executor.pool, old_pool)
        for worker in blocked_workers:
            self.assertFalse(worker.is_alive())
            self.assertIsNotNone(worker.exitcode)
        for pid in old_pids:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        self.assertFalse(old_pool._worker_handler.is_alive())
        self.assertFalse(old_pool._task_handler.is_alive())
        self.assertFalse(old_pool._result_handler.is_alive())
        valid = self.work_item()
        executor.submit(valid)
        executor.get_result(valid)
        self.assertEqual(valid.preprocess_result, ["image"])

    def wait_until(self, condition, message):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.01)
        self.fail(message)

    def test_local_preprocess_uses_config_and_rejects_expired_work(self):
        config = SimpleNamespace(prefix="local")
        preprocess = Mock(side_effect=lambda inputs, config: [config.prefix, *inputs])
        executor = LocalPreprocessExecutor(preprocess, config, {})
        expired = self.work_item(deadline=time.monotonic() - 1)
        with self.assertRaises(FtRuntimeException) as raised:
            executor.submit(expired)
        self.assertEqual(
            raised.exception.exception_type, ExceptionType.GENERATE_TIMEOUT
        )
        preprocess.assert_not_called()
        valid = self.work_item()
        executor.submit(valid)
        executor.get_result(valid)
        self.assertEqual(valid.preprocess_result, ["local", "image"])
        preprocess.assert_called_once_with(["image"], config)

    def test_invalid_image_keeps_pool_and_other_queued_request(self):
        executor = self.real_executor()
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
