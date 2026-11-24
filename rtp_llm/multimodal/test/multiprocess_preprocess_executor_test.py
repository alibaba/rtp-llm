import io
import multiprocessing
import pickle
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from PIL import Image, UnidentifiedImageError

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.metrics import AccMetrics
from rtp_llm.multimodal import mm_preprocess
from rtp_llm.multimodal.mm_preprocess import (
    MultiprocessPreprocessExecutor,
    _PreprocessTaskError,
    _worker_process_task,
)
from rtp_llm.multimodal.vit_metrics import record_vit_preprocess_value


class _FailingFuture:
    def __init__(self, error):
        self._error = error

    def get(self, timeout=None):
        raise self._error


class _WorkItem:
    def __init__(self, error):
        self.future = _FailingFuture(error)
        self.embedding_result = None
        self.mm_timeout_ms = 60000
        self.media_traces = ()

    def remaining_timeout_s(self):
        return 60.0


class MultiprocessPreprocessExecutorTest(unittest.TestCase):
    def test_image_pixel_limit_survives_worker_transport_as_input_error(self):
        data = io.BytesIO()
        Image.new("RGB", (8, 8)).save(data, format="PNG")
        data.seek(0)

        def preprocess(_inputs, _config):
            with Image.open(data):
                self.fail("Pillow should reject the image in open()")

        with patch.object(Image, "MAX_IMAGE_PIXELS", 16), patch.multiple(
            mm_preprocess,
            _worker_preprocess_func=preprocess,
            _worker_vit_config=SimpleNamespace(),
            _worker_preprocess_params={},
        ), self.assertRaises(_PreprocessTaskError) as raised:
            _worker_process_task([])
        error = pickle.loads(pickle.dumps(raised.exception)).error
        self.assertIsInstance(error, FtRuntimeException)
        self.assertEqual(
            error.exception_type, ExceptionType.MM_IMAGE_PIXEL_LIMIT_EXCEEDED
        )
        self.assertIn("64 pixels", error.message)
        self.assertIn("32 pixels", error.message)

    def test_download_error_survives_worker_transport_without_becoming_timeout(self):
        for with_samples in (False, True):
            with self.subTest(with_samples=with_samples):

                def preprocess(_inputs, _config, **_params):
                    if with_samples:
                        record_vit_preprocess_value(
                            AccMetrics.MM_DOWNLOAD_PROXY_QPS_METRIC, 1
                        )
                    raise FtRuntimeException(
                        ExceptionType.MM_DOWNLOAD_FAILED,
                        "Temporary multimodal download failure",
                    ) from TimeoutError("download timed out")

                mm_preprocess._worker_preprocess_func = preprocess
                mm_preprocess._worker_vit_config = SimpleNamespace()
                mm_preprocess._worker_preprocess_params = {}
                expected = _PreprocessTaskError if with_samples else FtRuntimeException
                with self.assertRaises(expected) as raised:
                    _worker_process_task([])
                error = pickle.loads(pickle.dumps(raised.exception))
                executor = MultiprocessPreprocessExecutor.__new__(
                    MultiprocessPreprocessExecutor
                )
                executor._preprocess_queue_lock = threading.Lock()
                executor._pending_preprocess_tasks = set()
                executor._rebuild_pool = Mock()
                executor._record_timeout_locked = Mock()
                with self.assertRaises(FtRuntimeException) as received:
                    executor.get_result(_WorkItem(error))
                self.assertEqual(
                    received.exception.exception_type, ExceptionType.MM_DOWNLOAD_FAILED
                )
                self.assertEqual(
                    received.exception.message, "Temporary multimodal download failure"
                )
                executor._rebuild_pool.assert_not_called()
                executor._record_timeout_locked.assert_not_called()

    def test_outer_preprocess_wait_timeout_remains_a_timeout(self):
        executor = MultiprocessPreprocessExecutor.__new__(
            MultiprocessPreprocessExecutor
        )
        executor._pool_lock = threading.Lock()
        executor._preprocess_queue_lock = threading.Lock()
        executor._pending_preprocess_tasks = set()
        executor._record_timeout_locked = Mock()
        executor._rebuild_pool = Mock()
        item = _WorkItem(mm_preprocess.multiprocessing.pool.TimeoutError())
        with self.assertRaisesRegex(
            TimeoutError, "Preprocessing timeout after 60000ms"
        ):
            executor.get_result(item)
        executor._record_timeout_locked.assert_called_once()
        executor._rebuild_pool.assert_not_called()

    def test_counter_samples_survive_success_and_failed_worker_results(self):
        metric = AccMetrics.MM_DOWNLOAD_PROXY_QPS_METRIC
        for failed in (False, True):
            with self.subTest(failed=failed), patch.object(
                mm_preprocess.kmonitor, "report"
            ) as report:

                def preprocess(_inputs, _config, **_params):
                    record_vit_preprocess_value(metric, 1, {"region": "cn-beijing"})
                    if failed:
                        raise ValueError("download failed")
                    return "embedding input"

                mm_preprocess._worker_preprocess_func = preprocess
                mm_preprocess._worker_vit_config = SimpleNamespace()
                mm_preprocess._worker_preprocess_params = {}
                if failed:
                    with self.assertRaises(_PreprocessTaskError) as raised:
                        _worker_process_task([])
                    error = pickle.loads(pickle.dumps(raised.exception))
                    report.assert_not_called()
                    executor = MultiprocessPreprocessExecutor.__new__(
                        MultiprocessPreprocessExecutor
                    )
                    executor._preprocess_queue_lock = threading.Lock()
                    executor._pending_preprocess_tasks = set()
                    with self.assertRaisesRegex(ValueError, "download failed"):
                        executor.get_result(_WorkItem(error))
                else:
                    payload = pickle.loads(pickle.dumps(_worker_process_task([])))
                    report.assert_not_called()
                    result, _, _ = mm_preprocess._unpack_preprocess_result(payload)
                    self.assertEqual(result, "embedding input")
                self.assertEqual(
                    [c for c in report.call_args_list if c.args[0] == metric],
                    [call(metric, 1.0, {"region": "cn-beijing"})],
                )

    def test_local_failure_reports_samples_and_preserves_error(self):
        metric = AccMetrics.MM_DOWNLOAD_PROXY_QPS_METRIC

        def fail(_inputs, _config, **_params):
            record_vit_preprocess_value(metric, 1, {"region": "cn-beijing"})
            raise ValueError("download failed")

        with patch.object(mm_preprocess.kmonitor, "report") as report:
            executor = mm_preprocess.LocalPreprocessExecutor(
                fail, SimpleNamespace(), {}
            )
            item = SimpleNamespace(
                should_preprocess=True,
                capture_inspection_sources=False,
                mm_inputs=[],
                region=None,
            )
            with self.assertRaisesRegex(ValueError, "download failed"):
                executor.submit(item)
            report.assert_any_call(metric, 1.0, {"region": "cn-beijing"})

    def setUp(self):
        self._saved = (
            mm_preprocess._worker_preprocess_func,
            mm_preprocess._worker_vit_config,
            mm_preprocess._worker_preprocess_params,
        )

    def tearDown(self):
        (
            mm_preprocess._worker_preprocess_func,
            mm_preprocess._worker_vit_config,
            mm_preprocess._worker_preprocess_params,
        ) = self._saved

    def test_worker_wraps_task_oserror(self):
        def fail(_inputs, _config, **_params):
            raise OSError("not an image")

        mm_preprocess._worker_preprocess_func = fail
        mm_preprocess._worker_vit_config = SimpleNamespace()
        mm_preprocess._worker_preprocess_params = {}

        with self.assertRaises(_PreprocessTaskError) as raised:
            _worker_process_task(["bad-image"])

        self.assertIsInstance(raised.exception.error, OSError)

    def test_get_result_unwraps_task_error_without_rebuilding_pool(self):
        executor = MultiprocessPreprocessExecutor.__new__(
            MultiprocessPreprocessExecutor
        )
        executor._pool_lock = threading.Lock()
        executor._preprocess_queue_lock = threading.Lock()
        executor._pending_preprocess_tasks = set()
        executor._consecutive_timeouts = 0
        executor._max_consecutive_timeouts = 1
        executor._rebuild_pool = Mock()

        original = OSError("not an image")
        work_item = _WorkItem(_PreprocessTaskError(original))

        with self.assertRaises(OSError) as raised:
            executor.get_result(work_item)

        self.assertIs(raised.exception, original)
        executor._rebuild_pool.assert_not_called()

    def test_queue_wait_timeout_without_progress_counts_toward_rebuild(self):
        executor = MultiprocessPreprocessExecutor.__new__(
            MultiprocessPreprocessExecutor
        )
        executor._pool_lock = threading.Lock()
        executor._consecutive_timeouts = 0
        executor._max_consecutive_timeouts = 1
        executor._completed_tasks = 0
        executor._last_timeout_completed = 0
        executor._rebuild_pool = Mock()

        executor._on_wait_timeout([])

        executor._rebuild_pool.assert_called_once()
        self.assertEqual(executor._consecutive_timeouts, 0)

    def test_queue_wait_timeout_with_progress_keeps_pool(self):
        executor = MultiprocessPreprocessExecutor.__new__(
            MultiprocessPreprocessExecutor
        )
        executor._pool_lock = threading.Lock()
        executor._consecutive_timeouts = 0
        executor._max_consecutive_timeouts = 1
        executor._completed_tasks = 1
        executor._last_timeout_completed = 0
        executor._rebuild_pool = Mock()

        executor._on_wait_timeout([])

        executor._rebuild_pool.assert_not_called()
        self.assertEqual(executor._consecutive_timeouts, 0)
        self.assertEqual(executor._last_timeout_completed, 1)


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


class PoolRecoveryTest(unittest.TestCase):
    @staticmethod
    def work_item(value="image"):
        return SimpleNamespace(
            should_preprocess=True,
            mm_inputs=[value],
            future=None,
            embedding_result=None,
            mm_timeout_ms=60000,
            remaining_timeout_s=lambda: 60,
            capture_inspection_sources=False,
            deadline=None,
            region=None,
            media_traces=(),
            preprocess_ready_event=threading.Event(),
        )

    @staticmethod
    def executor(old_pool, new_pool):
        executor = MultiprocessPreprocessExecutor.__new__(
            MultiprocessPreprocessExecutor
        )
        executor.pool = old_pool
        pool_factory = Mock(return_value=new_pool)

        def create_pool():
            executor.pool = pool_factory()

        executor._create_pool = create_pool
        executor.pool_factory = pool_factory
        executor._inspection_channel = None
        executor.vit_config = SimpleNamespace(mm_preprocess_max_workers=1)
        executor.preprocess_params = {}
        executor.preprocess_func = preprocess_image
        executor._pool_lock = threading.Lock()
        executor._preprocess_queue_lock = threading.Lock()
        executor._pending_preprocess_tasks = set()
        executor._next_preprocess_task_id = 0
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
        executor.pool_factory.assert_called_once()
        new_pool.apply_async.assert_called_once()
        self.assertIs(work.future, new_pool.apply_async.return_value)

    def test_submission_recovers_after_replacement_creation_failed(self):
        old_pool, new_pool = Mock(), Mock()
        old_pool.apply_async.side_effect = BrokenPipeError("broken pool pipe")
        executor = self.executor(old_pool, new_pool)
        executor.pool_factory.side_effect = [
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
        executor = self.executor(old_pool, new_pool)
        work = self.work_item()
        executor.submit(work)
        work.future.get.side_effect = EOFError("broken result pipe")
        with self.assertRaisesRegex(EOFError, "broken result pipe"):
            executor.get_result(work)
        old_pool.terminate.assert_called_once()
        old_pool.join.assert_called_once()
        self.assertIs(executor.pool, new_pool)
        self.assertEqual(executor._pending_preprocess_tasks, set())

    def test_retry_failure_clears_queue_accounting(self):
        old_pool, new_pool = Mock(), Mock()
        old_pool.apply_async.side_effect = BrokenPipeError("broken pool pipe")
        new_pool.apply_async.side_effect = OSError("retry failed")
        executor = self.executor(old_pool, new_pool)
        with self.assertRaisesRegex(OSError, "retry failed"):
            executor.submit(self.work_item())
        old_pool.apply_async.assert_called_once()
        new_pool.apply_async.assert_called_once()
        self.assertEqual(executor._pending_preprocess_tasks, set())

    def test_invalid_image_keeps_pool_and_other_queued_request(self):
        config = VitConfig()
        config.mm_preprocess_max_workers = 1
        config.url_cache_item_num = 0
        executor = MultiprocessPreprocessExecutor(
            multiprocessing.get_context("spawn"),
            config,
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
    unittest.main()
