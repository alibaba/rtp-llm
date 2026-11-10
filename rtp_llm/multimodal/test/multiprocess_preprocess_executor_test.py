import pickle
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
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


if __name__ == "__main__":
    unittest.main()
