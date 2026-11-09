import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from rtp_llm.multimodal import mm_preprocess
from rtp_llm.multimodal.mm_preprocess import (
    MultiprocessPreprocessExecutor,
    _PreprocessTaskError,
    _worker_process_task,
)


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

    def remaining_timeout_s(self):
        return 60.0


class MultiprocessPreprocessExecutorTest(unittest.TestCase):
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
