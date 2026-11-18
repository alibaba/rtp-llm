"""Touch hints must proceed independently of cache snapshot completion."""
import ast
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import threading
import time
import unittest


def observer():
    path = Path(__file__).parent / 'VLLMEngineObserver.py'
    names = {'_drain_v6d_cache_touches', '_on_v6d_cache_touch_done', '_poll_v6d_cache_source'}
    methods = [n for c in ast.parse(path.read_text()).body if isinstance(c, ast.ClassDef)
               for n in c.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    unit = ast.Module(body=[ast.ClassDef(name='Observer', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    scope = dict(logger=Mock(), time=time, _CACHE_SNAPSHOT_INTERVAL_NS=100000000)
    exec(compile(ast.fix_missing_locations(unit), str(path), 'exec'), scope)
    value = scope['Observer']()
    value._forward_stop = threading.Event()
    value._v6d_snapshot_future = Future()
    value._v6d_scheduler = SimpleNamespace(submit_local_cache_touches=Mock(return_value=Future()))
    value._bridge = SimpleNamespace(take_v6d_cache_touches=Mock(return_value=[(1, [11,22]),(2,[22,33])]))
    return value, scope['logger']


class TouchDrainTest(unittest.TestCase):
    def test_pending_snapshot_does_not_block_touch_submission_or_wait_for_result(self):
        value, log = observer()
        value._poll_v6d_cache_source()
        value._v6d_scheduler.submit_local_cache_touches.assert_called_once_with((11,22,33))
        self.assertFalse(value._v6d_snapshot_future.done())
        self.assertFalse(value._v6d_scheduler.submit_local_cache_touches.return_value.done())
        log.exception.assert_not_called()

    def test_submit_failure_does_not_escape_to_inference_loop(self):
        value, log = observer()
        value._v6d_scheduler.submit_local_cache_touches.side_effect = RuntimeError('closed')
        value._poll_v6d_cache_source()
        log.exception.assert_called_once()

    def test_failed_storage_node_and_failed_future_are_both_ignored(self):
        value, log = observer()
        value._drain_v6d_cache_touches()
        value._v6d_scheduler.submit_local_cache_touches.return_value.set_result([{'touched':2},TimeoutError()])
        log.warning.assert_called_once()
        failed = Future()
        failed.set_exception(RuntimeError('shutdown'))
        value._on_v6d_cache_touch_done(failed)
        log.exception.assert_called_once()

    def test_disabled_source_and_empty_queue_do_not_submit(self):
        value, log = observer()
        value._bridge.take_v6d_cache_touches.return_value=[]
        value._drain_v6d_cache_touches()
        value._v6d_scheduler.submit_local_cache_touches.assert_not_called()
        value._forward_stop.set()
        value._bridge.take_v6d_cache_touches.reset_mock()
        value._drain_v6d_cache_touches()
        value._bridge.take_v6d_cache_touches.assert_not_called()


if __name__ == '__main__':
    unittest.main()
