import importlib.util
import pathlib
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

SOURCE = pathlib.Path(__file__).resolve().parents[2] / 'adapters/vllm/VLLMEngineTransport.py'

def load_transport():
    dependencies = {
        'navi_sched.util': {'naviSchedModule': Mock()},
        'navi_sched.adapters.vllm.NaviSchedAdapterConfig': {'NaviSchedAdapterConfig': Mock()},
        'navi_sched.adapters.vllm.NaviSchedClient': {'NaviSchedClient': Mock()},
        'navi_sched.adapters.vllm.VLLMAdapterUtils': {'NAVI_SCHED_MEDIA_ERROR_PREFIX': '', 'make_kmon_tags': Mock()},
        'navi_sched.adapters.vllm.VLLMCachePolicy': {'get_vllm_hash_block_size': Mock()},
    }
    modules = {}
    for name, attributes in dependencies.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module
    name = 'navi_sched.adapters.vllm.VLLMEngineTransport'
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.VLLMEngineTransport

class TransportShutdownTest(unittest.TestCase):
    def make_transport(self):
        transport_class = load_transport()
        transport = transport_class.__new__(transport_class)
        transport._closed = False
        transport._local_output_task = Mock()
        transport._remote_output_task = Mock()
        transport._response_poll_lock = threading.Lock()
        transport._client = Mock()
        transport._response_executor = Mock()
        transport._request_ids = {'request': 1}
        transport._external_request_ids = {1: 'request'}
        transport._render_continuation = Mock()
        transport._engine_core = Mock()
        return transport

    def test_ppu_shutdown_with_none_timeout_cleans_once(self):
        transport = self.make_transport()
        transport.shutdown(timeout=None)
        transport.shutdown(timeout=None)
        self.assertTrue(transport._closed)
        self.assertEqual({}, transport._request_ids)
        self.assertEqual({}, transport._external_request_ids)
        transport._client.close.assert_called_once_with()
        transport._engine_core.shutdown.assert_called_once_with()
        transport._response_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        transport._local_output_task.cancel.assert_called_once_with()
        transport._remote_output_task.cancel.assert_called_once_with()
        transport._render_continuation.close.assert_called_once_with()

    def test_ppu_shutdown_forwards_explicit_timeout(self):
        transport = self.make_transport()
        transport.shutdown(timeout=0.25)
        transport._engine_core.shutdown.assert_called_once_with(timeout=0.25)

    def test_legacy_shutdown_calls_core_without_keywords(self):
        transport = self.make_transport()
        calls = []
        class LegacyCore:
            def shutdown(self):
                calls.append('shutdown')
        transport._engine_core = LegacyCore()
        transport.shutdown()
        self.assertEqual(['shutdown'], calls)

if __name__ == '__main__':
    unittest.main(verbosity=2)
