import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

from rtp_llm.async_decoder_engine.embedding.embedding_engine import EmbeddingCppEngine
from rtp_llm.server.backend_manager import BackendManager, _reset_ep_wrappers


class FakeEngine:
    def __init__(self, stop_error=None):
        self.stop_error = stop_error
        self.request_stop = Mock()

    def stop(self):
        if self.stop_error is not None:
            raise self.stop_error


class BackendManagerShutdownTest(unittest.TestCase):
    @staticmethod
    def make_manager(engine, ready_event=None):
        manager = object.__new__(BackendManager)
        manager.engine = engine
        manager._shutdown_requested = threading.Event()
        manager._shutdown_lock = threading.RLock()
        manager._serving = True
        manager._stopping = False
        manager._stopped = False
        manager._shutdown_ready_event = ready_event
        return manager

    def test_scheduler_ready_ack_precedes_blocking_stop(self):
        order = []
        engine = FakeEngine()
        engine.request_stop.side_effect = lambda: order.append("request-stop")
        engine.stop = Mock(side_effect=lambda: order.append("stop"))
        ready_event = Mock()
        ready_event.set.side_effect = lambda: order.append("ready")
        manager = self.make_manager(engine, ready_event)

        with patch("rtp_llm.server.backend_manager.BaseEngine", FakeEngine), patch(
            "rtp_llm.server.backend_manager._nfs_manager.unmount_all"
        ), patch(
            "rtp_llm.server.backend_manager.distributed_environment_initialized",
            return_value=False,
        ):
            manager.stop()

        self.assertEqual(order, ["request-stop", "ready", "stop"])

    def test_embedding_engine_request_stop_delegates_to_cpp_scheduler(self):
        engine = object.__new__(EmbeddingCppEngine)
        engine.cpp_engine = Mock()

        engine.request_stop()

        engine.cpp_engine.request_stop.assert_called_once_with()

    def test_embedding_tp_shutdown_ack_precedes_blocking_cpp_stop(self):
        order = []
        engine = object.__new__(EmbeddingCppEngine)
        engine.cpp_engine = Mock()
        engine.cpp_engine.request_stop.side_effect = lambda: order.append(
            "embedding-request-stop"
        )
        engine.cpp_engine.stop.side_effect = lambda: order.append("embedding-stop")
        engine.mm_process_engine = None
        ready_event = Mock()
        ready_event.set.side_effect = lambda: order.append("ready")
        manager = self.make_manager(engine, ready_event)

        with patch("rtp_llm.server.backend_manager._nfs_manager.unmount_all"), patch(
            "rtp_llm.server.backend_manager.distributed_environment_initialized",
            return_value=False,
        ):
            manager.stop()

        self.assertEqual(
            order,
            ["embedding-request-stop", "ready", "embedding-stop"],
        )

    def test_request_shutdown_propagates_engine_stop_failure(self):
        manager = self.make_manager(FakeEngine(RuntimeError("engine stop failed")))

        with patch("rtp_llm.server.backend_manager.BaseEngine", FakeEngine), patch(
            "rtp_llm.server.backend_manager._nfs_manager.unmount_all"
        ), patch(
            "rtp_llm.server.backend_manager.distributed_environment_initialized",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "engine stop failed"):
                manager.request_shutdown()

        self.assertTrue(manager._stopped)

    def test_request_shutdown_propagates_distributed_teardown_failure(self):
        manager = self.make_manager(FakeEngine())

        with patch("rtp_llm.server.backend_manager.BaseEngine", FakeEngine), patch(
            "rtp_llm.server.backend_manager._nfs_manager.unmount_all"
        ), patch(
            "rtp_llm.server.backend_manager.distributed_environment_initialized",
            return_value=True,
        ), patch(
            "rtp_llm.server.backend_manager.destroy_distributed_environment",
            side_effect=RuntimeError("distributed teardown failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "distributed teardown failed"):
                manager.request_shutdown()

        self.assertTrue(manager._stopped)

    def test_ep_wrappers_reset_before_distributed_teardown(self):
        order = []
        manager = self.make_manager(FakeEngine())

        with patch("rtp_llm.server.backend_manager.BaseEngine", FakeEngine), patch(
            "rtp_llm.server.backend_manager._nfs_manager.unmount_all"
        ), patch(
            "rtp_llm.server.backend_manager.distributed_environment_initialized",
            return_value=True,
        ), patch(
            "rtp_llm.server.backend_manager._reset_ep_wrappers",
            side_effect=lambda: order.append("ep"),
        ), patch(
            "rtp_llm.server.backend_manager.destroy_distributed_environment",
            side_effect=lambda: order.append("distributed"),
        ):
            manager.stop()

        self.assertEqual(order, ["ep", "distributed"])

    def test_reset_ep_wrappers_releases_each_initialized_backend(self):
        deepep = Mock()
        deepep.is_initialized.return_value = True
        moriep = Mock()
        moriep.is_initialized.return_value = True
        deepep_module = types.ModuleType("deepep_wrapper")
        deepep_module.DeepEPWrapper = deepep
        moriep_module = types.ModuleType("moriep_wrapper")
        moriep_module.MoriEPWrapper = moriep

        with patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.distributed.deepep_wrapper": deepep_module,
                "rtp_llm.models_py.distributed.moriep_wrapper": moriep_module,
            },
        ):
            _reset_ep_wrappers()

        deepep.reset.assert_called_once_with()
        moriep.reset.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
