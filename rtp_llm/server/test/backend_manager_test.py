import gc
import signal
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.server import backend_manager as backend_manager_module
from rtp_llm.server.backend_manager import BackendManager
from rtp_llm.start_backend_server import local_rank_start


class RecordingEngine(BaseEngine):
    def __init__(self, events, stop_error=None):
        self.events = events
        self.stop_error = stop_error
        self.started = True

    def _start(self):
        pass

    def _stop(self):
        self.events.append("engine")
        if self.stop_error is not None:
            raise self.stop_error


class RecordingDistributedServer:
    def __init__(self, events):
        self.events = events
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        self.events.append("store")


class BackendManagerTest(unittest.TestCase):
    def _manager(self, events, stop_error=None):
        manager = object.__new__(BackendManager)
        manager.engine = RecordingEngine(events, stop_error)
        manager._shutdown_requested = Mock()
        manager._stop_lock = threading.Lock()
        manager._stopped = False
        manager._stop_error = None
        manager._gc_frozen = True
        manager._owns_distributed_environment = True
        manager._distributed_server = RecordingDistributedServer(events)
        return manager

    def test_serve_forever_waits_directly_for_shutdown_event(self):
        manager = self._manager([])

        manager.serve_forever()

        manager._shutdown_requested.wait.assert_called_once_with()
        manager._shutdown_requested.is_set.assert_not_called()

    def test_stop_releases_resources_in_dependency_order_and_is_idempotent(self):
        events = []
        manager = self._manager(events)

        with patch.object(gc, "unfreeze", side_effect=lambda: events.append("gc")), patch(
            "rtp_llm.server.backend_manager._reset_moriep_wrapper",
            side_effect=lambda: events.append("mori"),
            create=True,
        ), patch(
            "rtp_llm.server.backend_manager._reset_deepep_wrapper",
            side_effect=lambda: events.append("deepep"),
            create=True,
        ), patch.object(
            backend_manager_module,
            "destroy_distributed_environment",
            side_effect=lambda: events.append("distributed"),
            create=True,
        ), patch.object(
            backend_manager_module._nfs_manager,
            "unmount_all",
            side_effect=lambda: events.append("nfs"),
        ):
            manager.stop()
            manager.stop()

        self.assertEqual(
            events,
            ["gc", "engine", "mori", "deepep", "distributed", "store", "nfs"],
        )
        self.assertIsNone(manager.engine)
        self.assertIsNone(manager._distributed_server)

    def test_concurrent_stop_releases_resources_once(self):
        events = []
        manager = self._manager(events)
        start_stop = threading.Barrier(3)
        stop_errors = []

        def stop_manager():
            start_stop.wait()
            try:
                manager.stop()
            except Exception as error:
                stop_errors.append(error)

        threads = [threading.Thread(target=stop_manager) for _ in range(2)]
        with patch.object(gc, "unfreeze"), patch(
            "rtp_llm.server.backend_manager._reset_moriep_wrapper"
        ) as reset_mori, patch(
            "rtp_llm.server.backend_manager._reset_deepep_wrapper"
        ) as reset_deepep, patch.object(
            backend_manager_module, "destroy_distributed_environment"
        ) as destroy_distributed, patch.object(
            backend_manager_module._nfs_manager, "unmount_all"
        ) as unmount_all:
            for thread in threads:
                thread.start()
            start_stop.wait()
            for thread in threads:
                thread.join(timeout=5)

        self.assertEqual(stop_errors, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(events, ["engine", "store"])
        reset_mori.assert_called_once_with()
        reset_deepep.assert_called_once_with()
        destroy_distributed.assert_called_once_with()
        unmount_all.assert_called_once_with()

    def test_ready_is_false_after_shutdown_is_requested(self):
        manager = self._manager([])
        manager._shutdown_requested.is_set.return_value = True

        self.assertFalse(manager.ready())

    def test_stop_finishes_cleanup_before_reporting_engine_error(self):
        events = []
        manager = self._manager(events, RuntimeError("engine stop failed"))

        with patch.object(gc, "unfreeze", side_effect=lambda: events.append("gc")), patch(
            "rtp_llm.server.backend_manager._reset_moriep_wrapper",
            side_effect=lambda: events.append("mori"),
            create=True,
        ), patch(
            "rtp_llm.server.backend_manager._reset_deepep_wrapper",
            side_effect=lambda: events.append("deepep"),
            create=True,
        ), patch.object(
            backend_manager_module,
            "destroy_distributed_environment",
            side_effect=lambda: events.append("distributed"),
            create=True,
        ), patch.object(
            backend_manager_module._nfs_manager,
            "unmount_all",
            side_effect=lambda: events.append("nfs"),
        ):
            with self.assertRaisesRegex(RuntimeError, "engine"):
                manager.stop()

        self.assertEqual(
            events,
            ["gc", "engine", "mori", "deepep", "distributed", "store", "nfs"],
        )

    def test_local_rank_start_stops_partial_backend_and_restores_handlers(self):
        manager = Mock()
        manager.start.side_effect = RuntimeError("startup failed")
        configs = SimpleNamespace(
            parallelism_config=SimpleNamespace(world_size=1, local_rank=0),
            ffn_disaggregate_config=SimpleNamespace(),
            prefill_cp_config=SimpleNamespace(),
            server_config=Mock(),
            distribute_config=Mock(),
        )
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        previous_sigint = signal.getsignal(signal.SIGINT)

        try:
            with patch(
                "rtp_llm.server.backend_manager.BackendManager", return_value=manager
            ), patch(
                "rtp_llm.utils.util.copy_gemm_config"
            ), patch(
                "rtp_llm.start_backend_server.set_parallelism_config"
            ), patch(
                "rtp_llm.start_backend_server.setup_cuda_device_and_accl_env"
            ), patch(
                "rtp_llm.start_backend_server.set_global_controller"
            ):
                with self.assertRaisesRegex(RuntimeError, "startup failed"):
                    local_rank_start(Mock(), configs)

            manager.stop.assert_called_once_with()
            self.assertIs(signal.getsignal(signal.SIGTERM), previous_sigterm)
            self.assertIs(signal.getsignal(signal.SIGINT), previous_sigint)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)

    def test_local_rank_start_does_not_lose_pre_initialization_signal(self):
        manager = Mock()
        configs = SimpleNamespace(
            parallelism_config=SimpleNamespace(world_size=1, local_rank=0),
            ffn_disaggregate_config=SimpleNamespace(),
            prefill_cp_config=SimpleNamespace(),
            server_config=Mock(),
            distribute_config=Mock(),
        )
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        previous_sigint = signal.getsignal(signal.SIGINT)

        try:
            with patch(
                "rtp_llm.server.backend_manager.BackendManager", return_value=manager
            ), patch(
                "rtp_llm.utils.util.copy_gemm_config",
                side_effect=lambda: signal.raise_signal(signal.SIGTERM),
            ), patch(
                "rtp_llm.start_backend_server.set_parallelism_config"
            ), patch(
                "rtp_llm.start_backend_server.setup_cuda_device_and_accl_env"
            ), patch(
                "rtp_llm.start_backend_server.set_global_controller"
            ):
                local_rank_start(Mock(), configs)

            manager.start.assert_not_called()
            manager.stop.assert_called_once_with()
            self.assertIs(signal.getsignal(signal.SIGTERM), previous_sigterm)
            self.assertIs(signal.getsignal(signal.SIGINT), previous_sigint)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    unittest.main()
