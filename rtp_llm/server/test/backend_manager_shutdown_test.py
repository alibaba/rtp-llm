import multiprocessing
import threading
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

from torch.distributed import TCPStore

from rtp_llm.async_decoder_engine.embedding.embedding_engine import EmbeddingCppEngine
from rtp_llm.async_decoder_engine.rpc_engine import LanguageCppEngine
from rtp_llm.ops.rtp_llm.rtp_llm_op import RtpLLMOp
from rtp_llm.server.backend_manager import BackendManager


class FakeEngine:
    def __init__(self, stop_error=None):
        self.stop_error = stop_error
        self.request_stop = Mock()
        self.stop_writeback = Mock()

    def stop(self):
        if self.stop_error is not None:
            raise self.stop_error


def stop_writeback_rank(
    rank,
    world_size,
    port,
    port_queue,
    ready,
    start_leaders,
    draining,
    drained,
    release,
    stopped,
):
    store = TCPStore(
        "127.0.0.1",
        port,
        world_size,
        rank == 0,
        timeout=timedelta(seconds=15),
        wait_for_workers=False,
    )
    if rank == 0:
        port_queue.put(store.port)
    engine = FakeEngine()
    if rank % 2 == 0:
        if not start_leaders.wait(15):
            raise RuntimeError("leader was not signalled")

        def drain():
            draining[rank // 2].set()
            if not release[rank // 2].wait(15):
                raise RuntimeError("writeback transfer did not stop")
            if stopped[rank + 1].is_set():
                raise RuntimeError("follower closed worker RPC before drain")
            drained[rank // 2].set()

        engine.stop_writeback.side_effect = drain
    engine.stop = Mock(side_effect=stopped[rank].set)
    manager = BackendManagerShutdownTest.make_manager(engine, ready)
    manager.py_env_configs.cache_store_config.p2p_writeback_enable = True
    manager.py_env_configs.parallelism_config = SimpleNamespace(
        world_size=world_size, world_rank=rank
    )
    manager._distributed_server = SimpleNamespace(store=store)
    with patch("rtp_llm.server.backend_manager.BaseEngine", FakeEngine), patch(
        "rtp_llm.server.backend_manager._nfs_manager.unmount_all"
    ), patch(
        "rtp_llm.server.backend_manager.distributed_environment_initialized",
        return_value=False,
    ):
        manager.stop()


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
        manager.py_env_configs = SimpleNamespace(
            cache_store_config=SimpleNamespace(p2p_writeback_enable=False),
            parallelism_config=SimpleNamespace(world_size=1, world_rank=0),
        )
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
        engine.stop_writeback.assert_not_called()

    def test_single_rank_drains_writeback_before_engine_stop(self):
        order = []
        engine = FakeEngine()
        engine.stop_writeback.side_effect = lambda: order.append("drain")
        engine.stop = Mock(side_effect=lambda: order.append("stop"))
        manager = self.make_manager(engine)
        manager.py_env_configs.cache_store_config.p2p_writeback_enable = True
        with patch("rtp_llm.server.backend_manager.BaseEngine", FakeEngine), patch(
            "rtp_llm.server.backend_manager._nfs_manager.unmount_all"
        ), patch(
            "rtp_llm.server.backend_manager.distributed_environment_initialized",
            return_value=False,
        ):
            manager.stop()
        self.assertEqual(order, ["drain", "stop"])

    def test_language_engine_writeback_drain_reaches_cpp_without_stopping_rpc(self):
        engine = object.__new__(LanguageCppEngine)
        engine.rtp_llm_op_ = object.__new__(RtpLLMOp)
        engine.rtp_llm_op_.ft_op = Mock()
        engine.stop_writeback()
        engine.rtp_llm_op_.ft_op.stop_writeback.assert_called_once_with()
        engine.rtp_llm_op_.ft_op.stop.assert_not_called()

    def test_followers_keep_worker_rpc_until_all_leaders_drain(self):
        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                self.run_writeback_shutdown(world_size)

    def run_writeback_shutdown(self, world_size):
        ctx = multiprocessing.get_context("spawn")
        port_queue = ctx.Queue()
        start_leaders = ctx.Event()
        ready = [ctx.Event() if rank % 2 else None for rank in range(world_size)]
        draining = [ctx.Event() for _ in range(world_size // 2)]
        drained = [ctx.Event() for _ in range(world_size // 2)]
        release = [ctx.Event() for _ in range(world_size // 2)]
        stopped = [ctx.Event() for _ in range(world_size)]
        processes = []

        def start_rank(rank, port):
            process = ctx.Process(
                target=stop_writeback_rank,
                args=(
                    rank,
                    world_size,
                    port,
                    port_queue,
                    ready[rank],
                    start_leaders,
                    draining,
                    drained,
                    release,
                    stopped,
                ),
            )
            process.start()
            processes.append(process)

        try:
            start_rank(0, 0)
            port = port_queue.get(timeout=20)
            for rank in range(1, world_size):
                start_rank(rank, port)
            for rank in range(1, world_size, 2):
                self.assertTrue(ready[rank].wait(20))
                self.assertFalse(stopped[rank].is_set())
            start_leaders.set()
            for event in draining:
                self.assertTrue(event.wait(10))
            for event in stopped:
                self.assertFalse(event.wait(0.1))
            for group, event in enumerate(release):
                event.set()
                self.assertTrue(drained[group].wait(10))
                if group + 1 < len(release):
                    for stopped_event in stopped:
                        self.assertFalse(stopped_event.wait(0.1))
            for process in processes:
                process.join(20)
                self.assertEqual(process.exitcode, 0)
            self.assertTrue(all(event.is_set() for event in stopped))
        finally:
            start_leaders.set()
            for event in release:
                event.set()
            for process in processes:
                process.join(2)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            port_queue.close()
            port_queue.join_thread()

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

        with patch(
            "rtp_llm.server.backend_manager._nfs_manager.unmount_all"
        ), patch(
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


if __name__ == "__main__":
    unittest.main()
