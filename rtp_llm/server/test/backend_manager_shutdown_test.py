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
from rtp_llm.ops import RoleType
from rtp_llm.server.backend_manager import BackendManager


class FakeEngine:
    def __init__(self, stop_error=None):
        self.stop_error = stop_error
        self.drain = Mock()
        self.close_rpc = Mock()

    def stop(self, before_rpc_shutdown=None):
        self.drain()
        if before_rpc_shutdown is not None:
            before_rpc_shutdown(15000)
        if self.stop_error is not None:
            raise self.stop_error
        self.close_rpc()

def stop_engine_rank(
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

        engine.drain.side_effect = drain
    engine.close_rpc.side_effect = stopped[rank].set
    manager = BackendManagerShutdownTest.make_manager(engine)
    manager.py_env_configs.cache_store_config.p2p_writeback_enable = True
    manager.py_env_configs.parallelism_config = SimpleNamespace(
        world_size=world_size, world_rank=rank, pp_size=1
    )
    manager._distributed_server = SimpleNamespace(store=store)
    if ready is not None:
        ready.set()
    with patch("rtp_llm.utils.fuser._nfs_manager.unmount_all"):
        manager.stop()

class BackendManagerShutdownTest(unittest.TestCase):
    @staticmethod
    def make_manager(engine):
        manager = object.__new__(BackendManager)
        manager._stop_lock = threading.Lock()
        manager._stopped = False
        manager._shutdown_requested = threading.Event()
        manager._status_watcher_stop = threading.Event()
        manager._mark_not_serving = Mock()
        manager.engine = engine
        manager.py_env_configs = SimpleNamespace(
            cache_store_config=SimpleNamespace(p2p_writeback_enable=False),
            parallelism_config=SimpleNamespace(world_size=1, world_rank=0, pp_size=1),
            role_config=SimpleNamespace(role_type=RoleType.DECODE),
        )
        return manager

    def test_single_rank_stop_drains_writeback_before_rpc_shutdown(self):
        order = []
        engine = FakeEngine()
        engine.drain.side_effect = lambda: order.append("drain")
        engine.close_rpc.side_effect = lambda: order.append("close_rpc")
        manager = self.make_manager(engine)
        manager.py_env_configs.cache_store_config.p2p_writeback_enable = True
        with patch("rtp_llm.utils.fuser._nfs_manager.unmount_all"):
            manager.stop()
        self.assertEqual(order, ["drain", "close_rpc"])

    def test_language_engine_stop_passes_shutdown_barrier_to_cpp(self):
        engine = object.__new__(LanguageCppEngine)
        engine.mm_process_engine = None
        engine.rtp_llm_op_ = object.__new__(RtpLLMOp)
        engine.rtp_llm_op_.ft_op = Mock()
        barrier = Mock()
        engine.stop(barrier)
        engine.rtp_llm_op_.ft_op.stop.assert_called_once_with(barrier)
        barrier.assert_not_called()

    def test_language_engine_default_stop_keeps_existing_call(self):
        engine = object.__new__(LanguageCppEngine)
        engine.mm_process_engine = None
        engine.rtp_llm_op_ = object.__new__(RtpLLMOp)
        engine.rtp_llm_op_.ft_op = Mock()
        engine.stop()
        engine.rtp_llm_op_.ft_op.stop.assert_called_once_with()

    def test_embedding_engine_stop_skips_writeback_barrier_and_stops_engines(self):
        for via_manager in (False, True):
            with self.subTest(via_manager=via_manager):
                engine = object.__new__(EmbeddingCppEngine)
                engine.cpp_engine = Mock()
                mm_engine = Mock()
                engine.mm_process_engine = mm_engine
                manager = self.make_manager(engine)
                manager.py_env_configs.cache_store_config.p2p_writeback_enable = True
                manager.py_env_configs.parallelism_config.world_size = 2
                store = Mock()
                manager._distributed_server = SimpleNamespace(store=store)
                if via_manager:
                    with patch("rtp_llm.utils.fuser._nfs_manager.unmount_all"):
                        manager.stop()
                else:
                    engine._stop(manager._wait_for_writeback_stop)
                engine.cpp_engine.stop.assert_called_once_with()
                mm_engine.stop.assert_called_once_with()
                self.assertIsNone(engine.mm_process_engine)
                store.set.assert_not_called()
                store.wait.assert_not_called()

    def test_writeback_disabled_does_not_enter_rank_barrier(self):
        engine = FakeEngine()
        engine.stop = Mock(wraps=engine.stop)
        manager = self.make_manager(engine)
        manager.py_env_configs.parallelism_config.world_size = 2
        manager._distributed_server = SimpleNamespace(store=Mock())
        with patch("rtp_llm.utils.fuser._nfs_manager.unmount_all"):
            manager.stop()
        engine.stop.assert_called_once_with()
        manager._distributed_server.store.wait.assert_not_called()

    def test_pipeline_parallel_drains_all_workers_before_shutdown(self):
        engine = FakeEngine()
        engine.stop = Mock(wraps=engine.stop)
        manager = self.make_manager(engine)
        manager.py_env_configs.cache_store_config.p2p_writeback_enable = True
        manager.py_env_configs.parallelism_config.pp_size = 2
        manager.py_env_configs.parallelism_config.world_size = 2
        manager._distributed_server = SimpleNamespace(store=Mock())
        with patch("rtp_llm.utils.fuser._nfs_manager.unmount_all"):
            manager.stop()
        engine.stop.assert_called_once_with(manager._wait_for_writeback_stop)

    def test_missing_rank_uses_only_remaining_shutdown_budget(self):
        manager = self.make_manager(FakeEngine())
        manager.py_env_configs.parallelism_config.world_size = 2
        store = Mock()
        store.wait.side_effect = TimeoutError("rank never drained")
        manager._distributed_server = SimpleNamespace(store=store)
        with self.assertRaisesRegex(TimeoutError, "rank never drained"):
            manager._wait_for_writeback_stop(35)
        keys, timeout = store.wait.call_args.args
        self.assertEqual(keys, ["shutdown_writeback_drained_0", "shutdown_writeback_drained_1"])
        self.assertGreater(timeout.total_seconds(), 0)
        self.assertLessEqual(timeout.total_seconds(), 0.035)

    def test_rank_barrier_failure_does_not_close_rpc(self):
        engine = FakeEngine()
        manager = self.make_manager(engine)
        manager.py_env_configs.cache_store_config.p2p_writeback_enable = True
        manager.py_env_configs.parallelism_config.world_size = 2
        store = Mock()
        store.wait.side_effect = RuntimeError("rank barrier failed")
        manager._distributed_server = SimpleNamespace(store=store)
        with patch("rtp_llm.utils.fuser._nfs_manager.unmount_all") as unmount:
            with self.assertRaisesRegex(RuntimeError, "rank barrier failed"):
                manager.stop()
            unmount.assert_called_once_with()
        engine.drain.assert_called_once_with()
        engine.close_rpc.assert_not_called()

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
                target=stop_engine_rank,
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


if __name__ == "__main__":
    unittest.main()
