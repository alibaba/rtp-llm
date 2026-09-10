import multiprocessing
import signal
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from rtp_llm.start_backend_server import (
    _wait_for_ranks_startup,
    local_rank_start,
    multi_rank_start,
    start_backend_server,
)


class _AliveProcess:
    pid = 12345
    exitcode = None
    name = "rank-0"

    def __init__(self):
        self.alive = True
        self.terminated = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def join(self, timeout=None):
        return None


class _TrackedConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _StartThenFailProcess(_AliveProcess):
    def start(self):
        raise KeyboardInterrupt("signal after child creation")


class _StartFailureContext:
    def __init__(self):
        self.reader = _TrackedConnection()
        self.writer = _TrackedConnection()
        self.process = _StartThenFailProcess()

    def Pipe(self, duplex=False):
        return self.reader, self.writer

    def Process(self, **_kwargs):
        return self.process


class _StubbornProcess(_AliveProcess):
    def __init__(self):
        super().__init__()
        self.killed = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class StartBackendServerLifecycleTest(unittest.TestCase):
    @patch("rtp_llm.start_backend_server._install_hot_hook_runtime")
    @patch("rtp_llm.start_backend_server.arm_parent_death_sigkill")
    @patch("rtp_llm.start_backend_server.setproctitle")
    @patch("rtp_llm.start_backend_server.load_gpu_nic_affinity")
    @patch("rtp_llm.start_backend_server.torch.cuda.is_available", return_value=False)
    @patch("rtp_llm.start_backend_server.local_rank_start", return_value=None)
    def test_inline_backend_does_not_arm_parent_death_signal(
        self,
        local_rank_start_mock,
        cuda_available_mock,
        load_gpu_nic_affinity_mock,
        setproctitle_mock,
        parent_death_signal_mock,
        install_hot_hook_runtime_mock,
    ):
        start_backend_server(None, Mock(), None)

        parent_death_signal_mock.assert_not_called()
        local_rank_start_mock.assert_called_once()

    @patch("rtp_llm.start_backend_server._install_hot_hook_runtime")
    @patch("rtp_llm.start_backend_server.arm_parent_death_sigkill")
    @patch("rtp_llm.start_backend_server.setproctitle")
    @patch("rtp_llm.start_backend_server.load_gpu_nic_affinity")
    @patch("rtp_llm.start_backend_server.torch.cuda.is_available", return_value=False)
    @patch("rtp_llm.start_backend_server.local_rank_start", return_value=None)
    def test_spawned_backend_arms_captured_parent_pid(
        self,
        local_rank_start_mock,
        cuda_available_mock,
        load_gpu_nic_affinity_mock,
        setproctitle_mock,
        parent_death_signal_mock,
        install_hot_hook_runtime_mock,
    ):
        start_backend_server(None, Mock(), None, expected_parent_pid=24680)

        parent_death_signal_mock.assert_called_once_with(24680)
        local_rank_start_mock.assert_called_once()

    def test_rank_failed_status_is_not_swallowed(self):
        reader, writer = multiprocessing.Pipe(duplex=False)
        writer.send({"status": "failed", "message": "rank copy timeout"})
        writer.close()
        started = time.monotonic()

        with self.assertRaisesRegex(Exception, "rank copy timeout"):
            _wait_for_ranks_startup([_AliveProcess()], [reader], 1)

        self.assertLess(time.monotonic() - started, 1)

    def test_process_start_base_exception_closes_pipes_and_cleans_rank(self):
        ctx = _StartFailureContext()
        configs = SimpleNamespace(
            parallelism_config=SimpleNamespace(world_rank=0),
            distribute_config=SimpleNamespace(fake_gang_env=False),
            server_config=SimpleNamespace(shutdown_timeout=1, monitor_interval=0.01),
        )

        with patch(
            "rtp_llm.start_backend_server.multiprocessing.get_context",
            return_value=ctx,
        ), patch(
            "rtp_llm.start_backend_server._get_local_world_size", return_value=1
        ), patch(
            "rtp_llm.start_backend_server._get_cuda_device_list", return_value=["0"]
        ), patch(
            "rtp_llm.start_backend_server._validate_dp_configuration"
        ), patch(
            "rtp_llm.start_backend_server.signal.signal"
        ):
            with self.assertRaises(KeyboardInterrupt):
                multi_rank_start(None, configs)

        self.assertTrue(ctx.process.terminated)
        self.assertFalse(ctx.process.is_alive())
        self.assertTrue(ctx.reader.closed)
        self.assertTrue(ctx.writer.closed)

    @patch("rtp_llm.start_backend_server._install_hot_hook_runtime")
    @patch("rtp_llm.start_backend_server.arm_parent_death_sigkill")
    @patch("rtp_llm.start_backend_server.setproctitle")
    @patch("rtp_llm.start_backend_server.load_gpu_nic_affinity")
    @patch("rtp_llm.start_backend_server.torch.cuda.is_available", return_value=False)
    @patch("rtp_llm.start_backend_server.local_rank_start", return_value=None)
    def test_backend_startup_signal_unwinds_immediately(
        self,
        _local_rank_start_mock,
        _cuda_available_mock,
        _load_gpu_nic_affinity_mock,
        _setproctitle_mock,
        _parent_death_signal_mock,
        _install_hot_hook_runtime_mock,
    ):
        handlers = {}

        def capture_handler(sig, handler):
            handlers[sig] = handler

        with patch(
            "rtp_llm.start_backend_server.signal.signal",
            side_effect=capture_handler,
        ):
            start_backend_server(None, Mock(), None)
            with self.assertRaisesRegex(KeyboardInterrupt, "backend startup"):
                handlers[signal.SIGTERM](signal.SIGTERM, None)

    @patch("rtp_llm.start_backend_server._install_hot_hook_runtime")
    @patch("rtp_llm.start_backend_server.arm_parent_death_sigkill")
    def test_rank_startup_signal_unwinds_into_cleanup(
        self,
        _parent_death_signal_mock,
        _install_hot_hook_runtime_mock,
    ):
        handlers = {}

        def capture_handler(sig, handler):
            handlers[sig] = handler

        def interrupt_startup():
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        fake_backend_module = SimpleNamespace(BackendManager=object)
        with patch.dict(
            sys.modules,
            {"rtp_llm.server.backend_manager": fake_backend_module},
        ), patch(
            "rtp_llm.start_backend_server.copy_gemm_config",
            side_effect=interrupt_startup,
        ), patch(
            "rtp_llm.start_backend_server.signal.signal",
            side_effect=capture_handler,
        ):
            with self.assertRaisesRegex(KeyboardInterrupt, "rank startup"):
                local_rank_start(None, Mock(), world_rank=0)

    def test_unreaped_rank_uses_immediate_exit_escape_hatch(self):
        stubborn_rank = _StubbornProcess()

        def fail_after_spawn(_global_controller, _configs, _ctx, processes, _readers):
            processes.append(stubborn_rank)
            raise RuntimeError("rank startup failed")

        configs = SimpleNamespace(
            distribute_config=SimpleNamespace(fake_gang_env=False),
            server_config=SimpleNamespace(shutdown_timeout=1, monitor_interval=0.01),
        )
        with patch(
            "rtp_llm.start_backend_server.multiprocessing.get_context",
            return_value=Mock(),
        ), patch(
            "rtp_llm.start_backend_server._create_rank_processes",
            side_effect=fail_after_spawn,
        ), patch(
            "rtp_llm.start_backend_server.signal.signal"
        ), patch(
            "rtp_llm.start_backend_server.os._exit",
            side_effect=SystemExit(1),
        ) as exit_mock:
            with self.assertRaises(SystemExit):
                multi_rank_start(None, configs)

        self.assertTrue(stubborn_rank.terminated)
        self.assertTrue(stubborn_rank.killed)
        exit_mock.assert_called_once_with(1)

    def test_fake_gang_env_waits_for_ranks_and_reports_backend_ready(self):
        """Fake gang mode must use the same rank-ready handshake as normal mode."""
        process = _AliveProcess()
        reader = _TrackedConnection()
        processes = []
        rank_pipe_readers = []

        def create_rank_processes(
            _controller, _configs, _ctx, out_processes, out_readers
        ):
            out_processes.append(process)
            out_readers.append(reader)

        configs = SimpleNamespace(
            distribute_config=SimpleNamespace(fake_gang_env=True),
            server_config=SimpleNamespace(shutdown_timeout=1, monitor_interval=0.01),
        )
        pipe_writer = Mock()
        manager = Mock()

        with patch(
            "rtp_llm.start_backend_server._create_rank_processes",
            side_effect=create_rank_processes,
        ), patch(
            "rtp_llm.start_backend_server._wait_for_ranks_startup"
        ) as wait_for_ranks_startup, patch(
            "rtp_llm.start_backend_server.ProcessManager", return_value=manager
        ) as process_manager, patch(
            "rtp_llm.start_backend_server._send_pipe_status"
        ) as send_pipe_status, patch(
            "rtp_llm.start_backend_server._close_readers"
        ) as close_readers:
            result = multi_rank_start(None, configs, pipe_writer)

        self.assertEqual(result, [process])
        wait_for_ranks_startup.assert_called_once_with([process], [reader], 1)
        process_manager.assert_called_once_with(
            shutdown_timeout=1,
            monitor_interval=0.01,
            allow_defer_first_sigterm=True,
            pre_exit_cleanup=ANY,
        )
        manager.set_processes.assert_called_once_with(
            [process], shutdown_group="backend"
        )
        send_pipe_status.assert_called_once_with(
            pipe_writer, "success", "All 1 backend ranks started successfully"
        )
        manager.monitor_and_release_processes.assert_called_once_with()
        close_readers.assert_not_called()


if __name__ == "__main__":
    unittest.main()
