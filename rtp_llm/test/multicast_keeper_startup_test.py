import os
import socket
import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import rtp_llm.start_backend_server as backend_startup
import rtp_llm.start_server as server_startup


class _FakeEndpoint:
    def close(self):
        pass


class _FakeRankProcess:
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get("name")
        self.spawn_environment = None


class _FakeKeeperRuntime:
    def __init__(self, events):
        self.events = events
        self.artifacts = SimpleNamespace(
            shim="/opt/rtp-llm/multicast_keeper/mc_shim_unified.so"
        )

    def start(self):
        self.events.append("keeper_start")
        return self

    @contextmanager
    def configure_subprocess(self):
        self.events.append("keeper_env_enter")
        with mock.patch.dict(
            os.environ,
            {
                server_startup.MULTICAST_KEEPER_ENV: "1",
                "NEKYIA_KEEPER_DIR": "/run/rtp-llm/keeper",
                "LD_PRELOAD": "/opt/rtp-llm/multicast_keeper/mc_shim_unified.so",
            },
            clear=False,
        ):
            yield
        self.events.append("keeper_env_exit")

    def stop(self):
        self.events.append("keeper_stop")


class _FakeProcessManager:
    def __init__(self, events):
        self.events = events
        self.shutdown_requested = False
        self.failure_detected = False

    def add_process(self, process, shutdown_group="default"):
        self.events.append(("add_process", shutdown_group))

    def add_processes(self, processes, shutdown_group="default"):
        self.events.append(("add_processes", shutdown_group))

    def run_health_checks(self):
        self.events.append("health_checks")
        return True

    def monitor_and_release_processes(self):
        self.events.append("business_cleanup")

    def request_failure_shutdown(self):
        self.shutdown_requested = True
        self.failure_detected = True
        self.events.append("failure_shutdown")


def _server_config():
    return SimpleNamespace(
        runtime_config=SimpleNamespace(enable_sleep_mode=True, sleep_mode_level=3),
        role_config=SimpleNamespace(role_type=server_startup.RoleType.PDFUSION),
        server_config=SimpleNamespace(shutdown_timeout=10, monitor_interval=0.01),
        parallelism_config=SimpleNamespace(dp_size=1, world_rank=0),
        concurrency_config=SimpleNamespace(),
    )


class MulticastKeeperRootStartupTest(unittest.TestCase):
    def test_keeper_gate_requires_level3_sleep_and_explicit_opt_in(self):
        config = _server_config()
        cases = (
            (True, 3, "1", True),
            (False, 3, "1", False),
            (True, 2, "1", False),
            (True, 3, "0", False),
        )
        for sleep_enabled, level, opt_in, expected in cases:
            config.runtime_config.enable_sleep_mode = sleep_enabled
            config.runtime_config.sleep_mode_level = level
            with mock.patch.dict(
                os.environ,
                {server_startup.MULTICAST_KEEPER_ENV: opt_in},
                clear=False,
            ):
                self.assertEqual(
                    server_startup._multicast_keeper_requested(config), expected
                )

    def test_backend_only_environment_and_cleanup_order(self):
        events = []
        runtime = _FakeKeeperRuntime(events)
        manager = _FakeProcessManager(events)
        fake_monitor_thread = mock.Mock()

        def start_backend(*args):
            events.append(
                (
                    "backend_start",
                    os.environ.get("NEKYIA_KEEPER_DIR"),
                    os.environ.get("LD_PRELOAD"),
                )
            )
            return object()

        def start_frontend(*args):
            events.append(
                (
                    "frontend_start",
                    os.environ.get(server_startup.MULTICAST_KEEPER_ENV),
                    os.environ.get("NEKYIA_KEEPER_DIR"),
                    os.environ.get("LD_PRELOAD"),
                )
            )
            return []

        def start_dash_sc(*args):
            events.append(
                (
                    "dash_sc_start",
                    os.environ.get(server_startup.MULTICAST_KEEPER_ENV),
                    os.environ.get("NEKYIA_KEEPER_DIR"),
                    os.environ.get("LD_PRELOAD"),
                )
            )
            return []

        patches = (
            mock.patch.object(server_startup.multiprocessing, "set_start_method"),
            mock.patch.object(server_startup, "init_controller", return_value=object()),
            mock.patch.object(server_startup, "ProcessManager", return_value=manager),
            mock.patch.object(
                server_startup, "_setup_startup_warmup_health_gate", return_value=None
            ),
            mock.patch.object(
                server_startup, "_create_multicast_keeper_runtime", return_value=runtime
            ),
            mock.patch.object(
                server_startup,
                "_start_multicast_keeper_monitor",
                return_value=fake_monitor_thread,
            ),
            mock.patch.object(
                server_startup, "start_backend_server_impl", side_effect=start_backend
            ),
            mock.patch.object(
                server_startup, "start_frontend_server_impl", side_effect=start_frontend
            ),
            mock.patch.object(
                server_startup,
                "start_dash_sc_server_impl",
                side_effect=start_dash_sc,
            ),
            mock.patch.object(
                server_startup, "_maybe_run_startup_real_warmup", return_value=False
            ),
            mock.patch.object(server_startup, "_mark_startup_warmup_health_gate_ready"),
            mock.patch.object(server_startup, "_start_post_startup_jit_cache_writer"),
        )

        root_environment = {
            server_startup.MULTICAST_KEEPER_ENV: "1",
            "NEKYIA_KEEPER_DIR": "/stale/keeper",
            "LD_PRELOAD": (
                "/opt/tms/libtorch_memory_saver.so:"
                "/opt/rtp-llm/multicast_keeper/mc_shim_unified.so:"
                "/opt/observability/libhook.so"
            ),
        }
        with mock.patch.dict(os.environ, root_environment, clear=True):
            for patcher in patches:
                patcher.start()
            try:
                server_startup.start_server(_server_config())
            finally:
                for patcher in reversed(patches):
                    patcher.stop()

        backend_event = next(event for event in events if event[0] == "backend_start")
        frontend_event = next(event for event in events if event[0] == "frontend_start")
        dash_sc_event = next(event for event in events if event[0] == "dash_sc_start")
        self.assertEqual(backend_event[1], "/run/rtp-llm/keeper")
        self.assertIn("mc_shim_unified.so", backend_event[2])
        for non_backend_event in (frontend_event, dash_sc_event):
            self.assertIsNone(non_backend_event[1])
            self.assertIsNone(non_backend_event[2])
            self.assertEqual(
                non_backend_event[3],
                "/opt/tms/libtorch_memory_saver.so:/opt/observability/libhook.so",
            )
        self.assertLess(events.index("keeper_start"), events.index(backend_event))
        self.assertLess(events.index("business_cleanup"), events.index("keeper_stop"))
        fake_monitor_thread.join.assert_called_once_with(timeout=1.0)

    def test_unhealthy_holder_requests_fail_closed_shutdown(self):
        requested = threading.Event()
        manager = mock.Mock()
        manager.request_failure_shutdown.side_effect = requested.set
        runtime = mock.Mock()
        runtime.creator_timeout_ms = 100
        runtime.client_timeout_ms = 100
        runtime.health.side_effect = RuntimeError("holder identity changed")
        stop_event = threading.Event()

        thread = server_startup._start_multicast_keeper_monitor(
            runtime, manager, stop_event, 0.01
        )
        self.assertTrue(requested.wait(timeout=1.0))
        stop_event.set()
        thread.join(timeout=1.0)

        manager.request_failure_shutdown.assert_called_once_with()
        runtime.health.assert_called()

    def test_live_holder_ping_timeout_is_bounded_but_not_immediately_fatal(self):
        requested = threading.Event()
        manager = mock.Mock()
        manager.request_failure_shutdown.side_effect = requested.set
        runtime = mock.Mock()
        runtime.creator_timeout_ms = 100
        runtime.client_timeout_ms = 100
        runtime.health.side_effect = socket.timeout("holder is creating")
        runtime.process.pid = 123
        runtime.process.poll.return_value = None
        stop_event = threading.Event()

        thread = server_startup._start_multicast_keeper_monitor(
            runtime, manager, stop_event, 0.01
        )
        self.assertFalse(requested.wait(timeout=0.1))
        self.assertTrue(requested.wait(timeout=1.0))
        stop_event.set()
        thread.join(timeout=1.0)

        manager.request_failure_shutdown.assert_called_once_with()


class MulticastKeeperRankStartupTest(unittest.TestCase):
    def _rank_config(self):
        return SimpleNamespace(
            parallelism_config=SimpleNamespace(
                world_rank=4, world_size=8, tp_size=2, dp_size=1
            )
        )

    def test_each_rank_gets_explicit_owner_and_parent_env_is_restored(self):
        created = []

        def make_process(*args, **kwargs):
            process = _FakeRankProcess(*args, **kwargs)
            created.append(process)
            return process

        def start_process(process):
            process.spawn_environment = dict(os.environ)

        original = {
            backend_startup.MULTICAST_KEEPER_ENV: "1",
            backend_startup.MULTICAST_KEEPER_OWNER_ID_ENV: "parent-owner",
            "WORLD_RANK": "parent-rank",
            "CUDA_VISIBLE_DEVICES": "parent-devices",
        }
        with mock.patch.dict(os.environ, original, clear=True), mock.patch.object(
            backend_startup, "_get_local_world_size", return_value=2
        ), mock.patch.object(
            backend_startup, "_get_cuda_device_list", return_value=["0", "1"]
        ), mock.patch.object(
            backend_startup.multiprocessing,
            "Pipe",
            side_effect=lambda **kwargs: (_FakeEndpoint(), _FakeEndpoint()),
        ), mock.patch.object(
            backend_startup, "Process", side_effect=make_process
        ), mock.patch.object(
            backend_startup,
            "start_memory_saver_configured_process",
            side_effect=start_process,
        ):
            backend_startup._create_rank_processes(object(), self._rank_config())
            self.assertEqual(dict(os.environ), original)

        self.assertEqual(
            [
                process.spawn_environment[backend_startup.MULTICAST_KEEPER_OWNER_ID_ENV]
                for process in created
            ],
            ["5", "6"],
        )
        self.assertEqual(
            [process.spawn_environment["WORLD_RANK"] for process in created],
            ["4", "5"],
        )

    def test_single_rank_gets_global_rank_owner_and_restores_parent(self):
        observed = {}

        def local_rank(*args):
            observed["world_rank"] = args[2]
            observed["owner_id"] = os.environ.get(
                backend_startup.MULTICAST_KEEPER_OWNER_ID_ENV
            )

        original = {
            backend_startup.MULTICAST_KEEPER_ENV: "1",
            backend_startup.MULTICAST_KEEPER_OWNER_ID_ENV: "parent-owner",
        }
        with mock.patch.dict(os.environ, original, clear=True), mock.patch.object(
            backend_startup, "local_rank_start", side_effect=local_rank
        ):
            backend_startup._start_single_rank(object(), self._rank_config(), 7)
            self.assertEqual(dict(os.environ), original)

        self.assertEqual(observed, {"world_rank": 7, "owner_id": "8"})


if __name__ == "__main__":
    unittest.main()
