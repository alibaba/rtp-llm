import multiprocessing
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm import start_backend_server, start_server
from rtp_llm.ops import RoleType
from rtp_llm.utils.backend_restart import RestartOutcome
from rtp_llm.utils.drain_flag import DrainFlag


class ServiceDrainingPropagationTest(unittest.TestCase):
    def test_each_launch_passes_a_fresh_event_to_all_process_groups(self):
        configs = Mock()
        configs.parallelism_config.dp_size = 1
        # Without in-place restart every process group shares the instance's event.
        configs.server_config.engine_restart_limit = 0
        with ExitStack() as stack:
            for name in (
                "init_controller",
                "_sync_server_shutdown_timeout",
                "_setup_startup_warmup_health_gate",
                "_mark_startup_warmup_health_gate_ready",
            ):
                stack.enter_context(patch.object(start_server, name))
            manager_class = stack.enter_context(
                patch.object(start_server, "ProcessManager")
            )
            starters = [
                stack.enter_context(patch.object(start_server, name))
                for name in (
                    "start_backend_server_impl",
                    "start_frontend_server_impl",
                    "start_dash_sc_server_impl",
                )
            ]
            vit = stack.enter_context(
                patch.object(start_server, "start_vit_server_impl")
            )
            previous = None
            for role in (RoleType.PDFUSION, RoleType.PDFUSION, RoleType.VIT):
                configs.role_config.role_type = role
                start_server.start_server(configs)
                event = manager_class.call_args.kwargs["service_draining"]
                # multiprocessing.Event.set() hangs once a waiter was killed.
                self.assertIsInstance(event, DrainFlag)
                self.assertIsNot(event, previous)
                self.assertFalse(event.is_set())
                for starter in [vit] if role == RoleType.VIT else starters:
                    self.assertIs(starter.call_args.args[-1], event)
                event.set()
                previous = event


class StartupBackendReadinessTest(unittest.TestCase):
    def test_restart_waits_for_each_backend_generation(self):
        config = SimpleNamespace(
            parallelism_config=SimpleNamespace(
                world_size=1, local_world_size=1, world_rank=0
            ),
            server_config=SimpleNamespace(
                engine_restart_limit=2,
                engine_restart_heartbeat_timeout_s=10,
                startup_health_check_timeout=30,
            ),
        )
        lengths = iter((4096, 8192))

        def spawn_backend(*args):
            status = {}

            def check_ready():
                status["max_seq_len"] = next(lengths)
                return True

            return Mock(), check_ready, status

        with patch.object(
            start_server, "_spawn_backend_process", side_effect=spawn_backend
        ):
            restarter = start_server._create_backend_restarter(
                None, config, Mock(shutdown_requested=False), None, DrainFlag()
            )
            self.addCleanup(restarter.coordinator.close)
            for _ in range(2):
                outcome, _ = restarter._start_generation()
                self.assertEqual(outcome, RestartOutcome.RESTARTED)
                self.assertIsNone(restarter._after_ready)

    def test_backend_ranks_report_resolved_model_length(self):
        from unittest.mock import Mock

        readers = []
        for _ in range(2):
            reader, writer = multiprocessing.Pipe(duplex=False)
            self.addCleanup(reader.close)
            readers.append(reader)
            start_backend_server._send_pipe_status(
                writer, "success", "ready", max_seq_len=4096
            )
        process = Mock(exitcode=None)
        process.is_alive.return_value = True
        self.assertEqual(
            start_backend_server._wait_for_ranks_startup(
                [process, process], readers, 2
            ),
            4096,
        )


if __name__ == "__main__":
    unittest.main()
