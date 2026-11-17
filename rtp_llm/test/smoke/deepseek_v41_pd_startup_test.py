import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.smoke import deepseek_v41_pd_contract_test as pd
from rtp_llm.test.smoke import startup_resource_monitor as resources


class RoleStartupError(RuntimeError):
    pass


class DeepSeekV41PDStartupTest(unittest.TestCase):
    @contextmanager
    def startup_case(self, managers):
        case = pd.DeepSeekV41PDContractTest()
        case.model_path = "/unused/checkpoint"
        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch.object(pd, "get_gpu_ids", return_value=[0, 1, 2, 3]),
            patch.object(pd, "MagaServerManager", side_effect=managers) as factory,
            patch.object(case, "server_args", return_value="--world_size 4"),
        ):
            case.output_dir = Path(output_dir)
            factory.get_free_port.side_effect = ["18000", "19000"]
            try:
                yield case
            finally:
                case.doCleanups()

    def test_failed_role_stops_waiting_sibling_before_startup_join(self):
        for failed_role in (0, 1):
            for raises in (False, True):
                with self.subTest(failed_role=failed_role, raises=raises):
                    entered = threading.Barrier(2)
                    release_waiter = threading.Event()
                    stopped = [threading.Event(), threading.Event()]
                    managers = [Mock(), Mock()]

                    def failed_start(**kwargs):
                        entered.wait(timeout=10)
                        if raises:
                            raise RoleStartupError
                        return False

                    def waiting_start(**kwargs):
                        entered.wait(timeout=10)
                        if not release_waiter.wait(timeout=20):
                            raise TimeoutError("waiting role was not stopped")
                        return False

                    def stop_waiter():
                        stopped[1 - failed_role].set()
                        release_waiter.set()

                    managers[failed_role].start_server.side_effect = failed_start
                    managers[failed_role].stop_server.side_effect = stopped[
                        failed_role
                    ].set
                    managers[1 - failed_role].start_server.side_effect = waiting_start
                    managers[1 - failed_role].stop_server.side_effect = stop_waiter
                    with self.startup_case(managers) as case:
                        with ThreadPoolExecutor(max_workers=1) as runner:
                            result = runner.submit(case.start_servers)
                            try:
                                self.assertTrue(
                                    stopped[1 - failed_role].wait(timeout=5),
                                    "failed startup left the sibling waiting for its own timeout",
                                )
                            finally:
                                release_waiter.set()
                            expected = RuntimeError if raises else AssertionError
                            with self.assertRaises(expected):
                                result.result(timeout=10)
                        self.assertTrue(all(event.is_set() for event in stopped))

    def test_success_keeps_both_concurrently_started_roles_running(self):
        entered = threading.Barrier(2)
        stopped = [threading.Event(), threading.Event()]
        managers = [Mock(), Mock()]

        def ready(**kwargs):
            entered.wait(timeout=10)
            return True

        for manager, event in zip(managers, stopped):
            manager.start_server.side_effect = ready
            manager.stop_server.side_effect = event.set
        with self.startup_case(managers) as case:
            case.start_servers()
            self.assertFalse(any(event.is_set() for event in stopped))
        self.assertTrue(all(event.is_set() for event in stopped))

    def test_resource_artifact_covers_launch_and_stop_then_closes(self):
        for outcome in ("success", "false", "exception"):
            with self.subTest(outcome=outcome):
                sampled = threading.Event()
                entered = threading.Barrier(2)
                streams = []
                snapshot = resources._snapshot
                open_path = Path.open
                managers = [Mock(), Mock()]

                def collect(root_pid):
                    result = snapshot(root_pid)
                    sampled.set()
                    return result

                def capture_stream(path, *args, **kwargs):
                    stream = open_path(path, *args, **kwargs)
                    if path == artifact:
                        streams.append(stream)
                    return stream

                def start(**kwargs):
                    self.assertTrue(streams, "resource artifact absent at role launch")
                    self.assertFalse(streams[0].closed)
                    entered.wait(timeout=10)
                    if outcome == "exception":
                        raise RoleStartupError
                    return outcome == "success"

                def stop():
                    self.assertTrue(streams, "resource artifact absent at role stop")
                    self.assertFalse(streams[0].closed)
                    sampled.clear()
                    self.assertTrue(
                        sampled.wait(timeout=5), "resource sampler stopped before roles"
                    )

                for manager in managers:
                    manager.start_server.side_effect = start
                    manager.stop_server.side_effect = stop
                with self.startup_case(managers) as case:
                    artifact = case.output_dir / "startup_resources.jsonl"
                    with (
                        patch.object(Path, "open", capture_stream),
                        patch.object(resources, "_snapshot", side_effect=collect),
                        patch.object(resources, "_INTERVAL_SECONDS", 0.01),
                    ):
                        if outcome == "success":
                            case.start_servers()
                        else:
                            expected = (
                                RuntimeError
                                if outcome == "exception"
                                else AssertionError
                            )
                            with self.assertRaises(expected):
                                case.start_servers()
                        self.assertTrue(case.doCleanups())
                    self.assertTrue(streams[0].closed)
                    samples = [
                        json.loads(line) for line in artifact.read_text().splitlines()
                    ]
                    self.assertTrue(samples)
                    self.assertTrue(
                        all(sample["root_pid"] == os.getpid() for sample in samples)
                    )


if __name__ == "__main__":
    unittest.main()
