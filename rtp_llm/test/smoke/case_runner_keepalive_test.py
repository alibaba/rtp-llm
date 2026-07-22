import json
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from smoke.case_runner import CaseRunner
from smoke.task_info import TaskStates


class FakeServerManager:
    def __init__(self, port: int, pid: int | None = None, fail_stop: bool = False):
        self.port = port
        self.server_pid = os.getpid() if pid is None else pid
        self.log_file_path = f"/tmp/fake_server_{port}.log"
        self.fail_stop = fail_stop
        self.stop_count = 0

    def stop_server(self):
        self.stop_count += 1
        if self.fail_stop:
            raise RuntimeError("injected stop failure")


def make_runner() -> CaseRunner:
    runner = object.__new__(CaseRunner)
    runner.task_info = SimpleNamespace(taskinfo_rel_path="data/model/fake.json")
    runner.smoke_args = {"prefill": "--role_type PREFILL"}
    runner.smoke_args_str = ""
    return runner


class KeepaliveTest(unittest.TestCase):
    def wait_for_file(self, path: str, thread: threading.Thread) -> None:
        deadline = time.time() + 5
        while time.time() < deadline:
            if os.path.exists(path):
                return
            if not thread.is_alive():
                break
            time.sleep(0.01)
        self.fail(f"timed out waiting for {path}")

    def test_pd_schema_and_stop_file_cleanup(self):
        runner = make_runner()
        prefill = FakeServerManager(12345)
        decode = FakeServerManager(12346)
        errors = []

        with tempfile.TemporaryDirectory() as temp_dir:
            live_info = os.path.join(temp_dir, "live.json")
            stop_file = os.path.join(temp_dir, "stop")
            env = {
                "SMOKE_LIVE_INFO": live_info,
                "SMOKE_STOP_FILE": stop_file,
            }

            def run_keepalive():
                try:
                    with patch.dict(os.environ, env, clear=False):
                        runner._keep_servers_alive(
                            {"prefill": prefill, "decode": decode}
                        )
                except Exception as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            thread = threading.Thread(target=run_keepalive)
            thread.start()
            self.wait_for_file(live_info, thread)
            with open(live_info, encoding="utf-8") as source:
                info = json.load(source)
            self.assertEqual(set(info["servers"]), {"prefill", "decode"})
            self.assertEqual(info["servers"]["prefill"]["port"], 12345)
            self.assertEqual(info["servers"]["decode"]["port"], 12346)
            self.assertEqual(info["stop_file"], stop_file)

            open(stop_file, "w", encoding="utf-8").close()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

        self.assertEqual(errors, [])
        self.assertEqual(prefill.stop_count, 1)
        self.assertEqual(decode.stop_count, 1)

    def test_single_server_backward_compatible_fields(self):
        runner = make_runner()
        manager = FakeServerManager(23456)

        with tempfile.TemporaryDirectory() as temp_dir:
            live_info = os.path.join(temp_dir, "live.json")
            stop_file = os.path.join(temp_dir, "stop")
            open(stop_file, "w", encoding="utf-8").close()
            with patch.dict(
                os.environ,
                {
                    "SMOKE_LIVE_INFO": live_info,
                    "SMOKE_STOP_FILE": stop_file,
                },
                clear=False,
            ):
                runner._keep_servers_alive({"main": manager})
            with open(live_info, encoding="utf-8") as source:
                info = json.load(source)

        self.assertEqual(info["port"], 23456)
        self.assertEqual(info["server_pid"], os.getpid())
        self.assertEqual(info["servers"]["main"]["port"], 23456)
        self.assertEqual(manager.stop_count, 1)

    def test_server_death_still_stops_all_servers(self):
        runner = make_runner()
        dead = FakeServerManager(30001, pid=999999999)
        healthy = FakeServerManager(30002)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {
                    "SMOKE_LIVE_INFO": os.path.join(temp_dir, "live.json"),
                    "SMOKE_STOP_FILE": os.path.join(temp_dir, "stop"),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "disappeared"):
                    runner._keep_servers_alive(
                        {"prefill": healthy, "decode": dead}
                    )

        self.assertEqual(healthy.stop_count, 1)
        self.assertEqual(dead.stop_count, 1)

    def test_one_stop_failure_does_not_skip_other_server(self):
        runner = make_runner()
        good = FakeServerManager(40001)
        bad = FakeServerManager(40002, fail_stop=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            stop_file = os.path.join(temp_dir, "stop")
            open(stop_file, "w", encoding="utf-8").close()
            with patch.dict(
                os.environ,
                {
                    "SMOKE_LIVE_INFO": os.path.join(temp_dir, "live.json"),
                    "SMOKE_STOP_FILE": stop_file,
                },
                clear=False,
            ):
                runner._keep_servers_alive({"good": good, "bad": bad})

        self.assertEqual(good.stop_count, 1)
        self.assertEqual(bad.stop_count, 1)

    def test_live_info_directory_failure_still_stops_all_servers(self):
        runner = make_runner()
        prefill = FakeServerManager(41001)
        decode = FakeServerManager(41002)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {
                    "SMOKE_LIVE_INFO": os.path.join(temp_dir, "live.json"),
                    "SMOKE_STOP_FILE": os.path.join(temp_dir, "stop"),
                },
                clear=False,
            ), patch(
                "smoke.case_runner.os.makedirs",
                side_effect=OSError("injected live-info directory failure"),
            ):
                with self.assertRaisesRegex(OSError, "directory failure"):
                    runner._keep_servers_alive(
                        {"prefill": prefill, "decode": decode}
                    )

        self.assertEqual(prefill.stop_count, 1)
        self.assertEqual(decode.stop_count, 1)

    def test_after_curl_failure_does_not_park(self):
        runner = make_runner()
        runner.env_args = []
        manager = FakeServerManager(42001)
        failed = TaskStates(ret=False, err_msg="injected curl failure")
        runner.create_env_from_args = Mock(return_value={})
        runner.start_server = Mock(return_value=manager)
        runner.curl_server = Mock(return_value=failed)
        runner._keep_server_alive = Mock(return_value=TaskStates())

        with patch.dict(
            os.environ,
            {
                "SMOKE_KEEP_SERVER_ALIVE": "False",
                "SMOKE_KEEP_SERVER_ALIVE_AFTER_CURL": "True",
            },
            clear=False,
        ):
            result = runner.run()

        self.assertFalse(result.ret)
        runner._keep_server_alive.assert_not_called()
        self.assertEqual(manager.stop_count, 1)

    def test_keepalive_modes_are_mutually_exclusive(self):
        with patch.dict(
            os.environ,
            {
                "SMOKE_KEEP_SERVER_ALIVE": "True",
                "SMOKE_KEEP_SERVER_ALIVE_AFTER_CURL": "True",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                CaseRunner._keepalive_enabled()


if __name__ == "__main__":
    unittest.main()
