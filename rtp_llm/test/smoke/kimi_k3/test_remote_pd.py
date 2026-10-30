import copy
import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from email.message import Message
from pathlib import Path
from unittest import mock

import remote_pd_test as remote

PINS = {"external_revision": "a" * 40, "internal_revision": "b" * 40}
RUN_ID = "k3-pd-unittest"


def registration(role):
    return {
        **PINS,
        "hostname": role + "-host",
        "ip": "10.0.0.1" if role == "prefill" else "10.0.0.2",
    }


def identity(role):
    checkpoint = {
        "config_sha256": "c" * 64,
        "index_sha256": "d" * 64,
        "shards": [["model-00001.safetensors", 100]],
    }
    return {
        **PINS,
        "hostname": role + "-host",
        "gpu_uuids": ["GPU-" + role + str(index) for index in range(8)],
        "checkpoint_identity": checkpoint,
        "draft_identity": copy.deepcopy(checkpoint),
        "guard_sha256": "e" * 64,
    }


def record(role, prepared=False):
    return {
        "registration": registration(role),
        "prepared": identity(role) if prepared else None,
        "ready": False,
        "result": None,
        "cleanup": None,
    }


def snapshot(prepared=False):
    return {
        "run_id": RUN_ID,
        "deadline_unix_s": time.time() + 60,
        "roles": {role: record(role, prepared) for role in remote.ROLES},
        "error": None,
    }


class Response(io.BytesIO):
    def __init__(self, value, content_type="application/json", raw=False):
        super().__init__(value if raw else json.dumps(value).encode())
        self.status = 200
        self.headers = Message()
        self.headers["Content-Type"] = content_type


class MemoryState:
    def __init__(self, expired=False):
        self.state = snapshot()
        self.state["roles"] = {}
        if expired:
            self.state["deadline_unix_s"] = time.time() - 1
        self.lock = threading.Lock()
        self.events = []
        self.fail_heartbeats = threading.Event()

    def client(self):
        owner = self

        class Client:
            run_id = RUN_ID
            timeout = 0.2
            deadline = None

            def request(self, operation=None, role=None, data=None):
                with owner.lock:
                    if operation == "heartbeat" and owner.fail_heartbeats.is_set():
                        raise RuntimeError("Injected broker disconnect")
                    if operation == "register":
                        owner.state["roles"][role] = record(role)
                        owner.state["roles"][role]["registration"] = copy.deepcopy(data)
                    elif operation not in (None, "heartbeat"):
                        owner.state["roles"][role][operation] = copy.deepcopy(data)
                    owner.events.append((operation, role))
                    self.deadline = owner.state["deadline_unix_s"]
                    return copy.deepcopy(owner.state)

        return Client()


class FakeWorker:
    def __init__(self, role, disconnected, output, state, failure=None):
        self.role, self.disconnected = role, disconnected
        self.task = output / "native"
        self.state, self.failure = state, failure
        self.started, self.closed, self.requested = False, False, False

    def prepare_native(self, config, role, run_id):
        if self.failure == "prepare":
            raise RuntimeError("Injected prepare failure")
        if (
            role != self.role
            or run_id != RUN_ID
            or config[self.role]["artifact_root"] != str(self.task.parent)
        ):
            raise AssertionError("Wrong native prepare inputs")
        self.task.mkdir()
        return identity(role)

    def start(self):
        self.started = True

    def status(self):
        if not self.started:
            raise AssertionError("Status before start")
        return {"healthy": True}

    def requests(self, message):
        if self.role != "prefill":
            raise AssertionError("Requests ran on Decode")
        self.requested = True
        if self.failure == "disconnect":
            self.state.fail_heartbeats.set()
            if not self.disconnected.wait(2):
                raise AssertionError("Watchdog failed to interrupt requests")
            raise RuntimeError("Injected requests interrupted by disconnect")
        return {"requests_passed": True}

    def close(self):
        self.closed = True
        if self.failure == "cleanup":
            raise RuntimeError("Injected cleanup failure")


class SchemaTest(unittest.TestCase):
    def test_json_rejects_duplicate_nonfinite_and_oversized_data(self):
        for payload in (
            b'{"a":1,"a":2}',
            b'{"a":NaN}',
            b" " * (remote.MAX_JSON_BYTES + 1),
        ):
            with self.subTest(payload=payload[:30]), self.assertRaises(ValueError):
                remote.decode_json(payload)

    def test_snapshot_schema_and_run_id(self):
        value = snapshot(True)
        self.assertIs(remote.validate_snapshot(value, RUN_ID), value)
        for mutation in (
            lambda state: state.update(run_id="another-run"),
            lambda state: state.update(unexpected=True),
            lambda state: state.update(deadline_unix_s=True),
            lambda state: state["roles"]["prefill"].update(ready="yes"),
            lambda state: state["roles"]["prefill"].update(
                result={"passed": True, "errors": ["failure"]}
            ),
        ):
            invalid = snapshot(True)
            mutation(invalid)
            with self.assertRaises(ValueError):
                remote.validate_snapshot(invalid, RUN_ID)

    def test_pair_rejects_same_host_ip_source_and_gpu(self):
        self.assertTrue(remote.check_pair(snapshot(True), PINS, prepared=True))
        for mutation in (
            lambda state: state["roles"]["decode"]["registration"].update(
                hostname="prefill-host"
            ),
            lambda state: state["roles"]["decode"]["registration"].update(
                ip="10.0.0.1"
            ),
            lambda state: state["roles"]["decode"]["registration"].update(
                external_revision="f" * 40
            ),
            lambda state: state["roles"]["decode"]["prepared"].update(
                gpu_uuids=identity("prefill")["gpu_uuids"]
            ),
            lambda state: state["roles"]["decode"]["prepared"].update(
                guard_sha256="f" * 64
            ),
        ):
            value = snapshot(True)
            mutation(value)
            with self.assertRaises(ValueError):
                remote.check_pair(value, PINS, prepared=True)

    def test_prepared_barrier_waits_for_both(self):
        state = snapshot()
        self.assertFalse(remote.check_pair(state, PINS, prepared=True))
        state["roles"].pop("decode")
        self.assertFalse(remote.check_pair(state, PINS))

    def test_rejects_nonroutable_address(self):
        for address in ("127.0.0.1", "0.0.0.0", "169.254.1.2", "224.0.0.1", "::1"):
            value = registration("prefill")
            value["ip"] = address
            with self.subTest(address=address), self.assertRaises(ValueError):
                remote.validate_registration(value)

    def test_input_resolves_runfiles_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "rtp_llm" / "settings.json"
            path.parent.mkdir()
            path.write_text("{}")
            self.assertEqual(
                remote.resolve_input("settings.json", {"TEST_SRCDIR": str(root)}), path
            )
            with self.assertRaises(ValueError):
                remote.resolve_input("../settings.json", {"TEST_SRCDIR": str(root)})


class BrokerTest(unittest.TestCase):
    def test_authorization_and_request_shape(self):
        broker = remote.Broker("http://broker:8080", "test-token", RUN_ID)
        broker.opener = mock.Mock()
        broker.opener.open.return_value = Response(snapshot())
        broker.request("register", "prefill", registration("prefill"))
        request = broker.opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertEqual(request.full_url, "http://broker:8080/v1/runs/" + RUN_ID)
        self.assertEqual(json.loads(request.data)["op"], "register")

    def test_rejects_changed_deadline_and_wrong_content_type(self):
        broker = remote.Broker("https://broker", "test-token", RUN_ID)
        state = snapshot()
        changed = copy.deepcopy(state)
        changed["deadline_unix_s"] += 1
        broker.opener = mock.Mock()
        broker.opener.open.side_effect = [Response(state), Response(changed)]
        broker.request()
        with self.assertRaisesRegex(ValueError, "deadline"):
            broker.request()
        broker.opener.open.side_effect = [Response(state, content_type="text/html")]
        with self.assertRaises(RuntimeError):
            broker.request()

    def test_rejects_bad_url_token_and_redirect(self):
        for url in (
            "file:///tmp/broker",
            "http://user:pass@broker",
            "http://broker/path",
            "http://broker?token=secret",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                remote.Broker(url, "test-token", RUN_ID)
        with self.assertRaises(ValueError):
            remote.Broker("http://broker", "bad\ntoken", RUN_ID)
        self.assertIsNone(
            remote._NoRedirect().redirect_request(
                None, None, 302, "", None, "http://elsewhere"
            )
        )


class LifecycleTest(unittest.TestCase):
    def run_pair(self, failures=None, checker_passes=True, expired=False):
        failures = failures or {}
        state = MemoryState(expired)
        results, workers, exceptions = {}, {}, []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def execute(role):
                try:
                    output = root / role
                    output.mkdir()
                    status = root / (role + ".shard-status")

                    def factory(disconnected):
                        worker = FakeWorker(
                            role, disconnected, output, state, failures.get(role)
                        )
                        workers[role] = worker
                        return worker

                    results[role] = remote.run_shard(
                        {**PINS, "prefill": {}, "decode": {}, "startup_timeout_s": 1},
                        registration(role),
                        state.client(),
                        role,
                        output,
                        factory,
                        lambda path: {
                            "passed": checker_passes,
                            "errors": [] if checker_passes else ["bad answer"],
                        },
                        status,
                        interval=0.002,
                    )
                    if not expired:
                        self.assertTrue(status.exists())
                except Exception as error:
                    exceptions.append(error)

            threads = [
                threading.Thread(target=execute, args=(role,), daemon=True)
                for role in remote.ROLES
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
            self.assertFalse(
                any(thread.is_alive() for thread in threads), "Coordination deadlocked"
            )
            self.assertEqual(exceptions, [])
            self.assertEqual(set(results), set(remote.ROLES))
            self.assertTrue(all(worker.closed for worker in workers.values()))
        return results, workers, state

    def test_both_shards_pass_only_after_both_cleanup(self):
        results, workers, state = self.run_pair()
        self.assertTrue(all(result["passed"] for result in results.values()))
        self.assertTrue(workers["prefill"].requested)
        self.assertFalse(workers["decode"].requested)
        for result in results.values():
            self.assertTrue(
                all(
                    record["cleanup"]["passed"]
                    for record in result["paired_state"]["roles"].values()
                )
            )
        self.assertIn(("ready", "decode"), state.events)

    def test_checker_failure_fails_both_shards_and_closes_services(self):
        results, _, _ = self.run_pair(checker_passes=False)
        self.assertTrue(all(not result["passed"] for result in results.values()))
        self.assertIn("bad answer", " ".join(results["prefill"]["errors"]))
        self.assertIn("result-check.json", " ".join(results["prefill"]["errors"]))

    def test_prepare_failure_fails_peer_before_model_start(self):
        results, workers, _ = self.run_pair({"prefill": "prepare"})
        self.assertTrue(all(not result["passed"] for result in results.values()))
        self.assertFalse(any(worker.started for worker in workers.values()))

    def test_broker_disconnect_interrupts_active_requests(self):
        results, workers, _ = self.run_pair({"prefill": "disconnect"})
        self.assertTrue(workers["prefill"].disconnected.is_set())
        self.assertTrue(all(not result["passed"] for result in results.values()))

    def test_cleanup_failure_fails_both_results(self):
        results, _, _ = self.run_pair({"decode": "cleanup"})
        self.assertTrue(all(not result["passed"] for result in results.values()))

    def test_expired_run_never_starts_model(self):
        results, workers, _ = self.run_pair(expired=True)
        self.assertTrue(all(not result["passed"] for result in results.values()))
        self.assertFalse(any(worker.started for worker in workers.values()))

    def test_failure_prints_logs_after_cleanup_and_success_is_silent(self):
        def close(worker):
            worker.closed = True
            (worker.task / "service.log").write_text(f"{worker.role} final log\n")

        for passes in (True, False):
            with self.subTest(passes=passes), mock.patch.object(
                FakeWorker, "close", close
            ), redirect_stdout(io.StringIO()) as output:
                results, _, _ = self.run_pair(checker_passes=passes)
            if passes:
                self.assertEqual(output.getvalue(), "")
                self.assertTrue(all(result["passed"] for result in results.values()))
            else:
                for role in remote.ROLES:
                    self.assertIn(f"{role} final log", output.getvalue())
                    self.assertFalse(results[role]["passed"])
                self.assertIn("bad answer", " ".join(results["prefill"]["errors"]))


class FailureLogTest(unittest.TestCase):
    def test_bounded_tail_redacts_token_and_reads_only_named_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            last = b"\nlast service line\n"
            (task / "service.log").write_bytes(
                b"discarded prefix\n" + b"x" * remote.FAILURE_LOG_TAIL_BYTES + last
            )
            (task / "requests.log").write_bytes(b"request \xff secret-broker-token\n")
            (task / "config.json").write_text("config must not be printed")
            (task / "core-test").write_text("core must not be printed")
            with mock.patch.dict(
                "os.environ", {"K3_PD_BROKER_TOKEN": "secret-broker-token"}
            ), redirect_stdout(io.StringIO()) as output:
                remote._print_failure_logs(task, "decode")
            text = output.getvalue()
            self.assertNotIn("discarded prefix", text)
            service_tail = text.split(
                f"(last {remote.FAILURE_LOG_TAIL_BYTES} bytes)\n", 1
            )[1].split("\nK3 decode failure log:", 1)[0]
            self.assertEqual(
                service_tail,
                "x" * (remote.FAILURE_LOG_TAIL_BYTES - len(last)) + last.decode() + "\n",
            )
            self.assertIn("request \ufffd <redacted>", text)
            self.assertNotIn("secret-broker-token", text)
            self.assertNotIn("must not be printed", text)

    def test_missing_and_unreadable_logs_do_not_replace_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            with redirect_stdout(io.StringIO()) as output:
                remote._print_failure_logs(task, "decode")
            self.assertEqual(output.getvalue(), "")
            (task / "service.log").mkdir()
            (task / "requests.log").write_text("request error\n")
            with redirect_stdout(io.StringIO()) as output:
                remote._print_failure_logs(task, "decode")
            self.assertIn("IsADirectoryError", output.getvalue())
            self.assertIn("request error", output.getvalue())
            with mock.patch("builtins.print", side_effect=BrokenPipeError):
                remote._print_failure_logs(task, "decode")


if __name__ == "__main__":
    unittest.main()
