"""CPU integration of real HTTP rendezvous with two concurrent role runners."""

import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import remote_pd_test as remote
from remote_pd_broker import Rendezvous, make_handler

PINS = {"external_revision": "a" * 40, "internal_revision": "b" * 40}
RUN_ID = "k3-pd-onlineprotocol"
TOKEN = "f" * 64


def registration(role):
    return {
        **PINS,
        "hostname": role + "-host",
        "ip": "10.0.0.1" if role == "prefill" else "10.0.0.2",
    }


def prepared_identity(role):
    def checkpoint(count):
        return {
            "config_sha256": "c" * 64,
            "index_sha256": "d" * 64,
            "shards": [
                [f"model-{index:05d}-of-{count:05d}.safetensors", 1024]
                for index in range(count)
            ],
        }

    return {
        **PINS,
        "hostname": role + "-host",
        "gpu_uuids": [f"GPU-{role}-{index}" for index in range(8)],
        "checkpoint_identity": checkpoint(96),
        "draft_identity": checkpoint(9),
        "guard_sha256": "e" * 64,
    }


class NativeWorker:
    """Replace only GPU/model operations, leaving the actual protocol intact."""

    def __init__(self, role, disconnected, output, requests_entered, failure):
        self.role, self.disconnected = role, disconnected
        self.task = output / "native"
        self.requests_entered, self.failure = requests_entered, failure
        self.started, self.closed, self.requested = False, False, False

    def prepare_native(self, config, role, run_id):
        if self.failure == "prepare":
            raise RuntimeError("injected prepare failure")
        assert role == self.role and run_id == RUN_ID
        assert config[role]["artifact_root"] == str(self.task.parent)
        assert config["prefill"]["ip"] == "10.0.0.1"
        assert config["decode"]["ip"] == "10.0.0.2"
        self.task.mkdir()
        return prepared_identity(role)

    def start(self):
        self.started = True

    def status(self):
        if not self.started:
            raise AssertionError("status before native start")
        return {"healthy": True}

    def requests(self, message):
        if self.role != "prefill":
            raise AssertionError("requests must execute on Prefill")
        self.requested = True
        self.requests_entered.set()
        if self.failure == "wait_disconnect":
            if not self.disconnected.wait(5):
                raise AssertionError("watchdog did not interrupt active requests")
            raise RuntimeError("native requests interrupted")
        # Keep requests alive for several real HTTP heartbeat exchanges.
        if self.disconnected.wait(0.15):
            raise RuntimeError("native requests interrupted")
        return {"requests_passed": True}

    def close(self):
        self.closed = True
        if self.failure == "cleanup":
            raise RuntimeError("injected cleanup failure")


class OnlineProtocolTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def serve(self, timeout=30):
        rendezvous = Rendezvous(RUN_ID, timeout, self.root / "broker")
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(rendezvous, TOKEN))
        server.daemon_threads = True
        server_thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        server_thread.start()
        closed = False

        def close():
            nonlocal closed
            if not closed:
                closed = True
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=2)

        self.addCleanup(close)
        url = f"http://127.0.0.1:{server.server_port}"
        return rendezvous, url, close

    def run_pair(
        self, failures=None, checker_passes=True, timeout=30, disconnect=False
    ):
        failures = failures or {}
        rendezvous, url, close_server = self.serve(timeout)
        results, workers, exceptions = {}, {}, []
        requests_entered = threading.Event()
        checked_paths = []

        def execute(role):
            try:
                output = self.root / role
                output.mkdir()

                def factory(disconnected):
                    worker = NativeWorker(
                        role, disconnected, output, requests_entered, failures.get(role)
                    )
                    workers[role] = worker
                    return worker

                def checker(path):
                    checked_paths.append(path)
                    return {
                        "passed": checker_passes,
                        "errors": [] if checker_passes else ["injected bad answer"],
                    }

                results[role] = remote.run_shard(
                    {**PINS, "prefill": {}, "decode": {}, "startup_timeout_s": 1},
                    registration(role),
                    remote.Broker(url, TOKEN, RUN_ID, timeout=0.3),
                    role,
                    output,
                    factory,
                    checker,
                    self.root / (role + ".shard-status"),
                    interval=0.03,
                )
            except Exception as error:
                exceptions.append(error)

        threads = [
            threading.Thread(target=execute, args=(role,), daemon=True)
            for role in remote.ROLES
        ]
        for thread in threads:
            thread.start()
        if disconnect:
            self.assertTrue(requests_entered.wait(3), "requests never started")
            close_server()
        for thread in threads:
            thread.join(timeout=8)
        self.assertFalse(
            any(thread.is_alive() for thread in threads), "protocol deadlock"
        )
        self.assertEqual(exceptions, [])
        self.assertEqual(set(results), set(remote.ROLES))
        self.assertTrue(all(worker.closed for worker in workers.values()))
        for role, result in results.items():
            self.assertEqual(
                json.loads((self.root / role / "summary.json").read_text()), result
            )
            self.assertTrue((self.root / (role + ".shard-status")).is_file())
        if not disconnect:
            state = remote.Broker(url, TOKEN, RUN_ID).request()
            self.assertEqual(state, rendezvous.snapshot())
            self.assertEqual(
                json.loads((self.root / "broker" / "state.json").read_text()), state
            )
        else:
            state = rendezvous.snapshot()
        return results, workers, state, checked_paths

    def test_real_http_success_has_both_terminal_roles_and_only_prefill_requests(self):
        results, workers, state, checked = self.run_pair()
        self.assertTrue(all(result["passed"] for result in results.values()))
        self.assertIsNone(state["error"])
        self.assertTrue(workers["prefill"].requested)
        self.assertFalse(workers["decode"].requested)
        self.assertEqual(len(checked), 1)
        self.assertEqual(
            checked[0], workers["prefill"].task / "requests" / "accuracy.json"
        )
        for role in remote.ROLES:
            self.assertTrue(state["roles"][role]["result"]["passed"])
            self.assertTrue(state["roles"][role]["cleanup"]["passed"])
            self.assertTrue(
                results[role]["paired_state"]["roles"]["prefill"]["cleanup"]["passed"]
            )
            self.assertTrue(
                results[role]["paired_state"]["roles"]["decode"]["cleanup"]["passed"]
            )
        events = [
            json.loads(line)
            for line in (self.root / "broker" / "events.jsonl").read_text().splitlines()
        ]
        for role in remote.ROLES:
            self.assertEqual(
                [event["op"] for event in events if event["role"] == role],
                ["register", "prepared", "ready", "result", "cleanup"],
            )

    def test_real_http_prepare_failure_stops_both_before_native_start(self):
        results, workers, state, checked = self.run_pair({"prefill": "prepare"})
        self.assertTrue(all(not result["passed"] for result in results.values()))
        self.assertFalse(any(worker.started for worker in workers.values()))
        self.assertEqual(checked, [])
        self.assertIn("injected prepare failure", state["error"])
        self.assertTrue(
            all(record["cleanup"]["passed"] for record in state["roles"].values())
        )

    def test_real_http_checker_failure_is_shared_without_overwriting_first_cause(self):
        results, _, state, checked = self.run_pair(checker_passes=False)
        self.assertTrue(all(not result["passed"] for result in results.values()))
        self.assertEqual(len(checked), 1)
        self.assertIn("prefill result failed", state["error"])
        self.assertIn("result check failed", state["error"])
        self.assertTrue(
            all(record["cleanup"]["passed"] for record in state["roles"].values())
        )

    def test_real_http_cleanup_failure_invalidates_both_passing_request_results(self):
        results, _, state, _ = self.run_pair({"decode": "cleanup"})
        self.assertTrue(all(not result["passed"] for result in results.values()))
        self.assertTrue(
            all(record["result"]["passed"] for record in state["roles"].values())
        )
        self.assertFalse(state["roles"]["decode"]["cleanup"]["passed"])
        self.assertIn("injected cleanup failure", state["error"])

    def test_real_http_deadline_interrupts_requests_and_accepts_final_cleanup(self):
        results, workers, state, _ = self.run_pair(
            {"prefill": "wait_disconnect"}, timeout=2
        )
        self.assertTrue(all(not result["passed"] for result in results.values()))
        self.assertTrue(workers["prefill"].disconnected.is_set())
        self.assertIn("deadline", state["error"].lower())
        self.assertTrue(
            all(record["cleanup"]["passed"] for record in state["roles"].values())
        )

    def test_real_transport_loss_interrupts_requests_and_closes_both_workers(self):
        results, workers, _, checked = self.run_pair(
            {"prefill": "wait_disconnect"}, disconnect=True
        )
        self.assertTrue(all(not result["passed"] for result in results.values()))
        self.assertTrue(
            all(worker.disconnected.is_set() for worker in workers.values())
        )
        self.assertEqual(checked, [])
        self.assertTrue(all(result["errors"] for result in results.values()))

    def test_real_http_wrong_token_rejected_without_role_registration(self):
        rendezvous, url, _ = self.serve()
        with self.assertRaisesRegex(RuntimeError, "Broker connection failed"):
            remote.Broker(url, "wrong-token", RUN_ID).request(
                "register", "prefill", registration("prefill")
            )
        self.assertEqual(rendezvous.snapshot()["roles"], {})


if __name__ == "__main__":
    unittest.main()
