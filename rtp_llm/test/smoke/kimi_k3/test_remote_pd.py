import copy
import io
import json
import os
import tempfile
import time
import unittest
import urllib.error
from contextlib import redirect_stdout
from email.message import Message
from pathlib import Path
from unittest import mock

import remote_pd_test as remote

RUN_ID = "k3-pd-unittest"


def registration(role):
    return {
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

    def test_pair_rejects_conflicting_worker_identities(self):
        self.assertTrue(remote.check_pair(snapshot(True), prepared=True))
        for mutation in (
            lambda state: state["roles"]["decode"]["registration"].update(
                hostname="prefill-host"
            ),
            lambda state: state["roles"]["decode"]["registration"].update(
                ip="10.0.0.1"
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
                remote.check_pair(value, prepared=True)

    def test_rejects_nonroutable_address(self):
        for address in ("127.0.0.1", "0.0.0.0", "169.254.1.2", "224.0.0.1", "::1"):
            value = registration("prefill")
            value["ip"] = address
            with self.subTest(address=address), self.assertRaises(ValueError):
                remote.validate_registration(value)


class EntryPointTest(unittest.TestCase):
    def test_runtime_metadata_and_shard_exit_status(self):
        for index, passed in enumerate((True, False)):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = {role: {} for role in remote.ROLES}
                environment = {
                    "TEST_TOTAL_SHARDS": "2",
                    "TEST_SHARD_INDEX": str(index),
                    "TEST_SHARD_STATUS_FILE": str(root / "shard-status"),
                    "TEST_UNDECLARED_OUTPUTS_DIR": str(root),
                    "K3_PD_BROKER_URL": "http://broker:8080",
                    "K3_PD_BROKER_TOKEN": "test-token",
                    "K3_PD_RUN_ID": RUN_ID,
                }
                summary = {
                    "passed": passed,
                    "role": remote.ROLES[index],
                    "errors": [] if passed else ["request failed"],
                    "cleanup_errors": [],
                }
                with (
                    mock.patch.dict(os.environ, environment, clear=True),
                    mock.patch.object(remote, "Broker") as broker,
                    mock.patch.object(
                        remote.socket, "gethostname", return_value="pd-host"
                    ),
                    mock.patch.object(remote, "advertise_ip", return_value="10.0.0.1"),
                    mock.patch.object(remote.signal, "signal"),
                    mock.patch.object(remote, "run_shard", return_value=summary) as run,
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(remote.main(config), 0 if passed else 1)
                broker.assert_called_once_with(
                    "http://broker:8080", "test-token", RUN_ID
                )
                self.assertEqual(run.call_args.args[0], config)
                self.assertEqual(
                    run.call_args.args[1], {"hostname": "pd-host", "ip": "10.0.0.1"}
                )
                self.assertEqual(run.call_args.args[3], remote.ROLES[index])
                self.assertTrue((root / "shard-status").is_file())
                self.assertEqual(config, {role: {} for role in remote.ROLES})


class BrokerTest(unittest.TestCase):
    def test_retries_identical_write_after_connection_loss(self):
        broker = remote.Broker("https://broker", "test-token", RUN_ID)
        broker.opener = mock.Mock()
        broker.opener.open.side_effect = [
            urllib.error.URLError("connection reset"),
            Response(snapshot()),
        ]
        with mock.patch.object(remote.time, "sleep"):
            broker.request("heartbeat", "prefill", {})
        first, second = broker.opener.open.call_args_list
        self.assertEqual(first.args[0].data, second.args[0].data)

    def test_rejected_request_is_not_retried(self):
        broker = remote.Broker("https://broker", "test-token", RUN_ID)
        broker.opener = mock.Mock()
        broker.opener.open.side_effect = urllib.error.HTTPError(
            broker.url, 403, "Forbidden", {}, io.BytesIO()
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP 403"):
            broker.request()
        broker.opener.open.assert_called_once()

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


class DiagnosticsTest(unittest.TestCase):
    def test_rejected_observation_does_not_prevent_next_record(self):
        broker = mock.Mock()
        broker.send_diagnostic.side_effect = [OSError("unavailable"), None]
        reporter = remote.ShardDiagnostics(broker, "decode")
        reporter.record("launcher_started", {"elapsed_ms": 1})
        reporter.record("health_wait", {"elapsed_ms": 2})
        reporter.close()
        self.assertEqual(
            [
                json.loads(call.args[1])
                for call in broker.send_diagnostic.call_args_list
            ],
            [
                {
                    "role": "decode",
                    "phase": "launcher_started",
                    "data": {"elapsed_ms": 1},
                },
                {"role": "decode", "phase": "health_wait", "data": {"elapsed_ms": 2}},
            ],
        )


class FailureLogTest(unittest.TestCase):
    def test_failure_logs_redact_the_broker_token_and_exclude_other_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            token = "secret-broker-token"
            (task / "service.log").write_text(f"request with {token}")
            (task / "config.json").write_text("private configuration")
            with redirect_stdout(io.StringIO()) as output:
                remote._print_failure_logs(task, "decode", token)
            text = output.getvalue()
            self.assertIn("request with <redacted>", text)
            self.assertNotIn(token, text)
            self.assertNotIn("private configuration", text)


if __name__ == "__main__":
    unittest.main()
