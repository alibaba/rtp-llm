import json
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from rtp_llm.test.perf_test.aone.broker import run_gang_broker


def _default_resolver(address):
    return _LOOPBACK_HOSTS.get(address)


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _request(
    broker,
    op,
    *,
    method="POST",
    payload=None,
    token=None,
    run_id=None,
    source_address=None,
    timeout=30.0,
):
    """Issue one HTTP request against the live broker socket.

    ``token`` semantics: ``None`` uses the broker token, ``False`` omits the
    authorization header, and any other string is sent verbatim.
    """
    target = urlsplit(broker.url)
    headers = {}
    if token is not False:
        value = broker.token if token is None else token
        headers["Authorization"] = f"Bearer {value}"
    connection = HTTPConnection(
        target.hostname,
        target.port,
        timeout=timeout,
        source_address=(source_address, 0) if source_address else None,
    )
    body = None if payload is None else json.dumps(payload)
    connection.request(
        method,
        f"/v1/runs/{(broker.run_id if run_id is None else run_id)}/{op}",
        body=body,
        headers=headers,
    )
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    parsed = json.loads(raw.decode("utf-8")) if raw else {}
    return (response.status, parsed)


def _register(
    broker,
    shard_index,
    hostname,
    port,
    *,
    token=None,
    run_id=None,
    source_address=None,
    config_fingerprint=None,
):
    payload = {"shard_index": shard_index, "hostname": hostname, "port": port}
    if config_fingerprint is not None:
        payload["config_fingerprint"] = config_fingerprint
    return _request(
        broker,
        "register",
        payload=payload,
        token=token,
        run_id=run_id,
        source_address=source_address,
    )


class GangBrokerTest(unittest.TestCase):

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state_dir = Path(temporary.name)

    def _start_broker(self, machine_count=1, resolver=None, **kwargs):
        broker = run_gang_broker(
            machine_count,
            state_dir=self.state_dir,
            host=_LOOPBACK,
            resolver=_default_resolver if resolver is None else resolver,
            **kwargs,
        )
        broker.start()
        self.addCleanup(broker.stop)
        return broker

    def test_duplicate_address_or_hostname_fails_the_whole_gang(self):
        for hostname, source in (("host-b", None), ("HOST-A.", "127.0.0.2")):
            with self.subTest(hostname=hostname):
                broker = self._start_broker(machine_count=2)
                self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])
                status, body = _register(
                    broker, 1, hostname, 9001, source_address=source
                )
                self.assertEqual((409, "duplicate_host"), (status, body["error"]))
                self.assertEqual("failed", broker.gang_state)

    def test_duplicate_shard_is_rejected_without_replacing_its_member(self):
        broker = self._start_broker(machine_count=2)
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])
        self.assertEqual(409, _register(broker, 0, "host-b", 9001)[0])
        self.assertEqual(["host-a"], [member.hostname for member in broker.members])

    def test_missing_member_and_run_deadlines_fail_the_gang(self):
        for deadline, reason in (
            ("register_deadline_s", "register_deadline_exceeded"),
            ("run_deadline_s", "run_deadline_exceeded"),
        ):
            with self.subTest(deadline=deadline):
                broker = self._start_broker(machine_count=2, **{deadline: 0.05})
                self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])
                self.assertTrue(_wait_for(lambda: broker.gang_state == "failed"))
                self.assertEqual(reason, broker.failure_reason)

    def test_rejected_auth_or_fingerprint_cannot_register_a_member(self):
        broker = self._start_broker(expected_config_fingerprint="planned")
        self.assertEqual(401, _register(broker, 0, "host-a", 9000, token="wrong")[0])
        self.assertEqual(
            400, _register(broker, 0, "host-a", 9000, config_fingerprint="wrong")[0]
        )
        self.assertEqual((), broker.members)
        self.assertEqual(
            200, _register(broker, 0, "host-a", 9000, config_fingerprint="planned")[0]
        )

    def test_heartbeat_expiry_marks_member_lost(self):
        broker = self._start_broker(
            machine_count=1,
            register_deadline_s=60.0,
            run_deadline_s=60.0,
            heartbeat_expiry_s=0.25,
        )
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])
        self.assertEqual(200, _request(broker, "members", method="GET")[0])
        self.assertEqual(
            200, _request(broker, "heartbeat", payload={"shard_index": 0})[0]
        )
        self.assertTrue(_wait_for(lambda: broker.gang_state == "failed"))
        self.assertEqual("heartbeat_expired", broker.failure_reason)
        state = json.loads((self.state_dir / "broker_state.json").read_text())
        self.assertEqual([0], state["lost_shard_indices"])

    def test_http_registration_readiness_completion_and_secret_safe_diagnostics(self):
        broker = self._start_broker(machine_count=2)
        self.assertEqual(
            401,
            _request(
                broker,
                "register",
                token=False,
                payload={"shard_index": 0, "hostname": "host-a", "port": 9000},
            )[0],
        )
        self.assertEqual("registering", broker.gang_state)
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])
        self.assertEqual("registering", broker.gang_state)
        response = []

        def members():
            response.append(_request(broker, "members", method="GET", timeout=3))

        waiter = threading.Thread(target=members, daemon=True)
        waiter.start()
        try:
            self.assertEqual(
                200, _register(broker, 1, "host-b", 9001, source_address="127.0.0.2")[0]
            )
            waiter.join(timeout=4)
            self.assertFalse(waiter.is_alive())
            status, body = response[0]
        finally:
            if waiter.is_alive():
                broker.stop()
                waiter.join(timeout=4)
        self.assertEqual(200, status)
        self.assertEqual("ready", body["gang_state"])
        self.assertEqual([0, 1], [member["shard_index"] for member in body["members"]])
        self.assertEqual(
            200,
            _request(
                broker, "result", payload={"shard_index": 0, "state": "completed"}
            )[0],
        )
        self.assertEqual("ready", broker.gang_state)
        self.assertEqual(
            200,
            _request(
                broker,
                "result",
                payload={"shard_index": 1, "state": "completed"},
                source_address="127.0.0.2",
            )[0],
        )
        self.assertEqual("completed", broker.gang_state)
        self.assertEqual(200, _request(broker, "cleanup")[0])
        state = json.loads((self.state_dir / "broker_state.json").read_text())
        self.assertTrue(state["cleaned"])
        self.assertEqual("completed", state["gang_state"])
        self.assertTrue((self.state_dir / "broker_events.jsonl").is_file())
        for path in self.state_dir.iterdir():
            if path.is_file():
                self.assertNotIn(broker.token, path.read_text())

    def test_failed_member_terminates_gang_and_rejects_waiting_members(self):
        broker = self._start_broker(machine_count=2)
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])
        self.assertEqual(
            200,
            _request(
                broker,
                "result",
                payload={
                    "shard_index": 0,
                    "state": "failed",
                    "detail": "engine crashed",
                },
            )[0],
        )
        self.assertEqual("failed", broker.gang_state)
        self.assertEqual("member_failed", broker.failure_reason)
        (status, body) = _request(broker, "members", method="GET", timeout=2)
        self.assertEqual(504, status)
        self.assertEqual("member_failed", body["error"])


_LOOPBACK = "127.0.0.1"
_LOOPBACK_HOSTS = {_LOOPBACK: "host-a", "127.0.0.2": "host-b"}

if __name__ == "__main__":
    unittest.main()
