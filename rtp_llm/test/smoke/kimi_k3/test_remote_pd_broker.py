"""CPU-only regressions for the single-run CI rendezvous state machine."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from remote_pd_broker import Rendezvous


class RendezvousTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.clock = patch("remote_pd_broker.time.time", return_value=1000.0)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        self.broker = Rendezvous("k3-pd-test123", 1000, self.output)

    def post(self, operation, role="prefill", data=None):
        return self.broker.update({"op": operation, "role": role, "data": data})

    @staticmethod
    def registration(role):
        return {
            "hostname": "worker-" + role,
            "ip": "10.0.0.1" if role == "prefill" else "10.0.0.2",
            "external_revision": "a" * 40,
            "internal_revision": "b" * 40,
        }

    def identity(self, role):
        registration = self.registration(role)
        checkpoint = {
            "config_sha256": "c" * 64,
            "index_sha256": "d" * 64,
            "shards": [["model-00001.safetensors", 1024]],
        }
        return {
            **{key: value for key, value in registration.items() if key != "ip"},
            "gpu_uuids": [f"GPU-{role}-{index}" for index in range(8)],
            "guard_sha256": "e" * 64,
            "checkpoint_identity": checkpoint,
            "draft_identity": copy.deepcopy(checkpoint),
        }

    def register_both(self):
        for role in ("prefill", "decode"):
            self.post("register", role, self.registration(role))

    def ready_both(self):
        self.register_both()
        for role in ("prefill", "decode"):
            self.post("prepared", role, self.identity(role))
        for role in ("prefill", "decode"):
            self.post("ready", role, True)

    def test_success_remains_terminal_after_deadline(self):
        self.ready_both()
        for operation in ("result", "cleanup"):
            for role in ("prefill", "decode"):
                self.post(operation, role, {"passed": True, "errors": []})
        before = self.broker.snapshot()
        self.now.return_value = 9000.0
        self.assertEqual(self.broker.snapshot(), before)
        self.broker.persist()
        self.assertEqual(json.loads((self.output / "state.json").read_text()), before)

    def test_late_heartbeat_cannot_revive_expired_worker(self):
        self.register_both()
        self.now.return_value = 1121.0
        state = self.post("heartbeat", data={})
        self.assertIn("heartbeat expired", state["error"])
        self.now.return_value = 1122.0
        self.assertEqual(self.post("heartbeat", data={})["error"], state["error"])

    def test_deadline_error_survives_late_registration(self):
        self.now.return_value = 2000.0
        state = self.post("register", data=self.registration("prefill"))
        self.assertEqual(state["error"], "CI rendezvous deadline exceeded")

    def test_conflicting_terminal_post_does_not_poison_accepted_state(self):
        self.ready_both()
        passed = {"passed": True, "errors": []}
        self.post("result", data=passed)
        before = self.broker.snapshot()
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.post("result", data={"passed": False, "errors": ["late conflict"]})
        self.assertEqual(self.broker.snapshot(), before)
        self.post("result", data=passed)
        self.assertEqual(self.broker.snapshot(), before)

    def test_failed_role_can_report_failure_and_cleanup_without_starting(self):
        self.post("register", data=self.registration("prefill"))
        failure = {"passed": False, "errors": ["source revision mismatch"]}
        state = self.post("result", data=failure)
        self.assertIn("source revision mismatch", state["error"])
        state = self.post("cleanup", data={"passed": True, "errors": []})
        self.assertEqual(state["roles"]["prefill"]["result"], failure)
        with self.assertRaisesRegex(ValueError, "finished role"):
            self.post("prepared", data=self.identity("prefill"))

    def test_passing_result_requires_started_ready_role(self):
        self.post("register", data=self.registration("prefill"))
        with self.assertRaisesRegex(ValueError, "ready role"):
            self.post("result", data={"passed": True, "errors": []})
        self.assertIsNone(self.broker.snapshot()["roles"]["prefill"]["result"])

    def test_strict_outcomes_match_runner_schema(self):
        self.post("register", data=self.registration("prefill"))
        for outcome in (
            {"passed": False, "errors": []},
            {"passed": False, "errors": [""]},
            {"passed": True, "errors": ["contradiction"]},
            {"passed": False, "errors": ["a" * 2049]},
            {"passed": False, "errors": ["error"] * 17},
            {"passed": 1, "errors": []},
        ):
            with self.subTest(outcome=outcome), self.assertRaises(ValueError):
                self.post("result", data=outcome)
        self.assertIsNone(self.broker.snapshot()["error"])
        state = self.post("result", data={"passed": False, "errors": ["x" * 2048] * 16})
        self.assertLessEqual(len(state["error"]), 2048)

    def test_exactly_eight_valid_gpu_uuids_are_required(self):
        self.post("register", data=self.registration("prefill"))
        original = self.identity("prefill")
        for uuids in (
            original["gpu_uuids"] + [original["gpu_uuids"][0]],
            original["gpu_uuids"][:-1],
            ["GPU-one"] * 8,
            list(range(8)),
            [None] * 8,
        ):
            with self.subTest(uuids=uuids), self.assertRaises(ValueError):
                self.post("prepared", data={**original, "gpu_uuids": uuids})

    def test_physical_gpu_overlap_is_a_recorded_pair_failure(self):
        self.register_both()
        self.post("prepared", data=self.identity("prefill"))
        decode = self.identity("decode")
        decode["gpu_uuids"][0] = "GPU-prefill-0"
        state = self.post("prepared", "decode", decode)
        self.assertIn("overlap physical GPUs", state["error"])

    def test_checkpoint_and_guard_mismatches_are_recorded(self):
        for key in ("checkpoint_identity", "draft_identity", "guard_sha256"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                self.broker = Rendezvous("k3-pd-test123", 1000, directory)
                self.register_both()
                self.post("prepared", data=self.identity("prefill"))
                decode = self.identity("decode")
                if key == "guard_sha256":
                    decode[key] = "f" * 64
                else:
                    decode[key]["config_sha256"] = "f" * 64
                state = self.post("prepared", "decode", decode)
                self.assertIn(key + " mismatch", state["error"])

    def test_checkpoint_metadata_must_match_wire_schema(self):
        self.post("register", data=self.registration("prefill"))
        for field, value in (
            ("config_sha256", None),
            ("index_sha256", "bad hash"),
            ("shards", []),
            ("shards", [["../model.safetensors", 1024]]),
            ("shards", [["model.safetensors", 0]]),
        ):
            with self.subTest(field=field, value=value):
                identity = self.identity("prefill")
                identity["checkpoint_identity"][field] = value
                with self.assertRaises(ValueError):
                    self.post("prepared", data=identity)

    def test_registration_conflicts_and_source_pin_mismatch(self):
        self.post("register", data=self.registration("prefill"))
        conflict = self.registration("prefill")
        conflict["hostname"] = "unexpected-worker"
        with self.assertRaisesRegex(ValueError, "different worker"):
            self.post("register", data=conflict)
        decode = self.registration("decode")
        decode["external_revision"] = "f" * 40
        self.assertIn(
            "external_revision mismatch",
            self.post("register", "decode", decode)["error"],
        )


if __name__ == "__main__":
    unittest.main()
