#!/usr/bin/env python3
"""CPU-only tests for the multi-cycle PD level-3 E2E harness."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sleep_mode_integration import run_level3_pd_scenario as scenario


class FakePdDeployment:
    def __init__(
        self,
        bad_awake_request: int = 0,
        fail_sleep_role: str = "",
        fail_wake_role: str = "",
        raise_sleeping_inference: bool = False,
    ) -> None:
        self.states = {"decode": "RUNNING", "prefill": "RUNNING"}
        self.epochs = {"decode": 0, "prefill": 0}
        self.pids = {"decode": {101, 102}, "prefill": {201, 202}}
        self.starttimes = {101: 1001, 102: 1002, 201: 2001, 202: 2002}
        self.bad_awake_request = bad_awake_request
        self.fail_sleep_role = fail_sleep_role
        self.fail_wake_role = fail_wake_role
        self.raise_sleeping_inference = raise_sleeping_inference
        self.awake_requests = 0
        self.sleeping_requests = 0
        self.transitions = []

    def http(self, role, method, path, body=None, timeout=120):
        del timeout
        if method == "GET" and path == "/health":
            return 200, {"status": "ok"}
        if method == "GET" and path == "/sleep_status":
            state = self.states[role]
            return 200, {
                "state": state,
                "supported_levels": [1, 2, 3],
                "process_id": min(self.pids[role]),
                "process_ids": sorted(self.pids[role]),
                "sleep_epoch": self.epochs[role],
                "gpu_resource_state": (
                    "RELEASED" if state == "CHECKPOINTED" else "ACTIVE"
                ),
                "device_kv_cache_valid": False,
                "kv_memory_state": "ACTIVE" if state == "RUNNING" else "DISCARDED",
            }
        if method == "POST" and path == "/sleep":
            self.transitions.append(("sleep", role, body["level"]))
            if role == self.fail_sleep_role:
                return 500, {"status": "error", "role": role}
            self.states[role] = "CHECKPOINTED"
            self.epochs[role] += 1
            return 200, {"status": "ok"}
        if method == "POST" and path == "/wake_up":
            self.transitions.append(("wake", role, None))
            if role == self.fail_wake_role:
                return 500, {"status": "error", "role": role}
            self.states[role] = "RUNNING"
            return 200, {"status": "ok"}
        raise AssertionError(f"unexpected HTTP call: {role} {method} {path}")

    def infer(self, timeout=600):
        del timeout
        if all(state == "CHECKPOINTED" for state in self.states.values()):
            self.sleeping_requests += 1
            if self.raise_sleeping_inference:
                raise RuntimeError("injected sleeping inference failure")
            return 503, {"error": "sleeping"}
        self.awake_requests += 1
        text = scenario.EXPECTED_TEXT
        if self.awake_requests == self.bad_awake_request:
            text = "different text"
        return 200, {"choices": [{"message": {"content": text}}]}

    def backend_statuses(self, role):
        return [
            {
                "address": f"{role}-{rank}",
                "state": self.states[role],
                "process_id": pid,
                "process_starttime": self.starttimes[pid],
            }
            for rank, pid in enumerate(sorted(self.pids[role]))
        ]

    def memory_snapshot(self, role_pids):
        checkpointed = all(state == "CHECKPOINTED" for state in self.states.values())
        per_pid = 0 if checkpointed else 1024
        per_gpu = 4 if checkpointed else 4096
        return {
            "rank_roles": {
                str(pid): role for role, pids in role_pids.items() for pid in pids
            },
            "rank_process_memory_mib": {
                str(pid): per_pid for pids in role_pids.values() for pid in pids
            },
            "rank_process_memory_by_physical_gpu_mib": {
                str(pid): {
                    gpu: per_pid for gpu in scenario._gpu_ids(scenario.ROLE_GPUS[role])
                }
                for role, pids in role_pids.items()
                for pid in pids
            },
            "physical_gpu_memory_mib": {gpu: per_gpu for gpu in scenario.all_gpu_ids()},
        }


class Level3PdScenarioTest(unittest.TestCase):
    def run_scenario(self, deployment):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.json"
            stdout = io.StringIO()
            with (
                mock.patch.object(scenario, "SLEEP_WAKE_CYCLES", 3),
                mock.patch.object(scenario, "EXPECTED_RANKS_PER_ROLE", 2),
                mock.patch.object(scenario, "MEMORY_SETTLE_SECONDS", 0),
                mock.patch.object(scenario, "SUMMARY_PATH", str(summary_path)),
                mock.patch.object(scenario, "http", side_effect=deployment.http),
                mock.patch.object(scenario, "infer", side_effect=deployment.infer),
                mock.patch.object(
                    scenario,
                    "backend_statuses",
                    side_effect=deployment.backend_statuses,
                ),
                mock.patch.object(
                    scenario,
                    "process_starttime",
                    side_effect=lambda pid: deployment.starttimes[pid],
                ),
                mock.patch.object(
                    scenario,
                    "memory_snapshot",
                    side_effect=deployment.memory_snapshot,
                ),
                contextlib.redirect_stdout(stdout),
            ):
                return_code = scenario.main()
            file_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        marker_lines = [
            line
            for line in stdout.getvalue().splitlines()
            if line.startswith("PD_LEVEL3_SUMMARY_JSON=")
        ]
        self.assertEqual(len(marker_lines), 1)
        stdout_summary = json.loads(marker_lines[0].split("=", 1)[1])
        self.assertEqual(stdout_summary, file_summary)
        return return_code, file_summary

    def test_three_cycles_preserve_output_processes_and_memory_baseline(self):
        deployment = FakePdDeployment()

        return_code, summary = self.run_scenario(deployment)

        self.assertEqual(return_code, 0)
        self.assertTrue(summary["success"])
        self.assertEqual(summary["cycles_requested"], 3)
        self.assertEqual(summary["cycles_completed"], 3)
        self.assertEqual(deployment.awake_requests, 7)
        self.assertEqual(deployment.sleeping_requests, 3)
        expected_transitions = []
        for _ in range(3):
            expected_transitions.extend(
                [
                    ("sleep", "decode", 3),
                    ("sleep", "prefill", 3),
                    ("wake", "prefill", None),
                    ("wake", "decode", None),
                ]
            )
        self.assertEqual(deployment.transitions, expected_transitions)
        for cycle in summary["cycles"]:
            self.assertTrue(cycle["completed"])
            self.assertTrue(cycle["pre_sleep_inference"]["matches_expected"])
            self.assertTrue(cycle["post_wake_inference"]["matches_expected"])
            self.assertEqual(
                set(cycle["checkpointed_memory"]["rank_process_memory_mib"].values()),
                {0},
            )
            self.assertEqual(
                set(cycle["checkpointed_memory"]["physical_gpu_memory_mib"].values()),
                {4},
            )

    def test_post_wake_text_mismatch_fails_scenario(self):
        # Awake request 1 is baseline, 2 is cycle-1 pre-sleep, and 3 is post-wake.
        deployment = FakePdDeployment(bad_awake_request=3)

        return_code, summary = self.run_scenario(deployment)

        self.assertEqual(return_code, 1)
        self.assertFalse(summary["success"])
        self.assertEqual(summary["cycles_completed"], 0)
        self.assertEqual(
            deployment.transitions,
            [
                ("sleep", "decode", 3),
                ("sleep", "prefill", 3),
                ("wake", "prefill", None),
                ("wake", "decode", None),
            ],
        )
        self.assertTrue(
            any(
                failure["name"]
                == "cycle 1 post-wake inference matches baseline and golden"
                for failure in summary["checks"]["failures"]
            )
        )

    def test_sleep_failure_stops_sequence_and_only_cleans_checkpointed_role(self):
        deployment = FakePdDeployment(fail_sleep_role="prefill")

        return_code, summary = self.run_scenario(deployment)

        self.assertEqual(return_code, 1)
        self.assertFalse(summary["success"])
        self.assertEqual(summary["cycles_completed"], 0)
        self.assertEqual(deployment.awake_requests, 2)
        self.assertEqual(deployment.sleeping_requests, 0)
        self.assertEqual(
            deployment.transitions,
            [
                ("sleep", "decode", 3),
                ("sleep", "prefill", 3),
                ("wake", "decode", None),
            ],
        )

    def test_decode_sleep_failure_does_not_sleep_or_wake_prefill(self):
        deployment = FakePdDeployment(fail_sleep_role="decode")

        return_code, summary = self.run_scenario(deployment)

        self.assertEqual(return_code, 1)
        self.assertFalse(summary["success"])
        self.assertEqual(summary["cycles_completed"], 0)
        self.assertEqual(deployment.awake_requests, 2)
        self.assertEqual(deployment.sleeping_requests, 0)
        self.assertEqual(deployment.transitions, [("sleep", "decode", 3)])

    def test_prefill_wake_failure_is_not_retried_and_blocks_decode(self):
        deployment = FakePdDeployment(fail_wake_role="prefill")

        return_code, summary = self.run_scenario(deployment)

        self.assertEqual(return_code, 1)
        self.assertFalse(summary["success"])
        self.assertEqual(summary["cycles_completed"], 0)
        self.assertEqual(deployment.awake_requests, 2)
        self.assertEqual(deployment.sleeping_requests, 1)
        self.assertEqual(
            deployment.transitions,
            [
                ("sleep", "decode", 3),
                ("sleep", "prefill", 3),
                ("wake", "prefill", None),
            ],
        )

    def test_decode_wake_failure_is_not_retried(self):
        deployment = FakePdDeployment(fail_wake_role="decode")

        return_code, summary = self.run_scenario(deployment)

        self.assertEqual(return_code, 1)
        self.assertFalse(summary["success"])
        self.assertEqual(summary["cycles_completed"], 0)
        self.assertEqual(deployment.awake_requests, 2)
        self.assertEqual(deployment.sleeping_requests, 1)
        self.assertEqual(
            deployment.transitions,
            [
                ("sleep", "decode", 3),
                ("sleep", "prefill", 3),
                ("wake", "prefill", None),
                ("wake", "decode", None),
            ],
        )

    def test_exception_cleanup_wakes_prefill_before_decode(self):
        deployment = FakePdDeployment(raise_sleeping_inference=True)

        return_code, summary = self.run_scenario(deployment)

        self.assertEqual(return_code, 1)
        self.assertFalse(summary["success"])
        self.assertEqual(summary["cycles_completed"], 0)
        self.assertEqual(
            deployment.transitions,
            [
                ("sleep", "decode", 3),
                ("sleep", "prefill", 3),
                ("wake", "prefill", None),
                ("wake", "decode", None),
            ],
        )

    def test_memory_snapshot_keeps_per_rank_and_physical_gpu_views(self):
        processes = {
            "4": {101: 12, 102: 0},
            "5": {102: 34},
            "6": {201: 56},
            "7": {202: 78},
        }
        cards = {"4": 100, "5": 200, "6": 300, "7": 400}
        with (
            mock.patch.object(scenario, "gpu_processes_by_gpu", return_value=processes),
            mock.patch.object(scenario, "gpu_memory", return_value=cards),
        ):
            snapshot = scenario.memory_snapshot(
                {"decode": {101, 102}, "prefill": {201, 202}}
            )

        self.assertEqual(
            snapshot["rank_process_memory_mib"],
            {"101": 12, "102": 34, "201": 56, "202": 78},
        )
        self.assertEqual(snapshot["physical_gpu_memory_mib"], cards)


if __name__ == "__main__":
    unittest.main()
