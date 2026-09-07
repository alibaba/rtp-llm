"""Recovery owner/log evidence and complete YAML with an independent model."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_ft.scenario.actions import engine_control
from flexlb_ft.scenario.actions import engine_recovery as recovery
from flexlb_ft.scenario.catalog import handlers
from flexlb_ft.scenario.compiler import compile_scenarios
from flexlb_ft.scenario.loader import load_scenarios
from flexlb_ft.scenario.runtime import Deadline, execute_instance
from test_scenario_backend import Ops
from test_scenario_runtime import Clock
from test_scenario_status_protocol import owner_frame


class Model:
    def __init__(self, mode="correct"):
        self.mode, self.restored, self.cleaned = mode, False, False
        self.ops = Ops()
        future = self.ops.future

        def compatible_future(*args, **kwargs):
            call = future(*args, **kwargs)
            result = call.result
            call.result = lambda timeout=None: result(timeout=timeout)
            return call

        self.ops.future = compatible_future
        self.engines = {}
        for role in ("prefill", "decode"):
            for i in range(2):
                name = f"{role}-{i}"
                self.engines[name] = dict(
                    name=name,
                    role=role,
                    http_addr=f"{name}:9000",
                    grpc_addr=f"{name}:9001",
                    port=9001,
                    stopped=False,
                )

    def setup(self, ctx, environment, deadline):
        self.ops.batch = "nonbatch" not in ctx.instance["profile"]
        directory = ctx.artifact_dir / "master-sync"
        directory.mkdir()
        self.log = directory / "sync.log"
        self.log.write_text(
            "Created WorkerStatus generation 1 for worker: prefill-0:9000\n"
        )
        return NS(master_sync_log_path=self.log), self.ops

    def teardown(self, ctx, deadline):
        self.cleaned = True

    def append(self, line):
        with self.log.open("a") as stream:
            stream.write(line + "\n")

    def control(self, ops, endpoint, deadline, body=None):
        if endpoint == "snapshot":
            return {"engines": copy.deepcopy(list(self.engines.values()))}
        name = body["engine"]
        if endpoint == "stop_engine":
            self.engines[name]["stopped"] = True
            self.append(
                f"worker {name}:9000 marked dead after 3 consecutive gRPC failures"
            )
        elif endpoint == "start_engine":
            self.engines[name]["stopped"] = False
            self.restored = True
            if self.mode != "no_generation":
                self.append(
                    f"Created WorkerStatus generation 2 for worker: {name}:9000"
                )
        else:
            raise RuntimeError(f"unexpected control {endpoint}")
        return dict(status="ok", engine=name, port=9001)

    def http(self, ctx, server, path, deadline, body=None, *args, **kwargs):
        if server == "mock":
            if path == "inject":
                if not body["enabled"] and self.mode == "jitter_bumps_generation":
                    self.append(
                        "Created WorkerStatus generation 2 for worker: prefill-0:9000"
                    )
                return 200, {"status": "ok"}
            return 200, self.control(self.ops, "snapshot", deadline)
        if path == "rtp_llm/master/info":
            alive = sum(
                not e["stopped"]
                for e in self.engines.values()
                if e["role"] == "prefill"
            )
            return 200, {
                "worker_summary": {
                    "PREFILL": {
                        "alive": alive,
                        "discovered": (
                            3 if self.restored and self.mode == "bad_topology" else 2
                        ),
                    }
                }
            }
        data = owner_frame()["inflight"]
        data["prefill_endpoints"] = [
            dict(
                ip_port=f"prefill-{i}:9000",
                inflight_batches=0,
                inflight_requests=int(
                    i == 0 and self.restored and self.mode == "old_members"
                ),
            )
            for i in range(2)
        ]
        return 200, data


class RecoveryTest(unittest.TestCase):
    def test_log_mark_excludes_previous_lines_and_similar_address(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sync.log"
            path.write_text("old for worker: p:10\n")
            stat = path.stat()
            mark = dict(
                path=str(path),
                device=stat.st_dev,
                inode=stat.st_ino,
                offset=stat.st_size,
                targets={"p": {"ip_port": "p:10"}},
            )
            with path.open("a") as stream:
                stream.write(
                    "Created WorkerStatus generation 2 for worker: p:100\nCreated WorkerStatus generation 3 for worker: p:10\nworker p:10 marked dead after 3 consecutive gRPC failures\n"
                )
            result = recovery._log_counts(mark, Deadline(100, lambda: 0))
            self.assertEqual({"created": 1, "retired": 1}, result["counts"]["p"])
            path.write_text("")
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                recovery._log_counts(mark, Deadline(100, lambda: 0))

    def test_missing_log_cannot_prove_no_generation_bump(self):
        mark = dict(
            path="/nonexistent/owned-sync.log",
            device=1,
            inode=1,
            offset=0,
            targets={"p": {"ip_port": "p:10"}},
        )
        with self.assertRaises(FileNotFoundError):
            recovery._log_counts(mark, Deadline(100, lambda: 0))

    def test_missing_target_ledger_is_error_not_zero(self):
        frame = {"targets": {"p": {"ip_port": "missing"}}, **owner_frame()}
        with self.assertRaises(RuntimeError):
            recovery.metric(frame, "target_prefill_requests")

    def test_legacy_generate_payload_does_not_change_schedule(self):
        ops = Ops(batch=False)
        proxy = recovery.GeneratePayload(ops, "legacy_default")
        self.assertEqual(
            (1, {}), proxy.build_generate_input(1, output_len=2, block_keys=[7])
        )
        self.assertEqual(
            (1, {"output_len": 2}), proxy.build_schedule_request(1, output_len=2)
        )

    def run_program(self, variant, mode, expected, failed=None):
        registry = handlers()
        registry.update({h.name: h for h in recovery.HANDLERS})
        path = (
            Path(__file__).resolve().parents[1]
            / "scenarios/engine_fault/engine_fault_recovery.yaml"
        )
        plans = [
            p
            for p in compile_scenarios(load_scenarios(path), handlers=registry)
            if p["variant_id"] == variant
        ]
        self.assertEqual(4, len(plans))
        for plan in plans:
            with self.subTest(
                profile=plan["profile"], mode=mode
            ), tempfile.TemporaryDirectory() as root:
                model, clock = Model(mode), Clock()
                with patch.object(
                    recovery.status, "_http", side_effect=model.http
                ), patch.object(engine_control, "_http", side_effect=model.control):
                    result = execute_instance(
                        plan,
                        model,
                        handlers=registry,
                        artifact_dir=root,
                        clock=clock,
                        sleeper=clock.sleep,
                    )
                self.assertEqual(expected, result["status"], result)
                self.assertTrue(model.cleaned)
                self.assertTrue(
                    all(row["status"] == "PASS" for row in result["cleanup"])
                )
                if failed:
                    self.assertEqual(
                        expected,
                        next(
                            s["status"] for s in result["stages"] if s["id"] == failed
                        ),
                    )

    def test_complete_down_phases_program(self):
        self.run_program("down_phases", "correct", "PASS")

    def test_complete_flap_program(self):
        self.run_program("flap", "correct", "PASS")

    def test_flap_alive_without_discovery_convergence_fails(self):
        self.run_program("flap", "bad_topology", "FAIL", "topology_discovered")

    def test_ttft_uses_upper_index_p50_and_missing_is_not_zero(self):
        records = []
        for latency in [1, 4, 2, 3]:
            records.append(
                {
                    "business_finished": True,
                    "business_error_code": None,
                    "cancel": {"requested_s": None},
                    "schedule": {"status": "OK", "started_s": 0},
                    "stream": {
                        "status": "OK",
                        "started_s": 0,
                        "first_output_s": latency,
                    },
                }
            )
        self.assertEqual(3000, recovery._ttft({"frames": [{"records": records}]}))
        for record in records:
            record["stream"]["first_output_s"] = None
        self.assertIsNone(recovery._ttft({"frames": [{"records": records}]}))

    def test_complete_generation_program(self):
        self.run_program("generation_bump", "correct", "PASS")

    def test_alive_recovery_without_generation_bump_fails(self):
        self.run_program(
            "generation_bump", "no_generation", "FAIL", "generation_is_new"
        )

    def test_old_prefill_members_fail_before_fresh_recovery_request(self):
        self.run_program(
            "generation_bump",
            "old_members",
            "FAIL",
            "recovered_prefill_member_ledger_zero",
        )

    def test_complete_short_gap_program(self):
        self.run_program("status_gap_no_bump", "correct", "PASS")

    def test_jitter_generation_churn_fails(self):
        self.run_program(
            "status_gap_no_bump", "jitter_bumps_generation", "FAIL", "no_new_generation"
        )


if __name__ == "__main__":
    unittest.main()
