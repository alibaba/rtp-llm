import copy
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flexlb_ft.scenario import compile_scenarios, load_scenarios
from flexlb_ft.scenario.actions import engine_fault, master
from flexlb_ft.scenario.actions.elastic import ClientRecords
from flexlb_ft.scenario.catalog import handlers
from flexlb_ft.scenario.runtime import execute_instance


class Backend:
    def __init__(self, injected_delta=1.5, corrupt=None):
        self.injected_delta, self.corrupt = injected_delta, corrupt
        self.active = set()
        self.rid = 0

    def setup(self, ctx, environment, deadline):
        return object(), object()

    def teardown(self, ctx, deadline):
        if self.active:
            raise RuntimeError("fault cleanup did not run before environment cleanup")

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "snapshot":
            return {
                "engines": [
                    dict(
                        name=name, role="prefill", grpc_addr="host:1234", stopped=False
                    )
                    for name in ("prefill-0", "prefill-1")
                ]
            }
        if body["enabled"]:
            self.active.add(body["engine"])
        else:
            self.active.discard(body["engine"])
        return dict(status="ok", engine=body["engine"], port=1234, type=body["type"])

    def start_requests(self, ctx, params, deadline):
        self.rid += 1
        records = ClientRecords(ctx.env_epoch)
        record = records.issue(self.rid, ctx.clock)
        start = record["issued_s"]
        delay = self.injected_delta if self.active else 0
        records.update(
            record,
            schedule=dict(status="OK", started_s=start, ended_s=start + 0.1),
            stream=dict(
                status="OK",
                started_s=start + 0.1,
                first_output_s=start + 0.2 + delay,
                ended_s=start + 0.3 + delay,
            ),
            transport_terminal_s=start + 0.3 + delay,
            consumer_exit_s=start + 0.3 + delay,
            consumer_done=True,
            consumer_completion_verified=True,
            business_finished=True,
        )
        if self.corrupt:
            self.corrupt(record)
        return ctx.register_resource("requests", records)

    def wait_requests(self, ctx, records, deadline):
        return dict(completed=True, error_count=0)


class RpcMeasurementTests(unittest.TestCase):
    def plans(self):
        return compile_scenarios(
            load_scenarios(ROOT / "scenarios/engine_fault"), handlers=handlers()
        )

    def run_plan(self, plan, backend):
        with tempfile.TemporaryDirectory() as root, patch.object(
            engine_fault, "_http", side_effect=backend.http
        ), patch.object(master, "_master_json", return_value={"scheduler_inflight": 0}):
            return execute_instance(
                plan,
                backend,
                artifact_dir=root,
                handlers=handlers(),
                clock=time.monotonic,
                sleeper=time.sleep,
            )

    def test_all_shipped_variants_execute_real_handlers_and_exact_checks(self):
        plans = self.plans()
        self.assertEqual(len(plans), 6)
        for plan in plans:
            with self.subTest(instance=plan["id"]):
                result = self.run_plan(plan, Backend())
                self.assertEqual(result["status"], "PASS", result["error"])
                checks = [
                    check for stage in result["stages"] for check in stage["checks"]
                ]
                self.assertEqual(
                    len(checks),
                    6 if plan["profile"] in ("batch-window", "single-batch") else 5,
                )

    def test_insufficient_delay_is_business_fail_not_execution_error(self):
        result = self.run_plan(self.plans()[0], Backend(injected_delta=0.5))
        self.assertEqual(result["status"], "FAIL", result)
        latency = next(row for row in result["stages"] if row["id"] == "latency")
        failed = [row["id"] for row in latency["checks"] if row["status"] == "FAIL"]
        self.assertEqual(failed, ["latency_increased"])
        self.assertTrue(all(row["status"] == "PASS" for row in result["cleanup"]))

    def test_missing_exit_or_timestamp_cannot_pass_latency_check(self):
        for corrupt in (
            lambda record: record.update(consumer_completion_verified=False),
            lambda record: record["stream"].update(first_output_s=None),
            lambda record: record.update(transport_terminal_s=float("nan")),
        ):
            plan = next(p for p in self.plans() if "generate_delay" in p["id"])
            result = self.run_plan(plan, Backend(corrupt=corrupt))
            # All terminal fields must be valid even when the selected metric
            # uses only a subset; corrupt terminal evidence is not a finding.
            self.assertEqual(result["status"], "ERROR", result)
            self.assertEqual(
                next(row for row in result["stages"] if row["id"] == "latency")[
                    "status"
                ],
                "ERROR",
            )

    def test_reused_cohort_is_rejected(self):
        plan = copy.deepcopy(self.plans()[0])
        stage = next(s for s in plan["stages"] if s["id"] == "latency")
        stage["params"]["recovery"] = stage["params"]["baseline"]
        self.assertEqual(self.run_plan(plan, Backend())["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
