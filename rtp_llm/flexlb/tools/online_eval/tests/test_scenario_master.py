"""Master fault ownership and observable recovery, without Java startup."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexlb_test_framework.scenario.actions import master
from flexlb_test_framework.scenario.contracts import PlanContext
from flexlb_test_framework.scenario.runtime import Deadline, RuntimeContext


class MasterActionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.proc = SimpleNamespace(
            pid=123,
            alive=Mock(return_value=True),
            proc=Mock(),
            freeze=Mock(),
            unfreeze=Mock(),
        )
        self.manager = Mock()
        self.ctx = RuntimeContext(
            {},
            SimpleNamespace(manager=self.manager),
            self.tmp.name,
            time.monotonic,
            time.sleep,
        )
        self.ctx.ops = Mock()
        self.ctx.ops.master_target.return_value = "master"
        self.ctx.env_epoch = 1
        self.ctx.env = SimpleNamespace(
            master=self.proc,
            masters={},
            master_specs={},
            master_http_port=18080,
            spec=SimpleNamespace(n_prefill=2, n_decode=4),
        )
        self.deadline = Deadline(time.monotonic() + 10, time.monotonic, time.sleep)

    def test_kill_retains_original_process_for_reap_after_registry_replacement(self):
        def kill(env):
            env.master = None

        self.manager.kill_master9.side_effect = kill
        out = master._fault(self.ctx, dict(target="single", mode="kill"), self.deadline)
        old = self.ctx.resource(out.output["fault"], "master_fault")
        new = SimpleNamespace(pid=456)
        self.manager.start_master.return_value = new
        result = master._restore(
            self.ctx, {"fault": out.output["fault"]}, self.deadline
        )
        self.assertEqual("PASS", result.checks[0].status)
        self.assertIs(old.process, self.proc)
        self.proc.proc.wait.assert_called()
        self.assertTrue(old.restored)
        self.assertEqual(2, self.ctx.ops.invalidate_channel.call_count)
        self.ctx.ops.invalidate_channel.assert_called_with("master")
        with self.assertRaisesRegex(ValueError, "already"):
            master._restore(self.ctx, {"fault": out.output["fault"]}, self.deadline)

    def test_freeze_cleanup_thaws_same_owned_process(self):
        master._fault(self.ctx, dict(target="single", mode="freeze"), self.deadline)
        self.proc.freeze.assert_called_once()
        result = self.ctx.cleanup(5)
        self.proc.unfreeze.assert_called_once()
        self.assertTrue(all(row["status"] == "PASS" for row in result))

    def test_replaced_process_does_not_satisfy_freeze_continuity(self):
        out = master._fault(
            self.ctx, dict(target="single", mode="freeze"), self.deadline
        )
        self.ctx.env.master = SimpleNamespace(pid=456, alive=lambda: True)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            master._restore(self.ctx, {"fault": out.output["fault"]}, self.deadline)
        self.ctx.cleanup(5)
        self.proc.unfreeze.assert_called_once()

    def test_absent_ha_layout_cannot_issue_fault(self):
        with self.assertRaisesRegex(ValueError, "actual dual"):
            master._fault(self.ctx, dict(target="A", mode="kill"), self.deadline)
        self.manager.kill_master9_instance.assert_not_called()

    def test_forged_or_stale_fault_is_rejected(self):
        out = master._fault(
            self.ctx, dict(target="single", mode="freeze"), self.deadline
        )
        self.ctx.env_epoch += 1
        with self.assertRaisesRegex(ValueError, "stale"):
            master._restore(self.ctx, {"fault": out.output["fault"]}, self.deadline)
        self.ctx.cleanup(5)

    def test_readiness_records_topology_and_scheduler_owner(self):
        info = dict(
            ready=True,
            worker_summary={
                "PREFILL": dict(discovered=2, alive=2),
                "DECODE": dict(discovered=4, alive=4),
            },
        )
        with patch.object(
            master,
            "_master_json",
            side_effect=[
                info,
                {
                    "scheduler_inflight": 0,
                    "prefill_endpoints": [{"inflight_batches": 0}],
                    "decode_endpoints": [{"total_load": 0}],
                },
            ],
        ):
            result = master._ready(
                self.ctx, dict(target="single", inflight_zero=True), self.deadline
            )
        self.assertEqual(["topology", "inflight"], [c.id for c in result.checks])
        self.assertEqual(
            0,
            json.loads(Path(result.artifacts[0]).read_text())[0]["scheduler_inflight"],
        )

    def test_rejoining_topology_waits_before_reading_absent_endpoint_ledgers(self):
        offline = dict(
            ready=False,
            worker_summary={
                "PREFILL": dict(discovered=2, alive=0),
                "DECODE": dict(discovered=4, alive=0),
            },
        )
        online = dict(
            ready=True,
            worker_summary={
                "PREFILL": dict(discovered=2, alive=2),
                "DECODE": dict(discovered=4, alive=4),
            },
        )
        empty = dict(scheduler_inflight=0, prefill_endpoints=[], decode_endpoints=[])
        clean = dict(
            scheduler_inflight=0,
            prefill_endpoints=[dict(inflight_batches=0)],
            decode_endpoints=[dict(total_load=0)],
        )
        with patch.object(
            master,
            "_master_json",
            side_effect=[
                offline,
                empty,
                dict(ready=False, worker_summary=None),
                empty,
                dict(ready=False, worker_summary={}),
                empty,
                online,
                empty,
                online,
                clean,
            ],
        ):
            result = master._ready(
                self.ctx, dict(target="single", inflight_zero=True), self.deadline
            )
        samples = json.loads(Path(result.artifacts[0]).read_text())
        self.assertEqual(5, len(samples))
        self.assertIsNone(samples[0]["endpoint_loads"])
        self.assertIsNone(samples[1]["topology"]["PREFILL"])
        self.assertIsNone(samples[2]["topology"]["DECODE"])
        with patch.object(
            master,
            "_master_json",
            side_effect=[online, dict(empty, prefill_endpoints="invalid")],
        ):
            with self.assertRaisesRegex(ValueError, "invalid endpoint ledger"):
                master._ready(
                    self.ctx, dict(target="single", inflight_zero=True), self.deadline
                )

    def test_missing_or_negative_inflight_is_error_not_zero(self):
        for value in ({}, {"scheduler_inflight": -1}, {"scheduler_inflight": True}):
            with patch.object(
                master, "_master_json", side_effect=[{"worker_summary": {}}, value]
            ):
                with self.assertRaises((KeyError, ValueError)):
                    master._ready(
                        self.ctx,
                        dict(target="single", inflight_zero=True),
                        self.deadline,
                    )

    def test_endpoint_owner_counts_are_not_replaced_by_scheduler_zero(self):
        data = {
            "scheduler_inflight": 0,
            "prefill_endpoints": [{"inflight_batches": 2}],
            "decode_endpoints": [{"total_load": 3}],
        }
        self.assertEqual({"prefill": [2], "decode": [3]}, master._endpoint_loads(data))
        for rows in ([], [{}], [{"total_load": -1}], [{"total_load": True}]):
            with self.assertRaises((ValueError, KeyError)):
                master._endpoint_loads({**data, "decode_endpoints": rows})

    def test_ha_and_coldstart_fail_compilation_for_wrong_environment(self):
        plan = SimpleNamespace(path="stage", environment={})
        with self.assertRaisesRegex(ValueError, "dual_standalone"):
            master._ha_validate({}, plan)
        with self.assertRaisesRegex(ValueError, "zero master"):
            master._batch_validate({"coldstart": True}, plan)
        plan.environment = {
            "master_layout": "dual_standalone",
            "master_stable_window_s": 0,
        }
        self.assertEqual(["A", "B"], master._ha_validate({}, plan)["targets"])
        plan.environment["master_layout"] = "single"
        self.assertTrue(master._batch_validate({"coldstart": True}, plan)["coldstart"])

    def test_empty_client_rows_fail_even_zero_error_assertion(self):
        handle = self.ctx.register_resource("ha_rows", [])
        result = master._client_check(
            self.ctx,
            dict(
                rows=handle, metric="failed_count", op="eq", expected=0, min_samples=1
            ),
            self.deadline,
        )
        self.assertEqual("FAIL", result.checks[0].status)

    def test_schedule_only_rows_are_not_successful_streams(self):
        handle = self.ctx.register_resource("ha_rows", [{"status": "scheduled"}])
        result = master._client_check(
            self.ctx,
            dict(
                rows=handle, metric="success_rate", op="eq", expected=1, min_samples=1
            ),
            self.deadline,
        )
        self.assertEqual(0, result.output["actual"])
        self.assertEqual("FAIL", result.checks[0].status)

    def test_client_nonzero_exit_and_malformed_rows_are_errors(self):
        root = Path(self.tmp.name)
        process = SimpleNamespace(proc=Mock())
        client = master.OwnedHaClient(SimpleNamespace(proc=process, out_dir=root))
        process.proc.wait.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "exit code 7"):
            client.finish(self.deadline)
        process.proc.wait.return_value = 0
        for content in ("", "not json", "{}"):
            (root / "client_events.jsonl").write_text(content)
            with self.assertRaises(ValueError):
                client.finish(self.deadline)

    def test_client_windows_use_issue_epoch_and_keep_full_rows(self):
        rows = [{"send_start_epoch_ms": t * 1000, "rid": t} for t in (1, 2, 3)]
        for row in rows:
            row["wall_clock_ts"] = 1000
        handle = self.ctx.register_resource("ha_rows", rows)
        out = master._window(
            self.ctx, {"rows": handle, "from": 2, "until": 3}, self.deadline
        )
        self.assertEqual([rows[1]], self.ctx.resource(out.output["rows"], "ha_rows"))

    def test_finite_cold_batch_keeps_every_request_and_validates_distribution(self):
        from flexlb_test_framework.scenario.actions.elastic import RecordedRequests

        self.ctx.env.spec.master_stable_window_s = 0
        self.ctx.ops = SimpleNamespace(next_request_id=Mock(side_effect=range(1, 21)))

        def run(records, row, shape, timeout_s):
            records.update(
                row,
                schedule={"status": "OK"},
                stream={"status": "OK"},
                prefill_addr="p" + str(row["wire_request_id"] % 2),
                business_finished=row["wire_request_id"] <= 16,
                consumer_exit_s=time.monotonic(),
                transport_terminal_s=time.monotonic(),
            )

        info = {
            "worker_summary": {
                "PREFILL": {"discovered": 2, "alive": 2},
                "DECODE": {"discovered": 4, "alive": 4},
            }
        }
        with patch.object(RecordedRequests, "run", run), patch.object(
            master, "_master_json", return_value=info
        ):
            out = master._batch(
                self.ctx,
                dict(
                    target="single",
                    count=20,
                    concurrency=10,
                    request_timeout_s=15,
                    sample_after_s=0,
                    coldstart=True,
                ),
                self.deadline,
            )
        self.assertEqual(0.8, out.output["success_rate"])
        verdict = master._cold_check(
            self.ctx, {"snapshot": out.output["snapshot"]}, self.deadline
        )
        self.assertTrue(all(c.status == "PASS" for c in verdict.checks))
        artifact = json.loads(Path(out.artifacts[0]).read_text())
        self.assertEqual(20, len(artifact["records"]))
        self.assertTrue(all(row["consumer_exit_s"] for row in artifact["records"]))
        self.assertTrue(all(row["status"] == "PASS" for row in self.ctx.cleanup(5)))

    def test_successive_dual_probes_share_environment_request_ids(self):
        from flexlb_test_framework.scenario.actions.elastic import RecordedRequests

        self.ctx.env.spec.master_stable_window_s = 0
        self.ctx.env.master_specs = {
            "A": SimpleNamespace(bind_ip="127.0.0.1", http_port=18080)
        }
        self.ctx.ops = SimpleNamespace(next_request_id=Mock(side_effect=range(1, 41)))

        def run(records, row, shape, timeout_s):
            records.update(
                row,
                schedule={"status": "OK"},
                stream={"status": "OK"},
                prefill_addr="p0",
                business_finished=True,
                consumer_exit_s=time.monotonic(),
                transport_terminal_s=time.monotonic(),
            )

        params = dict(
            target="A",
            count=20,
            concurrency=10,
            request_timeout_s=15,
            sample_after_s=0,
            sample_topology=False,
            coldstart=False,
        )
        self.ctx.env.mock_http_port = 19000
        fresh_ops = [Mock(), Mock()]
        with patch.object(master, "_process"), patch.object(
            RecordedRequests, "run", run
        ), patch("flexlb_test_framework.engine_ops.EngineOps", side_effect=fresh_ops):
            results = [master._batch(self.ctx, params, self.deadline) for _ in range(2)]
        ids = [json.loads(Path(x.artifacts[0]).read_text())["records"] for x in results]
        self.assertEqual(
            list(range(1, 41)), [r["wire_request_id"] for group in ids for r in group]
        )
        for ops in fresh_ops:
            ops.next_request_id.assert_not_called()
        self.ctx.cleanup(5)

    def test_scheduler_missing_response_cannot_satisfy_ttl_zero(self):
        with patch.object(master, "_master_json", return_value={}):
            with self.assertRaises(KeyError):
                master._inflight(
                    self.ctx, dict(target="single", op="eq", value=0), self.deadline
                )

    def test_finished_future_without_consumer_terminal_evidence_fails_cleanup(self):
        from concurrent.futures import Future

        records = Mock()
        records.snapshot_records.return_value = [{"consumer_started": True}]
        batch = master.FiniteMasterBatch(records, 1)
        future = Future()
        future.set_result(None)
        batch.futures = [future]
        batch.rows = [
            {
                "consumer_started": True,
                "consumer_exit_s": None,
                "transport_terminal_s": None,
            }
        ]
        batch.artifact = Path(self.tmp.name) / "incomplete.json"
        with self.assertRaises(RuntimeError):
            batch.cleanup(self.deadline)
        self.assertTrue(batch.artifact.exists())

    def test_negative_windows_filter_actual_error_rows_and_reject_empty_success(self):
        rows = [
            {"status": "ok", "error_kind": "none"},
            {"status": "schedule_error", "error_kind": "business", "error": "8431"},
            {"status": "schedule_error", "error_kind": "deadline"},
        ]
        for row in rows:
            row["wall_clock_ts"] = 1000
        handle = self.ctx.register_resource("ha_rows", rows)
        out = master._window(
            self.ctx,
            {"rows": handle, "status": "schedule_error", "error_kind": "business"},
            self.deadline,
        )
        self.assertEqual([rows[1]], self.ctx.resource(out.output["rows"], "ha_rows"))
        empty = self.ctx.register_resource("ha_rows", [])
        verdict = master._client_check(
            self.ctx,
            {
                "rows": empty,
                "metric": "wrong_error_code",
                "op": "eq",
                "expected": 0,
                "code": 8431,
                "min_samples": 5,
            },
            self.deadline,
        )
        self.assertEqual("FAIL", verdict.checks[0].status)

    def test_tail_inflight_tolerance_uses_actual_count(self):
        with patch.object(
            master, "_master_json", return_value={"scheduler_inflight": 8}
        ):
            out = master._inflight(
                self.ctx, {"target": "single", "op": "le", "value": 8}, self.deadline
            )
        self.assertEqual(8, out.output["count"])

    def test_quota_fill_requires_every_schedule_to_be_accepted(self):
        records = Mock()
        records.snapshot_records.return_value = [{"schedule": {"status": "OK"}}] * 3 + [
            {"schedule": {"status": "REJECTED"}}
        ]
        handle = self.ctx.register_resource("requests", records)
        verdict = master._admission(
            self.ctx, {"requests": handle, "count": 4}, self.deadline
        )
        self.assertEqual("FAIL", verdict.checks[0].status)
        self.assertEqual(3, verdict.output["admitted"])

    def test_one_switched_request_is_not_replaced_by_one_percent_threshold(self):
        self.manager.master_instance_target.return_value = "B"
        rows = [{"master_target": "A"}] * 999 + [{"master_target": "B"}]
        handle = self.ctx.register_resource("ha_rows", rows)
        verdict = master._client_check(
            self.ctx,
            {
                "rows": handle,
                "metric": "target_count",
                "target": "B",
                "op": "ge",
                "expected": 1,
                "min_samples": 1,
            },
            self.deadline,
        )
        self.assertEqual("PASS", verdict.checks[0].status)

    def test_dual_kill_steady_excludes_rescued_boundary_requests(self):
        rows = [
            {"wall_clock_ts": 1000, "failover": False, "master_target": "B"},
            {"wall_clock_ts": 1000, "failover": True, "master_target": "A"},
        ]
        handle = self.ctx.register_resource("ha_rows", rows)
        out = master._window(
            self.ctx, {"rows": handle, "failover": False}, self.deadline
        )
        self.assertEqual([rows[0]], self.ctx.resource(out.output["rows"], "ha_rows"))

    def test_master_programs_compile_with_explicit_registered_actions(self):
        from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
        from flexlb_test_framework.scenario.actions.engine_control import (
            HANDLERS as controls,
        )
        from flexlb_test_framework.scenario.actions.engine_fault import (
            HANDLERS as faults,
        )
        from flexlb_test_framework.scenario.actions.master_observation import (
            HANDLERS as observations,
        )

        root = Path(__file__).resolve().parents[1] / "scenarios/master"
        plans = compile_scenarios(
            load_scenarios(root),
            handlers={
                h.name: h for h in master.HANDLERS + observations + controls + faults
            },
        )
        self.assertEqual(29, len(plans))
        self.assertEqual(5, len({p["scenario_id"] for p in plans}))
        self.assertTrue(all(any(s["check_ids"] for s in p["stages"]) for p in plans))
        for plan in plans:
            if plan["variant_id"] in {"kill_single", "kill_dual_b_to_a"}:
                ids = [s["id"] for s in plan["stages"]]
                ready, clean = (
                    ("restored_topology", "restored_inflight")
                    if plan["variant_id"] == "kill_single"
                    else ("ready_b", "clean_b")
                )
                index = ids.index(ready)
                self.assertEqual(clean, ids[index + 1])
                self.assertEqual(60, plan["stages"][index]["timeout_s"])
                self.assertFalse(plan["stages"][index]["params"]["inflight_zero"])
                self.assertEqual(10, plan["stages"][index + 1]["timeout_s"])
                self.assertTrue(plan["stages"][index + 1]["params"]["inflight_zero"])
            self.assertEqual(0, plan["resource_budget"]["max_dynamic_additions"])
            if plan["scenario_id"] == "master_coldstart":
                self.assertEqual(0, plan["environment"]["master_stable_window_s"])
            if (
                plan["scenario_id"]
                in {
                    "master_ha_failover",
                    "client_fallback_failback",
                }
                and plan["variant_id"] != "direct_generate_error"
            ):
                self.assertEqual(
                    "dual_standalone", plan["environment"]["master_layout"]
                )

    def test_short_hang_empty_send_window_still_requires_real_recovery_burst(self):
        self.manager.master_instance_target.return_value = "B:18085"
        good = [{"status": "ok", "master_target": "B:18085"}] * 3
        empty = self.ctx.register_resource("ha_rows", [])
        burst = self.ctx.register_resource("ha_rows", good)
        self.assertEqual(
            "PASS",
            master._short(
                self.ctx,
                dict(hang=empty, burst=burst, post=burst, target="B"),
                self.deadline,
            )
            .checks[0]
            .status,
        )
        self.assertEqual(
            "FAIL",
            master._short(
                self.ctx,
                dict(hang=empty, burst=empty, post=empty, target="B"),
                self.deadline,
            )
            .checks[0]
            .status,
        )

    def test_direct_request_bypasses_schedule_and_records_actual_grpc_error(self):
        class RpcError(Exception):
            def code(self):
                return SimpleNamespace(name="UNKNOWN")

        class Call:
            def __iter__(self):
                raise RpcError("injected")

            def cancel(self):
                return True

        self.ctx.ops = SimpleNamespace(
            next_request_id=lambda: 42,
            _channel=lambda target: target,
            build_generate_input=lambda *args, **kw: object(),
            pb2_grpc=SimpleNamespace(
                RpcServiceStub=lambda channel: SimpleNamespace(
                    GenerateStreamCall=lambda *args, **kw: Call()
                )
            ),
        )
        snapshot = {
            "engines": [
                {
                    "name": "prefill-0",
                    "role": "prefill",
                    "grpc_addr": "127.0.0.1:55151",
                    "stopped": False,
                }
            ]
        }
        with patch.dict(
            sys.modules, {"grpc": SimpleNamespace(RpcError=RpcError)}
        ), patch(
            "flexlb_test_framework.scenario.actions.engine_control._http",
            return_value=snapshot,
        ):
            out = master._direct(self.ctx, {"engine": "prefill-0"}, self.deadline)
        self.assertTrue(out.output["error"])
        self.assertFalse(out.output["finished"])
        record = self.ctx.resource(out.output["result"], "direct_request")
        self.assertEqual("GenerateStreamCall", record["method"])
        self.assertEqual("direct", record["route"])
        self.assertTrue(record["consumer_done"])
        self.assertIsNotNone(record["consumer_exit_s"])

    def test_strict_parameters_and_typed_prior_fault(self):
        plan = PlanContext("fault", {})
        for params in (
            {"mode": "kill", "pid": 2},
            {"mode": "stop"},
            {"mode": "kill", "target": "foreign"},
        ):
            with self.assertRaises(ValueError):
                master._fault_validate(params, plan)
        with self.assertRaises(ValueError):
            master._restore_validate(
                {"fault": {"$ref": "stages.missing.output.fault"}}, plan
            )


if __name__ == "__main__":
    unittest.main()
