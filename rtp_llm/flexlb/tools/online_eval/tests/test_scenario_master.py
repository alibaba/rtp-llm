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
from scenario.actions import master
from cases.master_ha_failover import actions as ha
from scenario.contracts import PlanContext
from scenario.runtime import Deadline, RuntimeContext
from cases.master_ha_failover.runtime import HaMasterStateSampler, HaTrafficRunner


class MasterActionsTest(unittest.TestCase):
    def test_ha_client_receives_http_discovery_candidates(self):
        root = Path(self.tmp.name)
        env = SimpleNamespace(master_specs={
            "A": SimpleNamespace(bind_ip="127.0.0.1", http_port=18080),
            "B": SimpleNamespace(bind_ip="127.0.0.1", http_port=18083),
        })
        manager = Mock()
        manager.master_instance_target.side_effect = lambda _env, name: {
            "A": "127.0.0.1:18082", "B": "127.0.0.1:18085"
        }[name]
        with patch("cases.master_ha_failover.runtime.ClientOps"):
            runner = HaTrafficRunner(manager, env, root, "flow", [
                "127.0.0.1:18082", "127.0.0.1:18085"
            ])
        path = Path(runner._overrides["MASTER_DISCOVERY_FILE"])
        self.assertEqual({"hosts": [
            {"http": "127.0.0.1:18080", "grpc": "127.0.0.1:18082"},
            {"http": "127.0.0.1:18083", "grpc": "127.0.0.1:18085"},
        ]}, json.loads(path.read_text()))
        frozen = json.loads((runner.out_dir / "flow-input.json").read_text())
        from traffic.traffic_source import sha256_file
        self.assertEqual(sha256_file(Path(runner._overrides["TRACE_FILE"])), frozen["trace"]["sha256"])
        self.assertEqual(runner._overrides, frozen["environment"])
        self.assertEqual(runner.flow_identity["run_id"], frozen["run_id"])
        self.assertEqual("false", runner._overrides["LOOP"])
        self.assertEqual("false", runner._overrides["REPLAY_UNIQUE_PREFIX"])
        runner.stop_sending()
        command = json.loads((runner.control / "stop.json").read_text())
        runner.stop_sending()
        self.assertEqual(command, json.loads((runner.control / "stop.json").read_text()))
        state = dict(runner.flow_identity, state="DRAINED", submitted=2, terminal=2,
                     applied_command_id=command["command_id"])
        (runner.control / "status.json").write_text(json.dumps(state))
        runner.validate_drain([{}, {}])
        with self.assertRaisesRegex(ValueError, "every submitted"):
            runner.validate_drain([{}])
        state["applied_command_id"] = "wrong"
        (runner.control / "status.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "stop command"):
            runner.validate_drain([{}, {}])

    def test_ha_one_pass_rejects_trace_shorter_than_client_duration(self):
        root = Path(self.tmp.name)
        env = SimpleNamespace(master_specs={
            "A": SimpleNamespace(bind_ip="127.0.0.1", http_port=18080),
            "B": SimpleNamespace(bind_ip="127.0.0.1", http_port=18083),
        })
        manager = Mock()
        manager.master_instance_target.side_effect = lambda _env, name: {
            "A": "127.0.0.1:18082", "B": "127.0.0.1:18085"
        }[name]

        def short_trace(path, *_args, **_kwargs):
            path.write_text('{"ts":0}\n{"ts":1000}\n')
            return path

        with patch("cases.master_ha_failover.runtime.ClientOps"), patch(
            "traffic.traffic_source.materialize", side_effect=short_trace
        ), self.assertRaisesRegex(ValueError, "one-pass HA trace ends"):
            HaTrafficRunner(manager, env, root, "flow", [
                "127.0.0.1:18082", "127.0.0.1:18085"
            ], duration_s=10, replay_speed=2, source={"kind": "trace"})

    def test_ha_state_sampler_keeps_each_master_and_missing_inflight_distinct(self):
        root = Path(self.tmp.name)
        env = SimpleNamespace(master_specs={
            "A": SimpleNamespace(bind_ip="127.0.0.1", http_port=101),
            "B": SimpleNamespace(bind_ip="127.0.0.1", http_port=102),
        })
        sampler = HaMasterStateSampler(env, root / "master_states.jsonl", 0.01)

        def fetch(url, timeout):
            if url.endswith(":101/rtp_llm/inflight_status"):
                return dict(scheduler_inflight=4,
                            prefill_endpoints=[{"inflight_requests": 2}, {"inflight_requests": 3}],
                            decode_endpoints=[{"master_queued": 1, "confirmed_running": 6}])
            sampler._stop.set()
            return None

        with patch("cases.master_ha_failover.runtime.http_get_json", side_effect=fetch):
            sampler._run()
        rows = [json.loads(line) for line in sampler.path.read_text().splitlines()]
        self.assertEqual(["A", "B"], [row["master"] for row in rows])
        self.assertEqual(5, rows[0]["prefill_inflight_requests"])
        self.assertEqual(6, rows[0]["decode_confirmed_running"])
        self.assertEqual(0, rows[1]["http_up"])
        self.assertNotIn("scheduler_inflight", rows[1])

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
        from monitoring.session import PrometheusSession
        self.ctx.monitor = PrometheusSession(self.tmp.name, {"mock": "http://unused/metrics"}, query_plan="master_ha_failover.yaml")
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

    def test_ha_fails_compilation_for_wrong_environment(self):
        plan = SimpleNamespace(path="stage", environment={})
        with self.assertRaisesRegex(ValueError, "dual_standalone"):
            ha._ha_validate({}, plan)
        plan.environment = {
            "master_layout": "dual_standalone",
            "master_stable_window_s": 0,
        }
        self.assertEqual(["A", "B"], ha._ha_validate({}, plan)["targets"])

    def test_empty_client_rows_fail_even_zero_error_assertion(self):
        handle = self.ctx.register_resource("ha_rows", [])
        result = ha._client_check(
            self.ctx,
            dict(
                rows=handle, metric="ha_gate/failed_count", op="eq", expected=0, min_samples=1
            ),
            self.deadline,
        )
        self.assertEqual("ERROR", result.checks[0].status)
        self.assertEqual("INVALID", result.checks[0].actual["validity"])

    def test_schedule_only_rows_are_not_successful_streams(self):
        handle = self.ctx.register_resource("ha_rows", [{"status": "scheduled"}])
        result = ha._client_check(
            self.ctx,
            dict(
                rows=handle, metric="ha_gate/success_rate", op="eq", expected=1, min_samples=1
            ),
            self.deadline,
        )
        self.assertEqual(0, result.output["actual"])
        self.assertEqual("FAIL", result.checks[0].status)

    def test_rolling_error_gate_counts_transport_and_worker_failures(self):
        rows = self.ctx.register_resource("ha_rows", [
            {"status": "ok", "route_path": "master"},
            {"status": "exception", "route_path": "failed"},
            {"status": "schedule_error", "route_path": "master"},
        ])
        result = ha._client_check(
            self.ctx,
            dict(rows=rows, metric="ha_gate/non_ok_count", op="eq", expected=0,
                 min_samples=3),
            self.deadline,
        )
        self.assertEqual(2, result.output["actual"])
        self.assertEqual("FAIL", result.checks[0].status)

    def test_client_nonzero_exit_and_malformed_rows_are_errors(self):
        root = Path(self.tmp.name)
        process = SimpleNamespace(proc=Mock())
        client = ha.OwnedHaClient(SimpleNamespace(proc=process, out_dir=root))
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
        out = ha._window(
            self.ctx, {"rows": handle, "from": 2, "until": 3}, self.deadline
        )
        self.assertEqual([rows[1]], self.ctx.resource(out.output["rows"], "ha_rows"))


    def test_negative_windows_filter_actual_error_rows_and_reject_empty_success(self):
        rows = [
            {"status": "ok", "error_kind": "none"},
            {"status": "schedule_error", "error_kind": "business", "error": "8431"},
            {"status": "schedule_error", "error_kind": "deadline"},
        ]
        for row in rows:
            row["wall_clock_ts"] = 1000
        handle = self.ctx.register_resource("ha_rows", rows)
        out = ha._window(
            self.ctx,
            {"rows": handle, "status": "schedule_error", "error_kind": "business"},
            self.deadline,
        )
        self.assertEqual([rows[1]], self.ctx.resource(out.output["rows"], "ha_rows"))
        empty = self.ctx.register_resource("ha_rows", [])
        verdict = ha._client_check(
            self.ctx,
            {
                "rows": empty,
                "metric": "ha_gate/wrong_error_code",
                "op": "eq",
                "expected": 0,
                "code": 8431,
                "min_samples": 5,
            },
            self.deadline,
        )
        self.assertEqual("ERROR", verdict.checks[0].status)


    def test_one_switched_request_is_not_replaced_by_one_percent_threshold(self):
        self.manager.master_instance_target.return_value = "B"
        rows = [{"master_target": "A"}] * 999 + [{"master_target": "B"}]
        handle = self.ctx.register_resource("ha_rows", rows)
        verdict = ha._client_check(
            self.ctx,
            {
                "rows": handle,
                "metric": "ha_gate/target_count",
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
        out = ha._window(
            self.ctx, {"rows": handle, "failover": False}, self.deadline
        )
        self.assertEqual([rows[0]], self.ctx.resource(out.output["rows"], "ha_rows"))

    def test_master_programs_compile_with_explicit_registered_actions(self):
        from scenario import compile_scenarios, load_scenarios
        from scenario.actions.engine_control import (
            HANDLERS as controls,
        )
        from scenario.actions.engine_fault import (
            HANDLERS as faults,
        )

        root = Path(__file__).resolve().parents[1] / "config/scenarios"
        with patch("scenario.compiler.VICTIM_OFFSETS", (700, 701, 702)):
            plans = compile_scenarios(
                load_scenarios(root / "master_ha_failover.yaml"),
                handlers={
                    h.name: h for h in master.HANDLERS + ha.HANDLERS + controls + faults
                },
            )
        self.assertEqual(4, len(plans))
        self.assertEqual(1, len({p["scenario_id"] for p in plans}))
        self.assertTrue(all(any(s["check_ids"] for s in p["stages"]) for p in plans))
        for plan in plans:
            self.assertEqual(0, plan["resource_budget"]["max_dynamic_additions"])
            if plan["scenario_id"] == "master_ha_failover":
                self.assertEqual(
                    "dual_standalone", plan["environment"]["master_layout"]
                )
                ids = [stage["id"] for stage in plan["stages"]]
                self.assertLess(ids.index("kill_a"), ids.index("kill_b"))
                self.assertLess(ids.index("restart_a"), ids.index("a_ready"))
                if plan["variant_id"] == "non_rolling":
                    self.assertLess(ids.index("kill_b"), ids.index("restart_a"))
                    self.assertIn("outage_failures", ids)
                    continue
                self.assertLess(ids.index("a_ready"), ids.index("kill_b"))
                self.assertLess(ids.index("restart_a"), ids.index("restart_b"))
                self.assertNotIn("outage_failures", ids)
                self.assertIn("both_balance", ids)
                self.assertIn("handover_errors", ids)
                self.assertIn("late_errors", ids)
                self.assertIn("rolling_errors", ids)
                self.assertNotIn("to_b_balance", ids)
                self.assertNotIn("to_a_balance", ids)
                finish = next(s for s in plan["stages"] if s["id"] == "finish")
                self.assertTrue(finish["params"]["stop_sending"])
                flow = next(stage for stage in plan["stages"] if stage["id"] == "flow")
                self.assertEqual("prefix_lineage", flow["params"]["source"]["model"])
                self.assertEqual(20000, flow["params"]["max_requests"])
                self.assertEqual(240, flow["params"]["duration_s"])
                self.assertEqual(5, flow["params"]["replay_speed"])
                self.assertFalse(flow["params"]["loop"])
                self.assertEqual(120000, flow["params"]["timeout_ms"])
                self.assertEqual(125, plan["environment"]["n_prefill"])
                self.assertEqual(536, plan["environment"]["n_decode"])

    def test_ha_non_rolling_option_retains_full_outage_checks(self):
        from cases.config import configure_program
        from scenario.loader import load_document, ScenarioError

        path = Path(__file__).resolve().parents[1] / "config/scenarios/master_ha_failover.yaml"
        config = load_document(path)
        document = configure_program(config, str(path))
        ids = [stage["id"] for stage in document["variants"][1]["stages"]]
        self.assertLess(ids.index("kill_b"), ids.index("outage_start"))
        self.assertLess(ids.index("outage_end"), ids.index("restart_a"))
        self.assertIn("outage_failures", ids)
        self.assertNotIn("rolling_errors", ids)
        config["parameters"]["dual_master_cycle"]["restart_mode"] = "typo"
        with self.assertRaisesRegex(ScenarioError, "restart_mode"):
            configure_program(config, str(path))

    def test_ha_prefill_balance_uses_successful_requests_and_known_pool(self):
        rows = self.ctx.register_resource("ha_rows", [
            {"status": "ok", "prefill": "P0"},
            {"status": "ok", "prefill": "P1"},
            {"status": "schedule_error", "prefill": None},
        ])
        params = dict(rows=rows, metric="ha_gate/prefill_max_share", op="le",
                      expected=0.75, min_samples=3)
        with patch("runtime.mock_control.topology_pools",
                   return_value={"prefill": ["P0", "P1"]}):
            result = ha._client_check(self.ctx, params, self.deadline)
        self.assertEqual("PASS", result.checks[0].status)
        self.assertEqual(0.5, result.output["actual"])
        with patch("runtime.mock_control.topology_pools",
                   return_value={"prefill": ["P1", "P2"]}), self.assertRaisesRegex(
                       ValueError, "known Prefill"):
            ha._client_check(self.ctx, params, self.deadline)

    def test_ha_handover_peak_balance_counts_failed_assigned_requests(self):
        rows = self.ctx.register_resource("ha_rows", [
            {"send_start_epoch_ms": 11_000 + (i % 5) * 1000,
             "status": "exception" if i < 5 else "ok",
             "prefill": "P0" if i < 6 else "P1"}
            for i in range(8)
        ] + [{"send_start_epoch_ms": 11_020, "status": "schedule_error", "prefill": None}])
        params = dict(rows=rows, metric="ha_gate/prefill_peak_skew", op="le",
                      expected=2.0, min_samples=5)
        with patch("runtime.mock_control.topology_pools",
                   return_value={"prefill": ["P0", "P1", "P2", "P3"]}):
            result = ha._client_check(self.ctx, params, self.deadline)
            self.assertEqual("FAIL", result.checks[0].status)
            self.assertEqual(3.0, result.output["actual"])
            missing = ha._client_check(self.ctx, {**params, "min_samples": 10}, self.deadline)
            self.assertEqual("ERROR", missing.checks[0].status)
            self.assertEqual("INVALID", missing.checks[0].actual["validity"])

    def test_ha_handover_balance_uses_five_seconds_at_lower_qps(self):
        pool = [f"P{i}" for i in range(125)]
        rows = self.ctx.register_resource("ha_rows", [
            {"send_start_epoch_ms": 11_000 + (i // 60) * 1000,
             "status": "ok", "prefill": pool[i % len(pool)]}
            for i in range(300)
        ])
        params = dict(rows=rows, metric="ha_gate/prefill_peak_skew", op="le",
                      expected=4.0, min_samples=200)
        with patch("runtime.mock_control.topology_pools",
                   return_value={"prefill": pool}):
            result = ha._client_check(self.ctx, params, self.deadline)
        self.assertEqual("PASS", result.checks[0].status)
        self.assertEqual(1.25, result.output["actual"])


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
