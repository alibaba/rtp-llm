"""KV-full branch and terminal boundaries, plus complete ordered YAML fixture."""

import copy
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.actions import elastic as e
from flexlb_test_framework.scenario.actions import elastic_concurrent as concurrent
from flexlb_test_framework.scenario.actions import elastic_full as full
from flexlb_test_framework.scenario.actions import elastic_lifecycle as life
from flexlb_test_framework.scenario.actions import engine_control as control
from flexlb_test_framework.scenario.catalog import handlers as builtin_handlers
from flexlb_test_framework.scenario.loader import load_scenarios
from flexlb_test_framework.scenario.runtime import execute_instance


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        time.sleep(0.0001)


class FullTests(unittest.TestCase):
    def test_saturation_prices_next_request_without_exhausting_reserve(self):
        entry = dict(
            cache_blocks=24,
            available_blocks=4,
            block_size=1024,
            total_kv_tokens=24576,
            running=10,
        )
        self.assertTrue(full.saturated(entry))
        for key, value in [
            ("available_blocks", 5),
            ("available_blocks", 1),
            ("running", 0),
        ]:
            self.assertFalse(full.saturated(dict(entry, **{key: value})))
        with self.assertRaises(ValueError):
            full.saturated(dict(entry, total_kv_tokens=100))
        with self.assertRaises(ValueError):
            full.saturated(dict(entry, available_blocks=None))

    def terminal_result(
        self,
        branch="drain_ok",
        code=None,
        stamp=130,
        drain_ms=None,
        message="",
        transport="OK",
        cancel=False,
        data_error=False,
    ):
        records = e.ClientRecords(1)
        r = records.issue(1, lambda: 0)
        records.update(
            r,
            schedule=dict(status="OK"),
            stream=dict(status=transport),
            prefill_addr="p",
            business_finished=code is None,
            business_error_code=code,
            business_error_message=message,
            transport_terminal_s=stamp,
            consumer_exit_s=stamp,
            cancel=dict(requested_s=120 if cancel else None),
        )
        response = dict(
            drained=branch == "drain_ok",
            drain_ms=drain_ms if drain_ms is not None else 5000,
        )
        data = dict(
            records=records.snapshot_records(),
            timed_out=[],
            mutation=dict(started_s=100, response=response),
        )
        with tempfile.TemporaryDirectory() as temp:
            result = full.terminal(
                NS(resource=lambda *a: data, artifact_dir=Path(temp)),
                dict(
                    result="x",
                    branch=branch,
                    **(
                        dict(
                            data_error_codes=[8209],
                            data_error_tokens=["P->D link closed:"],
                        )
                        if data_error
                        else {}
                    ),
                ),
                NS(check=lambda: None),
            )
            return {c.id: c.status for c in result.checks}

    def test_retirement_requires_code_message_and_millisecond_bounds(self):
        for ms, expected in [
            (4999, "FAIL"),
            (5000, "PASS"),
            (10000, "PASS"),
            (10001, "FAIL"),
        ]:
            result = self.terminal_result(
                "drain_timeout",
                8510,
                drain_ms=ms,
                message="Decode endpoint generation retired: d1",
            )
            self.assertEqual(result["drain_branch"], expected)
            self.assertEqual(result["terminal_family"], "PASS")
        result = self.terminal_result(
            "drain_timeout", 8510, message="some other failure"
        )
        self.assertEqual(result["terminal_family"], "FAIL")
        self.assertEqual(result["retirement_contract"], "FAIL")
        self.assertEqual(
            self.terminal_result(
                "drain_ok", 8510, message="Decode endpoint generation retired"
            )["terminal_family"],
            "FAIL",
        )

    def test_data_error_requires_timeout_branch_code_message_and_removal(self):
        for branch, code, message, stamp, expected in [
            ("drain_timeout", 8209, "P->D link closed: decode stopped", 130, "PASS"),
            ("drain_ok", 8209, "P->D link closed: decode stopped", 130, "FAIL"),
            ("drain_timeout", 8209, "unrelated", 130, "FAIL"),
            ("drain_timeout", 8209, "P->D link closed: decode stopped", 99, "FAIL"),
            ("drain_timeout", 8210, "P->D link closed: decode stopped", 130, "FAIL"),
        ]:
            result = self.terminal_result(
                branch, code, stamp, message=message, data_error=True
            )
            self.assertEqual(
                expected,
                result["terminal_family"],
                (branch, code, message, stamp, result),
            )

    def test_fill_refusal_must_precede_remove_and_terminal_40_is_inclusive(self):
        self.assertEqual(
            self.terminal_result(code=8211, stamp=99)["terminal_family"], "PASS"
        )
        self.assertEqual(
            self.terminal_result(code=8211, stamp=101)["terminal_family"], "FAIL"
        )
        self.assertEqual(self.terminal_result(stamp=140)["terminal_40s"], "PASS")
        self.assertEqual(self.terminal_result(stamp=140.001)["terminal_40s"], "FAIL")
        self.assertEqual(
            self.terminal_result(transport="ERROR")["terminal_family"], "FAIL"
        )
        self.assertEqual(self.terminal_result(cancel=True)["terminal_40s"], "FAIL")

    def test_transient_zero_reject_budget_and_reserve_boundary(self):
        for available, rejects, expected in [
            (50, 0, ("PASS", "PASS")),
            (49, 0, ("FAIL", "PASS")),
            (50, 1, ("PASS", "FAIL")),
        ]:
            data = dict(
                errors=[],
                samples=[
                    dict(
                        time_s=t,
                        engines={
                            "survivor": dict(
                                mock_engine_cache_blocks=1000,
                                mock_engine_available_blocks=available,
                                mock_engine_lack_mem_rejects_total=0,
                                mock_engine_kv_admission_fails_total=(
                                    rejects if t == 20 else 0
                                ),
                            )
                        },
                    )
                    for t in range(21)
                ],
            )
            resources = dict(
                observation=NS(snapshot=lambda: data),
                mutation=dict(started_s=0),
                fill=dict(
                    victim="victim",
                    survivor="survivor",
                    snapshot=dict(
                        victim=dict(cache_blocks=24, available_blocks=4),
                        survivor=dict(available_blocks=24),
                    ),
                ),
            )
            with tempfile.TemporaryDirectory() as temp:
                ctx = NS(
                    clock=lambda: 21,
                    resource=lambda name, *args: resources[name],
                    artifact_dir=Path(temp),
                )
                result = full.transient(
                    ctx,
                    dict(observation="observation", mutation="mutation", fill="fill"),
                    NS(check=lambda: None),
                )
            self.assertEqual(tuple(c.status for c in result.checks), expected)

    def run_program(self, bad_retirement=False, stop_drift=False, stop_queue=False):
        handlers = builtin_handlers()
        plan = next(
            p
            for p in compile_scenarios(
                load_scenarios(ROOT / "scenarios/elastic/lifecycle.yaml"),
                handlers=handlers,
            )
            if p["variant_id"] == "kv_full_shrink"
        )
        self.assertEqual(plan["environment"]["decode_cache_blocks"], 24)
        self.assertEqual(len(plan["stages"]), 36)
        self.assertEqual(plan["resource_budget"]["max_dynamic_additions"], 1)
        clock = Clock()
        lock = threading.RLock()
        gates = [threading.Event(), threading.Event()]
        state = dict(
            phase=0,
            next_rid=0,
            active={},
            removed={},
            added=None,
            fill_shapes=[],
            recovery=0,
            effective_scale=1,
        )
        engines = {
            f"{role}-{i}": dict(
                name=f"{role}-{i}",
                role=role,
                grpc_addr=f"127.0.0.1:{10001+offset+i*2}",
                stopped=False,
                cache_blocks=24,
                available_blocks=24,
                block_size=1024,
                total_kv_tokens=24576,
                running=0,
                waiting=0,
            )
            for role, offset in [("prefill", 0), ("decode", 100)]
            for i in range(2)
        }

        def next_rid():
            with lock:
                state["next_rid"] += 1
                return state["next_rid"]

        with tempfile.TemporaryDirectory() as temp:
            discovery = Path(temp) / "discovery.json"

            def sync():
                discovery.write_text(
                    json.dumps(
                        {
                            "mock.decode.hosts.address": [
                                "127.0.0.1:"
                                + str(int(v["grpc_addr"].split(":")[-1]) - 1)
                                for v in engines.values()
                                if v["role"] == "decode"
                            ]
                        }
                    )
                )

            class Backend:
                def setup(self, ctx, environment, deadline):
                    sync()
                    return NS(discovery_file=discovery), NS(
                        next_request_id=next_rid, master_http_port=2
                    )

                def teardown(self, ctx, deadline):
                    state["cleaned"] = True

            class Metrics:
                def __init__(self, ctx, max_duration_s):
                    self.thread = NS(start=lambda: None, ident=None)

                def stop(self, deadline):
                    state["metrics_stopped"] = True

                def snapshot(self):
                    samples = []
                    for t in range(int(clock()) + 1):
                        rows = {}
                        for name in [
                            "prefill-0",
                            "prefill-1",
                            "decode-0",
                            "decode-1",
                            "decode-2",
                        ]:
                            if name in state["removed"] and t >= state["removed"][name]:
                                continue
                            if name == "decode-2" and (
                                state["added"] is None or t < state["added"]
                            ):
                                continue
                            rows[name] = dict(
                                role=name.split("-")[0],
                                mock_engine_completed_total=t * 30
                                + (
                                    10000
                                    if stop_drift
                                    and name == "decode-1"
                                    and state.get("stop_at") is not None
                                    and t >= state["stop_at"]
                                    else 0
                                ),
                                mock_engine_cache_blocks=24,
                                mock_engine_available_blocks=20,
                                mock_engine_waiting=(
                                    3
                                    if stop_queue
                                    and state.get("stop_at") is not None
                                    and t >= state["stop_at"]
                                    else 1
                                ),
                                mock_engine_lack_mem_rejects_total=0,
                                mock_engine_kv_admission_fails_total=0,
                                rtp_llm_context_tps=10,
                                rtp_llm_generate_tps=10,
                            )
                        samples.append(dict(time_s=float(t), engines=rows))
                    return dict(samples=samples, errors=[], env_epoch=1)

            def run(records, record, shape, **kwargs):
                fill = shape["input_len"] == 2035
                phase = state["phase"]
                target = (
                    ("decode-0" if phase == 0 else "decode-1")
                    if record["wire_request_id"] % 2
                    else ("decode-1" if phase == 0 else "decode-2")
                )
                if fill:
                    self.assertEqual(state["effective_scale"], 60)
                    with lock:
                        state["fill_shapes"].append(shape)
                        state["active"][target] = state["active"].get(target, 0) + 1
                    records.update(record, schedule=dict(status="OK"), prefill_addr="p")
                    records._activate(
                        record, NS(cancel=lambda: gates[phase].set() or True)
                    )
                    gates[phase].wait(timeout=3)
                    code = 8510 if phase == 1 and target == "decode-1" else None
                    message = (
                        "wrong reason"
                        if bad_retirement
                        else "Decode endpoint generation retired"
                    )
                else:
                    code = None
                    message = ""
                    if len(shape["block_keys"]) == 3:
                        state["recovery"] += 1
                    else:
                        self.assertEqual(kwargs["stream_timeout_s"], 10)
                records.update(
                    record,
                    schedule=dict(status="OK"),
                    stream=dict(status="OK"),
                    business_finished=code is None,
                    business_error_code=code,
                    business_error_message=message,
                    transport_terminal_s=clock(),
                    consumer_exit_s=clock(),
                )
                with records._lock:
                    records._calls.pop(record["wire_request_id"], None)
                if fill:
                    with lock:
                        state["active"][target] -= 1

            def http(ops, endpoint, deadline, body=None):
                with lock:
                    if endpoint == "snapshot":
                        result = copy.deepcopy(list(engines.values()))
                        for row in result:
                            row["running"] = state["active"].get(row["name"], 0)
                            row["available_blocks"] = max(2, 24 - row["running"] * 2)
                        return dict(engines=result)
                    if endpoint == "set_perf":
                        state["effective_scale"] = body["decode_scale"]
                        state.setdefault("perf_writes", []).append(
                            (body["engine"], body["decode_scale"])
                        )
                        return dict(
                            status="ok",
                            engine=body["engine"],
                            port=int(
                                engines[body["engine"]]["grpc_addr"].split(":")[-1]
                            ),
                        )
                    if endpoint == "add_engine":
                        self.assertEqual(body, dict(role="decode"))
                        state["added"] = clock()
                        engines["decode-2"] = dict(
                            name="decode-2",
                            role="decode",
                            grpc_addr="127.0.0.1:10105",
                            stopped=False,
                            cache_blocks=24,
                            available_blocks=24,
                            block_size=1024,
                            total_kv_tokens=24576,
                            running=0,
                            waiting=0,
                        )
                        sync()
                        return dict(
                            status="ok",
                            action="added",
                            engine="decode-2",
                            port=10105,
                            http_port=10104,
                        )
                    self.assertEqual(endpoint, "remove_engine")
                    phase = state["phase"]
                    self.assertEqual(state["effective_scale"], 60)
                    name = body["engine"]
                    row = engines.pop(name)
                    state["removed"][name] = clock()
                    sync()
                    self.assertEqual(
                        body["drain_timeout_ms"], 60000 if phase == 0 else 5000
                    )
                    state["phase"] += 1
                    gates[phase].set()
                    return dict(
                        status="ok",
                        action="removed",
                        engine=name,
                        port=int(row["grpc_addr"].split(":")[-1]),
                        mode="graceful",
                        drained=phase == 0,
                        drain_ms=6300 if phase == 0 else 5000,
                    )

            def master(*args, **kwargs):
                n = sum(r["role"] == "decode" for r in engines.values())
                return 200, dict(
                    worker_summary=dict(DECODE=dict(discovered=n, alive=n))
                )

            original_stop = e.BoundedFlow.stop

            def delayed_stop(flow, deadline, **kwargs):
                result = original_stop(flow, deadline, **kwargs)
                if (
                    state["added"] is not None
                    and "stop_at" not in state
                    and (stop_drift or stop_queue)
                ):
                    state["stop_at"] = clock() + 1
                    clock.sleep(2)
                return result

            original_collect = full.collect

            def observed_collect(ctx, params, deadline):
                self.assertEqual(state["effective_scale"], 60)
                return original_collect(ctx, params, deadline)

            handlers["elastic_full_collect"] = type(handlers["elastic_full_collect"])(
                "elastic_full_collect",
                full.collect_validate,
                observed_collect,
                {"result": "snapshot"},
            )
            with patch.object(e.BoundedFlow, "stop", delayed_stop), patch.object(
                e, "ElasticMetrics", Metrics
            ), patch.object(e.RecordedRequests, "run", run), patch.object(
                e, "_http", side_effect=http
            ), patch.object(
                control, "_http", side_effect=http
            ), patch.object(
                concurrent, "mutation_http", side_effect=http
            ), patch(
                "flexlb_test_framework.harness.http_post_json", side_effect=master
            ), patch.object(
                life,
                "_master_get",
                return_value=dict(
                    scheduler_inflight=0,
                    prefill_endpoints=[dict(inflight_batches=0)],
                    decode_endpoints=[dict(total_load=0)],
                ),
            ):
                result = execute_instance(
                    plan,
                    Backend(),
                    handlers=handlers,
                    artifact_dir=Path(temp) / "artifacts",
                    clock=clock,
                    sleeper=clock.sleep,
                )
            self.assertTrue(state["cleaned"])
            self.assertTrue(state["metrics_stopped"])
            self.assertFalse(
                any(
                    t.name.startswith("elastic-full-fill-") and t.is_alive()
                    for t in threading.enumerate()
                )
            )
            return result, state

    def test_full_program_constructs_both_branches_with_stable_two_block_shapes(self):
        result, state = self.run_program()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(sum(len(s["checks"]) for s in result["stages"]), 33)
        self.assertEqual(
            state["perf_writes"],
            [
                ("decode-0", 60),
                ("decode-1", 60),
                ("decode-1", 1000),
                ("decode-2", 60),
                ("decode-2", 1),
            ],
        )
        self.assertEqual(state["phase"], 2)
        self.assertEqual(state["recovery"], 20)
        self.assertGreaterEqual(len(state["fill_shapes"]), 20)
        self.assertTrue(
            all(
                s["input_len"] + s["output_len"] == 2048 and len(s["block_keys"]) == 2
                for s in state["fill_shapes"]
            )
        )

    def test_steady_includes_stop_period_completion_drift_and_waiting_peak(self):
        for option, failed in [
            ("stop_drift", "share_max"),
            ("stop_queue", "waiting_peak"),
        ]:
            result, state = self.run_program(**{option: True})
            row = next(s for s in result["stages"] if s["id"] == "steady_bounds")
            self.assertEqual(row["status"], "FAIL", result)
            self.assertEqual(
                next(c for c in row["checks"] if c["id"] == failed)["status"], "FAIL"
            )
            self.assertEqual(
                next(s for s in result["stages"] if s["id"] == "slow_victim")["status"],
                "BLOCKED",
            )

    def test_timeout_retirement_message_failure_is_not_relabeled(self):
        result, _ = self.run_program(bad_retirement=True)
        row = next(s for s in result["stages"] if s["id"] == "terminal_timeout")
        self.assertEqual(row["status"], "FAIL", result)
        self.assertEqual(
            next(s for s in result["stages"] if s["id"] == "recovery")["status"],
            "BLOCKED",
        )


if __name__ == "__main__":
    unittest.main()
