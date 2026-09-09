"""Transient capacities and ownership, including a complete formal YAML trace."""

import copy
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from environment_expectations import environment as expected_environment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.actions import elastic as e
from flexlb_test_framework.scenario.actions import elastic_transient as tr
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
        time.sleep(0.001)


def snapshots():
    return {
        name: dict(
            name=name,
            role=name.split("-")[0],
            grpc_addr=f"127.0.0.1:{10001+i*2}",
            http_addr=f"127.0.0.1:{10000+i*2}",
            cache_blocks=1000,
            available_blocks=800,
        )
        for i, name in enumerate((tr.VICTIM,) + tr.SURVIVORS)
    }


def metrics(start, end, override=None):
    samples = []
    for t in range(int(start), int(end) + 1):
        rows = {}
        for name in (tr.VICTIM,) + tr.SURVIVORS:
            rows[name] = dict(
                mock_engine_accepted_total=30 * t,
                mock_engine_cache_blocks=1000,
                mock_engine_available_blocks=800,
                mock_engine_waiting=1,
                mock_engine_lack_mem_rejects_total=0,
                mock_engine_kv_admission_fails_total=0,
                rtp_llm_context_tps=10,
                rtp_llm_generate_tps=10,
            )
            if override:
                override(t, name, rows[name])
        samples.append(dict(time_s=t, engines=rows))
    return dict(start_s=start, end_s=end, data=dict(samples=samples, errors=[]))


class TransientTests(unittest.TestCase):
    def test_master_requires_unique_http_identity_and_integer_request_count(self):
        self.assertEqual(
            tr.master_rows(
                dict(prefill_endpoints=[dict(ip_port="h:10000", inflight_requests=3)])
            ),
            {"h:10000": {"inflight_requests": 3}},
        )
        for payload in [
            {},
            dict(prefill_endpoints=[dict(ip_port="h:1", inflight_requests=None)]),
            dict(prefill_endpoints=[dict(ip_port="h:1", inflight_requests=3)] * 2),
        ]:
            with self.assertRaises(ValueError):
                tr.master_rows(payload)

    def bounds(self, *, p=16, d=128, master=64, free=50, rejects=0, missing=False):
        pre = snapshots()

        def override(t, name, row):
            row["mock_engine_waiting"] = p if name in tr.P else d
            row["mock_engine_available_blocks"] = free
            row["mock_engine_lack_mem_rejects_total"] = (
                rejects * t / 20 if name == tr.P[0] else 0
            )

        win = metrics(0, 20, override)
        mdata = dict(
            samples=[
                dict(
                    time_s=t,
                    engines={
                        pre[n]["http_addr"]: dict(
                            inflight_requests=999 if n == tr.VICTIM else master
                        )
                        for n in (tr.VICTIM,) + tr.P
                        if not (missing and n == tr.P[0])
                    },
                )
                for t in range(21)
            ],
            errors=[],
        )
        resources = dict(
            pre=pre, base=metrics(0, 20), win=win, master=NS(snapshot=lambda: mdata)
        )
        with tempfile.TemporaryDirectory() as temp:
            result = tr.bounds(
                NS(resource=lambda key, kind: resources[key], artifact_dir=Path(temp)),
                dict(
                    pre_event="pre",
                    baseline="base",
                    transient="win",
                    master_observation="master",
                ),
                NS(check=lambda: None),
            )
            return {c.id: c.status for c in result.checks}

    def test_capacity_boundaries_and_victim_http_exclusion(self):
        self.assertTrue(all(s == "PASS" for s in self.bounds().values()))
        for option, value, failure in [
            ("p", 17, "prefill_waiting"),
            ("d", 129, "decode_waiting"),
            ("master", 65, "master_inflight"),
            ("free", 49, "occupancy"),
            ("rejects", 1, "rejects"),
        ]:
            self.assertEqual(self.bounds(**{option: value})[failure], "FAIL")
        with self.assertRaisesRegex(ValueError, "lacks engine"):
            self.bounds(missing=True)

    def test_whole_window_accepted_share_and_tail_queue(self):
        def override(t, name, row):
            if name == tr.P[0]:
                row["mock_engine_accepted_total"] += min(t, 30) * 100

        resources = dict(base=metrics(0, 20), steady=metrics(0, 60, override))
        with tempfile.TemporaryDirectory() as temp:
            result = tr.steady(
                NS(resource=lambda key, kind: resources[key], artifact_dir=Path(temp)),
                dict(baseline="base", steady="steady"),
                NS(check=lambda: None),
            )
        self.assertEqual(
            next(c for c in result.checks if c.id == "share_max").status, "FAIL"
        )
        self.assertEqual(
            next(c for c in result.checks if c.id == "waiting").status, "PASS"
        )

    def test_locality_does_not_hide_unfinished_unrouted_consumers(self):
        for address, terminal, expected in [
            ("", True, ("PASS", "PASS")),
            ("", False, ("FAIL", "PASS")),
            ("127.0.0.1:10001", True, ("PASS", "PASS")),
            ("survivor-rpc", True, ("PASS", "FAIL")),
        ]:
            records = e.ClientRecords(1)
            record = records.issue(1, lambda: 0)
            records.update(
                record,
                schedule=dict(status="ERROR"),
                prefill_addr=address,
                transport_terminal_s=1 if terminal else None,
                consumer_exit_s=1 if terminal else None,
            )
            records.done = threading.Event()
            records.done.set()
            resources = dict(pre=snapshots(), flow=records, burst=e.ClientRecords(1))
            with tempfile.TemporaryDirectory() as temp:
                result = tr.locality(
                    NS(
                        resource=lambda key, kind: resources[key],
                        artifact_dir=Path(temp),
                    ),
                    dict(pre_event="pre", flow="flow", requests="burst"),
                    NS(check=lambda: None),
                )
            self.assertEqual(tuple(c.status for c in result.checks), expected)

    def test_burst_consumer_exception_is_error_and_cleanup_joins_jobs(self):
        resources, cleanups = {}, []

        def register(kind, value, cleanup=None, **kwargs):
            resources[kind] = value
            if cleanup:
                cleanups.append(cleanup)
            return kind

        rid = iter(range(1, 100))
        limit = time.monotonic() + 5
        deadline = NS(
            check=lambda: None, remaining=lambda: max(0.001, limit - time.monotonic())
        )

        def crash(*args, **kwargs):
            raise RuntimeError("consumer fixture crash")

        with tempfile.TemporaryDirectory() as temp:
            ctx = NS(
                ops=NS(next_request_id=lambda: next(rid)),
                env_epoch=1,
                clock=time.monotonic,
                artifact_dir=Path(temp),
                register_resource=register,
                resource=lambda key, kind: resources[key],
            )
            with patch.object(e.RecordedRequests, "run", crash):
                tr.burst_start(ctx, {}, deadline)
                with self.assertRaisesRegex(RuntimeError, "consumer fixture crash"):
                    tr.burst_collect(ctx, dict(requests="requests"), deadline)
                for cleanup in reversed(cleanups):
                    cleanup(deadline)
            self.assertTrue(
                all(
                    event.is_set() for future, event in resources["requests"].burst_jobs
                )
            )

    def test_actual_master_sampler_retains_errors_and_exits(self):
        from flexlb_test_framework.scenario.contracts import StageOutput

        for payload, valid in [
            (
                dict(prefill_endpoints=[dict(ip_port="h:10000", inflight_requests=3)]),
                True,
            ),
            ({}, False),
        ]:
            resources, cleanups = {}, []

            def register(kind, value, cleanup=None, **kwargs):
                resources[kind] = value
                if cleanup:
                    cleanups.append(cleanup)
                return kind

            limit = time.monotonic() + 5
            deadline = NS(
                check=lambda: None,
                remaining=lambda: max(0.001, limit - time.monotonic()),
            )
            with tempfile.TemporaryDirectory() as temp:
                ctx = NS(
                    ops=NS(master_http_port=2),
                    env_epoch=1,
                    clock=time.monotonic,
                    artifact_dir=Path(temp),
                    register_resource=register,
                )

                def fetch(url, timeout):
                    self.assertEqual(url, "http://127.0.0.1:2/rtp_llm/inflight_status")
                    self.assertGreater(timeout, 0)
                    self.assertLessEqual(timeout, 2)
                    return io.BytesIO(json.dumps(payload).encode())

                with patch.object(
                    tr.balance,
                    "observe_start",
                    return_value=StageOutput(
                        output=dict(observation="mock", started_s=time.monotonic())
                    ),
                ), patch.object(tr.urllib.request, "urlopen", fetch):
                    tr.observe(ctx, {}, deadline)
                    sampler = resources["observation"]
                    while (
                        not sampler.samples
                        and not sampler.errors
                        and time.monotonic() < limit
                    ):
                        time.sleep(0.001)
                    if valid:
                        cleanups[0](deadline)
                    else:
                        with self.assertRaisesRegex(ValueError, "acquisition errors"):
                            cleanups[0](deadline)
                self.assertTrue(sampler.done.is_set())
                data = json.loads(
                    next(
                        Path(temp).glob("elastic-transient-master-metrics-*.json")
                    ).read_text()
                )
                self.assertTrue(data["complete"])
                self.assertEqual(bool(data["errors"]), not valid)
                if valid:
                    self.assertEqual(
                        data["samples"][0]["engines"]["h:10000"]["inflight_requests"], 3
                    )

    def test_birth_spec_preserves_expected_performance_topology_and_config(self):
        from flexlb_cfg import render_env
        from flexlb_test_framework.scenario.backend import make_env_spec

        handlers = builtin_handlers()
        plan = next(
            p
            for p in compile_scenarios(
                load_scenarios(ROOT / "scenarios/elastic/lifecycle.yaml"),
                handlers=handlers,
            )
            if p["variant_id"] == "transient_imbalance"
        )
        old = expected_environment("transient_imbalance", NS(profile="batch-window"))
        new = make_env_spec(plan["environment"], "batch-window", {"master_base": 28000})
        self.assertEqual(new.perf, old.perf)
        self.assertEqual(
            json.loads(render_env(new.master_profile, new.config_overrides)),
            old.resolved_config,
        )
        for field in (
            "n_prefill",
            "n_decode",
            "discovery",
            "prefill_cache_blocks",
            "decode_cache_blocks",
        ):
            self.assertEqual(getattr(new, field), getattr(old, field))
        self.assertEqual(old.master_env, {"FLEXLB_FT_SPEC_ID": "transient_bound"})
        self.assertEqual(new.master_env, {})

    def run_program(self, bad_survivor=False):
        handlers = builtin_handlers()
        plan = next(
            p
            for p in compile_scenarios(
                load_scenarios(ROOT / "scenarios/elastic/lifecycle.yaml"),
                handlers=handlers,
            )
            if p["variant_id"] == "transient_imbalance"
        )
        self.assertEqual(plan["environment"]["prefill_max_waiting_batches"], 16)
        self.assertEqual(
            plan["environment"]["config_overrides"]["queue_timeout_ms"], 60000
        )
        self.assertEqual(len(plan["stages"]), 19)
        clock = Clock()
        lock = threading.RLock()
        state = dict(removed=None, recovery=0, burst=0, concurrency=0, peak=0, issued=0)
        engines = snapshots()
        original = copy.deepcopy(engines)

        def rid():
            with lock:
                state["issued"] += 1
                return state["issued"]

        with tempfile.TemporaryDirectory() as temp:
            discovery = Path(temp) / "discovery.json"

            def sync():
                discovery.write_text(
                    json.dumps(
                        {
                            "mock.prefill.hosts.address": [
                                v["http_addr"]
                                for v in engines.values()
                                if v["role"] == "prefill"
                            ]
                        }
                    )
                )

            class Backend:
                def setup(self, ctx, environment, deadline):
                    state["environment"] = environment
                    sync()
                    return NS(discovery_file=discovery), NS(
                        next_request_id=rid, master_http_port=2, mock_http_port=3
                    )

                def teardown(self, ctx, deadline):
                    state["cleaned"] = True

            class Metrics:
                def __init__(self, ctx, max_duration_s):
                    self.thread = NS(start=lambda: None, ident=None)
                    self._stop = threading.Event()
                    self.done = threading.Event()
                    self.errors = []

                def stop(self, deadline):
                    state["metrics_stopped"] = True

                def snapshot(self):
                    data = metrics(0, int(clock()))["data"]
                    for sample in data["samples"]:
                        for name, row in original.items():
                            sample["engines"][row["http_addr"]] = dict(
                                inflight_requests=999 if name == tr.VICTIM else 1
                            )
                        if (
                            state["removed"] is not None
                            and sample["time_s"] >= state["removed"]
                        ):
                            sample["engines"].pop(tr.VICTIM, None)
                    return data

            def run(records, record, shape, **kw):
                is_burst = kw["stream_timeout_s"] == 45
                if is_burst:
                    self.assertEqual(kw["schedule_timeout_s"], 30)
                    self.assertEqual(kw["timeout_s"], 75)
                    self.assertEqual(
                        shape,
                        dict(
                            input_len=2048,
                            output_len=2,
                            block_keys=[record["wire_request_id"] * 100 + 1],
                        ),
                    )
                    with lock:
                        state["burst"] += 1
                        state["concurrency"] += 1
                        state["peak"] = max(state["peak"], state["concurrency"])
                    time.sleep(0.005)
                elif len(shape["block_keys"]) == 3:
                    state["recovery"] += 1
                else:
                    self.assertEqual(kw["stream_timeout_s"], 30)
                failed = bad_survivor and is_burst
                records.update(
                    record,
                    schedule=dict(status="OK"),
                    stream=dict(status="OK"),
                    prefill_addr=original[tr.P[0]]["grpc_addr"],
                    business_finished=not failed,
                    business_error_code=8510 if failed else None,
                    transport_terminal_s=clock(),
                    consumer_exit_s=clock(),
                )
                if is_burst:
                    with lock:
                        state["concurrency"] -= 1

            def http(ops, endpoint, deadline, body=None):
                self.assertEqual(endpoint, "snapshot")
                return dict(engines=copy.deepcopy(list(engines.values())))

            def urlopen(request, timeout):
                self.assertEqual(
                    json.loads(request.data), dict(engine=tr.VICTIM, mode="abrupt")
                )
                self.assertEqual(timeout, 5)
                state["removed"] = clock()
                engines.pop(tr.VICTIM)
                sync()
                return io.BytesIO(
                    json.dumps(
                        dict(
                            status="ok",
                            action="removed",
                            engine=tr.VICTIM,
                            mode="abrupt",
                        )
                    ).encode()
                )

            def master(*args, **kwargs):
                count = sum(v["role"] == "prefill" for v in engines.values())
                return 200, dict(
                    worker_summary=dict(PREFILL=dict(discovered=count, alive=count))
                )

            with patch.object(e, "ElasticMetrics", Metrics), patch.object(
                e.RecordedRequests, "run", run
            ), patch.object(e, "_http", http), patch.object(
                tr.urllib.request, "urlopen", urlopen
            ), patch(
                "flexlb_test_framework.harness.http_post_json", master
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
            self.assertEqual(state["concurrency"], 0)
            evidence = {
                p.name: json.loads(p.read_text())
                for p in (Path(temp) / "artifacts").glob("elastic-transient-*.json")
            }
            return result, state, evidence

    def test_formal_program_preserves_burst_caps_windows_and_all_issued(self):
        result, state, evidence = self.run_program()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(state["burst"], 30)
        self.assertLessEqual(state["peak"], 15)
        self.assertGreater(state["peak"], 1)
        self.assertEqual(state["recovery"], 20)
        self.assertEqual(sum(len(s["checks"]) for s in result["stages"]), 20)
        locality = next(
            v
            for k, v in evidence.items()
            if k.startswith("elastic-transient-locality-")
        )
        self.assertGreater(len(locality["records"]), 30)
        self.assertTrue(locality["summary"]["result_complete"])

    def test_formal_survivor_failure_is_not_an_allowed_unrouted_failure(self):
        result, _, _ = self.run_program(bad_survivor=True)
        row = next(s for s in result["stages"] if s["id"] == "locality")
        self.assertEqual(row["status"], "FAIL", result)
        self.assertEqual(
            next(c for c in row["checks"] if c["id"] == "survivor_failures")["status"],
            "FAIL",
        )


if __name__ == "__main__":
    unittest.main()
