"""Run the complete ordered YAML using simulated external services."""

import copy
import json
import sys
import tempfile
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.actions import elastic as e
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
        if seconds < 0:
            raise AssertionError("negative sleep")
        self.now += seconds


class LifecycleTests(unittest.TestCase):
    def test_share_thresholds_preserve_inclusive_preference_and_exclusive_rebalance(
        self,
    ):
        before = dict(counts=dict(old0=0, old1=0, new=0))
        after = dict(counts=dict(old0=20, old1=20, new=60))
        for ceiling, exclusive, expected in [
            (0.6, False, "PASS"),
            (0.5, False, "FAIL"),
            (0.6, True, "FAIL"),
        ]:
            with tempfile.TemporaryDirectory() as root:
                ctx = NS(
                    resolve=lambda v: v,
                    resource=lambda v, *a: before if v == "before" else after,
                    artifact_dir=Path(root),
                )
                result = life.share(
                    ctx,
                    dict(
                        before="before",
                        after="after",
                        engine="new",
                        max_share=ceiling,
                        old_floor=0.1,
                        exclusive=exclusive,
                        require_new=True,
                    ),
                    NS(check=lambda: None),
                )
            self.assertEqual(result.checks[1].status, expected)

    def test_old_worker_starvation_is_independent_of_newcomer_ceiling(self):
        before = dict(counts=dict(old0=0, old1=0, new=0))
        after = dict(counts=dict(old0=5, old1=85, new=10))
        with tempfile.TemporaryDirectory() as root:
            ctx = NS(
                resolve=lambda v: v,
                resource=lambda v, *a: before if v == "before" else after,
                artifact_dir=Path(root),
            )
            result = life.share(
                ctx,
                dict(
                    before="before",
                    after="after",
                    engine="new",
                    max_share=0.6,
                    old_floor=0.1,
                    exclusive=False,
                    require_new=False,
                ),
                NS(check=lambda: None),
            )
        self.assertEqual(result.checks[1].status, "PASS")
        self.assertEqual(result.checks[2].status, "FAIL")

    def test_post_window_requires_fresh_traffic_but_allows_transient_only(self):
        for increment, expected in [(0, "FAIL"), (1, "PASS")]:
            samples = [
                dict(
                    offset_s=t,
                    counts=dict(
                        old0=t * 5, old1=t * 5, new=10 + (increment if t >= 5 else 0)
                    ),
                )
                for t in [0, 5, 10, 17, 24, 31, 38, 45]
            ]
            with tempfile.TemporaryDirectory() as root:
                data = dict(
                    series=dict(samples=samples), before=samples[0], after=samples[-1]
                )
                ctx = NS(
                    resolve=lambda v: v,
                    resource=lambda v, *a: data[v],
                    artifact_dir=Path(root),
                )
                share = life.share(
                    ctx,
                    dict(
                        series="series",
                        engine="new",
                        max_share=0.6,
                        old_floor=0.1,
                        exclusive=False,
                        require_new=False,
                    ),
                    NS(check=lambda: None),
                )
                self.assertTrue(all(c.status == "PASS" for c in share.checks))
                received = life.window_received(
                    ctx,
                    dict(before="before", after="after", engine="new"),
                    NS(check=lambda: None),
                )
                self.assertEqual(received.checks[0].status, expected)

    def test_remove_requires_fresh_traffic_despite_historical_accepts(self):
        result, state = self.run_program(no_remove_traffic=True)
        rows = {row["id"]: row for row in result["stages"]}
        self.assertEqual(rows["remove_traffic"]["status"], "FAIL")
        self.assertEqual(rows["remove"]["status"], "BLOCKED")
        self.assertTrue(state["cleaned"])

    def test_pre_add_successes_do_not_dilute_add_window_failure(self):
        flow = e.ClientRecords(1)
        for timestamp in range(110):
            record = flow.issue(timestamp, lambda: timestamp)
            flow.update(
                record,
                business_finished=timestamp < 108,
                schedule=dict(status="OK"),
                stream=dict(status="OK"),
                consumer_exit_s=110,
                transport_terminal_s=110,
            )
        with tempfile.TemporaryDirectory() as root:
            ctx = NS(
                clock=lambda: 120,
                resolve=lambda x: x,
                resource=lambda value, *a: (
                    flow if value == "flow" else dict(started_s=101)
                ),
                artifact_dir=Path(root),
            )
            result = life.add_availability(
                ctx,
                dict(flow="flow", mutation="mutation", received_s=109),
                NS(check=lambda: None),
            )
        self.assertEqual(result.checks[0].actual, 10)
        self.assertEqual(result.checks[2].actual, 0.8)
        self.assertEqual(result.checks[2].status, "FAIL")

    def run_program(
        self,
        variant="normal",
        fail_remove=False,
        batch_error=False,
        no_remove_traffic=False,
        profile="batch-window",
        driver_factory=None,
        delayed_old=0,
    ):
        clock = Clock()
        state = dict(
            engines={
                "prefill-0": dict(
                    role="prefill", grpc_addr="127.0.0.1:10001", accepted=0
                ),
                "prefill-1": dict(
                    role="prefill", grpc_addr="127.0.0.1:10003", accepted=0
                ),
            },
            active=False,
            last=0.0,
            adds=0,
            batches=0,
            next_rid=1,
            flow_count=0,
        )
        lock = threading.Lock()
        flows = []
        handlers = builtin_handlers()
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/elastic/lifecycle.yaml"), handlers=handlers
        )
        plan = next(
            p for p in plans if p["variant_id"] == variant and p["profile"] == profile
        )
        self.assertEqual(
            len(plan["stages"]), 12 if variant.startswith("rebalance") else 57
        )
        self.assertEqual(
            plan["resource_budget"]["max_dynamic_additions"],
            1 if variant.startswith("rebalance") else 4,
        )

        with tempfile.TemporaryDirectory() as root:
            discovery = Path(root) / "discovery.json"

            def sync_file():
                discovery.write_text(
                    json.dumps(
                        {
                            "mock.prefill.hosts.address": [
                                row["grpc_addr"].rsplit(":", 1)[0]
                                + ":"
                                + str(int(row["grpc_addr"].rsplit(":", 1)[1]) - 1)
                                for row in state["engines"].values()
                            ]
                        }
                    )
                )

            def next_rid():
                with lock:
                    rid = state["next_rid"]
                    state["next_rid"] += 1
                    return rid

            class Backend:
                def setup(self, ctx, environment, deadline):
                    sync_file()
                    ops = NS(master_http_port=1, next_request_id=next_rid)
                    if driver_factory:
                        driver_factory(ops, environment, state, clock)
                    return NS(discovery_file=discovery), ops

                def start_requests(self, ctx, params, deadline):
                    return ctx.register_resource("requests", object())

                def wait_requests(self, ctx, resource, deadline):
                    return dict(completed=True, error_count=0)

                def teardown(self, ctx, deadline):
                    state["cleaned"] = True

            class Flow(e.RecordedRequests if driver_factory else e.ClientRecords):
                pump_error = None

                def __init__(self, ops, epoch, clock):
                    if driver_factory:
                        super().__init__(ops, epoch, clock)
                    else:
                        super().__init__(epoch)
                    state["flow_count"] += 1
                    self.ordinal = state["flow_count"]
                    flows.append(self)

                def start(self):
                    state["active"] = True
                    state["active_flow"] = self
                    state["last"] = clock()
                    if driver_factory:
                        self.sample()
                        return
                    record = self.issue(next_rid(), clock)
                    self.update(
                        record,
                        business_finished=not (fail_remove and self.ordinal == 2),
                        prefill_addr="127.0.0.1:10001",
                        schedule=dict(status="OK"),
                        stream=dict(
                            status="OK",
                            method=(
                                "GenerateStreamCall"
                                if "nonbatch" in profile
                                else "FetchResponse"
                            ),
                        ),
                        consumer_exit_s=clock(),
                        transport_terminal_s=clock(),
                    )

                def sample(self):
                    previous = state.get("request_source")
                    state["request_source"] = "flow"
                    try:
                        rid = next_rid()
                        record = self.issue(rid, clock)
                        self.run(
                            record,
                            dict(output_len=2, block_keys=[rid * 100 + 1]),
                            stream_timeout_s=10,
                        )
                    finally:
                        state["request_source"] = previous

                def stop(self, deadline, cancel=False):
                    state["active"] = False
                    self.stopped = True
                    return e.completeness(self.snapshot_records())

            def http(ops, endpoint, deadline, body=None):
                if state["active"]:
                    elapsed = clock() - state["last"]
                    for row in [] if driver_factory else state["engines"].values():
                        if not (
                            no_remove_traffic and state["active_flow"].ordinal == 2
                        ):
                            row["accepted"] += round(elapsed * 10)
                    state["last"] = clock()
                    if elapsed > 0 and driver_factory:
                        for _ in range(max(len(state["engines"]), round(elapsed * 5))):
                            state["active_flow"].sample()
                    if elapsed > 0 and not driver_factory:
                        flow = state["active_flow"]
                        record = flow.issue(next_rid(), clock)
                        flow.update(
                            record,
                            business_finished=True,
                            prefill_addr="127.0.0.1:10001",
                            schedule=dict(status="OK"),
                            stream=dict(
                                status="OK",
                                method=(
                                    "GenerateStreamCall"
                                    if "nonbatch" in profile
                                    else "FetchResponse"
                                ),
                            ),
                            consumer_exit_s=clock(),
                            transport_terminal_s=clock(),
                        )
                if endpoint == "snapshot":
                    return dict(
                        engines=[
                            dict(name=name, **copy.deepcopy(row))
                            for name, row in state["engines"].items()
                        ]
                    )
                if endpoint == "add_engine":
                    state["adds"] += 1
                    state["engines"]["prefill-0"]["accepted"] += delayed_old
                    name = f"prefill-{state['adds']+1}"
                    port = 10003 + 2 * state["adds"]
                    state["engines"][name] = dict(
                        role="prefill", grpc_addr=f"127.0.0.1:{port}", accepted=0
                    )
                    sync_file()
                    return dict(
                        status="ok",
                        action="added",
                        engine=name,
                        port=port,
                        http_port=port - 1,
                    )
                if endpoint == "remove_engine":
                    name = body["engine"]
                    row = state["engines"].pop(name)
                    sync_file()
                    return dict(
                        status="ok",
                        action="removed",
                        engine=name,
                        port=int(row["grpc_addr"].rsplit(":", 1)[1]),
                        mode="graceful",
                        drained=True,
                    )
                raise AssertionError(endpoint)

            def master(*args, **kwargs):
                n = len(state["engines"])
                return 200, dict(
                    worker_summary={"PREFILL": dict(discovered=n, alive=n)}
                )

            def accounting(*args, **kwargs):
                return dict(
                    scheduler_inflight=0,
                    prefill_endpoints=[
                        dict(inflight_batches=0) for _ in state["engines"]
                    ],
                    decode_endpoints=[dict(total_load=0) for _ in range(4)],
                )

            def run_record(records, record, shape, **kwargs):
                with lock:
                    index = state["batches"]
                    state["batches"] += 1
                    names = sorted(state["engines"])
                    name = names[index % len(names)]
                    if not (
                        no_remove_traffic
                        and state.get("active_flow")
                        and state["active_flow"].ordinal == 2
                    ):
                        state["engines"][name]["accepted"] += 1
                records.update(
                    record,
                    business_finished=not (batch_error and index == 50),
                    prefill_addr=state["engines"][name]["grpc_addr"],
                    schedule=dict(status="OK"),
                    stream=dict(
                        status="OK",
                        method=(
                            "GenerateStreamCall"
                            if "nonbatch" in profile
                            else "FetchResponse"
                        ),
                    ),
                    consumer_exit_s=clock(),
                    transport_terminal_s=clock(),
                )

            with patch.object(e, "_http", side_effect=http), patch.object(
                e, "ColdFlow", Flow
            ), (
                nullcontext()
                if driver_factory
                else patch.object(e.RecordedRequests, "run", run_record)
            ), patch.object(
                life, "_master_get", side_effect=accounting
            ), patch(
                "flexlb_test_framework.harness.http_post_json", side_effect=master
            ):
                result = execute_instance(
                    plan,
                    Backend(),
                    handlers=handlers,
                    artifact_dir=Path(root) / "artifacts",
                    clock=clock,
                    sleeper=clock.sleep,
                )
            self.assertTrue(state["cleaned"])
            self.assertTrue(all(flow.stopped for flow in flows))
            state["artifacts"] = [
                json.loads(p.read_text())
                for p in (Path(root) / "artifacts").glob("elastic-batch-*.json")
            ]
            state["plan"] = plan
            state["flow_records"] = [f.snapshot_records() for f in flows]
            state["recovery_artifacts"] = [
                json.loads(p.read_text())
                for p in (Path(root) / "artifacts").glob(
                    "elastic-cycle-recovery-*.json"
                )
            ]
            state["probe_artifacts"] = [
                json.loads(p.read_text())
                for p in (Path(root) / "artifacts").glob("elastic-added-probe-*.json")
            ]
        return result, state

    def test_normal_and_strict_complete_all_stages_and_four_additions(self):
        for variant in ("normal", "strict"):
            result, state = self.run_program(variant)
            self.assertEqual(result["status"], "PASS", result)
            self.assertEqual(state["adds"], 4)
            self.assertGreaterEqual(state["batches"], 5)
            self.assertEqual(sum(len(s["checks"]) for s in result["stages"]), 75)
            self.assertTrue(all(s["status"] == "PASS" for s in result["stages"]))

    def test_remove_failure_cannot_be_hidden_by_preference_90_percent_floor(self):
        result, state = self.run_program(fail_remove=True)
        self.assertEqual(result["status"], "FAIL")
        rows = {s["id"]: s for s in result["stages"]}
        self.assertEqual(rows["preference_availability"]["status"], "PASS")
        self.assertEqual(rows["remove_zero_errors"]["status"], "FAIL")
        self.assertEqual(rows["cycle1_add"]["status"], "BLOCKED")

    def test_rebalance_is_immediate_without_preference_warmup(self):
        result, state = self.run_program("rebalance")
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(state["flow_count"], 0)
        self.assertEqual(state["batches"], 100)
        self.assertEqual(state["adds"], 1)
        rows = {row["id"]: row for row in result["stages"]}
        self.assertEqual(rows["rebalance_share"]["status"], "PASS")

    def test_rebalance_request_error_fails_independently_of_share(self):
        result, _ = self.run_program("rebalance", batch_error=True)
        self.assertEqual(result["status"], "FAIL")
        row = next(s for s in result["stages"] if s["id"] == "rebalance_after_add")
        self.assertEqual(
            {c["id"]: c["status"] for c in row["checks"]},
            {"complete": "PASS", "no_errors": "FAIL", "protocol": "PASS"},
        )

    def test_missing_decode_owner_field_is_error(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = NS(clock=lambda: 0, artifact_dir=Path(root))
            deadline = NS(check=lambda: None, remaining=lambda: 100)
            with patch.object(
                life,
                "_master_get",
                return_value=dict(
                    scheduler_inflight=0,
                    prefill_endpoints=[dict(inflight_batches=0)],
                    decode_endpoints=[{}],
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "decode inflight_requests/total_load"
                ):
                    life.accounting(ctx, {}, deadline)
