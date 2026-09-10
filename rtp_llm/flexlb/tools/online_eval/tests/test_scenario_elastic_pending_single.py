"""Independent SINGLE+BATCH ledger model; not Java scheduler acceptance."""

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

from environment_expectations import environment as expected_environment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.actions import elastic as e
from flexlb_test_framework.scenario.actions import elastic_concurrent as concurrent
from flexlb_test_framework.scenario.actions import elastic_lifecycle as life
from flexlb_test_framework.scenario.actions import engine_control as control
from flexlb_test_framework.scenario.backend import make_env_spec
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


class SingleBatchTests(unittest.TestCase):
    def plans(self):
        handlers = {h.name: h for h in e.HANDLERS + control.HANDLERS}
        return (
            compile_scenarios(
                load_scenarios(ROOT / "scenarios/elastic/pending_drain.yaml"),
                handlers=handlers,
            ),
            handlers,
        )

    def test_profile_registration_and_entire_expected_config_are_preserved(self):
        plans, _ = self.plans()
        self.assertEqual(
            {(p["variant_id"], p["profile"]) for p in plans},
            {
                ("legacy_terminal", "batch-window"),
                ("zero_errors", "batch-window"),
                ("single_batch_terminal", "single-batch"),
            },
        )
        for p in plans:
            old = expected_environment("pending_drain", NS(profile=p["profile"]))
            new = make_env_spec(p["environment"], p["profile"], {"master_base": 28000})
            self.assertEqual(
                p["environment"]["resolved_config"],
                old.resolved_config,
            )
            self.assertEqual(new.perf, old.perf)
            for field in [
                "n_prefill",
                "n_decode",
                "discovery",
                "prefill_cache_blocks",
                "decode_cache_blocks",
            ]:
                self.assertEqual(getattr(old, field), getattr(new, field))
        sb = next(p for p in plans if p["profile"] == "single-batch")
        self.assertEqual(
            sb["environment"]["effective_axes"],
            dict(
                scheduler="QUEUE",
                ordering="PRIORITY",
                decision="SINGLE",
                dispatcher="BATCH",
            ),
        )
        self.assertEqual(
            sb["environment"]["resolved_config"]["scheduler"]["decision"],
            {"type": "SINGLE"},
        )
        self.assertEqual(len(sb["stages"]), 13)
        self.assertEqual(sum(len(s["check_ids"]) for s in sb["stages"]), 13)

    def run_program(
        self,
        *,
        nonbatch=False,
        bypass_cap=False,
        completed_interference=False,
        recovery_errors=0,
    ):
        plans, handlers = self.plans()
        plan = next(p for p in plans if p["variant_id"] == "single_batch_terminal")
        clock = Clock()
        lock = threading.RLock()
        removed = threading.Event()
        addresses = {f"prefill-{i}": f"127.0.0.1:{10001+2*i}" for i in range(2)}
        state = dict(
            rid=0,
            requests={},
            shapes=[],
            ledgers={n: [] for n in addresses},
            queued={n: [] for n in addresses},
            enqueue_batches=[],
            fetch=[],
            generate=[],
            completed={n: 0 for n in addresses},
            recovery=0,
            active=0,
            peak=0,
        )
        barrier = threading.Barrier(10)
        engines = {
            n: dict(name=n, role="prefill", grpc_addr=a, stopped=False)
            for n, a in addresses.items()
        }

        def next_rid():
            with lock:
                state["rid"] += 1
                return state["rid"]

        def build(rid, **shape):
            state["shapes"].append(shape)
            return NS(rid=rid, **shape)

        class Future:
            def __init__(self, response):
                self.response = response

            def result(self):
                if self.response.pending and not removed.wait(timeout=3):
                    raise TimeoutError("pending Schedule never released")
                return self.response

            def cancel(self):
                return True

        class Stream:
            def __init__(self, rid):
                self.rid = rid
                self.cancelled = threading.Event()

            def cancel(self):
                self.cancelled.set()
                return True

            def __iter__(self):
                response = state["requests"][self.rid]
                if response.recovery:
                    with lock:
                        state["active"] += 1
                        state["peak"] = max(state["peak"], state["active"])
                    barrier.wait(timeout=3)
                    with lock:
                        state["active"] -= 1
                    error = response.ordinal <= recovery_errors
                else:
                    while not removed.is_set() and not self.cancelled.is_set():
                        time.sleep(0.001)
                    error = self.rid in state.get("retired_queue", [])
                if self.cancelled.is_set():
                    return
                yield NS(
                    HasField=lambda name: error,
                    error_info=NS(
                        error_code=8431, error_message="modeled queue retirement"
                    ),
                    flatten_output=NS(finished=[not error]),
                )

        def schedule(request, timeout):
            self.assertEqual(timeout, 30)
            with lock:
                recovery = removed.is_set()
                name = "prefill-1" if recovery or request.rid % 2 == 0 else "prefill-0"
                if recovery:
                    state["recovery"] += 1
                else:
                    self.assertEqual(request.input_len, 1024)
                    self.assertEqual(request.output_len, 2)
                    self.assertEqual(len(request.block_keys), 3)
                    # SINGLE makes exactly one member per decision. Capacity
                    # belongs to BATCH ledgers, not outstanding request count.
                    if len(state["ledgers"][name]) < 2 or bypass_cap:
                        batch = [request.rid]
                        state["ledgers"][name].append(batch)
                        state["enqueue_batches"].append(
                            dict(engine=name, members=batch, reason="single_request")
                        )
                    else:
                        state["queued"][name].append(request.rid)
                response = NS(
                    pending=not recovery and request.rid in state["queued"][name],
                    code=200,
                    success=True,
                    error_message="",
                    target=addresses[name],
                    recovery=recovery,
                    ordinal=state["recovery"],
                    enqueued_by_master=not nonbatch,
                )
                state["requests"][request.rid] = response
                return Future(response)

        def fetch(request, timeout):
            state["fetch"].append((request.request_id, timeout))
            return Stream(request.request_id)

        def generate(request, timeout):
            state["generate"].append((request.rid, timeout))
            return Stream(request.rid)

        ops = NS(
            next_request_id=next_rid,
            mock_http_port=1,
            master_http_port=2,
            schedule_pb2_grpc=NS(
                FlexlbServiceStub=lambda ch: NS(Schedule=NS(future=schedule))
            ),
            pb2_grpc=NS(
                RpcServiceStub=lambda ch: NS(
                    FetchResponse=fetch, GenerateStreamCall=generate
                )
            ),
            pb2=NS(FetchRequestPB=lambda **kw: NS(**kw)),
            _channel=lambda x: x,
            master_target=lambda: "master",
            build_schedule_request=build,
            prefill_addr=lambda r: r.target,
            build_generate_input=lambda rid, **shape: NS(rid=rid, **shape),
            _copy_role_addrs=lambda *args: None,
        )
        with tempfile.TemporaryDirectory() as temp:
            discovery = Path(temp) / "discovery.json"

            def sync():
                discovery.write_text(
                    json.dumps(
                        {
                            "mock.prefill.hosts.address": [
                                "127.0.0.1:"
                                + str(int(row["grpc_addr"].split(":")[-1]) - 1)
                                for row in engines.values()
                            ]
                        }
                    )
                )

            class Backend:
                def setup(self, ctx, environment, deadline):
                    if (
                        environment["effective_axes"]["decision"] != "SINGLE"
                        or environment["effective_axes"]["dispatcher"] != "BATCH"
                    ):
                        raise ValueError(
                            "fixture requires SINGLE+BATCH, not a renamed BW trace"
                        )
                    sync()
                    return NS(discovery_file=discovery), ops

                def teardown(self, ctx, deadline):
                    state["cleaned"] = True

            def http(ops, endpoint, deadline, body=None):
                with lock:
                    if endpoint == "snapshot":
                        rows = copy.deepcopy(list(engines.values()))
                        for row in rows:
                            name = row["name"]
                            row.update(
                                waiting=0,
                                running=sum(
                                    len(batch) for batch in state["ledgers"][name]
                                ),
                                completed=state["completed"][name],
                            )
                            if (
                                completed_interference
                                and state["rid"] >= 6
                                and name == "prefill-0"
                            ):
                                row["completed"] += 1
                        return dict(engines=rows)
                    if endpoint == "set_perf":
                        state.setdefault("perf", []).append(body)
                        return dict(
                            status="ok",
                            engine=body["engine"],
                            port=int(addresses[body["engine"]].split(":")[-1]),
                        )
                    self.assertEqual(endpoint, "remove_engine")
                    self.assertEqual(
                        body,
                        dict(
                            engine="prefill-0", mode="graceful", drain_timeout_ms=60000
                        ),
                    )
                    state["pre_remove_ledgers"] = copy.deepcopy(state["ledgers"])
                    state["pre_remove_queued"] = copy.deepcopy(state["queued"])
                    state["retired_queue"] = list(state["queued"]["prefill-0"])
                    state["remove_s"] = clock()
                    # Model both active 8s batches and the survivor's next wave
                    # finishing within 16s. This is not a Java timing claim.
                    clock.sleep(16)
                    for name in addresses:
                        state["ledgers"][name].clear()
                        state["queued"][name].clear()
                    engines.pop("prefill-0")
                    sync()
                    removed.set()
                    return dict(
                        status="ok",
                        action="removed",
                        engine="prefill-0",
                        port=10001,
                        mode="graceful",
                        drained=True,
                    )

            def accounting(*args):
                self.assertTrue(removed.is_set())
                self.assertFalse(any(state["ledgers"].values()))
                self.assertFalse(any(state["queued"].values()))
                return dict(
                    scheduler_inflight=0,
                    prefill_endpoints=[dict(inflight_batches=0)],
                    decode_endpoints=[dict(inflight_requests=0, total_load=0)],
                )

            with patch.object(e, "_http", http), patch.object(
                control, "_http", http
            ), patch.object(concurrent, "mutation_http", http), patch.object(
                life, "_master_get", accounting
            ), patch(
                "flexlb_test_framework.harness.http_post_json",
                return_value=(
                    200,
                    dict(worker_summary=dict(PREFILL=dict(discovered=1, alive=1))),
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
            state["artifacts"] = {
                p.name: json.loads(p.read_text())
                for p in (Path(temp) / "artifacts").glob("elastic-pending-*.json")
            }
            self.assertTrue(state["cleaned"])
            self.assertFalse(
                any(
                    t.name.startswith("elastic-pending-") and t.is_alive()
                    for t in threading.enumerate()
                )
            )
            return result, state

    def test_complete_single_batch_program_proves_modeled_two_ledgers_and_pending(self):
        result, state = self.run_program()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(state["pre_remove_ledgers"]["prefill-0"], [[1], [3]])
        self.assertEqual(state["pre_remove_queued"]["prefill-0"], [5])
        self.assertTrue(
            all(len(batch["members"]) == 1 for batch in state["enqueue_batches"])
        )
        self.assertEqual(len(state["fetch"]), 26)
        self.assertFalse(state["generate"])
        self.assertEqual(state["recovery"], 20)
        self.assertEqual(state["peak"], 10)
        self.assertEqual(sum(len(s["checks"]) for s in result["stages"]), 13)
        construction = next(
            v
            for k, v in state["artifacts"].items()
            if k.startswith("elastic-pending-wave-") and "-cleanup" not in k
        )
        self.assertEqual(construction["pending_estimate"], 2)
        self.assertEqual(construction["engine_waiting_running"], [0, 2])
        self.assertEqual(
            construction["inference"],
            "Schedule started but has not returned; engine counts are diagnostic only",
        )

    def test_capacity_bypass_and_completed_interference_cannot_manufacture_pending(
        self,
    ):
        for options, check in [
            (dict(bypass_cap=True), "pending_nonempty"),
            (dict(completed_interference=True), "no_completed_interference"),
        ]:
            result, state = self.run_program(**options)
            row = next(s for s in result["stages"] if s["id"] == "wave")
            self.assertEqual(
                next(c for c in row["checks"] if c["id"] == check)["status"],
                "FAIL",
                result,
            )
            self.assertNotIn("remove_s", state)

    def test_generate_stream_cannot_impersonate_single_batch_fetch_path(self):
        result, state = self.run_program(nonbatch=True)
        row = next(s for s in result["stages"] if s["id"] == "batch_path")
        self.assertEqual(row["status"], "FAIL", result)
        self.assertTrue(state["generate"])
        self.assertFalse(state["fetch"])
        self.assertNotIn("remove_s", state)

    def test_single_batch_recovery_keeps_19_of_20_boundary(self):
        for failures, status in [(1, "PASS"), (2, "FAIL")]:
            result, _ = self.run_program(recovery_errors=failures)
            self.assertEqual(
                next(s for s in result["stages"] if s["id"] == "recovery")["status"],
                status,
                result,
            )


if __name__ == "__main__":
    unittest.main()
