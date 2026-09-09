"""Controlled crash transport plus real owned survivor consumers for TOMBSTONED."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_scenario_priority_preemption as programs
from environment_expectations import environment as expected_environment
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import priority_preemption as preempt
from flexlb_test_framework.scenario.backend import make_env_spec
from flexlb_test_framework.scenario.runtime import execute_instance


class CrashError(Exception):
    def code(self):
        return NS(name="UNAVAILABLE")

    def trailing_metadata(self):
        return None


class CrashStream:
    def __init__(self, backend):
        self.backend = backend
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()
        self.backend.crashed.set()
        return True

    def __iter__(self):
        yield NS(HasField=lambda key: False, flatten_output=NS(finished=[False]))
        if not self.backend.crashed.wait(10):
            raise RuntimeError("fixture crash never triggered")
        if self.backend.victim_completed:
            yield NS(HasField=lambda key: False, flatten_output=NS(finished=[True]))
        else:
            raise CrashError("fixture original Prefill transport lost")


class TombstoneBackend(programs.Backend):
    def __init__(
        self,
        fence_code=8429,
        grow=False,
        victim_completed=False,
        lose_final_count=False,
        trigger_error=False,
    ):
        super().__init__()
        self.ops.batch = True
        self.fence_code, self.grow = fence_code, grow
        self.victim_completed, self.lose_final_count = (
            victim_completed,
            lose_final_count,
        )
        self.trigger_error = trigger_error
        self.crashed = threading.Event()
        self.stopped = False
        self.restarted = False
        self.armed = False
        self.fetches = []
        self.controls = []
        self.probes = []
        self.after_snapshots = 0
        self.residue_samples = []
        original = self.ops.future

        def future(req, timeout, metadata=None):
            call = original(req, timeout, metadata)
            if req[0] == 2:
                if not self.armed:
                    raise AssertionError("sacrifice issued before crash_after")
                self.stopped = True
                self.crashed.set()
                if self.trigger_error:

                    def result(timeout):
                        raise CrashError("sacrificial RPC cut")

                    call.result = result
            return call

        self.ops.future = future
        fetch = self.ops.fetch

        def wrapped(req, timeout):
            self.fetches.append(
                dict(
                    rid=req["request_id"],
                    timeout=timeout,
                    scheduled=len(self.ops.responses),
                )
            )
            if req["request_id"] == 1:
                self.ops.fetch_count += 1
                stream = CrashStream(self)
                self.ops.streams.append(stream)
                return stream
            return fetch(req, timeout)

        self.ops.fetch = wrapped
        self.ops.prefill_addr = lambda response: "127.0.0.1:1234"
        self.ops.pb2.EnqueueBatchRequestPB = lambda **kw: NS(**kw)
        self.ops.pb2.EnqueueBatchDpSlotPB = lambda **kw: NS(**kw)
        self.ops.pb2.EnqueueBatchExternalInputPB = lambda **kw: NS(**kw)
        self.ops.pb2_grpc.RpcServiceStub = lambda channel: NS(
            FetchResponse=self.ops.fetch,
            GenerateStreamCall=self.ops.generate,
            EnqueueBatch=self.enqueue,
        )

    def enqueue(self, req, timeout):
        inp = req.dp_slots[0].requests[0].input
        self.probes.append(
            dict(
                rid=inp[0],
                shape=inp[1],
                batch_id=req.batch_id,
                timeout=timeout,
                fetch_attach_timeout_ms=req.fetch_attach_timeout_ms,
            )
        )
        return NS(
            successes=[],
            errors=[NS(request_id=inp[0], error_info=NS(error_code=self.fence_code))],
        )

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "inject":
            self.controls.append(dict(body))
            self.armed = body["enabled"]
            return dict(
                status="ok", engine=body["engine"], type=body["type"], port=1234
            )
        if endpoint == "start_engine":
            self.controls.append(dict(body, operation="start"))
            self.stopped = False
            self.restarted = True
            self.armed = False
            return dict(status="ok", engine=body["engine"], port=1234)
        if endpoint != "snapshot":
            raise AssertionError(endpoint)
        arrived = len(self.ops.responses) >= 3
        cancel = 0
        if arrived:
            self.after_snapshots += 1
            cancel = 0 if self.lose_final_count and self.after_snapshots > 1 else 1
        return dict(
            engines=[
                dict(
                    name=role + "-0",
                    role=role,
                    grpc_addr=f"127.0.0.1:{1234+i}",
                    stopped=self.stopped if i == 0 else False,
                    inflight=0,
                    leak_detected=False,
                    rpc_counts={"cancel": cancel if i == 0 else 0},
                    request_lifecycle=(
                        {}
                        if i == 0 and self.restarted
                        else {
                            "1": dict(
                                running_ms=100,
                                end_state="finished" if arrived else "running",
                            )
                        }
                    ),
                    cancelled_rids=[],
                )
                for i, role in enumerate(("prefill", "decode"))
            ]
        )

    def status_http(
        self, ctx, server, path, deadline, body=None, allowed=(200,), text=False
    ):
        if server != "master":
            raise AssertionError(server)
        if path == "rtp_llm/master/info":
            if body != {}:
                raise AssertionError("Master info requires POST JSON {}")
            return 200, dict(
                worker_summary={"PREFILL": dict(alive=0 if self.stopped else 1)}
            )
        if path == "rtp_llm/inflight_status":
            count = 2 if self.grow and self.residue_samples else 1
            self.residue_samples.append(count)
            return 200, dict(scheduler_inflight=count)
        raise AssertionError(path)


class TombstonedPrograms(unittest.TestCase):
    def test_health_uses_real_master_info_post_contract(self):
        from flexlb_test_framework.scenario.actions import status_protocol

        for state, alive in (("dropped", 0), ("restored", 1)):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                ctx = NS(
                    clock=lambda: 0,
                    artifact_dir=Path(tmp),
                    env=NS(master_http_port=12345),
                )
                deadline = NS(check=lambda: None, remaining=lambda: 30)

                class Response:
                    status = 200

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        pass

                    def read(self, limit):
                        return json.dumps(
                            {"worker_summary": {"PREFILL": {"alive": alive}}}
                        ).encode()

                def urlopen(request, timeout):
                    self.assertEqual(
                        request.full_url, "http://127.0.0.1:12345/rtp_llm/master/info"
                    )
                    self.assertEqual(request.get_method(), "POST")
                    self.assertEqual(json.loads(request.data), {})
                    return Response()

                with patch.object(status_protocol.urllib.request, "urlopen", urlopen):
                    result = preempt._ts_health(ctx, {"state": state}, deadline)
                self.assertEqual(result.checks[0].status, "PASS")

    def test_health_accepts_successful_sparse_roles_and_keeps_invalid_raw(self):
        from flexlb_test_framework.scenario.actions import status_protocol

        for raw, valid in (
            ({"success": True, "code": 200, "worker_summary": None}, True),
            ({"success": True, "code": 200, "worker_summary": {}}, True),
            (
                {
                    "success": True,
                    "code": 200,
                    "worker_summary": {"DECODE": {"discovered": 1, "alive": 1}},
                },
                True,
            ),
            ({"worker_summary": {}}, False),
            ({"success": False, "code": 500, "worker_summary": None}, False),
            (
                {
                    "success": True,
                    "code": 200,
                    "worker_summary": {"DECODE": {"discovered": 1, "alive": 2}},
                },
                False,
            ),
            ({"success": True, "code": 200, "worker_summary": []}, False),
        ):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as tmp:
                ctx = NS(clock=lambda: 0, artifact_dir=Path(tmp))
                with patch.object(status_protocol, "_http", return_value=(200, raw)):
                    if valid:
                        result = preempt._ts_health(ctx, {"state": "dropped"}, None)
                        self.assertEqual(result.checks[0].status, "PASS")
                    else:
                        with self.assertRaises((ValueError, RuntimeError)):
                            preempt._ts_health(ctx, {"state": "dropped"}, None)
                samples = json.loads(
                    next(Path(tmp).glob("preemption-ts-health-*.json")).read_text()
                )
                self.assertEqual(samples[0]["raw"], raw)
                self.assertEqual(samples[0]["matched"], valid)
                if not valid:
                    self.assertNotIn("alive", samples[0])

    def run_program(self, profile="single-batch", **kwargs):
        _, registry = programs.PreemptionPrograms().plan("cancel_tombstoned")
        plans = compile_scenarios(
            load_scenarios(
                programs.ROOT / "scenarios/priority/priority_preemption.yaml"
            ),
            handlers=registry,
        )
        plan = next(
            p
            for p in plans
            if p["variant_id"] == "cancel_tombstoned" and p["profile"] == profile
        )
        backend = TombstoneBackend(**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            preempt, "_http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.engine_fault._http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.engine_control._http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.status_protocol._http",
            backend.status_http,
        ):
            result = execute_instance(
                plan, backend, handlers=registry, artifact_dir=tmp
            )
            artifacts = {}
            for kind in ("cut", "cancel", "fence", "residue", "trigger"):
                artifacts[kind] = [
                    json.loads(p.read_text())
                    for p in Path(tmp).glob(f"preemption-ts-{kind}-*.json")
                ]
        checks = {c["id"]: c["status"] for s in result["stages"] for c in s["checks"]}
        return result, artifacts, checks, backend, plan

    def test_both_batch_profiles_keep_single_decision_crash_and_fence_contract(self):
        for profile in ("single-batch", "batch-window"):
            with self.subTest(profile=profile):
                result, a, checks, b, plan = self.run_program(profile)
                self.assertEqual("PASS", result["status"], result)
                old = expected_environment("cancel_tombstoned", NS(profile=profile))
                env = b.environments[0]
                actual = make_env_spec(env, profile, {"master_base": 28000})
                self.assertEqual(
                    old.resolved_config,
                    env["resolved_config"],
                )
                self.assertEqual(
                    (old.n_prefill, old.n_decode, old.perf, old.master_env),
                    (actual.n_prefill, actual.n_decode, actual.perf, actual.master_env),
                )
                self.assertEqual(
                    "SINGLE", env["resolved_config"]["scheduler"]["decision"]["type"]
                )
                self.assertEqual(
                    [(30, 512, 5000), (None, 2048, 10), (70, 512, 2), (None, 2048, 2)],
                    [
                        (r.get("priority"), r["input_len"], r["output_len"])
                        for r in b.shapes
                    ],
                )
                self.assertEqual([1, 3, 4], [f["rid"] for f in b.fetches])
                self.assertEqual(1, b.fetches[0]["scheduled"])
                self.assertEqual(0, b.ops.generate_count)
                self.assertTrue(all(59 < f["timeout"] <= 60 for f in b.fetches[:2]))
                row = a["cut"][0]["row"]
                self.assertTrue(a["cut"][0]["cut"])
                self.assertTrue(
                    row["consumer_done"] and row["consumer_completion_verified"]
                )
                self.assertEqual("UNAVAILABLE", row["stream"]["status"])
                self.assertFalse(row["business_finished"])
                self.assertIsNone(row["cancel"]["requested_s"])
                self.assertTrue(b.restarted)
                self.assertFalse(b.stopped)
                self.assertFalse(b.armed)
                self.assertEqual(1, len(b.probes))
                self.assertEqual(1, b.probes[0]["rid"])
                self.assertEqual(11, b.probes[0]["batch_id"])
                self.assertEqual({"output_len": 2}, b.probes[0]["shape"])
                self.assertEqual(30000, b.probes[0]["fetch_attach_timeout_ms"])
                self.assertEqual([1, 1], b.residue_samples)
                self.assertEqual(
                    ("PASS", "PASS", "PASS"),
                    tuple(checks[k] for k in ("PR10", "PR6", "P6")),
                )
                self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_wrong_fence_code_does_not_pass_exact_8429(self):
        result, _, checks, _, _ = self.run_program(fence_code=8400)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("PASS", "FAIL", "PASS"), tuple(checks[k] for k in ("PR10", "PR6", "P6"))
        )

    def test_residue_growth_fails_p6_without_requiring_zero(self):
        result, _, checks, b, _ = self.run_program(grow=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual([1, 2], b.residue_samples)
        self.assertEqual(
            ("PASS", "PASS", "FAIL"), tuple(checks[k] for k in ("PR10", "PR6", "P6"))
        )

    def test_completed_victim_cannot_be_reported_as_crash_cut(self):
        result, a, checks, _, _ = self.run_program(victim_completed=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertFalse(a["cut"][0]["cut"])
        self.assertEqual(
            ("FAIL", "PASS", "PASS"), tuple(checks[k] for k in ("PR10", "PR6", "P6"))
        )

    def test_reached_poll_and_final_delta_are_separate_expected_facts(self):
        result, a, checks, _, _ = self.run_program(lose_final_count=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertTrue(a["cancel"][0]["reached"])
        self.assertEqual(0, a["cancel"][0]["delta"])
        self.assertEqual(
            ("FAIL", "PASS", "PASS"), tuple(checks[k] for k in ("PR10", "PR6", "P6"))
        )

    def test_sacrificial_rpc_error_does_not_replace_health_crash_proof(self):
        result, a, _, b, _ = self.run_program(trigger_error=True)
        self.assertEqual("PASS", result["status"], result)
        self.assertIn("sacrificial RPC cut", a["trigger"][0]["error"])
        self.assertTrue(b.restarted)
