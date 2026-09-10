"""Recovery owner/log evidence and complete YAML with an independent model."""

import copy
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_test_framework.scenario.actions import engine_control
from flexlb_test_framework.scenario.actions import engine_recovery as recovery
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.compiler import compile_scenarios
from flexlb_test_framework.scenario.contracts import PlanContext
from flexlb_test_framework.scenario.loader import load_scenarios
from flexlb_test_framework.scenario.runtime import (
    Deadline,
    RuntimeContext,
    execute_instance,
)
from test_scenario_backend import Ops, Stream
from test_scenario_runtime import Clock
from test_scenario_status_protocol import owner_frame


class Model:
    def __init__(self, mode="correct"):
        self.mode, self.restored, self.cleaned = mode, False, False
        self.wiped = set()
        self.schedule_shapes = []
        self.routed = {}
        self.cancel_failure_ids = set()
        self.master_cancel_attempts = 0
        self.worker_cancel_attempts = 0
        self.scheduler_samples = 0
        self.holder_view = None
        self.round_robin = 0
        self.ops = Ops()
        future = self.ops.future

        def compatible_future(*args, **kwargs):
            if self.mode == "ttft_regression":
                self.ctx.sleeper(0.1 if self.restored else 0.001)
            call = future(*args, **kwargs)
            req = args[0]
            rid, shape = req
            self.schedule_shapes.append(dict(shape))
            if (
                self.mode == "master_cancel_unavailable"
                and shape.get("input_len") == 512
            ):
                self.cancel_failure_ids.add(rid)
            keys = shape.get("block_keys", [])
            if len(keys) == 10:
                index = (
                    self.holder_view
                    if self.holder_view is not None
                    else self.round_robin % 2
                )
                self.round_robin += 1
            else:
                index = 0
            if self.engines[f"prefill-{index}"]["stopped"]:
                index = 1 - index
            name = f"prefill-{index}"
            response = call.result(timeout=None)
            response.route = name + ":9001"
            response.HasField = lambda field: False
            self.routed.setdefault(name, set()).add(rid)
            if self.ops.batch:
                self.engines[name]["rpc_counts"]["enqueue_batch"] += 1
            threshold = self.engines[name].get("crash_after")
            if (
                self.ops.batch
                and threshold
                and self.engines[name]["rpc_counts"]["enqueue_batch"] >= threshold
            ):
                self.engines[name]["stopped"] = True
                self.wiped.update(self.routed[name])
                self.append(
                    f"worker {name}:9000 marked dead after 3 consecutive gRPC failures"
                )
                for field in ["accepted", "running", "inflight", "held_blocks"]:
                    self.engines[name][field] = 0
                self.engines[name]["cache_key_set"] = []
                response.code, response.success, response.error_message = (
                    503,
                    False,
                    "empty acknowledgement",
                )
                result = call.result
                call.result = lambda timeout=None: result(timeout=timeout)
                return call
            self.engines[name]["cache_key_set"] = sorted(
                set(self.engines[name]["cache_key_set"]) | set(keys)
            )
            self.engines[name]["accepted"] += 1
            if self.restored and self.mode == "first_wave_lack_mem":
                self.engines[name]["lack_mem_rejects"] += 1
            result = call.result
            call.result = lambda timeout=None: result(timeout=timeout)
            return call

        self.ops.future = compatible_future
        self.ops.prefill_addr = lambda response: response.route
        self.ops.role_addr = lambda response, owner: response.route
        self.ops.schedule_pb2 = NS(
            FlexlbCancelRequestPB=lambda **kw: NS(**kw),
            CANCEL_REASON_CLIENT_CANCELLED=1,
        )
        factory = self.ops.schedule_pb2_grpc.FlexlbServiceStub

        class UnavailableRpc(RuntimeError):
            def code(self):
                return NS(name="UNAVAILABLE")

        def master_cancel(request, timeout):
            self.master_cancel_attempts += 1
            if request.request_id in self.cancel_failure_ids:
                raise UnavailableRpc("master cancel unavailable")
            return NS(found=True)

        def service(channel):
            stub = factory(channel)
            stub.Cancel = master_cancel
            return stub

        self.ops.schedule_pb2_grpc.FlexlbServiceStub = service
        fetch = self.ops.fetch

        def fetch_after_crash(request, timeout):
            if (
                request["request_id"] in self.wiped
                and self.mode != "old_request_resurrects"
            ):
                self.ops.fetch_count += 1

                class MissingRpc(RuntimeError):
                    def code(self):
                        return NS(name="NOT_FOUND")

                return Stream(error=MissingRpc("wiped request"))
            return fetch(request, timeout)

        self.ops.fetch = fetch_after_crash
        fetch_with_crash = self.ops.fetch
        generate = self.ops.generate

        def controlled_fetch(request, timeout):
            if request["request_id"] in self.cancel_failure_ids:
                return Stream(error=UnavailableRpc("payload unavailable"))
            return fetch_with_crash(request, timeout)

        def controlled_generate(request, timeout):
            if request[0] in self.cancel_failure_ids:
                return Stream(error=UnavailableRpc("payload unavailable"))
            return generate(request, timeout)

        self.ops.fetch, self.ops.generate = controlled_fetch, controlled_generate
        worker_factory = self.ops.pb2_grpc.RpcServiceStub
        self.ops.pb2.CancelRequestPB = lambda **kw: NS(**kw)
        self.ops.pb2.CANCEL_STATUS_NOT_FOUND = 1

        def worker_cancel(request, timeout):
            self.worker_cancel_attempts += 1
            return NS(status=0)

        for name in (
            "EnqueueBatchRequestPB",
            "EnqueueBatchDpSlotPB",
            "EnqueueBatchExternalInputPB",
        ):
            setattr(self.ops.pb2, name, lambda **kw: NS(**kw))

        def enqueue_crash(channel, request, timeout):
            name = channel.split(":")[0]
            engine = self.engines[name]
            engine["rpc_counts"]["enqueue_batch"] += 1
            if not engine.get("crash_after"):
                raise AssertionError("crash must be armed before direct trigger")
            engine["stopped"] = True
            self.wiped.update(self.routed.get(name, set()))
            self.append(
                f"worker {name}:9000 marked dead after 3 consecutive gRPC failures"
            )
            for field in ["accepted", "running", "inflight", "held_blocks"]:
                engine[field] = 0
            engine["cache_key_set"] = []
            return NS(successes=[], errors=[])

        def worker_service(channel):
            stub = worker_factory(channel)
            stub.Cancel = worker_cancel
            stub.EnqueueBatch = lambda request, timeout: enqueue_crash(
                channel, request, timeout
            )
            return stub

        self.ops.pb2_grpc.RpcServiceStub = worker_service
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
                    cache_key_set=[],
                    kv_tokens_used=0,
                    accepted=0,
                    lack_mem_rejects=0,
                    running=0,
                    inflight=0,
                    held_blocks=0,
                    leak_detected=False,
                    rpc_counts={"enqueue_batch": 0},
                )

    def setup(self, ctx, environment, deadline):
        self.ctx = ctx
        self.ops.batch = "nonbatch" not in ctx.instance["profile"]
        self.variant = ctx.instance["variant_id"]
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
            self.engines[name].pop("crash_after", None)
            self.engines[name]["rpc_counts"]["enqueue_batch"] = 0
            if self.mode == "wipe_keeps_accepted":
                self.engines[name]["accepted"] = 1
            self.engines[name]["kv_tokens_used"] = (
                9 if self.mode == "reset_used_nonzero" else 0
            )
            if self.engines[name]["cache_key_set"]:
                self.holder_view = int(name.rsplit("-", 1)[1])
            if self.mode != "no_generation":
                self.append(
                    f"Created WorkerStatus generation 2 for worker: {name}:9000"
                )
        elif endpoint == "inject":
            if body["enabled"]:
                self.engines[name]["crash_after"] = body["n"]
            else:
                self.engines[name].pop("crash_after", None)
            return dict(status="ok", engine=name, port=9001, type=body["type"])
        elif endpoint == "set_perf":
            self.engines[name]["prefill_fixed_ms"] = body["prefill_fixed_ms"]
        elif endpoint == "cache_evict":
            self.engines[name]["cache_key_set"] = sorted(
                set(self.engines[name]["cache_key_set"]) - set(body["keys"])
            )
            if self.mode != "stale_holder_after_wipe":
                self.holder_view = None
        elif endpoint == "set_kv_pressure":
            self.engines[name]["kv_tokens_used"] = body["active_kv_tokens"]
        else:
            raise RuntimeError(f"unexpected control {endpoint}")
        return dict(status="ok", engine=name, port=9001)

    def http(self, ctx, server, path, deadline, body=None, *args, **kwargs):
        if server == "mock":
            if path == "inject":
                if self.variant == "status_gap_long_retire":
                    if body["enabled"]:
                        self.append(
                            "worker prefill-0:9000 marked dead after 3 consecutive gRPC failures"
                        )
                    else:
                        self.append(
                            "Created WorkerStatus generation 2 for worker: prefill-0:9000"
                        )
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
        if self.mode == "residue_grows" and self.restored:
            self.scheduler_samples += 1
            data["scheduler_inflight"] = min(2, self.scheduler_samples)
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

    def test_poll_does_not_accept_a_new_sample_at_the_expired_window(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as root:
            ctx = RuntimeContext(
                {"environment": {"n_prefill": 2}}, None, root, clock, clock.sleep
            )
            ctx.env_epoch = 1
            params = {
                "duration_s": 0.5,
                "interval_s": 0.5,
                "include": ["info"],
                "until": {"metric": "alive_prefill", "op": "ge", "value": 2},
            }
            frames = [
                {"info": {"worker_summary": {"PREFILL": {"alive": n}}}} for n in (1, 2)
            ]
            with patch.object(recovery.status, "_frame", side_effect=frames) as capture:
                result = recovery.execute_observe(
                    ctx, params, Deadline(clock() + 10, clock, clock.sleep)
                )
            snapshot = recovery._snapshot(ctx, result.output["snapshot"])
            self.assertEqual(1, capture.call_count)
            self.assertEqual(
                1, recovery.metric(snapshot["frames"][-1], "alive_prefill")
            )

    def test_missing_target_ledger_is_error_not_zero(self):
        frame = {"targets": {"p": {"ip_port": "missing"}}, **owner_frame()}
        with self.assertRaises(RuntimeError):
            recovery.metric(frame, "target_prefill_requests")

    def test_expected_generate_payload_does_not_change_schedule(self):
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
        self.assertEqual(
            (
                3
                if variant == "kv_resync"
                else (2 if variant in {"crash_after", "no_resurrect"} else 4)
            ),
            len(plans),
        )
        for plan in plans:
            if variant == "crash_after":
                prepared = [
                    st for st in plan["stages"] if st["action"] == "recovery_prepare"
                ]
                self.assertEqual(
                    ["trigger", "takeover", "recovery"], [st["id"] for st in prepared]
                )
                self.assertEqual(
                    [10, 10, 10], [st["params"]["output_len"] for st in prepared]
                )
            with self.subTest(
                profile=plan["profile"], mode=mode
            ), tempfile.TemporaryDirectory() as root:
                model, clock = Model(mode), Clock()
                with patch.object(
                    recovery.status, "_http", side_effect=model.http
                ), patch.object(
                    engine_control, "_http", side_effect=model.control
                ), patch(
                    "flexlb_test_framework.scenario.actions.kv._http",
                    side_effect=model.control,
                ), patch(
                    "flexlb_test_framework.scenario.actions.engine_fault._http",
                    side_effect=model.control,
                ):
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
                if variant == "crash_after":
                    self.assertEqual(7, len(model.schedule_shapes))
                    self.assertEqual(
                        {10}, {shape["output_len"] for shape in model.schedule_shapes}
                    )
                if mode == "master_cancel_unavailable":
                    self.assertGreater(model.master_cancel_attempts, 0)
                    self.assertEqual(0, model.worker_cancel_attempts)
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

    def test_flap_stop_allows_pending_rpc_budget_before_proving_worker_exit(self):
        from flexlb_test_framework.scenario.actions.elastic import ColdFlow

        registry = handlers()
        registry.update({h.name: h for h in recovery.HANDLERS})
        path = (
            Path(__file__).resolve().parents[1]
            / "scenarios/engine_fault/engine_fault_recovery.yaml"
        )
        plans = [
            p
            for p in compile_scenarios(load_scenarios(path), handlers=registry)
            if p["variant_id"] == "flap"
        ]
        caps = {
            next(st["timeout_s"] for st in p["stages"] if st["id"] == "stop_flow")
            for p in plans
        }
        self.assertEqual({50}, caps)
        # Real pump thread and cancellation/exit, with a 100x wall-clock scale:
        # Schedule consumes 28 logical seconds and stream consumes six more.
        # Only waiting and fake RPC latency are scaled; production request
        # limits remain Schedule30/stream10 and their real code paths execute.
        scale = 100

        class ScaledWaitDeadline(Deadline):
            def remaining(self):
                return super().remaining() / scale

        for cap in (20, next(iter(caps))):
            with self.subTest(stop_cap=cap):
                origin = time.monotonic()
                clock = lambda: (time.monotonic() - origin) * scale
                entered, released, cancelled = (
                    threading.Event(),
                    threading.Event(),
                    threading.Event(),
                )
                ops = Ops()
                rpc_caps = []

                def future(req, timeout, metadata=None):
                    rpc_caps.append(timeout)
                    timer = threading.Timer(0.28, released.set)
                    timer.start()
                    entered.set()

                    def result():
                        if not released.wait(1):
                            raise RuntimeError("fixture release was lost")
                        timer.cancel()
                        if cancelled.is_set():
                            raise RuntimeError("cancelled pending Schedule")
                        return NS(code=200, success=True, enqueued_by_master=True)

                    def cancel():
                        cancelled.set()
                        released.set()
                        return True

                    return NS(result=result, cancel=cancel)

                ops.schedule_pb2_grpc.FlexlbServiceStub = lambda channel: NS(
                    Schedule=NS(future=future)
                )

                class DelayedStream:
                    def __iter__(self):
                        cancelled.wait(0.06)
                        if cancelled.is_set():
                            raise RuntimeError("cancelled stream")
                        yield NS(
                            HasField=lambda field: False,
                            flatten_output=NS(finished=[True]),
                        )

                    def cancel(self):
                        cancelled.set()
                        return True

                def fetch(req, timeout):
                    rpc_caps.append(timeout)
                    return DelayedStream()

                ops.fetch = fetch
                flow = ColdFlow(ops, 1, clock=clock)
                if cap == 50:
                    real_done = flow.done
                    flow.done = NS(
                        wait=lambda seconds: real_done.wait(seconds / scale),
                        set=real_done.set,
                        is_set=real_done.is_set,
                    )
                flow.start()
                try:
                    self.assertTrue(entered.wait(1))
                    limit = ScaledWaitDeadline(clock() + cap, clock)
                    if cap == 20:
                        with self.assertRaises(TimeoutError):
                            flow.stop(limit)
                        self.assertTrue(cancelled.is_set())
                    else:
                        with tempfile.TemporaryDirectory() as root:
                            ctx = RuntimeContext({}, None, root, clock, time.sleep)
                            ctx.env_epoch = 1
                            handle = ctx.register_resource("flow", flow)
                            output = recovery.execute_flow_stop(
                                ctx, {"flow": handle}, Deadline(clock() + cap, clock)
                            )
                            evidence = ctx.resource(output.output["result"], "snapshot")
                            self.assertEqual(0, evidence["observed_total"])
                            self.assertIsNone(
                                evidence["frozen_records"][0]["consumer_exit_s"]
                            )
                            result = evidence["final"]
                            self.assertTrue(result["result_complete"])
                            self.assertEqual(1, result["completed"])
                        self.assertFalse(cancelled.is_set())
                        self.assertAlmostEqual(10, rpc_caps[1], delta=0.1)
                    self.assertAlmostEqual(30, rpc_caps[0], delta=0.1)
                finally:
                    flow.stop(Deadline(clock() + 200, clock), cancel=True)
                self.assertTrue(flow.done.is_set())
                self.assertTrue(
                    all(
                        r["consumer_exit_s"] is not None
                        and r["transport_terminal_s"] is not None
                        for r in flow.snapshot_records()
                    )
                )

    def test_flap_freezes_business_counters_before_late_success_or_failure(self):
        from flexlb_test_framework.scenario.actions.elastic import (
            ClientRecords,
            completeness,
        )

        for early_successes, early_failures, late_success, expected in (
            (0, 1, True, "FAIL"),
            (1, 1, False, "PASS"),
            (0, 0, True, "FAIL"),
        ):
            with self.subTest(
                early=(early_successes, early_failures), late=late_success
            ):
                clock = Clock()
                ledger = ClientRecords(1)

                def finish(record, success):
                    ledger.update(
                        record,
                        business_finished=success,
                        schedule={"status": "OK"},
                        stream={"status": "OK"},
                        transport_terminal_s=clock(),
                        consumer_exit_s=clock(),
                    )

                for success in [True] * early_successes + [False] * early_failures:
                    finish(
                        ledger.issue(len(ledger.snapshot_records()) + 1, clock), success
                    )
                pending = ledger.issue(99, clock)
                stop_event = threading.Event()
                waits = []

                def wait(seconds):
                    self.assertTrue(stop_event.is_set())
                    waits.append(seconds)
                    clock.sleep(seconds)
                    return False

                def stop(deadline):
                    self.assertEqual(20, clock())
                    clock.sleep(14)
                    deadline.check()
                    finish(pending, late_success)
                    return completeness(ledger.snapshot_records())

                flow = NS(
                    _stop=stop_event,
                    done=NS(wait=wait),
                    stop=stop,
                    snapshot_records=ledger.snapshot_records,
                    pump_error=None,
                )
                with tempfile.TemporaryDirectory() as root:
                    ctx = RuntimeContext({}, None, root, clock, clock.sleep)
                    ctx.env_epoch = 1
                    handle = ctx.register_resource("flow", flow)
                    output = recovery.execute_flow_stop(
                        ctx, {"flow": handle}, Deadline(50, clock)
                    )
                    result = ctx.resource(output.output["result"], "snapshot")
                    verdict = recovery.execute_flow_assert(
                        ctx,
                        {
                            "result": output.output["result"],
                            "min_success_rate": 0.5,
                        },
                        Deadline(50, clock),
                    )
                    checks = {c.id: c for c in verdict.checks}
                    self.assertEqual(expected, checks["success_rate"].status)
                    self.assertEqual("PASS", checks["complete"].status)
                    self.assertEqual(early_successes, result["observed_ok"])
                    self.assertEqual(
                        early_successes + early_failures, result["observed_total"]
                    )
                    self.assertEqual(
                        early_successes + early_failures + 1, result["final"]["issued"]
                    )
                    self.assertIsNone(result["frozen_records"][-1]["consumer_exit_s"])
                    self.assertEqual(34, result["records"][-1]["consumer_exit_s"])
                    self.assertEqual([20], waits)
                    self.assertTrue(Path(output.artifacts[0]).exists())

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
            record["recovery_first_output_observed_s"] = (
                record["stream"]["first_output_s"] + 0.1
            )
        self.assertEqual(3100, recovery._ttft({"frames": [{"records": records}]}))
        for record in records:
            record["stream"]["first_output_s"] = None
        self.assertIsNone(recovery._ttft({"frames": [{"records": records}]}))

    def test_complete_kv_resync_program(self):
        self.run_program("kv_resync", "correct", "PASS")

    def test_wiped_holder_stickiness_is_a_failure(self):
        self.run_program(
            "kv_resync",
            "stale_holder_after_wipe",
            "FAIL",
            "memory_lost_old_holder_spreads",
        )

    def test_complete_kv_usage_program(self):
        self.run_program("kv_usage_reset", "correct", "PASS")

    def test_used_after_reset_remains_observational(self):
        self.run_program("kv_usage_reset", "reset_used_nonzero", "PASS")

    def test_first_wave_lack_mem_is_a_master_contract_failure(self):
        self.run_program(
            "kv_usage_reset", "first_wave_lack_mem", "FAIL", "first_wave_no_lack_mem"
        )

    def test_complete_crash_takeover_program(self):
        self.run_program("crash_after", "correct", "PASS")

    def test_crash_residue_growth_fails(self):
        self.run_program("crash_after", "residue_grows", "FAIL", "residue_window")

    def test_complete_no_resurrection_program(self):
        self.run_program("no_resurrect", "correct", "PASS")

    def test_old_request_completion_is_resurrection_failure(self):
        self.run_program(
            "no_resurrect",
            "old_request_resurrects",
            "FAIL",
            "old_requests_never_complete",
        )

    def test_wipe_must_clear_accepted_counter_before_consumption(self):
        self.run_program(
            "no_resurrect", "wipe_keeps_accepted", "FAIL", "wiped_accepted"
        )

    def test_complete_long_gap_program(self):
        self.run_program("status_gap_long_retire", "correct", "PASS")

    def test_master_cancel_failure_does_not_add_worker_cancel(self):
        self.run_program("status_gap_long_retire", "master_cancel_unavailable", "PASS")

    def test_recovered_ttft_regression_fails(self):
        self.run_program("down_phases", "ttft_regression", "FAIL", "ttft_recovers")

    def test_observation_timeout_does_not_cancel_transport_or_accept_late_success(self):
        released = threading.Event()
        cancelled = threading.Event()

        class GatedStream:
            def cancel(self):
                cancelled.set()
                released.set()
                return True

            def __iter__(self):
                if not released.wait(2):
                    raise RuntimeError("test did not release stream")
                yield NS(
                    HasField=lambda field: False, flatten_output=NS(finished=[True])
                )

        with tempfile.TemporaryDirectory() as root:
            ctx = RuntimeContext({}, None, root, time.monotonic, time.sleep)
            ctx.env_epoch = 1
            ctx.ops = Ops()
            ctx.instance_deadline_s = time.monotonic() + 120
            rpc_limits = []
            ctx.ops.fetch = lambda request, timeout: (
                rpc_limits.append(timeout) or GatedStream()
            )
            params = recovery.validate_prepare(
                {"stream_timeout_s": 0.05},
                PlanContext("test", {}, profiles=("batch-window",)),
            )
            cohort = recovery.RecoveryRequests(ctx, params)
            try:
                cohort.dispatch(Deadline(time.monotonic() + 5))
                self.assertFalse(cancelled.is_set())
                self.assertGreater(rpc_limits[0], 50)
                self.assertEqual(
                    0,
                    recovery.metric({"records": cohort.snapshot_records()}, "success"),
                )
                released.set()
                for child in cohort.children:
                    for entry in child.entries:
                        child._await_consumer(entry, Deadline(time.monotonic() + 2))
                cohort.prove_ended(Deadline(time.monotonic() + 2))
                self.assertTrue(cohort.snapshot_records()[0]["business_finished"])
                self.assertEqual(
                    0,
                    recovery.metric({"records": cohort.snapshot_records()}, "success"),
                )
            finally:
                released.set()
                cohort.cleanup(Deadline(time.monotonic() + 2))

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
