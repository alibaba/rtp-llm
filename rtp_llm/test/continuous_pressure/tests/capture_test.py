"""CPU-only native capture, file identity and profile correlation checks."""

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.continuous_pressure.adapters.capture import KinetoCapture
from rtp_llm.test.continuous_pressure.analysis.kineto import validate_kineto
from rtp_llm.test.continuous_pressure.analysis.nsys import normalize_nsys
from rtp_llm.test.continuous_pressure.diagnostics import CoverageAssessment


def nsys_fixture(path, step_ids=(1, 2, 3)):
    # Nsight 2025.6.3 column projection. Preserve NVIDIA's encoded PID/TID join.
    with sqlite3.connect(path) as db:
        db.executescript(
            """
          CREATE TABLE StringIds(id INTEGER, value TEXT);
          CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,
            globalTid INTEGER,uint64Value INTEGER);
          CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,
            globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,
            globalPid INTEGER,correlationId INTEGER,deviceId INTEGER,contextId INTEGER,
            streamId INTEGER,demangledName INTEGER);
          INSERT INTO StringIds VALUES(1,'cudaLaunchKernel'),(2,'model_matmul');
        """
        )
        for rank in (0, 1):
            pid = (1 << 48) | ((629566 + rank) << 24)
            tid = pid | (629566 + rank)
            for step in step_ids:
                start = step * 1000
                db.execute(
                    "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?)",
                    (start, start + 100, f"pressure.step/{step}", None, tid, rank),
                )
                db.execute(
                    "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?)",
                    (start + 1, start + 99, "executor.model_forward", None, tid, None),
                )
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)",
                    (start + 10, start + 20, tid, step, 1),
                )
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?)",
                    (start + 30, start + 90, pid, step, rank, 1, 7, 2),
                )


def steps():
    return [
        dict(step_id=i, context_batch=1, decode_batch=0, complete=True)
        for i in (1, 2, 3)
    ]


def actual_trace():
    # Compact projection of kineto-cuda-schema-v1.json, PyTorch 2.11/cu130.
    # Preserve real timestamps, categories, CPU/GPU namespaces and correlations.
    values = [
        (
            1,
            1652691158900.723,
            168952.889,
            1652691158955.581,
            168885.321,
            1652691327624.188,
            58.041,
            1652691327680.280,
            42.880,
            1364,
        ),
        (
            2,
            1652691327960.095,
            154.582,
            1652691327987.552,
            122.944,
            1652691328088.032,
            11.248,
            1652691328101.273,
            42.145,
            1381,
        ),
        (
            3,
            1652691328152.240,
            33.630,
            1652691328157.810,
            24.540,
            1652691328170.765,
            5.131,
            1652691328179.002,
            42.368,
            1398,
        ),
    ]
    events = []
    for i, ts, dur, fts, fdur, rts, rdur, kts, kdur, corr in values:
        base = dict(ph="X", pid=644596, tid=644596)
        events.extend(
            [
                dict(
                    base,
                    cat="user_annotation",
                    name=f"pressure.step/{i}",
                    ts=ts,
                    dur=dur,
                ),
                dict(
                    base,
                    cat="user_annotation",
                    name="executor.model_forward",
                    ts=fts,
                    dur=fdur,
                ),
                dict(
                    base,
                    cat="cuda_driver",
                    name="cuLaunchKernelEx",
                    ts=rts,
                    dur=rdur,
                    args={"External id": i * 4, "correlation": corr},
                ),
                dict(
                    ph="X",
                    pid=0,
                    tid=7,
                    cat="kernel",
                    name="cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_nnn_align1_bias_f32_relu",
                    ts=kts,
                    dur=kdur,
                    args={
                        "External id": i * 4,
                        "device": 0,
                        "context": 1,
                        "stream": 7,
                        "correlation": corr,
                    },
                ),
            ]
        )
    events.append(
        dict(name="Record Window End", ph="i", ts=1652691328405.796, pid="", tid="")
    )
    return dict(
        schemaVersion=1,
        baseTimeNanoseconds=1790857026000000000,
        displayTimeUnit="ms",
        traceEvents=events,
    )


class PressureProfileTest(unittest.TestCase):
    def capture_case(
        self, fault=None, check_online=False, remote_points=None, write_scopes=False
    ):

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        pages = []
        for rank in (0, 1):
            path = root / f"native_export_{rank}.json"
            trace = actual_trace()
            if write_scopes:
                from rtp_llm.test.continuous_pressure.analysis.kineto import (
                    REMOTE_NATIVE_WRITE_SCOPES,
                )

                for name in REMOTE_NATIVE_WRITE_SCOPES:
                    trace["traceEvents"].append(
                        dict(trace["traceEvents"][0], name=name, dur=1)
                    )
            path.write_text(
                "{" if fault == "truncated" and rank == 1 else json.dumps(trace)
            )
            pages.append(
                dict(
                    world_rank=rank,
                    service_instance=f"instance{rank}",
                    profile=dict(
                        supported=True,
                        state="finished",
                        trace_name="pressure_fixture",
                        collection_complete=True,
                        export_complete=True,
                        file_name=str(path),
                        capture_begin_ns="1000000000",
                        capture_end_ns="4000000000",
                    ),
                )
            )
        if fault == "rank":
            pages.pop()
        if fault == "identity":
            pages[1]["service_instance"] = "other"
        if fault == "export":
            pages[1]["profile"]["export_error"] = "disk full"
        raw = dict(
            response=dict(ranks=pages),
            rank_offset_bounds_s={"0": [10, 10.1], "1": [10, 10.1]},
        )
        controller = KinetoCapture(
            start_capture=lambda: {},
            read_status=lambda timeout: raw,
            stop_capture=lambda: {},
            trace_name="pressure_fixture",
            directory=root,
            ranks=(0, 1),
            instances={0: "instance0", 1: "instance1"},
            session_id="session",
            timeout_s=2,
            requested_steps=3,
        )
        if check_online:
            with patch(
                "json.load", side_effect=AssertionError("online JSON parsing")
            ) as load:
                controller.start(99)
                controller.controller.result(2)
                load.assert_not_called()
        else:
            controller.start(99)
        sample = copy.deepcopy(raw)
        for page in sample["response"]["ranks"]:
            page["steps"] = [
                dict(
                    step_id=str(i),
                    session_id="session",
                    completed=True,
                    context_count="1",
                    preparation_begin_ns=str(i * 1000000000),
                    forward_enter_ns=str(i * 1000000000 + 1000000),
                    end_ns=str(i * 1000000000 + 2000000),
                )
                for i in (1, 2, 3)
            ]
        controller.consume(dict(event="pressure", raw=sample))
        controller.controller.result(2)
        coverage = Mock()
        coverage.assess.return_value = CoverageAssessment(
            True, 4096, 0.9, 3, False, (), ()
        )
        if remote_points:
            coverage.config.evidence_policy = "tree_then_mock_remote"
            coverage.mock_remote_summary.return_value = {
                "reasons": [],
                "ranks": {
                    str(r): dict(
                        remote_read_payload_bytes=0,
                        remote_write_acknowledged_payload_bytes=512,
                    )
                    for r in (0, 1)
                },
            }
            for ns, ack, *duration in remote_points:
                for page in sample["response"]["ranks"]:
                    page.update(
                        sampled_monotonic_ns=str(ns),
                        collect_duration_ns=str(duration[0] if duration else 100),
                        remote_runtime={"version": 1},
                        remote_write_acknowledged_payload_bytes=str(ack),
                    )
                controller.consume(dict(event="pressure", raw=sample))
        return controller.finish(coverage), root, coverage

    def test_drained_receipt_after_capture_and_overlap_are_separate(self):
        points = [
            (500000000, 0),
            (1500000000, 0),
            (3500000000, 0),
            (4100000000, 0),
            (4500000000, 512),
        ]
        result, _, coverage = self.capture_case(remote_points=points)
        self.assertTrue(result["trace_valid"], result["reasons"])
        self.assertEqual(
            result["pressure"]["mock_remote"]["ranks"]["0"]["write_bytes"], 0
        )
        coverage.mock_remote_summary.assert_not_called()
        ambiguous, _, _ = self.capture_case(
            remote_points=points[:3] + [(4100000000, 512)]
        )
        self.assertFalse(ambiguous["trace_valid"])
        self.assertIn("remote_capture_boundary_ambiguous", ambiguous["reasons"])
        crossing, _, _ = self.capture_case(
            remote_points=points[:2] + [(3500000000, 512, 700000000), (4300000000, 512)]
        )
        self.assertIn("remote_capture_boundary_ambiguous", crossing["reasons"])
        inside, _, _ = self.capture_case(
            remote_points=points[:2] + [(3500000000, 512), (4300000000, 512)]
        )
        self.assertIn("missing_native_remote_write_ack", inside["reasons"])

    def test_confirmed_remote_scopes_survive_uncertainty_in_additional_edge_bytes(self):
        for upper in (512, 1024):
            points = [
                (500000000, 0),
                (1500000000, 0),
                (3500000000, 512),
                (4100000000, upper),
            ]
            result, _, _ = self.capture_case(remote_points=points, write_scopes=True)
            self.assertTrue(result["trace_valid"], result["reasons"])
            rank = result["pressure"]["mock_remote"]["ranks"]["0"]
            self.assertEqual(
                (rank["write_bytes"], rank["possible_write_bytes"]), (512, upper)
            )
            self.assertEqual(
                "write_bytes" in rank["uncertain_byte_counts"], upper != 512
            )
        points = [(500000000, 0), (1500000000, 0), (3500000000, 0), (4100000000, 512)]
        result, _, _ = self.capture_case(remote_points=points, write_scopes=True)
        self.assertTrue(result["trace_valid"], result["reasons"])

    def test_online_wait_only_checks_artifact_readiness(self):
        result, _, _ = self.capture_case(check_online=True)
        self.assertTrue(result["trace_valid"], result)

    def test_rank_identity_and_real_capture_pressure_window_are_independent(self):
        result, _, coverage = self.capture_case()
        self.assertTrue(result["trace_valid"], result)
        self.assertFalse(result["pressure_valid"])
        coverage.assess.assert_called_once_with(11, 14.1)
        self.assertEqual(set(result["ranks"]), {"0", "1"})

    def test_missing_rank_identity_export_and_partial_json_fail_preserving_files(self):
        for fault in ("rank", "identity", "export", "truncated"):
            with self.subTest(fault=fault):
                result, root, _ = self.capture_case(fault)
                self.assertFalse(result["trace_valid"], result)
                self.assertTrue((root / "native_export_0.json").exists())
                self.assertTrue((root / "native_export_1.json").exists())
                if fault == "truncated":
                    self.assertEqual((root / "native_export_1.json").read_text(), "{")

    def test_actual_driver_schema_and_first_gather_boundary(self):
        result = validate_kineto(actual_trace(), steps(), 3)
        self.assertTrue(result["trace_valid"], result)
        self.assertEqual(result["usable_step_ids"], [2, 3])
        self.assertEqual(result["usable_pairs"], 1)
        self.assertEqual(result["exclusions"]["1"], "first_step_gather_omitted")

    def test_missing_correlation_and_flush_boundary(self):
        trace = actual_trace()
        trace["traceEvents"][3]["args"].pop("correlation")
        self.assertIn(
            "missing_cuda_correlation", validate_kineto(trace, steps(), 3)["reasons"]
        )
        trace = actual_trace()
        trace["traceEvents"][-1]["ts"] = 1652691328180.0
        result = validate_kineto(trace, steps(), 3)
        self.assertEqual(result["exclusions"]["3"], "flush_boundary")
        self.assertFalse(result["trace_valid"])

    def test_unknown_schema_and_missing_trace_end_fail_closed(self):
        trace = actual_trace()
        trace["schemaVersion"] = 999
        self.assertFalse(validate_kineto(trace, steps(), 3)["trace_valid"])
        trace = actual_trace()
        trace["traceEvents"].pop()
        self.assertFalse(validate_kineto(trace, steps(), 3)["trace_valid"])


class NsightPressureTest(unittest.TestCase):
    def test_internal_kernel_requires_unique_launch(self):

        faults = (
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL SELECT start+1,end-1,globalPid,99999,deviceId,contextId,streamId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=2;",
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT start+1,end-1,globalTid,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=2;",
            "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=2001 WHERE correlationId=2;",
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT 2000,2001,globalTid,99999,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=2; INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL SELECT start+1,end-1,globalPid,99999,deviceId,contextId,streamId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=2;",
        )
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.sqlite"
                nsys_fixture(path)
                with sqlite3.connect(path) as db:
                    db.executescript(fault)
                result = normalize_nsys(path, {0: steps(), 1: steps()})
                self.assertFalse(result["trace_valid"], result["reasons"])

    def test_nested_launch_and_non_forward_async_kernel(self):

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            nsys_fixture(path)
            with sqlite3.connect(path) as db:
                db.executescript(
                    """
                    CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER AS SELECT start+1 AS start,end-1 AS end,globalTid,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME;
                    UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=start-15;
                    INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT 1500,1520,globalTid,99999,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=2;
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL SELECT 2015,2050,globalPid,99999,deviceId,contextId,streamId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=2;
                """
                )
            result = normalize_nsys(path, {0: steps(), 1: steps()})
            self.assertTrue(result["trace_valid"], result["reasons"])
            extra = [
                e
                for e in result["events"]
                if e.get("correlation") == 99999 and e["kind"] == "kernel"
            ]
            self.assertEqual(len(extra), 2)
            self.assertTrue(all(e.get("association") == "non_forward" for e in extra))


if __name__ == "__main__":
    unittest.main()
