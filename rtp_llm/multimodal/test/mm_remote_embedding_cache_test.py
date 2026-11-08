import concurrent.futures
import json
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalInputsPB
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.kvcm.tensor_object import pack_object, plan_object
from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCache
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.mm_remote_embedding_cache import MMRemoteEmbeddingCache
from rtp_llm.multimodal.multimodal_util import trans_mm_input
from rtp_llm.multimodal.test.multimodal_process_engine_test import (
    FakeEmbeddingLengthInterface,
)
from rtp_llm.multimodal.test.remote_embedding_test_helpers import FakeClient, Store
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.grpc_util import trans_from_tensor


class RemoteCacheTest(unittest.TestCase):
    def remote(self, store, **kwargs):
        options = dict(
            max_object_bytes=65536,
            max_inflight_bytes=1024 * 1024,
            max_pending=8,
            read_timeout_ms=500,
        )
        options.update(kwargs)
        result = MMRemoteEmbeddingCache(store, **options)
        self.addCleanup(result.close)
        return result

    def metric_values(self, reporter, metric):
        return [
            (c.args[1], c.args[2])
            for c in reporter.call_args_list
            if c.args[0] == metric
        ]

    def test_repeated_acquisition_creates_one_client(self):
        store = Store()
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report, patch.object(
            self, "factory", return_value=store, create=True
        ) as factory:
            remote = self.remote(
                None, client_factory=factory, namespace="test-revision"
            )
            for _ in range(5):
                self.assertIsNone(remote.load("missing"))
            factory.assert_called_once()
        creations = self.metric_values(
            report, AccMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_QPS_METRIC
        )
        self.assertEqual(len(creations), 1)
        self.assertEqual(creations[0][1]["status"], "success")
        stages = self.metric_values(report, AccMetrics.VIT_REMOTE_EMB_STAGE_QPS_METRIC)
        self.assertEqual(sum(v for v, t in stages if t["stage"] == "client_acquire"), 5)
        self.assertFalse(any(t["stage"] == "client_init" for _, t in stages))

    def test_local_validation_does_not_retire_healthy_client(self):
        from unittest.mock import Mock

        store = Store()
        factory = Mock(return_value=store)
        remote = self.remote(None, client_factory=factory, namespace="test-revision")
        for invalid in (0, remote.max_object_bytes + 1):
            with patch.object(store, "object_size", return_value=invalid):
                self.assertIsNone(remote.load("invalid"))
            self.assertIsNone(remote.load("missing"))
            self.assertFalse(store.closed)
        factory.assert_called_once()

    def test_deadline_after_initialization_keeps_client_for_next_request(self):
        from unittest.mock import Mock

        entered, release = threading.Event(), threading.Event()
        store = Store()

        def create():
            entered.set()
            release.wait(3)
            return store

        factory = Mock(side_effect=create)
        remote = self.remote(
            None, client_factory=factory, namespace="test-revision", read_timeout_ms=20
        )
        try:
            self.assertIsNone(remote.load("missing"))
            self.assertTrue(entered.is_set())
        finally:
            release.set()
        deadline = time.monotonic() + 3
        while remote.stats()["inflight_tasks"] and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(remote.stats()["inflight_tasks"], 0)
        remote.read_timeout_ms = 500
        self.assertIsNone(remote.load("missing"))
        factory.assert_called_once()
        self.assertFalse(store.closed)

    def test_audit_is_opt_in_and_logging_failure_does_not_break_cache(self):
        from unittest.mock import Mock

        remote = self.remote(Store())
        remote._audit_enabled = False
        buffer = Mock()
        self.assertEqual(remote._audit_payload(buffer), {})
        buffer.numpy.assert_not_called()
        with patch("rtp_llm.multimodal.mm_remote_embedding_cache.logging.info") as log:
            remote.record("cache_admitted")
            log.assert_not_called()
        remote._audit_enabled = True
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.logging.info",
            side_effect=RuntimeError("sink"),
        ):
            remote.record("cache_admitted")
        self.assertEqual(remote.stats()["cache_admitted"], 2)

    def test_real_constructor_failure_is_separate_from_reuse(self):
        from unittest.mock import Mock

        factory = Mock(side_effect=[RuntimeError("create failed"), Store()])
        remote = self.remote(None, client_factory=factory, namespace="test-revision")
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            for _ in range(3):
                self.assertIsNone(remote.load("missing"))
        self.assertEqual(factory.call_count, 2)
        creations = self.metric_values(
            report, AccMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_QPS_METRIC
        )
        self.assertEqual([t["status"] for _, t in creations], ["error", "success"])
        errors = self.metric_values(
            report, AccMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_ERROR_QPS_METRIC
        )
        self.assertEqual([v for v, _ in errors], [1, 0])
        self.assertEqual(
            len(
                self.metric_values(
                    report, GaugeMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_RT_METRIC
                )
            ),
            2,
        )

    def test_metadata_outcomes_do_not_confuse_lookup_with_payload_hit(self):
        payload = pack_object(plan_object((torch.ones(3, 4), None)))
        store = Store({"key": payload})
        remote = self.remote(store)
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            self.assertIsNotNone(remote.load("key"))
            self.assertIsNone(remote.load("missing"))
            for result in (
                -1,
                TimeoutError("metadata timeout"),
                RuntimeError("metadata error"),
            ):
                with patch.object(store, "object_size", side_effect=[result]):
                    self.assertIsNone(remote.load("key"))
        values = self.metric_values(
            report, AccMetrics.VIT_REMOTE_EMB_METADATA_QPS_METRIC
        )
        self.assertEqual(
            [t["status"] for _, t in values],
            ["hit", "miss", "invalid", "timeout", "error"],
        )
        self.assertEqual(remote.stats()["hit"], 1)

    def test_executor_queue_delay_is_observable(self):
        from concurrent.futures import ThreadPoolExecutor

        remote = self.remote(Store())
        entered = [threading.Event(), threading.Event()]
        release = threading.Event()

        def block(index):
            entered[index].set()
            release.wait(3)

        blockers = [remote._executor.submit(block, i) for i in range(2)]
        self.assertTrue(all(e.wait(2) for e in entered))
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            with ThreadPoolExecutor(max_workers=1) as caller:
                pending = caller.submit(remote.load, "missing")
                try:
                    deadline = time.monotonic() + 2
                    while (
                        remote.stats()["inflight_tasks"] != 1
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.001)
                    self.assertEqual(remote.stats()["inflight_tasks"], 1)
                    time.sleep(0.03)
                finally:
                    release.set()
                self.assertIsNone(pending.result(timeout=2))
            for blocker in blockers:
                blocker.result(timeout=2)
        waits = self.metric_values(
            report, GaugeMetrics.VIT_REMOTE_EMB_QUEUE_WAIT_METRIC
        )
        self.assertEqual(len(waits), 1)
        self.assertEqual(waits[0][1], {"operation": "read"})
        self.assertGreaterEqual(waits[0][0], 20)

    def test_remote_read_metrics_separate_hit_miss_and_error(self):
        data = pack_object(plan_object((torch.ones(3, 4), None)))
        store = Store({"key": data})
        remote = self.remote(store)
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            # Neither a bypass nor its absent remote key is a remote read.
            self.assertIsNone(remote.load(None))
            self.assertFalse(report.called)
            self.assertIsNotNone(remote.load("key"))
            self.assertIsNone(remote.load("missing"))
            data[-1] ^= 1  # Corrupt payload is an error, not a miss/hit.
            self.assertIsNone(remote.load("key"))
        self.assertEqual(
            self.metric_values(report, AccMetrics.VIT_REMOTE_EMB_READ_QPS_METRIC),
            [(1, {"status": s}) for s in ("hit", "miss", "error")],
        )
        self.assertEqual(
            self.metric_values(report, AccMetrics.VIT_REMOTE_EMB_READ_ERROR_QPS_METRIC),
            [(0, {}), (0, {}), (1, {})],
        )
        rts = self.metric_values(report, GaugeMetrics.VIT_REMOTE_EMB_READ_RT_METRIC)
        self.assertEqual(
            [t for _, t in rts], [{"status": s} for s in ("hit", "miss", "error")]
        )
        self.assertTrue(all(0 <= v < 1000 for v, _ in rts))
        self.assertEqual(remote.stats()["inflight_tasks"], 0)

    def test_timed_out_read_reports_once_even_when_worker_fails_later(self):
        store = Store({"key": pack_object(plan_object((torch.ones(3, 4), None)))})
        store.read_gate.clear()
        store.fail_read = True
        remote = self.remote(store, read_timeout_ms=50)
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            try:
                self.assertIsNone(remote.load("key"))
                self.assertTrue(store.read_started.is_set())
                self.assertGreater(remote.stats()["inflight_bytes"], 0)
            finally:
                store.read_gate.set()
                remote.close()
        self.assertEqual(
            self.metric_values(report, AccMetrics.VIT_REMOTE_EMB_READ_QPS_METRIC),
            [(1, {"status": "timeout"})],
        )
        self.assertEqual(
            self.metric_values(report, AccMetrics.VIT_REMOTE_EMB_READ_ERROR_QPS_METRIC),
            [(1, {})],
        )
        rt, tags = self.metric_values(
            report, GaugeMetrics.VIT_REMOTE_EMB_READ_RT_METRIC
        )[0]
        self.assertGreaterEqual(rt, 40)  # Milliseconds, not seconds or microseconds.
        self.assertLess(rt, 500)
        self.assertEqual(tags, {"status": "timeout"})
        self.assertEqual(remote.stats()["inflight_tasks"], 0)

    def test_late_metadata_miss_is_not_a_second_caller_outcome(self):
        gate = threading.Event()
        started = threading.Event()
        remote = self.remote(Store(), read_timeout_ms=30)

        def late_miss(*args, **kwargs):
            started.set()
            self.assertTrue(gate.wait(3))
            return None

        with patch.object(remote._client, "object_size", side_effect=late_miss):
            with patch.object(remote, "record", wraps=remote.record) as record:
                try:
                    self.assertIsNone(remote.load("missing", trace_id="late-miss"))
                    self.assertTrue(started.is_set())
                finally:
                    gate.set()
                    remote.close()
        outcomes = [c.args[0] for c in record.call_args_list]
        self.assertEqual(outcomes, ["read_error"])
        self.assertEqual(remote.stats().get("miss", 0), 0)
        self.assertEqual(remote.stats()["inflight_tasks"], 0)

    def test_write_rt_includes_snapshot_queue_and_failure_keeps_trace(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                store = Store()
                store.fail_write = failed
                remote = self.remote(store)
                gate = threading.Event()
                blockers = [remote._executor.submit(gate.wait, 3) for _ in range(2)]
                entry = SimpleNamespace(
                    pool_owners=[],
                    result=(torch.ones(3, 4), None),
                    error=None,
                    original_devices=(torch.device("cpu"), None),
                    generation="queued",
                    remote_cache_context={"key": "key", "trace_id": "queued-write"},
                )
                with patch(
                    "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
                ) as report:
                    with patch.object(remote, "record", wraps=remote.record) as record:
                        try:
                            remote.submit_result(
                                entry.remote_cache_context["key"],
                                entry.result,
                                trace_id=entry.remote_cache_context["trace_id"],
                            )
                            time.sleep(0.06)
                        finally:
                            gate.set()
                            for blocker in blockers:
                                blocker.result(timeout=3)
                            deadline = time.monotonic() + 3
                            while (
                                remote.stats()["inflight_tasks"]
                                and time.monotonic() < deadline
                            ):
                                time.sleep(0.005)
                            remote.close()
                rt, tags = self.metric_values(
                    report, GaugeMetrics.VIT_REMOTE_EMB_WRITE_RT_METRIC
                )[0]
                self.assertGreaterEqual(rt, 50)
                self.assertEqual(tags, {"status": "error" if failed else "success"})
                terminal = [
                    c
                    for c in record.call_args_list
                    if c.args[0] in ("store_success", "write_error")
                ]
                self.assertEqual(len(terminal), 1)
                self.assertEqual(terminal[0].kwargs["trace_id"], "queued-write")
                self.assertEqual(remote.stats()["inflight_bytes"], 0)

    def test_sdk_timeout_types_are_classified_in_read_and_stage_metrics(self):
        for error_type in (TimeoutError, concurrent.futures.TimeoutError):
            for method, stage in (
                ("object_size", "metadata"),
                ("load_one", "transfer"),
            ):
                with self.subTest(error_type=error_type, stage=stage):
                    data = pack_object(plan_object((torch.ones(3, 4), None)))
                    store = Store({"key": data})
                    remote = self.remote(store)
                    with patch.object(
                        store, method, side_effect=error_type("SDK deadline")
                    ):
                        with patch(
                            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
                        ) as report:
                            self.assertIsNone(remote.load("key"))
                            remote.close()
                    self.assertEqual(remote.stats()["read_timeout"], 1)
                    self.assertEqual(remote.stats()["inflight_tasks"], 0)
                    self.assertEqual(remote.stats()["inflight_bytes"], 0)
                    self.assertEqual(
                        self.metric_values(
                            report, AccMetrics.VIT_REMOTE_EMB_READ_QPS_METRIC
                        ),
                        [(1, {"status": "timeout"})],
                    )
                    self.assertEqual(
                        self.metric_values(
                            report, AccMetrics.VIT_REMOTE_EMB_READ_ERROR_QPS_METRIC
                        ),
                        [(1, {})],
                    )
                    self.assertIn(
                        (1, {"operation": "read", "stage": stage, "status": "timeout"}),
                        self.metric_values(
                            report, AccMetrics.VIT_REMOTE_EMB_STAGE_QPS_METRIC
                        ),
                    )
                    self.assertEqual(
                        self.metric_values(
                            report,
                            AccMetrics.VIT_REMOTE_EMB_PAYLOAD_BYTES_PER_SECOND_METRIC,
                        ),
                        [(0, {"operation": "read"})],
                    )

    def test_sdk_timeout_types_are_classified_in_write_and_stage_metrics(self):
        for error_type in (TimeoutError, concurrent.futures.TimeoutError):
            with self.subTest(error_type=error_type):
                store = Store()
                remote = self.remote(store)
                entry = SimpleNamespace(
                    pool_owners=[],
                    result=(torch.ones(3, 4), None),
                    error=None,
                    original_devices=(torch.device("cpu"), None),
                    generation=1,
                    remote_cache_context={"key": "key", "trace_id": None},
                )
                with patch.object(
                    store, "save_one", side_effect=error_type("SDK deadline")
                ):
                    with patch(
                        "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
                    ) as report:
                        remote.submit_result(
                            entry.remote_cache_context["key"],
                            entry.result,
                            trace_id=entry.remote_cache_context["trace_id"],
                        )
                        deadline = time.monotonic() + 2
                        while (
                            remote.stats()["inflight_tasks"]
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.01)
                        self.assertEqual(remote.stats()["inflight_tasks"], 0)
                        remote.close()
                self.assertEqual(remote.stats()["inflight_tasks"], 0)
                self.assertEqual(remote.stats()["inflight_bytes"], 0)
                self.assertEqual(
                    self.metric_values(
                        report, AccMetrics.VIT_REMOTE_EMB_WRITE_QPS_METRIC
                    ),
                    [(1, {"status": "timeout"})],
                )
                self.assertEqual(
                    self.metric_values(
                        report, AccMetrics.VIT_REMOTE_EMB_WRITE_ERROR_QPS_METRIC
                    ),
                    [(1, {})],
                )
                self.assertIn(
                    (
                        1,
                        {
                            "operation": "write",
                            "stage": "transfer",
                            "status": "timeout",
                        },
                    ),
                    self.metric_values(
                        report, AccMetrics.VIT_REMOTE_EMB_STAGE_QPS_METRIC
                    ),
                )
                self.assertEqual(
                    self.metric_values(
                        report,
                        AccMetrics.VIT_REMOTE_EMB_PAYLOAD_BYTES_PER_SECOND_METRIC,
                    ),
                    [(0, {"operation": "write"})],
                )

    def test_admission_skips_do_not_increment_error_qps(self):
        data = pack_object(plan_object((torch.ones(3, 4), None)))
        remote = self.remote(Store({"key": data}), max_inflight_bytes=1, max_pending=1)
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            self.assertIsNone(remote.load("key"))  # Metadata lookup, no payload budget.
            self.assertTrue(remote._reserve(0))
            try:
                self.assertIsNone(remote.load("key"))  # No admitted IO at all.
            finally:
                remote._release(0)
        self.assertEqual(
            self.metric_values(report, AccMetrics.VIT_REMOTE_EMB_READ_QPS_METRIC),
            [(1, {"status": "admission_skip"})],
        )
        self.assertEqual(
            self.metric_values(report, AccMetrics.VIT_REMOTE_EMB_READ_ERROR_QPS_METRIC),
            [(0, {})],
        )
        self.assertEqual(
            self.metric_values(report, AccMetrics.VIT_REMOTE_EMB_SKIP_QPS_METRIC),
            [
                (1, {"operation": "read", "reason": "payload_budget"}),
                (1, {"operation": "read", "reason": "admission"}),
            ],
        )

    def test_metric_reporting_failure_preserves_result_and_releases_budget(self):
        data = pack_object(plan_object((torch.ones(3, 4), None)))
        remote = self.remote(Store({"key": data}))
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report",
            side_effect=RuntimeError("reporter unavailable"),
        ):
            self.assertIsNotNone(remote.load("key"))
        self.assertEqual(remote.stats()["inflight_tasks"], 0)
        self.assertEqual(remote.stats()["inflight_bytes"], 0)

    def test_remote_write_metrics_follow_completed_background_operations(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                store = Store()
                store.fail_write = failed
                remote = self.remote(store)
                entry = SimpleNamespace(
                    pool_owners=[],
                    result=(torch.ones(3, 4), None),
                    error=None,
                    original_devices=(torch.device("cpu"), None),
                    generation=1,
                    remote_cache_context={"key": "key", "trace_id": None},
                )
                with patch(
                    "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
                ) as report:
                    remote.submit_result(
                        entry.remote_cache_context["key"],
                        entry.result,
                        trace_id=entry.remote_cache_context["trace_id"],
                    )
                    deadline = time.monotonic() + 2
                    while (
                        remote.stats()["inflight_tasks"] and time.monotonic() < deadline
                    ):
                        time.sleep(0.01)
                    self.assertEqual(remote.stats()["inflight_tasks"], 0)
                    remote.close()
                status = "error" if failed else "success"
                self.assertEqual(
                    self.metric_values(
                        report, AccMetrics.VIT_REMOTE_EMB_WRITE_QPS_METRIC
                    ),
                    [(1, {"status": status})],
                )
                self.assertEqual(
                    self.metric_values(
                        report, AccMetrics.VIT_REMOTE_EMB_WRITE_ERROR_QPS_METRIC
                    ),
                    [(int(failed), {})],
                )
                self.assertEqual(
                    len(
                        self.metric_values(
                            report, GaugeMetrics.VIT_REMOTE_EMB_WRITE_RT_METRIC
                        )
                    ),
                    1,
                )
                nbytes = 0 if failed else store.objects["key"].numel()
                tags = {"operation": "write"}
                self.assertEqual(
                    self.metric_values(
                        report,
                        AccMetrics.VIT_REMOTE_EMB_SDK_OBJECT_BYTES_PER_SECOND_METRIC,
                    ),
                    [(nbytes, tags)],
                )
                self.assertEqual(
                    self.metric_values(
                        report,
                        AccMetrics.VIT_REMOTE_EMB_PAYLOAD_BYTES_PER_SECOND_METRIC,
                    ),
                    [(nbytes, tags)],
                )
                self.assertEqual(remote.stats()["inflight_tasks"], 0)

    def test_sdk_and_accepted_bytes_distinguish_corruption_and_miss(self):
        data = pack_object(plan_object((torch.ones(3, 4), None)))
        remote = self.remote(Store({"key": data}))
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            self.assertIsNotNone(remote.load("key"))
            self.assertIsNone(remote.load("missing"))
            data[-1] ^= 1
            self.assertIsNone(remote.load("key"))
        tags = {"operation": "read"}
        self.assertEqual(
            self.metric_values(
                report, AccMetrics.VIT_REMOTE_EMB_SDK_OBJECT_BYTES_PER_SECOND_METRIC
            ),
            [(data.numel(), tags), (data.numel(), tags)],
        )
        self.assertEqual(
            self.metric_values(
                report, AccMetrics.VIT_REMOTE_EMB_PAYLOAD_BYTES_PER_SECOND_METRIC
            ),
            [(data.numel(), tags), (0, tags), (0, tags)],
        )
        self.assertEqual(
            self.metric_values(
                report, GaugeMetrics.VIT_REMOTE_EMB_PAYLOAD_BYTES_METRIC
            ),
            [(data.numel(), tags)],
        )
        errors = self.metric_values(
            report, AccMetrics.VIT_REMOTE_EMB_STAGE_ERROR_QPS_METRIC
        )
        self.assertEqual(
            [t for v, t in errors if v], [{"operation": "read", "stage": "restore"}]
        )
        self.assertGreater(
            data.numel(), 3 * 4 * 4
        )  # Serialized header/alignment count too.

    def test_late_success_counts_sdk_bytes_but_not_accepted_payload(self):
        data = pack_object(plan_object((torch.ones(3, 4), None)))
        store = Store({"key": data})
        store.read_gate.clear()
        remote = self.remote(store, read_timeout_ms=30)
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            try:
                self.assertIsNone(remote.load("key"))
                self.assertTrue(store.read_started.is_set())
                self.assertFalse(
                    self.metric_values(
                        report, GaugeMetrics.VIT_REMOTE_EMB_SDK_OBJECT_BYTES_METRIC
                    )
                )
            finally:
                store.read_gate.set()
                remote.close()
        tags = {"operation": "read"}
        self.assertEqual(
            self.metric_values(report, AccMetrics.VIT_REMOTE_EMB_READ_ERROR_QPS_METRIC),
            [(1, {})],
        )
        self.assertEqual(
            self.metric_values(
                report, AccMetrics.VIT_REMOTE_EMB_PAYLOAD_BYTES_PER_SECOND_METRIC
            ),
            [(0, tags)],
        )
        self.assertEqual(
            self.metric_values(
                report, AccMetrics.VIT_REMOTE_EMB_SDK_OBJECT_BYTES_PER_SECOND_METRIC
            ),
            [(data.numel(), tags)],
        )
        self.assertFalse(
            self.metric_values(report, GaugeMetrics.VIT_REMOTE_EMB_PAYLOAD_BYTES_METRIC)
        )
        self.assertEqual(remote.stats()["inflight_bytes"], 0)

    def test_partial_transfer_failure_does_not_report_successful_bytes(self):
        data = pack_object(plan_object((torch.ones(3, 4), None)))
        store = FakeClient({"key": data})
        store.fail_load = True
        remote = self.remote(store)
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            self.assertIsNone(remote.load("key"))
        self.assertFalse(
            self.metric_values(
                report, GaugeMetrics.VIT_REMOTE_EMB_SDK_OBJECT_BYTES_METRIC
            )
        )
        self.assertEqual(
            self.metric_values(
                report, AccMetrics.VIT_REMOTE_EMB_SDK_OBJECT_BYTES_PER_SECOND_METRIC
            ),
            [(0, {"operation": "read"})],
        )
        errors = self.metric_values(
            report, AccMetrics.VIT_REMOTE_EMB_STAGE_ERROR_QPS_METRIC
        )
        self.assertEqual(
            [t for v, t in errors if v], [{"operation": "read", "stage": "transfer"}]
        )

    def test_sdk_metric_units_use_wall_time_and_mib(self):
        remote = self.remote(Store())
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            with patch(
                "rtp_llm.multimodal.mm_remote_embedding_cache.time.perf_counter",
                side_effect=[10, 10.25],
            ):
                with remote._observe_stage("write", "transfer", 2 * 1024 * 1024):
                    pass
        self.assertEqual(
            self.metric_values(
                report, GaugeMetrics.VIT_REMOTE_EMB_SDK_EFFECTIVE_SPEED_METRIC
            ),
            [(8, {"operation": "write"})],
        )
        self.assertEqual(
            self.metric_values(report, GaugeMetrics.VIT_REMOTE_EMB_STAGE_RT_METRIC),
            [(250, {"operation": "write", "stage": "transfer", "status": "success"})],
        )

    def test_result_shared_with_second_worker_without_local_cache(self):
        store = Store()
        writer = self.remote(store)
        result = (
            torch.arange(8, dtype=torch.float32).reshape(2, 4),
            torch.arange(4).reshape(2, 2),
            [torch.tensor([7], dtype=torch.int32), torch.tensor([0.5])],
        )
        writer.submit_result("url_config_a", result, trace_id="writer")
        self.assertTrue(store.saved.wait(2))
        reader = self.remote(Store(store.objects))
        loaded = reader.load("url_config_a")
        self.assertIsInstance(loaded, tuple)
        self.assertIsInstance(loaded[2], list)
        for actual, expected in [
            (loaded[0], result[0]),
            (loaded[1], result[1]),
            (loaded[2][0], result[2][0]),
            (loaded[2][1], result[2][1]),
        ]:
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected))
        self.assertIsNone(reader.load("url_config_missing"))
        self.assertEqual(reader.stats()["hit"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gpu_remote_gpu_survives_source_mutation(self):
        store = Store()
        store.write_gate.clear()
        writer = self.remote(store)
        source = (
            torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4),
            torch.arange(4, dtype=torch.int64, device="cuda"),
            [torch.arange(8, dtype=torch.float32, device="cuda")],
        )
        expected = [source[0].cpu(), source[1].cpu(), source[2][0].cpu()]
        writer.submit_result("existing_key_0", source, trace_id="writer")
        # The original producer tensors may be reused while the owned write waits.
        for tensor in (source[0], source[1], source[2][0]):
            tensor.zero_()
        store.write_gate.set()
        self.assertTrue(store.saved.wait(2))
        writer.close()
        reader = self.remote(Store(store.objects), cuda_device=torch.device("cuda", 0))
        loaded = reader.load("existing_key_0")
        self.assertIsNotNone(loaded)
        for actual, value in zip([loaded[0], loaded[1], loaded[2][0]], expected):
            self.assertEqual(actual.device, torch.device("cuda", 0))
            self.assertEqual(actual.dtype, value.dtype)
            self.assertTrue(torch.equal(actual.cpu(), value))

    def test_owned_snapshot_and_pack_failure_release_admission(self):
        store = Store()
        store.write_gate.clear()
        remote = self.remote(store)
        value = torch.arange(12).reshape(3, 4)
        remote.submit_result("snapshot", (value, None))
        value.zero_()
        store.write_gate.set()
        self.assertTrue(store.saved.wait(2))
        self.assertTrue(
            torch.equal(remote.load("snapshot")[0], torch.arange(12).reshape(3, 4))
        )
        remote.close()
        self.assertEqual(remote.stats()["inflight_tasks"], 0)
        other = self.remote(Store())
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.pack_object",
            side_effect=RuntimeError("copy failed"),
        ):
            other.submit_result("failed", (value, None))
        self.assertEqual(other.stats()["inflight_bytes"], 0)
        self.assertEqual(other.stats()["inflight_tasks"], 0)
        self.assertFalse(other._pending_writes)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_deferred_gpu_snapshot_waits_for_producer_and_is_submitted_once(self):
        from rtp_llm.multimodal.mm_work_item import MMWorkItem

        store = Store()
        writer = self.remote(store, cuda_device=torch.device("cuda", 0))
        item = MultimodalInput(
            "data:image/png;base64,AQID", 1, torch.empty(0), MMPreprocessConfig()
        )
        work = MMWorkItem([item], defer_cache_complete=True, defer_feature_hashes=True)
        work.remote_cache, work.remote_cache_key = writer, "deferred"
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            value = torch.empty((3, 4), device="cuda")
            torch.cuda._sleep(10000000)
            value.fill_(37)
            result = (value, None)
            work.complete_cache(result)
        self.assertFalse(store.saved.is_set())
        work.complete_cache(result, force=True)
        self.assertTrue(store.saved.wait(2))
        work.complete_cache(result, force=True)
        loaded = writer.load("deferred")
        self.assertTrue(torch.equal(loaded[0], torch.full((3, 4), 37.0, device="cuda")))
        self.assertEqual(writer.stats()["write_success"], 1)

    def test_key_bypass_has_low_cardinality_skip_metric(self):
        remote = self.remote(Store())
        item = SimpleNamespace(
            url="data:image/png;base64,AQID", tensor=torch.ones(1), mm_type=1
        )
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            self.assertIsNone(remote.key([item]))
        self.assertEqual(
            self.metric_values(report, AccMetrics.VIT_REMOTE_EMB_SKIP_QPS_METRIC),
            [(1, {"operation": "key", "reason": "nonempty_tensor"})],
        )

    def test_timeout_retains_budget_until_actual_io_completion(self):
        store = Store({"key": pack_object(plan_object((torch.ones(3, 4), None)))})
        store.read_gate.clear()
        remote = self.remote(store, read_timeout_ms=15)
        self.assertIsNone(remote.load("key"))
        self.assertTrue(store.read_started.is_set())
        self.assertGreater(remote.stats()["inflight_bytes"], 0)
        store.read_gate.set()
        remote.close()
        self.assertEqual(remote.stats()["inflight_bytes"], 0)
        self.assertEqual(remote.stats()["inflight_tasks"], 0)

    def test_failed_or_corrupt_read_is_not_published(self):
        data = pack_object(plan_object((torch.ones(3, 4), None)))
        store = Store({"key": data})
        remote = self.remote(store)
        store.fail_read = True
        self.assertIsNone(remote.load("key"))
        store.fail_read = False
        data[-1] ^= 1
        self.assertIsNone(remote.load("key"))
        self.assertEqual(remote.stats()["read_error"], 2)
        self.assertEqual(remote.stats()["inflight_bytes"], 0)

    def test_write_budget_dedup_and_unknown_failure_do_not_remove(self):
        store = Store()
        store.write_gate.clear()
        remote = self.remote(store, max_pending=1)
        entry = SimpleNamespace(
            pool_owners=[],
            result=(torch.ones(2, 4), None),
            error=None,
            original_devices=(torch.device("cpu"), None),
            generation="test-generation",
        )
        remote.bind_entry(entry, "same", "writer")
        remote.submit_result(
            entry.remote_cache_context["key"],
            entry.result,
            trace_id=entry.remote_cache_context["trace_id"],
        )
        remote.submit_result(
            entry.remote_cache_context["key"],
            entry.result,
            trace_id=entry.remote_cache_context["trace_id"],
        )
        other = SimpleNamespace(**vars(entry))
        remote.bind_entry(other, "different", "writer")
        remote.submit_result(
            other.remote_cache_context["key"],
            other.result,
            trace_id=other.remote_cache_context["trace_id"],
        )
        self.assertEqual(remote.stats()["inflight_tasks"], 1)
        self.assertEqual(remote.stats()["admission_skip"], 1)
        store.fail_write = True
        store.write_gate.set()
        remote.close()
        self.assertEqual(remote.stats()["write_error"], 1)
        self.assertEqual(remote.stats()["inflight_bytes"], 0)
        self.assertEqual(store.objects, {})

    def test_clear_does_not_upload_or_delete_shared_values(self):
        store = Store()
        remote = self.remote(store)
        cache = MMEmbeddingCache(0, 256)
        _, entry = cache.try_acquire("key")
        remote.bind_entry(entry, "key", "writer")
        cache.complete("key", entry, (torch.ones(2, 4), None))
        cache.clear()
        self.assertEqual(remote.stats()["inflight_tasks"], 0)
        self.assertEqual(store.objects, {})

    def test_oversized_read_and_write_bypass(self):
        store = Store({"key": torch.empty(1024, dtype=torch.uint8)})
        remote = self.remote(store, max_object_bytes=128)
        self.assertIsNone(remote.load("key"))
        self.assertFalse(store.read_started.is_set())
        remote.submit_result("key", (torch.ones(1024), None))
        self.assertEqual(remote.stats()["inflight_tasks"], 0)


class EngineRemoteCacheTest(unittest.TestCase):
    def engine(self, store, *, cpu_bytes=0, request_scoped=False):
        config = VitConfig()
        config.use_local_preprocess = True
        config.mm_cache_gpu_max_bytes = 0
        config.mm_cache_cpu_max_bytes = cpu_bytes
        config.mm_hash_key_cache_max_bytes = 4096
        config.mm_remote_cache_enable = True
        config.mm_remote_cache_namespace = "test-model-weights-preprocessor"
        config.mm_remote_cache_read_timeout_ms = 500
        model = ModelConfig()
        model.mm_related_params.preprocess_batch_size = 1
        model.ckpt_path = "/nonexistent/test-model"
        interface = FakeEmbeddingLengthInterface()
        interface.inspection_verdict_is_request_scoped = request_scoped
        factory = patch(
            "rtp_llm.multimodal.kvcm.RtpKvMetaObjectClient.from_env", return_value=store
        )
        factory.start()
        self.addCleanup(factory.stop)
        engine = MMProcessEngine(
            interface, model, config, ProfilingDebugLoggingConfig()
        )
        self.addCleanup(engine.stop)
        return engine

    @staticmethod
    def input(url):
        config = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000)
        return MultimodalInput(url, 1, torch.empty(0), config)

    def test_followers_of_remote_lookup_respect_producer_inspection_override(self):
        from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
        from rtp_llm.multimodal.test.multimodal_process_engine_test import (
            _StubGreenNetProvider,
        )

        item = self.input("data:image/png;base64,AQID")
        value = (torch.ones(2, 4), None)
        for producer_checks in (True, False):
            with self.subTest(producer_checks=producer_checks):
                store = Store()
                store.read_gate.clear()
                engine = self.engine(store, cpu_bytes=1024)
                key = engine._remote_embedding_cache.key([item])
                store.objects[key] = pack_object(plan_object(value))
                provider = _StubGreenNetProvider(GreenNetVerdict(True))
                engine._inspection._greennet_provider = provider
                engine.async_submit(
                    [item],
                    request_id=1,
                    user_id="producer",
                    input_inspection_enabled=producer_checks,
                )
                self.assertTrue(store.read_started.wait(2))
                # Release the producer only after the follower has joined the same lookup.
                claim = engine._claim_and_submit_async

                def join(*args, **kwargs):
                    entries = claim(*args, **kwargs)
                    store.read_gate.set()
                    return entries

                with patch.object(engine, "_claim_and_submit_async", side_effect=join):
                    engine.get_embedding_result(
                        [item],
                        request_id=2,
                        user_id="follower",
                        input_inspection_enabled=True,
                    )
                self.assertEqual(
                    provider.user_ids, ["producer" if producer_checks else "follower"]
                )
                self.assertEqual(engine._remote_embedding_cache.stats()["hit"], 1)

    def test_request_scoped_writer_or_reader_requires_fresh_verdict(self):
        from rtp_llm.config.exceptions import FtRuntimeException
        from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
        from rtp_llm.multimodal.test.multimodal_process_engine_test import (
            _StubGreenNetProvider,
        )

        item = self.input("data:image/png;base64,AQID")
        for scoped_writer, scoped_reader in ((True, False), (False, True)):
            with self.subTest(scoped_writer=scoped_writer, scoped_reader=scoped_reader):
                store = Store()
                writer = self.engine(store, request_scoped=scoped_writer)
                writer._inspection._greennet_provider = _StubGreenNetProvider(
                    GreenNetVerdict(True)
                )
                writer._embed_with_greennet_sync([item])
                self.assertTrue(store.saved.wait(2))
                reader = self.engine(Store(store.objects), request_scoped=scoped_reader)
                provider = _StubGreenNetProvider(GreenNetVerdict(False, 2, "blocked"))
                reader._inspection._greennet_provider = provider
                with self.assertRaisesRegex(FtRuntimeException, "blocked"):
                    reader.get_embedding_result([item], user_id="new-user")
                self.assertEqual(provider.user_ids, ["new-user"])
                self.assertIsNone(reader._embedding_cache.peek(item.cache_key()))

    def test_sync_hit_preserves_extras_and_skips_preprocess_and_forward(self):
        mm_input = MultimodalInput(
            "data:image/jpeg;base64,AQID",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )
        result = (
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
            torch.arange(6).reshape(3, 2),
            [torch.tensor([10, 20], dtype=torch.int32)],
        )
        store = Store()
        engine = self.engine(store)
        key = engine._remote_embedding_cache.key([mm_input])
        store.objects[key] = pack_object(plan_object(result))
        with patch.object(
            engine.preprocess_executor,
            "preprocess_func",
            side_effect=AssertionError("preprocess called"),
        ), patch.object(
            engine._scheduler,
            "submit_and_wait",
            side_effect=AssertionError("forward called"),
        ):
            response = engine.mm_embedding_impl([mm_input])
        self.assertTrue(torch.equal(response.embeddings[0], result[0]))
        self.assertTrue(torch.equal(response.position_ids[0], result[1]))
        self.assertTrue(torch.equal(response.extra_input[0], result[2][0]))
        self.assertEqual(engine._remote_embedding_cache.stats()["hit"], 1)

    def test_async_remote_failure_falls_back_to_normal_compute(self):
        store = Store()
        engine = self.engine(store)
        mm_input = MultimodalInput(
            "data:image/jpeg;base64,AQID",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )
        response = engine.get_embedding_result([mm_input], request_id=101)
        self.assertEqual(response[0].embeddings[0].shape, (len(mm_input.url), 4))
        self.assertEqual(engine._remote_embedding_cache.stats()["miss"], 1)

    def test_encoder_writes_without_cpu_cache_and_next_request_hits_remote(self):
        store = Store()
        engine = self.engine(store)
        item = MultimodalInput(
            "data:image/jpeg;base64,AQID",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )
        first = engine.mm_embedding_impl([item])
        self.assertTrue(store.saved.wait(2))
        self.assertIsNone(engine._embedding_cache.peek(item.cache_key()))
        with patch.object(
            engine._scheduler,
            "submit_and_wait",
            side_effect=AssertionError("encoder called on hit"),
        ):
            second = engine.mm_embedding_impl([item])
        self.assertTrue(torch.equal(first.embeddings[0], second.embeddings[0]))
        self.assertEqual(engine._remote_embedding_cache.stats()["write_success"], 1)
        self.assertEqual(engine._remote_embedding_cache.stats()["hit"], 1)

    def test_remote_hit_does_not_bypass_greennet_rejection(self):
        from rtp_llm.config.exceptions import FtRuntimeException
        from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
        from rtp_llm.multimodal.test.multimodal_process_engine_test import (
            _StubGreenNetProvider,
        )

        mm_input = MultimodalInput(
            "data:image/jpeg;base64,AQIE",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )
        store = Store()
        engine = self.engine(store)
        key = engine._remote_embedding_cache.key([mm_input])
        store.objects[key] = pack_object(plan_object((torch.ones(3, 4), None)))
        engine._inspection._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=False, code=2, message="blocked"), delay=0.05
        )
        with self.assertRaises(FtRuntimeException):
            engine.get_embedding_result([mm_input], request_id=202, user_id="test-user")
        self.assertIsNone(engine._embedding_cache.peek(mm_input.cache_key()))
        self.assertFalse(engine._hash_key_cache.contains(mm_input.cache_key()))
        self.assertFalse(store.saved.is_set())

    def test_disabled_never_constructs_optional_client(self):
        config = VitConfig()
        config.use_local_preprocess = True
        config.mm_cache_gpu_max_bytes = 0
        config.mm_cache_cpu_max_bytes = 0
        config.mm_hash_key_cache_max_bytes = 0
        with patch("rtp_llm.multimodal.kvcm.RtpKvMetaObjectClient.from_env") as create:
            engine = MMProcessEngine(
                FakeEmbeddingLengthInterface(),
                ModelConfig(),
                config,
                ProfilingDebugLoggingConfig(),
            )
            self.addCleanup(engine.stop)
            self.assertIsNone(engine._remote_embedding_cache)
            create.assert_not_called()


class RemoteIdentityAndRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.storage = {}
        self.clients = []

        def factory():
            client = FakeClient(self.storage)
            self.clients.append(client)
            return client

        self.factory = factory
        self.cache = self.make_cache()
        self.value = (
            torch.arange(24, dtype=torch.bfloat16).reshape(3, 8),
            torch.arange(9, dtype=torch.int64).reshape(3, 3),
            torch.arange(72, dtype=torch.bfloat16),
        )

    def make_cache(self, namespace="model-weights-processor-v1", max_bytes=65536):
        value = MMRemoteEmbeddingCache(
            None,
            client_factory=self.factory,
            namespace=namespace,
            model_identity={"dtype": "torch.bfloat16"},
            max_object_bytes=max_bytes,
            max_inflight_bytes=1024 * 1024,
            read_timeout_ms=500,
        )
        self.addCleanup(value.close)
        return value

    def item(self, url="data:image/jpeg;base64,AQID"):
        return SimpleNamespace(
            url=url,
            mm_type=1,
            tensor=torch.empty(0),
            mm_preprocess_config=SimpleNamespace(
                width=-1,
                height=-1,
                min_pixels=256,
                max_pixels=1024,
                fps=-1,
                min_frames=-1,
                max_frames=-1,
                crop_positions=[],
            ),
        )

    def save(self, remote, key="key", value=None):
        value = self.value if value is None else value
        entry = SimpleNamespace(
            pool_owners=[],
            result=value,
            error=None,
            original_devices=tuple(
                t.device if isinstance(t, torch.Tensor) else None for t in value
            ),
            generation="unit-generation",
        )
        remote.bind_entry(entry, key, "origin")
        remote.submit_result(
            entry.remote_cache_context["key"],
            entry.result,
            trace_id=entry.remote_cache_context["trace_id"],
        )
        deadline = time.monotonic() + 3
        while remote.stats()["inflight_tasks"] and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(remote.stats()["inflight_tasks"], 0)

    def test_complete_result_roundtrip_and_fresh_client(self):
        key = self.cache.key([self.item()])
        self.assertIsNone(self.cache.load(key, trace_id="writer"))
        self.cache._audit_enabled = True
        with patch.dict(os.environ, {"MM_EMB_REMOTE_AUDIT": "1"}), self.assertLogs(
            level="INFO"
        ) as logs:
            self.save(self.cache, key)
            reader = self.make_cache()
            actual = reader.load(key, trace_id="reader")
        events = [
            json.loads(line.split("MM_EMB_REMOTE_JSON ")[1])
            for line in logs.output
            if "MM_EMB_REMOTE_JSON " in line
        ]
        stored = next(e for e in events if e["event"] == "store_success")
        loaded = next(e for e in events if e["event"] == "load_hit")
        self.assertEqual(stored["payload_sha256"], loaded["payload_sha256"])
        self.assertEqual(stored["write_source"], "encoder_result")
        for a, expected in zip(actual, self.value):
            self.assertEqual(a.dtype, expected.dtype)
            self.assertTrue(torch.equal(a, expected))

    def test_retired_client_drains_concurrent_write_without_blocking_new_reader(self):
        blob = pack_object(plan_object(self.value))
        self.storage["existing"] = blob
        started, release, closing = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        class DrainingClient(FakeClient):
            def save_one(client, key, value, **kwargs):
                started.set()
                release.wait(3)
                super(DrainingClient, client).save_one(key, value, **kwargs)

            def load_one(client, key, output, **kwargs):
                output.fill_(9)
                raise TimeoutError("native partial transfer")

            def close(client):
                closing.set()
                release.wait(3)
                client.closed = True

        failed = DrainingClient(self.storage)
        self.cache._client = failed
        self.cache.read_timeout_ms = 20
        entry = SimpleNamespace(
            pool_owners=[],
            result=self.value,
            error=None,
            original_devices=tuple(t.device for t in self.value),
            generation="concurrent",
        )
        self.cache.bind_entry(entry, "pending-write", "writer")
        self.cache.submit_result(
            entry.remote_cache_context["key"],
            entry.result,
            trace_id=entry.remote_cache_context["trace_id"],
        )
        self.assertTrue(started.wait(1))
        try:
            self.assertIsNone(self.cache.load("existing"))
            self.assertFalse(closing.is_set())
            self.assertFalse(failed.closed)
            # The failed read releases its IO worker, while the old client
            # stays alive for the writer. New requests use a fresh client.
            self.cache.read_timeout_ms = 500
            actual = self.cache.load("existing")
            self.assertTrue(torch.equal(actual[0], self.value[0]))
        finally:
            release.set()
            self.cache.close()
        self.assertTrue(closing.is_set())
        self.assertTrue(failed.closed)
        self.assertEqual(self.cache.stats()["inflight_tasks"], 0)
        self.assertEqual(self.cache.stats()["inflight_bytes"], 0)

    def test_metadata_failure_and_blocking_close_remain_bounded_off_request_thread(
        self,
    ):
        release = threading.Event()
        retired = []

        class SlowCleanup(FakeClient):
            def object_size(client, key, **kwargs):
                raise TimeoutError("metadata deadline")

            def close(client):
                retired.append(client)
                release.wait(3)
                client.closed = True

        self.cache._factory = lambda: SlowCleanup(self.storage)
        self.cache.read_timeout_ms = 20
        self.cache.max_pending = 2
        try:
            started = time.monotonic()
            self.assertIsNone(self.cache.load("one"))
            self.assertIsNone(self.cache.load("two"))
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(len(retired), 2)
            self.assertFalse(any(client.closed for client in retired))
            self.assertEqual(self.cache.stats()["inflight_tasks"], 2)
            self.assertIsNone(self.cache.load("not-admitted"))
            self.assertEqual(self.cache.stats()["admission_skip"], 1)
            self.assertEqual(len(retired), 2)
        finally:
            release.set()
            self.cache.close()
        self.assertTrue(all(client.closed for client in retired))
        self.assertEqual(self.cache.stats()["inflight_tasks"], 0)
        self.assertEqual(self.cache.stats()["inflight_bytes"], 0)
        self.assertEqual(self.cache._client_users, {})

    def test_slow_client_factory_is_also_inside_read_deadline(self):
        started, release = threading.Event(), threading.Event()

        def factory():
            started.set()
            release.wait(3)
            return FakeClient(self.storage)

        self.cache._factory = factory
        self.cache.read_timeout_ms = 20
        try:
            before = time.monotonic()
            self.assertIsNone(self.cache.load("missing"))
            self.assertLess(time.monotonic() - before, 0.5)
            self.assertTrue(started.is_set())
            self.assertEqual(self.cache.stats()["inflight_tasks"], 1)
        finally:
            release.set()
            self.cache.close()
        self.assertEqual(self.cache.stats()["inflight_tasks"], 0)
        self.assertEqual(self.cache._client_users, {})

    def test_partial_native_failure_discards_buffer_and_recreates_client(self):
        self.save(self.cache)
        failed = self.clients[-1]
        failed.fail_load = True
        self.assertIsNone(self.cache.load("key"))
        self.assertTrue(failed.closed)
        restored = self.cache.load("key")
        self.assertEqual(len(self.clients), 2)
        self.assertTrue(torch.equal(restored[0], self.value[0]))

    def test_failed_result_write_has_no_implicit_retry(self):
        self.cache.load("missing")
        self.clients[-1].fail_save = True
        self.save(self.cache)
        self.assertNotIn("key", self.storage)
        self.assertEqual(self.cache.stats()["write_error"], 1)
        self.assertEqual(len(self.clients), 1)
        self.save(self.cache)
        self.assertIsNotNone(self.cache.load("key"))

    def test_oversize_admission_before_native_client(self):
        tiny = self.make_cache(max_bytes=20)
        self.save(tiny)
        self.assertFalse(self.clients)
        self.assertEqual(tiny.stats()["encode_skip"], 1)

    def test_missing_remote_key_cannot_upload(self):
        self.cache.submit_result(None, self.value)
        self.assertEqual(self.cache.stats()["inflight_tasks"], 0)
        self.assertFalse(self.clients)

    def test_benchmark_logger_failure_preserves_io_and_releases_resources(self):
        def log(marker, *args, **kwargs):
            if marker.startswith("MM_EMB_BENCHMARK_JSON"):
                raise RuntimeError("diagnostic sink unavailable")

        with patch.dict("os.environ", {"MM_EMB_BENCHMARK_TIMING": "1"}), patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.logging.info", side_effect=log
        ):
            self.save(self.cache)
            value = self.cache.load("key")
        self.assertTrue(torch.equal(value[0], self.value[0]))
        self.assertEqual(self.cache.stats()["inflight_tasks"], 0)
        self.assertEqual(self.cache.stats()["inflight_bytes"], 0)
        self.assertEqual(self.cache._client_users, {})

    def test_immutable_key_namespace_and_all_preprocessing_fields(self):
        item = self.item()
        key = self.cache.key([item])
        self.assertEqual(key, self.make_cache().key([self.item()]))
        self.assertNotEqual(key, self.make_cache(namespace="new-weights").key([item]))
        item.mm_preprocess_config.crop_positions = [0.123456789]
        self.assertNotEqual(key, self.cache.key([item]))
        first_crop = self.cache.key([item])
        item.mm_preprocess_config.crop_positions = [0.123456790]
        self.assertNotEqual(first_crop, self.cache.key([item]))
        item.url = "data:image/jpeg;base64,AQIE"
        self.assertNotEqual(first_crop, self.cache.key([item]))

    def test_mutable_urls_tensor_inputs_and_batched_groups_bypass(self):
        self.assertIsNone(self.cache.key([self.item("https://host/mutable.jpg")]))
        self.assertIsNone(self.cache.key([self.item("/tmp/mutable.jpg")]))
        item = self.item()
        item.tensor = torch.ones(3)
        self.assertIsNone(self.cache.key([item]))
        self.assertIsNone(self.cache.key([self.item(), self.item()]))
        self.assertIsNone(self.cache.load(None))
        self.assertFalse(self.clients)

    def test_inline_identity_hashes_decoded_bytes_and_reports_rejections(self):
        self.assertEqual(
            self.cache.key([self.item("data:image/jpeg;base64,AQID")]),
            self.cache.key([self.item("data:image/png;base64,AQID")]),
        )
        self.assertIsNone(self.cache.key([self.item("data:image/jpeg;base64,!")]))
        self.assertIsNone(self.cache.key([self.item("data:image/jpeg;base64,")]))
        self.cache._audit_enabled = True
        with patch.dict(os.environ, {"MM_EMB_REMOTE_AUDIT": "1"}), self.assertLogs(
            level="INFO"
        ) as logs:
            item = self.item()
            item.tensor = torch.empty(1)
            self.assertIsNone(self.cache.key([item], trace_id="reject"))
        event = json.loads(logs.output[0].split("MM_EMB_REMOTE_JSON ")[1])
        self.assertEqual(event["reason"], "nonempty_tensor")
        self.assertEqual(event["tensor_shape"], [1])
        self.assertEqual(event["trace_id"], "reject")
        self.assertNotIn("AQID", logs.output[0])

    def test_backend_registration_failure_falls_back(self):
        def fail():
            raise RuntimeError("unavailable")

        self.cache._factory = fail
        self.assertIsNone(self.cache.load("key"))
        self.save(self.cache)
        self.assertEqual(self.cache.stats()["read_error"], 1)
        self.assertEqual(self.cache.stats()["write_error"], 1)


class VitRpcRemoteIdentityTest(unittest.TestCase):
    def setUp(self):
        self.cache = MMRemoteEmbeddingCache(
            None,
            client_factory=lambda: self.fail("identity checks must not open a client"),
            namespace="deepseek-v41-test-weight-revision",
            model_identity={"model_type": "deepseek_v41"},
            max_object_bytes=65536,
            max_inflight_bytes=1024 * 1024,
        )
        self.addCleanup(self.cache.close)

    def decode(self, url="data:image/jpeg;base64,AQID", tensor=None, mm_type=1):
        wire = MultimodalInputsPB(request_id=123)
        image = wire.multimodal_inputs.add()
        image.multimodal_url = url
        image.multimodal_type = mm_type
        if tensor is not None:
            image.multimodal_tensor.CopyFrom(trans_from_tensor(tensor))
        # Exercise the actual protobuf wire decoder used by the ViT RPC server.
        decoded = MultimodalInputsPB.FromString(wire.SerializeToString())
        return trans_mm_input(decoded)

    def test_url_only_empty_tensor_wire_is_admitted_for_deepseek_v41(self):
        for tensor in (None, torch.empty(0)):
            with self.subTest(tensor_present=tensor is not None):
                inputs = self.decode(tensor=tensor)
                self.assertEqual(inputs[0].tensor.numel(), 0)
                self.assertIsNotNone(self.cache.key(inputs))

    def test_real_jpeg_bytes_are_admitted_after_wire_roundtrip(self):
        import base64
        import io

        from PIL import Image

        content = io.BytesIO()
        Image.new("RGB", (3, 2), (11, 22, 33)).save(content, format="JPEG")
        image_bytes = content.getvalue()
        with Image.open(io.BytesIO(image_bytes)) as image:
            self.assertEqual(image.size, (3, 2))
        url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
        inputs = self.decode(url=url, tensor=torch.empty(0))
        self.assertEqual(inputs[0].tensor.numel(), 0)
        self.assertEqual(inputs[0].url, url)
        key = self.cache.key(inputs)
        self.assertIsNotNone(key)
        self.assertEqual(key, self.cache.key(self.decode(url=url)))

    def test_renderer_png_and_jpeg_octet_stream_keep_original_image_key(self):
        import base64
        import io

        from PIL import Image

        from rtp_llm.config.generate_config import GenerateConfig
        from rtp_llm.cpp.model_rpc.model_rpc_client import trans_input
        from rtp_llm.openai.renderers.deepseekv41_renderer import _image_records_to_urls
        from rtp_llm.utils.base_model_datatypes import GenerateInput, MMUrlType

        for image_format, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg")):
            with self.subTest(image_format=image_format):
                content = io.BytesIO()
                Image.new("RGB", (3, 2), (11, 22, 33)).save(
                    content, format=image_format
                )
                image_bytes = content.getvalue()
                encoded = base64.b64encode(image_bytes).decode("ascii")
                original_url = f"data:{mime};base64,{encoded}"
                original_key = self.cache.key(self.decode(url=original_url))
                self.assertIsNotNone(original_key)
                for record in (
                    {"source": {"url": original_url}},
                    {"data": image_bytes},
                ):
                    urls = _image_records_to_urls([1234], [record], 1234)
                    self.assertEqual(
                        urls, ["data:application/octet-stream;base64," + encoded]
                    )
                    # Invoke the actual frontend serializer, not a hand-built
                    # absent TensorPB. URL-only requests must reach C++ with
                    # has_multimodal_tensor() false.
                    frontend = trans_input(
                        GenerateInput(
                            request_id=123,
                            token_ids=torch.tensor([1234], dtype=torch.int32),
                            mm_inputs=[
                                MultimodalInput(
                                    url=urls[0],
                                    mm_type=MMUrlType.IMAGE,
                                    tensor=torch.empty(0),
                                    mm_preprocess_config=MMPreprocessConfig(),
                                )
                            ],
                            generate_config=GenerateConfig(max_new_tokens=1),
                        )
                    )
                    frontend = type(frontend).FromString(frontend.SerializeToString())
                    self.assertEqual(len(frontend.multimodal_inputs), 1)
                    image = frontend.multimodal_inputs[0]
                    self.assertEqual(image.multimodal_url, urls[0])
                    self.assertEqual(image.multimodal_type, MMUrlType.IMAGE)
                    self.assertFalse(image.HasField("multimodal_tensor"))

                    # QueryConverterMultimodalInputTest independently exercises
                    # the actual C++ converter for this absent-TensorPB contract.
                    # Reproduce its present zero-length tensor on the ViT wire,
                    # then invoke the actual Python ViT decoder and cache key.
                    wire = MultimodalInputsPB(request_id=123)
                    forwarded = wire.multimodal_inputs.add()
                    forwarded.CopyFrom(image)
                    forwarded.multimodal_tensor.CopyFrom(
                        trans_from_tensor(torch.empty(0))
                    )
                    self.assertTrue(forwarded.HasField("multimodal_tensor"))
                    self.assertEqual(list(forwarded.multimodal_tensor.shape), [0])
                    inputs = trans_mm_input(
                        MultimodalInputsPB.FromString(wire.SerializeToString())
                    )
                    self.assertEqual(inputs[0].tensor.numel(), 0)
                    # Compare the same serialized preprocessing defaults.
                    expected = [
                        MultimodalInput(
                            url=original_url,
                            mm_type=MMUrlType.IMAGE,
                            tensor=torch.empty(0),
                            mm_preprocess_config=inputs[0].mm_preprocess_config,
                        )
                    ]
                    expected_key = self.cache.key(expected)
                    self.assertIsNotNone(expected_key)
                    self.assertEqual(self.cache.key(inputs), expected_key)

    def test_octet_stream_requires_image_media_type(self):
        url = "data:application/octet-stream;base64,AQID"
        for mm_type in (0, 2, 3, 4):
            with self.subTest(mm_type=mm_type):
                self.assertIsNone(self.cache.key(self.decode(url=url, mm_type=mm_type)))

    def test_other_data_media_and_parameters_still_bypass(self):
        for prefix in (
            "data:application/json;base64,",
            "data:application/pdf;base64,",
            "data:application/octet-stream-extra;base64,",
            "data:text/plain;base64,",
            "data:audio/wav;base64,",
            "data:video/mp4;base64,",
            "data:application/octet-stream;charset=utf-8;base64,",
            "data:application/octet-stream,",
        ):
            with self.subTest(prefix=prefix):
                self.assertIsNone(self.cache.key(self.decode(url=prefix + "AQID")))

    def test_octet_stream_base64_and_nonempty_tensor_fail_closed(self):
        prefix = "data:application/octet-stream;base64,"
        for payload in ("", "!", "AQI", "AQID\n", "AQ ID"):
            with self.subTest(payload=payload):
                self.assertIsNone(self.cache.key(self.decode(url=prefix + payload)))
        expected = torch.tensor([42.0])
        inputs = self.decode(url=prefix + "AQID", tensor=expected)
        self.assertIsNone(self.cache.key(inputs))
        self.assertTrue(torch.equal(inputs[0].tensor, expected))
        self.assertIsNone(
            self.cache.key(self.decode(url=prefix + "AQID") + self.decode())
        )

    def test_octet_stream_keeps_preprocessing_and_source_bytes_in_identity(self):
        url = "data:application/octet-stream;base64,AQID"
        inputs = self.decode(url=url)
        original_key = self.cache.key(inputs)
        self.assertIsNotNone(original_key)
        inputs[0].mm_preprocess_config.width = 256
        self.assertNotEqual(self.cache.key(inputs), original_key)
        self.assertNotEqual(
            self.cache.key(
                self.decode(url="data:application/octet-stream;base64,AQIE")
            ),
            original_key,
        )

    def test_explicit_one_float_is_preserved_and_bypasses_remote_cache(self):
        expected = torch.tensor([42.0])
        inputs = self.decode(tensor=expected)
        self.assertTrue(torch.equal(inputs[0].tensor, expected))
        self.assertIsNone(self.cache.key(inputs))
        self.assertTrue(torch.equal(inputs[0].tensor, expected))

    def test_mutable_urls_and_grouped_inputs_still_bypass(self):
        for url in (
            "https://host/image.jpg",
            "/tmp/image.jpg",
            "data:image/jpeg;base64,!",
        ):
            with self.subTest(url=url):
                self.assertIsNone(self.cache.key(self.decode(url=url)))
        self.assertIsNone(self.cache.key(self.decode() + self.decode()))

    def test_tensor_only_rpc_stays_a_tensor_and_bypasses_remote_cache(self):
        expected = torch.arange(6, dtype=torch.int32).reshape(2, 3)
        inputs = self.decode(url="", tensor=expected, mm_type=4)
        self.assertTrue(torch.equal(inputs[0].tensor, expected))
        self.assertIsNone(self.cache.key(inputs))


class DownloadedSourceModel(FakeEmbeddingLengthInterface):
    supports_inspection_sources = True
    supports_remote_source_identity = True

    def __init__(self):
        super().__init__()
        self.forward_count = 0

    def get_preprocess_params(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
            V41ImageProcessorConfig,
        )

        return dict(processor_config=V41ImageProcessorConfig(vision_min_pixels=1764))

    @staticmethod
    def preprocess_input(mm_inputs, vit_config, **kwargs):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
            DeepSeekV41VisionEmbedding,
        )

        return DeepSeekV41VisionEmbedding.preprocess_input(
            mm_inputs, vit_config, **kwargs
        )

    def batched_embedding(self, data_list, mm_types, **kwargs):
        self.forward_count += len(data_list)
        return [
            (torch.full((2, 4), float(data.rgb[0, 0, 0])), None) for data in data_list
        ]


class DownloadedImageCacheTest(unittest.TestCase):
    def image_bytes(self, color="red"):
        import io

        from PIL import Image

        stream = io.BytesIO()
        Image.new("RGB", (42, 42), color).save(stream, format="PNG")
        return stream.getvalue()

    def item(self, url="https://example.invalid/mutable.png"):
        return MultimodalInput(
            url,
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )

    def engine(self, payload, *, enabled=True, passed=True, hold=False):
        import io

        from rtp_llm.multimodal.test.remote_embedding_test_helpers import SourceProvider

        config = VitConfig()
        config.use_local_preprocess = True
        config.mm_cache_gpu_max_bytes = 0
        config.mm_cache_cpu_max_bytes = 1024 * 1024
        config.mm_hash_key_cache_max_bytes = 4096
        config.mm_remote_cache_enable = enabled
        config.mm_remote_cache_namespace = "downloaded-source-test-model-v1"
        config.mm_remote_cache_read_timeout_ms = 500
        model = ModelConfig()
        model.model_type = "deepseek_v41"
        model.ckpt_path = "/nonexistent/test-model"
        model.mm_related_params.preprocess_batch_size = 1
        store = Store()
        factory = patch(
            "rtp_llm.multimodal.kvcm.RtpKvMetaObjectClient.from_env", return_value=store
        )
        factory.start()
        self.addCleanup(factory.stop)
        interface = DownloadedSourceModel()
        engine = MMProcessEngine(
            interface, model, config, ProfilingDebugLoggingConfig(), device="cpu"
        )
        self.addCleanup(engine.stop)
        provider = SourceProvider(passed, hold)

        async def inspect_completed_request(request, inputs):
            # dev rechecks completed entries for the requesting caller. Only
            # emulate that route after this request's worker inspected the
            # controlled HTTP payload; an initial source-path bypass still fails.
            self.assertIn(request.id, provider.request_ids)
            handle = await provider.submit_sources(
                request, inputs, [payload] * len(inputs)
            )
            handle.rewritten_inputs = inputs
            return handle

        provider.preprocess_and_submit = inspect_completed_request
        engine._inspection._greennet_provider = provider
        self.addCleanup(provider.release.set)
        download = patch(
            "rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision.load_image_source",
            side_effect=lambda *args, **kw: io.BytesIO(payload),
        )
        download.start()
        self.addCleanup(download.stop)
        return engine, store, provider, interface, download

    def seed(self, engine, store, payload, value=None):
        import base64

        item = self.item("data:image/png;base64," + base64.b64encode(payload).decode())
        key = engine._remote_embedding_cache.key([item])
        value = (torch.full((2, 4), 37.0), None) if value is None else value
        store.objects[key] = pack_object(plan_object(value))
        return key, value

    def test_fresh_inspection_remote_hit_waits_for_matching_verdict_without_encoder(
        self,
    ):
        payload = self.image_bytes()
        engine, store, provider, model, _ = self.engine(payload, hold=True)
        item = self.item()
        key, value = self.seed(engine, store, payload)
        metric_patch = patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        )
        report = metric_patch.start()
        self.addCleanup(metric_patch.stop)
        engine.async_submit([item], request_id=901)
        self.assertTrue(store.read_started.wait(5))
        limit = time.monotonic() + 5
        while not provider.calls and time.monotonic() < limit:
            time.sleep(0.005)
        self.assertEqual(provider.calls, [([item.url], [payload])])
        entry = engine._embedding_cache.peek(item.cache_key())
        self.assertIsNotNone(entry)
        self.assertFalse(entry.is_done)
        self.assertFalse(store.saved.is_set())
        self.assertEqual(model.forward_count, 0)
        used = lambda: [
            c
            for c in report.call_args_list
            if c.args[0] == AccMetrics.VIT_REMOTE_EMB_RESULT_ACCEPTED_QPS_METRIC
        ]
        self.assertEqual(used(), [])
        provider.release.set()
        result = engine.get_embedding_result([item], request_id=901)
        self.assertEqual(
            [(c.args[1], c.args[2]) for c in used()],
            [(1, {"origin": "remote", "output": "embeddings"})],
        )
        self.assertTrue(torch.equal(result[0].embeddings[0], value[0]))
        self.assertTrue(entry.greennet_passed)
        self.assertEqual(entry.remote_cache_context["key"], key)
        self.assertEqual(engine._remote_embedding_cache.stats()["hit"], 1)

    def test_fresh_remote_hit_rejection_never_publishes_or_returns(self):
        from rtp_llm.config.exceptions import FtRuntimeException

        payload = self.image_bytes()
        engine, store, provider, model, _ = self.engine(payload, passed=False)
        self.seed(engine, store, payload)
        item = self.item()
        with patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            with self.assertRaises(FtRuntimeException):
                engine.get_embedding_result([item], request_id=902)
        self.assertFalse(
            any(
                c.args[0] == AccMetrics.VIT_REMOTE_EMB_RESULT_ACCEPTED_QPS_METRIC
                for c in report.call_args_list
            )
        )
        self.assertEqual(provider.calls, [([item.url], [payload])])
        self.assertEqual(engine._remote_embedding_cache.stats()["hit"], 1)
        self.assertEqual(model.forward_count, 0)
        self.assertIsNone(engine._embedding_cache.peek(item.cache_key()))
        self.assertFalse(store.saved.is_set())

    def test_fresh_remote_hit_inspection_timeout_never_publishes(self):
        import concurrent.futures

        payload = self.image_bytes()
        engine, store, provider, model, _ = self.engine(payload, hold=True)
        self.seed(engine, store, payload)
        item = self.item()
        key = item.cache_key()
        _, entry = engine._embedding_cache.try_acquire(key)
        engine._run_async_compute([item], key, entry, 903, time.monotonic() + 0.3)
        with self.assertRaises(concurrent.futures.TimeoutError):
            entry.wait(timeout=5)
        self.assertEqual(engine._remote_embedding_cache.stats()["hit"], 1)
        self.assertEqual(model.forward_count, 0)
        self.assertIsNone(engine._embedding_cache.peek(key))
        self.assertFalse(store.saved.is_set())

    def test_same_url_new_bytes_do_not_reuse_old_identity_or_url_verdict(self):
        from rtp_llm.config.exceptions import FtRuntimeException

        old, new = self.image_bytes("red"), self.image_bytes("blue")
        engine, store, provider, model, _ = self.engine(new, passed=False)
        old_key, _ = self.seed(engine, store, old)
        item = self.item()
        with patch.object(engine._hash_key_cache, "greennet_passed", return_value=True):
            verdict = engine.wait_greennet_verdict([item], request_id=904)
            self.assertFalse(verdict.passed)
            with self.assertRaises(FtRuntimeException):
                engine.get_embedding_result([item], request_id=904)
        self.assertTrue(provider.calls)
        self.assertTrue(all(sources == [new] for _, sources in provider.calls))
        self.assertGreaterEqual(engine._remote_embedding_cache.stats()["miss"], 1)
        self.assertEqual(engine._remote_embedding_cache.stats().get("hit", 0), 0)
        self.assertGreaterEqual(model.forward_count, 1)
        self.assertEqual(list(store.objects), [old_key])
        self.assertFalse(store.saved.is_set())

    def test_remote_transfer_failure_uses_already_decoded_source_and_encoder(self):
        payload = self.image_bytes()
        engine, store, provider, model, _ = self.engine(payload)
        self.seed(engine, store, payload)
        store.fail_read = True
        result = engine.get_embedding_result([self.item()], request_id=905)
        self.assertEqual(model.forward_count, 1)
        self.assertEqual(result[0].embeddings[0][0, 0].item(), 255)
        self.assertEqual(provider.calls[0][1], [payload])
        self.assertEqual(engine._remote_embedding_cache.stats()["read_error"], 1)

    def test_remote_disabled_keeps_encoder_and_has_no_optional_client(self):
        payload = self.image_bytes()
        engine, store, provider, model, _ = self.engine(payload, enabled=False)
        result = engine.get_embedding_result([self.item()], request_id=906)
        self.assertEqual(model.forward_count, 1)
        self.assertEqual(result[0].embeddings[0][0, 0].item(), 255)
        self.assertEqual(provider.calls[0][1], [payload])
        self.assertIsNone(engine._remote_embedding_cache)
        self.assertFalse(store.read_started.is_set())
        self.assertFalse(store.saved.is_set())

    def test_http_and_inline_actual_bytes_share_key_but_changed_bytes_do_not(self):
        import base64

        from rtp_llm.multimodal.mm_preprocess_result import ImageSourceIdentity

        payload = self.image_bytes()
        engine, _, _, _, _ = self.engine(payload)
        remote = engine._remote_embedding_cache
        url = self.item()
        identity = ImageSourceIdentity.from_bytes(payload)
        http_key = remote.key([url], source_identity=identity)
        inline = self.item(
            "data:application/octet-stream;base64," + base64.b64encode(payload).decode()
        )
        self.assertEqual(http_key, remote.key([inline]))
        self.assertNotEqual(
            http_key,
            remote.key(
                [url],
                source_identity=ImageSourceIdentity.from_bytes(
                    self.image_bytes("blue")
                ),
            ),
        )
        self.assertIsNone(remote.key([url]))
        self.assertIsNone(
            remote.key(
                [inline], source_identity=ImageSourceIdentity.from_bytes(b"mismatch")
            )
        )
        for media_type in (2, 3, 4):
            url.mm_type = media_type
            self.assertIsNone(remote.key([url], source_identity=identity))
        url.mm_type = 1
        url.tensor = torch.ones(1)
        self.assertIsNone(remote.key([url], source_identity=identity))


class GdrStartupLifetimeTest(unittest.TestCase):
    def test_startup_client_creation_registers_before_first_report(self):
        from unittest import mock

        from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.metric_factory import (
            MetricFactory,
        )
        from rtp_llm.metrics.kmonitor_metric_reporter import MetricReporter
        from rtp_llm.multimodal.mm_remote_embedding_cache import MMRemoteEmbeddingCache

        backend = mock.Mock()
        backend.register_acc_metric.side_effect = (
            lambda name, tags: MetricFactory.create_metric("AccMetric", name, tags)
        )
        backend.register_gauge_metric.side_effect = (
            lambda name, tags: MetricFactory.create_metric("GaugeMetric", name, tags)
        )
        reporter = MetricReporter(backend)
        client = object()
        with mock.patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor", reporter
        ), mock.patch(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.acc_metric.time"
        ) as clock:
            clock.time.return_value = 100.0
            self.assertFalse(reporter._inited)
            self.assertIs(
                MMRemoteEmbeddingCache._create_client(lambda: client, "gdr"), client
            )
            with self.assertRaises(TimeoutError):
                MMRemoteEmbeddingCache._create_client(
                    mock.Mock(side_effect=TimeoutError("startup timeout")), "gdr"
                )
            # The later server initialization neither resets nor re-registers.
            reporter.init()
            self.assertEqual(backend.register_acc_metric.call_count, len(AccMetrics))
            clock.time.return_value = 101.0
            points = reporter._matic_map[
                AccMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_QPS_METRIC.value
            ].fetch_reported_data()
            self.assertEqual(
                {p.tags["status"]: p.value for p in points},
                {"success": 1.0, "timeout": 1.0},
            )
            errors = reporter._matic_map[
                AccMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_ERROR_QPS_METRIC.value
            ].fetch_reported_data()
            self.assertEqual(errors[0].value, 1.0)

    def test_startup_registration_is_not_recreated_after_request_timeout(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        client = Mock(instance_id="registered-serving", instance_group="test")
        client.object_size.side_effect = [TimeoutError("metadata timeout"), None]
        vit = SimpleNamespace(
            mm_remote_cache_gdr_enable=None,
            mm_remote_cache_audit=None,
            mm_remote_cache_namespace="immutable-model-revision",
            mm_image_max_file_size_kb=1024,
            mm_video_max_file_size_kb=1024,
            mm_remote_cache_max_object_bytes=1024,
            mm_remote_cache_max_inflight_bytes=4096,
            mm_remote_cache_max_pending=2,
            mm_remote_cache_read_timeout_ms=100,
            mm_remote_cache_write_timeout_ms=100,
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"MM_REMOTE_CACHE_GDR_ENABLE": "1"}
        ), patch(
            "rtp_llm.multimodal.kvcm.registered_transport.RegisteredObjectTransport",
            return_value=client,
        ) as constructor, patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.kmonitor.report"
        ) as report:
            model = SimpleNamespace(
                model_type="test", compute_dtype="bf16", ckpt_path=directory
            )
            cache = MMRemoteEmbeddingCache.from_configs(model, vit, device="cuda:0")
            constructor.assert_called_once()  # Registration precedes any request.
            self.assertIsNone(cache.load("first"))
            self.assertIsNone(cache.load("second"))
            self.assertEqual(client.object_size.call_count, 2)
            client.close.assert_not_called()
            constructor.assert_called_once()
            cache.close()
            client.close.assert_called_once()
            created = [
                c
                for c in report.call_args_list
                if c.args[0] == AccMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_QPS_METRIC
            ]
            self.assertEqual(
                [(c.args[1], c.args[2]) for c in created],
                [(1, {"transport": "gdr", "status": "success"})],
            )
            acquired = [
                c
                for c in report.call_args_list
                if c.args[0] == AccMetrics.VIT_REMOTE_EMB_STAGE_QPS_METRIC
                and c.args[2]["stage"] == "client_acquire"
            ]
            self.assertEqual(len(acquired), 2)


class V41CppPlaceholderTest(unittest.TestCase):
    def cache(self, model_type="deepseek_v41"):
        cache = MMRemoteEmbeddingCache(
            Store(),
            model_identity={"model_type": model_type},
            namespace="immutable-v41-revision",
            max_object_bytes=65536,
            max_inflight_bytes=1048576,
            max_pending=8,
            read_timeout_ms=500,
        )
        self.addCleanup(cache.close)
        return cache

    def item(
        self, url="data:application/octet-stream;base64,AQID", tensor=None, mm_type=1
    ):
        return MultimodalInput(
            url=url,
            mm_type=mm_type,
            tensor=torch.empty(1) if tensor is None else tensor,
            mm_preprocess_config=MMPreprocessConfig(),
        )

    def test_v41_cpp_url_placeholder_matches_empty_wire_key(self):
        for model in ("deepseek_v41", "deepseek_v41_dspark"):
            cache = self.cache(model)
            for mime in ("image/png", "application/octet-stream"):
                item = self.item(url="data:" + mime + ";base64,AQID")
                empty = self.item(url=item.url, tensor=torch.empty(0))
                expected = cache.key([empty])
                self.assertIsNotNone(expected)
                cache.normalize_cpp_url_input(item)
                self.assertEqual(item.tensor.numel(), 0)
                self.assertEqual(cache.key([item]), expected)

    def test_qwen_dense_and_moe_cpp_placeholder_share_supported_contract(self):
        for model in ("qwen3_vl", "qwen3_vl_moe"):
            cache = self.cache(model)
            item = self.item(url="data:image/png;base64,AQID")
            expected = cache.key([self.item(url=item.url, tensor=torch.empty(0))])
            cache.normalize_cpp_url_input(item)
            self.assertEqual(item.tensor.numel(), 0)
            self.assertIsNotNone(expected)
            self.assertEqual(cache.key([item]), expected)

    def test_placeholder_normalization_keeps_other_tensor_contracts(self):
        cache = self.cache()
        for item in (
            self.item(tensor=torch.empty(2)),
            self.item(tensor=torch.empty(1, dtype=torch.int64)),
            self.item(mm_type=2),
            self.item(url="https://example.invalid/image.png"),
            self.item(url="data:application/pdf;base64,AQID"),
        ):
            original = item.tensor
            cache.normalize_cpp_url_input(item)
            self.assertIs(item.tensor, original)
        unknown = self.cache("unknown")
        item = self.item()
        unknown.normalize_cpp_url_input(item)
        self.assertEqual(item.tensor.numel(), 1)
        self.assertIsNone(unknown.key([item]))


if __name__ == "__main__":
    unittest.main()
