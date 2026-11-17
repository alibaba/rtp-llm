"""Verify abort trace stages using production methods without a GPU runtime."""

import ast
import copy
import json
import importlib.util
import logging
import queue
import threading
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest import mock

SOURCE = Path(__file__).resolve().parents[2] / "adapters/vllm/VLLMEngineObserver.py"
LOGGER = logging.getLogger("navi-abort-test")


class VLLMAbortLoggingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text())
        observer = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "VLLMEngineObserver"
        )
        cls.methods = {
            node.name: node
            for node in observer.body
            if isinstance(node, ast.FunctionDef)
        }

    def observer(self, cancelled=(), mapping=None):
        obj = SimpleNamespace(
            _bridge=SimpleNamespace(
                take_cancelled_requests=lambda: list(cancelled),
                log_request_lifecycle_audit=mock.Mock(),
            ),
            _engine_core=SimpleNamespace(
                aborts_queue=queue.Queue(), input_queue=queue.Queue()
            ),
            _response_lock=threading.Lock(),
            _pending_engine_abort_request_ids=[],
            _response_request_ids=mapping or {},
            _inactive_response_request_ids=set(),
            _cancelled_before_decode=set(),
            _pending_forward_batches={},
            _scheduled_forward_batches={},
            _discard_cache_touch_requests=mock.Mock(),
            _cause_batch_ids=lambda request: (7, 8),
            _request_audit_session="test-session",
            _request_audit_failures=0,
        )
        for name in (
            "_abort_engine_requests",
            "_deliver_engine_aborts",
            "_drain_cancelled_requests",
            "on_request_lifecycle_event",
        ):
            module = ast.parse("from __future__ import annotations\n")
            module.body.append(copy.deepcopy(self.methods[name]))
            namespace = dict(json=json, logger=LOGGER)
            exec(
                compile(ast.fix_missing_locations(module), str(SOURCE), "exec"),
                namespace,
            )
            setattr(obj, name, MethodType(namespace[name], obj))
        return obj

    def test_mapped_abort_logs_receipt_mapping_and_dual_queue_delivery(self):
        obj = self.observer((123, 456), {123: "external-request"})
        with self.assertLogs(LOGGER, level="INFO") as logs:
            self.assertTrue(
                obj._drain_cancelled_requests(SimpleNamespace(ABORT="abort"))
            )
        self.assertEqual(
            ["external-request"], obj._engine_core.aborts_queue.get_nowait()
        )
        self.assertEqual(
            ("abort", ["external-request"]), obj._engine_core.input_queue.get_nowait()
        )
        self.assertIn(456, obj._cancelled_before_decode)
        self.assertIn("request_id_map={123: 'external-request'}", logs.output[0])
        self.assertIn("deferred_navi_request_ids=[456]", logs.output[0])
        self.assertIn("[navi-vllm-abort-enqueued]", logs.output[1])

    def test_idle_poll_does_not_emit_abort_logs(self):
        obj = self.observer()
        with self.assertNoLogs(LOGGER, level="INFO"):
            self.assertTrue(
                obj._drain_cancelled_requests(SimpleNamespace(ABORT="abort"))
            )

    def test_queue_failure_does_not_log_enqueued_and_retry_remains(self):
        obj = self.observer()
        obj._engine_core.input_queue = SimpleNamespace(
            put_nowait=mock.Mock(side_effect=queue.Full)
        )
        with self.assertLogs(LOGGER, level="ERROR") as logs:
            self.assertFalse(
                obj._deliver_engine_aborts(
                    ["request", "request"], SimpleNamespace(ABORT="abort")
                )
            )
        self.assertEqual(["request"], obj._pending_engine_abort_request_ids)
        self.assertFalse(
            any("[navi-vllm-abort-enqueued]" in line for line in logs.output)
        )
        obj._engine_core.input_queue = queue.Queue()
        with self.assertLogs(LOGGER, level="INFO") as logs:
            self.assertTrue(
                obj._drain_cancelled_requests(SimpleNamespace(ABORT="abort"))
            )
        self.assertEqual([], obj._pending_engine_abort_request_ids)
        self.assertEqual(
            ("abort", ["request"]), obj._engine_core.input_queue.get_nowait()
        )
        self.assertIn("[navi-vllm-abort-enqueued]", logs.output[-1])

    def test_unbound_engine_logs_non_delivery(self):
        obj = self.observer()
        obj._engine_core = None
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            obj._abort_engine_requests(["request"], SimpleNamespace(ABORT="abort"))
        self.assertIn("reason=engine_core_unbound", logs.output[0])
        self.assertNotIn("abort-enqueued", logs.output[0])

    def test_applied_log_requires_terminal_aborted_status(self):
        obj = self.observer()
        for event, status in (
            ("admitted", "RUNNING"),
            ("terminal", "FINISHED_STOPPED"),
        ):
            with self.assertNoLogs(LOGGER, level="INFO"):
                obj.on_request_lifecycle_event(
                    dict(event=event, status=status, request_id="request")
                )
        with self.assertLogs(LOGGER, level="INFO") as logs:
            obj.on_request_lifecycle_event(
                dict(event="terminal", status="FINISHED_ABORTED", request_id="request")
            )
        self.assertIn(
            "[navi-vllm-abort-applied] vllm_request_id=request", logs.output[0]
        )
        self.assertIn("batch_id=7 schedule_batch_id=8", logs.output[0])
        self.assertEqual(3, obj._bridge.log_request_lifecycle_audit.call_count)

    def test_ppu_request_without_cached_token_field_still_logs_applied(self):
        path = SOURCE.with_name("VLLMSchedulerObserver.py")
        spec = importlib.util.spec_from_file_location("abort_scheduler_callback", path)
        callbacks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(callbacks)
        obj = self.observer()
        reporter = callbacks.VLLMSchedulerObserver(obj.on_request_lifecycle_event)
        scheduler = SimpleNamespace(request_observer=reporter, running=[], waiting=[],
                                    max_num_running_reqs=192)
        request = SimpleNamespace(
            request_id="ppu-aborted", arrival_time=0,
            status=SimpleNamespace(name="FINISHED_ABORTED"), num_prompt_tokens=64,
            num_computed_tokens=65, num_output_tokens=1, num_output_placeholders=0,
            max_tokens=1024, priority=0, num_preemptions=0,
        )
        with self.assertLogs(LOGGER, level="INFO") as logs:
            reporter.on_transition(scheduler, "terminal", request)
        self.assertIn("[navi-vllm-abort-applied] vllm_request_id=ppu-aborted", logs.output[0])
        audit = json.loads(obj._bridge.log_request_lifecycle_audit.call_args.args[0])
        self.assertIsNone(audit["cached_tokens"])
        self.assertEqual("FINISHED_ABORTED", audit["status"])
        self.assertEqual(0, reporter.errors)

    def test_applied_log_survives_audit_sink_failure(self):
        obj = self.observer()
        obj._bridge.log_request_lifecycle_audit.side_effect = RuntimeError(
            "audit unavailable"
        )
        with self.assertLogs(LOGGER, level="INFO") as logs:
            obj.on_request_lifecycle_event(
                dict(event="terminal", status="FINISHED_ABORTED", request_id="request")
            )
        self.assertIn("[navi-vllm-abort-applied]", logs.output[0])
        self.assertEqual(1, obj._request_audit_failures)


if __name__ == "__main__":
    unittest.main()
