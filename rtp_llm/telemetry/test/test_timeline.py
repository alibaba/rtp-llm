"""Shared request instrumentation contracts without Torch or serving/native dependencies."""

import asyncio
import unittest
from unittest import mock

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rtp_llm.telemetry import RequestTimeline, stage, traced_stage, tracing


class RequestTimelineTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tracing.reset_telemetry_for_test()
        self.exporter = InMemorySpanExporter()
        self.assertTrue(tracing.init_telemetry_for_test(self.exporter))

    def tearDown(self):
        tracing.reset_telemetry_for_test()

    def spans(self):
        tracing._provider.force_flush()
        return self.exporter.get_finished_spans()

    def test_default_timeline_and_custom_stage_names_have_no_dash_dependency(self):
        timeline = RequestTimeline()
        state = tracing.start_server_span("worker", {})
        timeline.attach(state)
        with stage("worker.decode"):
            pass
        timeline.finished("OK")
        state.finish()
        (span,) = self.spans()
        self.assertEqual(
            [event.name for event in span.events],
            [
                "rtp_llm.request_received",
                "worker.decode_started",
                "worker.decode_completed",
                "rtp_llm.request_finished",
            ],
        )

    def test_ingress_replay_and_first_token_are_per_request(self):
        timeline = RequestTimeline("test")
        timeline.query_logged()
        timeline.frame_received()
        state = tracing.start_server_span("request", {}, start_time=timeline.started_at)
        timeline.attach(state)
        timeline.attach(state)
        timeline.response_yielded(has_tokens=False, finished=False, finish_reason=0)
        timeline.response_yielded(has_tokens=True, finished=False, finish_reason=0)
        timeline.response_yielded(has_tokens=True, finished=True, finish_reason=1)
        timeline.finished("OK")
        state.finish()
        (span,) = self.spans()
        self.assertEqual(
            [e.name for e in span.events],
            [
                "test.request_received",
                "test.query_log_completed",
                "test.request_frame_received",
                "test.first_response_yielded",
                "test.first_token_response_yielded",
                "test.finished_response_yielded",
                "test.request_finished",
            ],
        )
        self.assertEqual(span.events[0].timestamp, span.start_time)
        times = [e.timestamp for e in span.events]
        self.assertEqual(times, sorted(times))
        self.assertEqual(span.events[-2].attributes["finish_reason"], 1)

    def test_sync_stage_preserves_result_and_reports_failure_without_payload(self):
        state = tracing.start_server_span("request", {})
        error = ValueError("private prompt")

        @traced_stage("test.parse")
        def parse(value):
            if value is error:
                raise error
            return value

        value = object()
        self.assertIs(parse(value), value)
        with self.assertRaises(ValueError) as caught:
            parse(error)
        self.assertIs(caught.exception, error)
        state.finish()
        (span,) = self.spans()
        self.assertEqual(
            [e.name for e in span.events],
            [
                "test.parse_started",
                "test.parse_completed",
                "test.parse_started",
                "test.parse_failed",
            ],
        )
        self.assertEqual(span.events[-1].attributes["error.type"], "ValueError")
        self.assertNotIn("private", str(span.events))

    async def test_async_cancellation_records_failure_and_reraises_same_exception(self):
        state = tracing.start_server_span("request", {})
        error = asyncio.CancelledError()

        @traced_stage("test.grammar")
        async def validate():
            await asyncio.sleep(0)
            raise error

        with self.assertRaises(asyncio.CancelledError) as caught:
            await validate()
        self.assertIs(caught.exception, error)
        state.finish()
        (span,) = self.spans()
        self.assertEqual(
            [e.name for e in span.events],
            [
                "test.grammar_started",
                "test.grammar_failed",
            ],
        )

    async def test_concurrent_requests_keep_timeline_and_stage_events_separate(self):
        @traced_stage("test.parse")
        async def parse():
            await asyncio.sleep(0)

        async def request(name):
            timeline = RequestTimeline("test")
            state = tracing.start_server_span(name, {}, start_time=timeline.started_at)
            timeline.attach(state)
            await parse()
            timeline.response_yielded(has_tokens=True, finished=True, finish_reason=1)
            state.finish()

        await asyncio.gather(request("first"), request("second"))
        spans = self.spans()
        self.assertEqual(len(spans), 2)
        self.assertNotEqual(spans[0].context.trace_id, spans[1].context.trace_id)
        for span in spans:
            self.assertEqual(len(span.events), 6)
            self.assertEqual(
                sum(
                    e.name.endswith("first_token_response_yielded") for e in span.events
                ),
                1,
            )

    def test_disabled_or_failing_recording_does_not_change_stage(self):
        value = object()
        with mock.patch.object(tracing, "CURRENT_TRACE_STATE") as carrier:
            carrier.get.return_value.add_event.side_effect = RuntimeError("SDK failure")
            with stage("parse") as result:
                result["accepted"] = True
            self.assertIs(traced_stage("prepare")(lambda: value)(), value)
        timeline = RequestTimeline("test")
        timeline.attach(None)
        timeline.response_yielded(has_tokens=True, finished=True, finish_reason=1)
        timeline.finished("OK")

    def test_generator_decoration_is_rejected_before_it_can_span_a_yield(self):
        def generate():
            yield 1

        async def async_generate():
            yield 1

        for function in (generate, async_generate):
            with self.assertRaises(TypeError):
                traced_stage("invalid")(function)

    def test_recording_failure_does_not_interrupt_response_or_completion(self):
        state = mock.Mock()
        state.add_event.side_effect = RuntimeError("exporter unavailable")
        timeline = RequestTimeline("test")
        timeline.attach(state)
        timeline.response_yielded(has_tokens=True, finished=True, finish_reason=1)
        timeline.finished("OK")
        self.assertEqual(state.add_event.call_count, 5)

    def test_stage_records_input_and_result_attributes_on_separate_events(self):
        state = tracing.start_server_span("request", {})
        with stage("test.discovery", attempt=2) as result:
            result["found"] = False
        state.finish()
        (span,) = self.spans()
        self.assertEqual(dict(span.events[0].attributes), {"attempt": 2})
        self.assertEqual(dict(span.events[1].attributes), {"found": False})

    async def test_disabled_tracing_preserves_success_and_cancellation(self):
        tracing.reset_telemetry_for_test()
        error = asyncio.CancelledError()

        @traced_stage("test.disabled")
        async def execute(cancel):
            with tracing.internal_span("test.internal", attempt=1) as span:
                self.assertIsNone(span)
                if cancel:
                    raise error
                return 42

        self.assertEqual(await execute(False), 42)
        with self.assertRaises(asyncio.CancelledError) as caught:
            await execute(True)
        self.assertIs(caught.exception, error)
