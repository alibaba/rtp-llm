import asyncio
import json
from typing import Any
from unittest import TestCase, main
from unittest.mock import MagicMock

import torch
from pydantic import BaseModel

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    ModelRpcClient,
    _selected_pd_separation,
)
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.openai.api_datatype import ChatCompletionRequest, FinisheReason
from rtp_llm.ops import RoleType
from rtp_llm.utils.base_model_datatypes import GenerateInput
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import init_controller, set_global_controller


class FakePipelinResponse(BaseModel):
    res: str


class FakeFrontendWorker(object):
    class FakeBackendRpcServerVisitor:
        def __init__(self):
            self.refresh_calls = []

        def is_backend_service_ready(self, refresh: bool = False):
            self.refresh_calls.append(refresh)
            return True

    def __init__(self):
        self.backend_rpc_server_visitor = self.FakeBackendRpcServerVisitor()
        self.close_called = False

    async def close(self):
        self.close_called = True

    def inference(self, prompt: str, *args: Any, **kwargs: Any):
        response_generator = self._inference(prompt, *args, **kwargs)
        return CompleteResponseAsyncGenerator(
            response_generator, CompleteResponseAsyncGenerator.get_last_value
        )

    def tokenizer_encode(self, prompt: str):
        return [1, 2, 3, 4], ["b", "c", "d", "e"]

    async def _inference(self, prompt: str, *args: Any, **kwargs: Any):
        yield FakePipelinResponse(res=prompt)

    def is_streaming(self, *args: Any, **kwargs: Any):
        return False


class FakeRawRequest(object):
    headers: dict[str, str]

    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}

    async def is_disconnected(self):
        return False


class FrontendServerTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Create PyEnvConfigs with default values for testing
        py_env_configs = PyEnvConfigs()
        set_global_controller(init_controller(py_env_configs.concurrency_config))
        py_env_configs.server_config.start_port = 0
        py_env_configs.server_config.rank_id = 0
        self.frontend_server = FrontendServer(
            rank_id=0,
            server_id=0,
            py_env_configs=py_env_configs,
        )
        self.frontend_server._frontend_worker = FakeFrontendWorker()

    def setUp(self):
        super().setUp()
        from rtp_llm.telemetry import tracing

        self.addCleanup(tracing.reset_telemetry_for_test)
        self.assertTrue(tracing.reset_telemetry_for_test())

    async def _async_run(self, *args: Any, **kwargs: Any):
        res = await self.frontend_server.inference(*args, **kwargs)
        return res

    def test_simple(self):
        loop = asyncio.new_event_loop()
        res = loop.run_until_complete(
            self._async_run(req={"prompt": "hello"}, raw_request=FakeRawRequest())
        )
        self.assertEqual(
            res.body.decode("utf-8"), '{"res":"hello"}', res.body.decode("utf-8")
        )
        res = loop.run_until_complete(
            self._async_run(req='{"prompt": "hello"}', raw_request=FakeRawRequest())
        )
        self.assertEqual(
            res.body.decode("utf-8"), '{"res":"hello"}', res.body.decode("utf-8")
        )

    def test_response_chunk_event_is_streaming_only(self):
        try:
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
        except ImportError:
            self.skipTest("opentelemetry not available")

        from rtp_llm.telemetry import attributes as trace_attrs
        from rtp_llm.telemetry import tracing

        async def _generate():
            yield FakePipelinResponse(res="first")
            yield FakePipelinResponse(res="second")

        def _generate_call():
            return CompleteResponseAsyncGenerator(
                _generate(), CompleteResponseAsyncGenerator.get_last_value
            )

        async def _run_case(is_streaming: bool):
            state = tracing.start_server_span(f"stream={is_streaming}", {})
            wrapped = await self.frontend_server._call_generate_with_report(
                _generate_call, is_streaming
            )
            async for _ in wrapped:
                pass
            state.finish()

        exporter = InMemorySpanExporter()
        self.assertTrue(
            tracing.init_telemetry_for_test(exporter, role="test", tp_rank=0)
        )
        try:
            asyncio.run(_run_case(False))
            asyncio.run(_run_case(True))
        finally:
            tracing.shutdown_telemetry()

        spans = {span.name: span for span in exporter.get_finished_spans()}
        self.assertEqual(list(spans["stream=False"].events), [])
        self.assertEqual(
            [event.name for event in spans["stream=True"].events],
            [trace_attrs.EVENT_FIRST_RESPONSE_CHUNK],
        )

    def test_streaming_latency_counts_only_visible_output_tokens(self):
        from rtp_llm.telemetry import CURRENT_TRACE_STATE

        class FakeTraceState:
            def __init__(self):
                self.events = []
                self.token_counts = []

            def add_event(self, name):
                self.events.append(name)

            def record_frontend_output_tokens(self, token_count):
                self.token_counts.append(token_count)

        responses = (
            {"choices": [{"delta": {"role": "assistant"}}]},
            {
                "choices": [{"delta": {"content": ""}}],
                "usage": {"completion_tokens": 1},
            },
            {
                "choices": [{"delta": {"content": "hello"}}],
                "usage": {"completion_tokens": 2},
            },
            {
                "choices": [{"delta": {"reasoning_content": " world"}}],
                "usage": {"completion_tokens": 4},
            },
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 4},
            },
        )

        async def _run_case(is_streaming: bool):
            async def _generate():
                for response in responses:
                    yield response

            def _generate_call():
                return CompleteResponseAsyncGenerator(
                    _generate(), CompleteResponseAsyncGenerator.get_last_value
                )

            trace_state = FakeTraceState()
            token = CURRENT_TRACE_STATE.set(trace_state)
            try:
                wrapped = await self.frontend_server._call_generate_with_report(
                    _generate_call, is_streaming
                )
                async for _ in wrapped:
                    pass
            finally:
                CURRENT_TRACE_STATE.reset(token)
            return trace_state

        streaming = asyncio.run(_run_case(True))
        non_streaming = asyncio.run(_run_case(False))

        self.assertEqual(streaming.events, ["first_response_chunk"])
        self.assertEqual(streaming.token_counts, [2, 2])
        self.assertEqual(non_streaming.events, [])
        self.assertEqual(non_streaming.token_counts, [])

    def test_streaming_reconciles_missing_and_rebased_token_accounting(self):
        from rtp_llm.telemetry import CURRENT_TRACE_STATE

        class FakeTraceState:
            def __init__(self):
                self.token_counts = []

            def add_event(self, name):
                pass

            def record_frontend_output_tokens(self, token_count):
                self.token_counts.append(token_count)

        async def _run(responses):
            async def _generate():
                for response in responses:
                    yield response

            trace_state = FakeTraceState()
            token = CURRENT_TRACE_STATE.set(trace_state)
            try:
                wrapped = await self.frontend_server._call_generate_with_report(
                    lambda: CompleteResponseAsyncGenerator(
                        _generate(), CompleteResponseAsyncGenerator.get_last_value
                    ),
                    True,
                )
                async for _ in wrapped:
                    pass
            finally:
                CURRENT_TRACE_STATE.reset(token)
            return trace_state.token_counts

        cases = (
            (
                (
                    {"choices": [{"delta": {"content": "a"}}]},
                    {
                        "choices": [{"delta": {"content": "b"}}],
                        "usage": {"completion_tokens": 2},
                    },
                ),
                [1, 1],
            ),
            (
                (
                    {
                        "choices": [{"delta": {"content": "a"}}],
                        "usage": {"completion_tokens": 1},
                    },
                    {"choices": [{"delta": {"content": "b"}}]},
                    {
                        "choices": [{"delta": {"content": "c"}}],
                        "usage": {"completion_tokens": 3},
                    },
                ),
                [1, 1, 1],
            ),
            (
                (
                    {
                        "choices": [{"delta": {"content": "ab"}}],
                        "usage": {"completion_tokens": 2},
                    },
                    {
                        "choices": [{"delta": {"content": "c"}}],
                        "usage": {"completion_tokens": 1},
                    },
                ),
                [2, 1],
            ),
        )
        for responses, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(asyncio.run(_run(responses)), expected)

    def test_streaming_falls_back_to_visible_lanes_without_token_accounting(self):
        """Tool/function frames count as delivered output even with no usage delta."""
        from rtp_llm.telemetry import CURRENT_TRACE_STATE

        class FakeTraceState:
            def __init__(self):
                self.token_counts = []

            def add_event(self, name):
                pass

            def record_frontend_output_tokens(self, token_count):
                self.token_counts.append(token_count)

        responses = (
            # No usage at all: one visible lane is the only available lower bound.
            {
                "choices": [
                    {"delta": {"tool_calls": [{"function": {"name": "get_weather"}}]}}
                ]
            },
            {"choices": [{"delta": {"function_call": {"arguments": '{"city":'}}}]},
            # Usage present but not advancing: still one visible lane.
            {
                "choices": [{"delta": {"content": "!"}}],
                "usage": {"completion_tokens": 0},
            },
            # Structural-only closing frame contributes nothing.
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"completion_tokens": 0},
            },
        )

        async def _run():
            async def _generate():
                for response in responses:
                    yield response

            def _generate_call():
                return CompleteResponseAsyncGenerator(
                    _generate(), CompleteResponseAsyncGenerator.get_last_value
                )

            trace_state = FakeTraceState()
            token = CURRENT_TRACE_STATE.set(trace_state)
            try:
                wrapped = await self.frontend_server._call_generate_with_report(
                    _generate_call, True
                )
                async for _ in wrapped:
                    pass
            finally:
                CURRENT_TRACE_STATE.reset(token)
            return trace_state

        self.assertEqual(asyncio.run(_run()).token_counts, [1, 1, 1])

    def test_streaming_token_observation_is_fail_open(self):
        from rtp_llm.telemetry import CURRENT_TRACE_STATE

        class FakeTraceState:
            def add_event(self, name):
                pass

            def record_frontend_output_tokens(self, token_count):
                pass

        class ChoicesPropertyRaises:
            def model_dump_json(self):
                return "{}"

            @property
            def choices(self):
                raise RuntimeError("choices unavailable")

        class SerializableResponse:
            def __init__(self, content):
                self.choices = [{"delta": {"content": content}}]

            def model_dump_json(self):
                return "{}"

        class BoolRaises:
            def __bool__(self):
                raise RuntimeError("truth value unavailable")

        class ModelDumpPropertyRaises:
            @property
            def model_dump(self):
                raise RuntimeError("model_dump unavailable")

        class ModelDumpCallRaises:
            def model_dump(self, **kwargs):
                raise RuntimeError("model_dump failed")

        responses = (
            ChoicesPropertyRaises(),
            SerializableResponse(BoolRaises()),
            SerializableResponse(ModelDumpPropertyRaises()),
            SerializableResponse(ModelDumpCallRaises()),
        )

        async def _run():
            async def _generate():
                for response in responses:
                    yield response

            def _generate_call():
                return CompleteResponseAsyncGenerator(
                    _generate(), CompleteResponseAsyncGenerator.get_last_value
                )

            token = CURRENT_TRACE_STATE.set(FakeTraceState())
            try:
                wrapped = await self.frontend_server._call_generate_with_report(
                    _generate_call, True
                )
                return [response async for response in wrapped]
            finally:
                CURRENT_TRACE_STATE.reset(token)

        observed = asyncio.run(_run())
        self.assertEqual(len(observed), len(responses))
        self.assertTrue(
            all(actual is expected for actual, expected in zip(observed, responses))
        )

    def test_encode(self):
        res = self.frontend_server.tokenizer_encode('{"prompt": "b c d e"}')
        self.assertEqual(
            res.body.decode("utf-8"),
            '{"token_ids":[1,2,3,4],"tokens":["b","c","d","e"],"error":""}',
        )
        # test error input
        res = self.frontend_server.tokenizer_encode('{"text": "b c d e"}')
        self.assertEqual(json.loads(res.body.decode("utf-8"))["error_code"], 514)

    def test_check_health_uses_cached_service_discovery(self):
        self.assertTrue(self.frontend_server.check_health())
        visitor = self.frontend_server._frontend_worker.backend_rpc_server_visitor
        self.assertEqual(visitor.refresh_calls, [False])

    def test_close_uses_production_frontend_server_contract(self):
        asyncio.run(self.frontend_server.close())

        self.assertTrue(self.frontend_server._frontend_worker.close_called)

    def test_infer_wrap_swallowed_error_marks_span_error(self):
        # Regression: _infer_wrap swallows exceptions into ORJSONResponse(500);
        # the span must still end with status ERROR + http.status_code=500
        # (OTel semconv: 5xx on SERVER spans), not the previous OK.
        try:
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
            from opentelemetry.trace import StatusCode
        except ImportError:
            self.skipTest("opentelemetry not available")

        from rtp_llm.structure.request_constants import request_id_field_name
        from rtp_llm.telemetry import tracing

        exporter = InMemorySpanExporter()
        self.assertTrue(
            tracing.init_telemetry_for_test(exporter, role="test", tp_rank=0)
        )
        try:

            async def _boom(req, raw_request, generate_call):
                raise RuntimeError("engine exploded")

            original_impl = self.frontend_server._infer_impl
            self.frontend_server._infer_impl = _boom
            try:

                async def _run():
                    state = tracing.start_server_span("POST /v1/chat/completions", {})
                    self.assertIsNotNone(state)
                    rep = await self.frontend_server._infer_wrap(
                        {request_id_field_name: 1}, None, None
                    )
                    self.assertEqual(rep.status_code, 500)
                    # the follow-up success-path finish() must stay a no-op
                    state.finish()

                loop = asyncio.new_event_loop()
                loop.run_until_complete(_run())
            finally:
                self.frontend_server._infer_impl = original_impl
        finally:
            tracing.shutdown_telemetry()

        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, StatusCode.ERROR)
        self.assertEqual(spans[0].attributes["http.status_code"], 500)
        self.assertEqual(spans[0].attributes["error.type"], "RuntimeError")

    def test_stream_cancel_records_committed_http_status(self):
        try:
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
            from opentelemetry.trace import StatusCode
        except ImportError:
            self.skipTest("opentelemetry not available")

        from rtp_llm.telemetry import tracing

        async def _cancelled_response():
            raise asyncio.CancelledError("client disconnected")
            yield  # pragma: no cover - keeps this an async generator

        async def _run():
            from rtp_llm.structure.request_constants import request_id_field_name

            state = tracing.start_server_span("stream-cancel", {})
            self.assertIsNotNone(state)
            response = CompleteResponseAsyncGenerator(
                _cancelled_response(), CompleteResponseAsyncGenerator.get_last_value
            )
            async for _ in self.frontend_server.stream_response(
                {request_id_field_name: 1}, response
            ):
                pass

        exporter = InMemorySpanExporter()
        self.assertTrue(
            tracing.init_telemetry_for_test(exporter, role="test", tp_rank=0)
        )
        try:
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(_run())
        finally:
            tracing.shutdown_telemetry()

        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, StatusCode.ERROR)
        self.assertEqual(spans[0].attributes["http.response.status_code"], 200)
        self.assertEqual(spans[0].attributes["http.status_code"], 200)

    def test_stream_aclose_does_not_yield_after_generator_exit(self):
        async def _response():
            try:
                yield FakePipelinResponse(res="first")
                await asyncio.Event().wait()
            finally:
                response_closed.set()

        async def _run():
            from rtp_llm.structure.request_constants import request_id_field_name

            response = CompleteResponseAsyncGenerator(
                _response(), CompleteResponseAsyncGenerator.get_last_value
            )
            stream = self.frontend_server.stream_response(
                {"stream": True, request_id_field_name: 1}, response
            )
            self.assertIn("first", await stream.__anext__())
            await stream.aclose()

        response_closed = asyncio.Event()
        original_controller = self.frontend_server._global_controller
        controller = MagicMock()
        self.frontend_server._global_controller = controller
        try:
            asyncio.run(_run())
        finally:
            self.frontend_server._global_controller = original_controller

        self.assertTrue(response_closed.is_set())
        controller.decrement.assert_called_once_with()

    def test_chat_failure_has_initial_llm_attributes_and_model_priority(self):
        try:
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
        except ImportError:
            self.skipTest("opentelemetry not available")

        from rtp_llm.telemetry import attributes as trace_attrs
        from rtp_llm.telemetry import tracing

        exporter = InMemorySpanExporter()
        self.assertTrue(
            tracing.init_telemetry_for_test(exporter, role="test", tp_rank=0)
        )
        original_impl = self.frontend_server._infer_impl
        original_endpoint = self.frontend_server._openai_endpoint

        async def _boom(req, raw_request, generate_call):
            raise RuntimeError("engine exploded")

        self.frontend_server._infer_impl = _boom
        self.frontend_server._openai_endpoint = type(
            "FakeOpenaiEndpoint", (), {"model_name": "loaded-model"}
        )()
        try:

            async def _run():
                for model in ("requested-model", None):
                    request = ChatCompletionRequest(
                        model=model,
                        messages=[{"role": "user", "content": "hello"}],
                    )
                    response = await self.frontend_server.chat_completion(
                        request, FakeRawRequest()
                    )
                    self.assertEqual(response.status_code, 500)

            loop = asyncio.new_event_loop()
            loop.run_until_complete(_run())
        finally:
            self.frontend_server._infer_impl = original_impl
            self.frontend_server._openai_endpoint = original_endpoint
            tracing.shutdown_telemetry()

        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        for span, expected_model in zip(spans, ("requested-model", "loaded-model")):
            self.assertEqual(span.attributes[trace_attrs.GEN_AI_SPAN_KIND], "LLM")
            self.assertEqual(span.attributes[trace_attrs.GEN_AI_OPERATION_NAME], "chat")
            self.assertEqual(span.attributes[trace_attrs.GEN_AI_SYSTEM], "rtp_llm")
            self.assertIs(span.attributes[trace_attrs.LINGJI_FLAG], True)
            self.assertEqual(
                span.attributes[trace_attrs.ACS_ARMS_TENANT_SPAN_POLICY], "mask"
            )
            self.assertEqual(
                span.attributes[trace_attrs.GEN_AI_REQUEST_MODEL], expected_model
            )
            self.assertNotIn(trace_attrs.GEN_AI_USAGE_TOTAL_TOKENS, span.attributes)

    def test_real_finish_reason_enum_uses_protocol_value(self):
        try:
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
        except ImportError:
            self.skipTest("opentelemetry not available")

        from rtp_llm.telemetry import attributes as trace_attrs
        from rtp_llm.telemetry import tracing

        exporter = InMemorySpanExporter()
        self.assertTrue(
            tracing.init_telemetry_for_test(exporter, role="test", tp_rank=0)
        )
        try:
            state = tracing.start_server_span("server", {})
            tracing.record_response_attributes(
                {"choices": [{"finish_reason": FinisheReason.length}]}
            )
            state.finish()
        finally:
            tracing.shutdown_telemetry()

        spans = {span.name: span for span in exporter.get_finished_spans()}
        reasons = spans["server"].attributes[trace_attrs.GEN_AI_RESPONSE_FINISH_REASONS]
        self.assertEqual(tuple(reasons), ("length",))

    def test_selected_pd_separation_matches_prefill_fallback_contract(self):
        pd_config = GenerateConfig(
            max_new_tokens=2,
            num_beams=1,
            variable_num_beams=[],
            num_return_sequences=1,
            can_use_pd_separation=True,
        )
        self.assertIs(_selected_pd_separation(RoleType.PREFILL, pd_config), True)
        self.assertIs(_selected_pd_separation(RoleType.PDFUSION, pd_config), False)
        self.assertIsNone(_selected_pd_separation(RoleType.DECODE, pd_config))

        fallback_updates = (
            {"max_new_tokens": 1},
            {"num_beams": 2},
            {"variable_num_beams": [2]},
            {"num_return_sequences": 2},
            {"can_use_pd_separation": False},
        )
        for update in fallback_updates:
            with self.subTest(update=update):
                fallback_config = pd_config.model_copy(update=update)
                self.assertIs(
                    _selected_pd_separation(RoleType.PREFILL, fallback_config),
                    False,
                )

    def test_enqueue_writes_selected_route_attributes_before_dial_failure(self):
        try:
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
            from opentelemetry.trace import StatusCode
        except ImportError:
            self.skipTest("opentelemetry not available")

        from rtp_llm.telemetry import attributes as trace_attrs
        from rtp_llm.telemetry import tracing

        class FailingChannelPool:
            def __init__(self):
                self.target = None

            async def get(self, target):
                self.target = target
                raise RuntimeError("dial failed")

        async def _consume(client, generate_input):
            async for _ in client.enqueue(generate_input):
                pass

        exporter = InMemorySpanExporter()
        self.assertTrue(
            tracing.init_telemetry_for_test(exporter, role="test", tp_rank=0)
        )
        cases = (
            (RoleType.PDFUSION, "127.0.0.1", 51001, False),
            (RoleType.PREFILL, "prefill.test", 51002, True),
        )
        try:
            for request_id, (role, host, port, expected_pd_sep) in enumerate(
                cases, start=1
            ):
                state = tracing.start_server_span(f"root-{request_id}", {})
                self.assertIsNotNone(state)
                config = GenerateConfig(
                    max_new_tokens=2,
                    num_beams=1,
                    variable_num_beams=[],
                    num_return_sequences=1,
                    can_use_pd_separation=True,
                    role_addrs=[
                        RoleAddr(
                            role=role,
                            ip=host,
                            http_port=0,
                            grpc_port=port,
                        )
                    ],
                )
                generate_input = GenerateInput(
                    request_id=request_id,
                    token_ids=torch.tensor([1, 2]),
                    mm_inputs=[],
                    generate_config=config,
                )
                client = ModelRpcClient([], {})
                channel_pool = FailingChannelPool()
                client._channel_pool = channel_pool

                with self.assertRaisesRegex(RuntimeError, "dial failed"):
                    asyncio.run(_consume(client, generate_input))
                self.assertEqual(channel_pool.target, f"{host}:{port}")
                state.finish()
        finally:
            tracing.shutdown_telemetry()

        spans = exporter.get_finished_spans()
        for request_id, (_, host, port, expected_pd_sep) in enumerate(cases, start=1):
            root_span = next(
                span for span in spans if span.name == f"root-{request_id}"
            )
            client_span = next(
                span
                for span in spans
                if span.name == "rtp_llm.generate_stream_call"
                and span.attributes["server.port"] == port
            )
            self.assertIs(
                root_span.attributes[trace_attrs.RTP_LLM_PD_SEP], expected_pd_sep
            )
            self.assertEqual(client_span.attributes["server.address"], host)
            self.assertEqual(client_span.status.status_code, StatusCode.ERROR)
            self.assertEqual(client_span.attributes["error.type"], "RuntimeError")


if __name__ == "__main__":
    main()
