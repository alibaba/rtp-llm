import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest import TestCase, main
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from pydantic import BaseModel

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    ModelRpcClient,
    _selected_pd_separation,
)
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.frontend.frontend_worker import (
    BatchPipelineResponse,
    FrontendWorker,
    PipelineResponse,
)
from rtp_llm.metrics import AccMetrics, GaugeMetrics
from rtp_llm.openai.api_datatype import ChatCompletionRequest, FinisheReason
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.ops import RoleType
from rtp_llm.pipeline.pipeline import Pipeline
from rtp_llm.structure.request_constants import request_id_field_name
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
)
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import init_controller, set_global_controller


class FakePipelinResponse(BaseModel):
    res: str


class FakeBatchResponse(BaseModel):
    response_batch: list[Any] = []


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
        self.batch_calls = []

    async def close(self):
        self.close_called = True

    def inference(self, batch=False, /, **kwargs: Any):
        if batch:
            self.batch_calls.append(kwargs)
            response_generator = CompleteResponseAsyncGenerator.generate_from_list(
                [FakeBatchResponse()]
            )
        else:
            response_generator = self._inference(**kwargs)
        return CompleteResponseAsyncGenerator(
            response_generator, CompleteResponseAsyncGenerator.get_last_value
        )

    def tokenizer_encode(self, prompt: str):
        return [1, 2, 3, 4], ["b", "c", "d", "e"]

    async def _inference(self, prompt: str, *args: Any, **kwargs: Any):
        yield FakePipelinResponse(res=prompt)

    def is_streaming(self, *args: Any, **kwargs: Any):
        return False


class BatchFrontendWorkerTest(TestCase):
    def test_chat_batch_preserves_master_scheduling_and_static_batch_rpc(self):
        for mode in ("static", "master", "failure"):
            with self.subTest(mode=mode):
                second_finished = asyncio.Event()
                closed = set()

                async def generate(item):
                    try:
                        yield "intermediate"
                        if item == 700:
                            await second_finished.wait()
                            if mode == "failure":
                                raise RuntimeError("batch member failed")
                        else:
                            second_finished.set()
                            if mode == "failure":
                                await asyncio.Future()
                        yield item
                    finally:
                        closed.add(item)

                endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
                endpoint._prepare_chat_input = lambda rid, _: (rid, GenerateConfig())
                endpoint._render_single_output = AsyncMock(
                    side_effect=lambda out, *_: out
                )
                visitor = endpoint.backend_rpc_server_visitor = MagicMock()
                visitor.host_service.service_available = mode != "static"
                visitor.enqueue = AsyncMock(side_effect=generate)
                visitor.batch_enqueue = AsyncMock(return_value=[700, 701])
                request = SimpleNamespace(
                    requests=[SimpleNamespace(stream=False) for _ in range(2)]
                )

                async def check():
                    call = endpoint.batch_chat_completion(700, request)
                    if mode == "failure":
                        with self.assertRaisesRegex(
                            RuntimeError, "batch member failed"
                        ):
                            await call
                        self.assertEqual({700, 701}, closed)
                    else:
                        self.assertEqual([700, 701], await call)

                asyncio.run(asyncio.wait_for(check(), 1))
                self.assertEqual(
                    0 if mode == "static" else 2, visitor.enqueue.await_count
                )
                self.assertEqual(
                    1 if mode == "static" else 0, visitor.batch_enqueue.await_count
                )

    def test_batch_endpoint_preserves_request_and_selects_execution_by_topology(self):
        cases = [
            (RoleType.PDFUSION, False, [], True),
            (RoleType.PDFUSION, True, [], False),
            (RoleType.FRONTEND, True, [], False),
            (RoleType.FRONTEND, True, [RoleType.PDFUSION], True),
            (RoleType.FRONTEND, True, [RoleType.PREFILL, RoleType.DECODE], False),
            (RoleType.PREFILL, True, [], False),
        ]
        for role, scheduling, assignments, atomic in cases:
            for key in ("generate_config", "generation_config"):
                with self.subTest(
                    role=role, scheduling=scheduling, assignments=assignments, key=key
                ):
                    expected = BatchPipelineResponse(response_batch=[])

                    async def generate(*args, **kwargs):
                        yield expected

                    worker = FrontendWorker.__new__(FrontendWorker)
                    visitor = worker.backend_rpc_server_visitor = MagicMock()
                    visitor.pd_sep_config.role_type = role
                    visitor.host_service.service_available = scheduling
                    worker._yield_batch_generate = MagicMock(side_effect=generate)
                    worker._inference = MagicMock(side_effect=generate)
                    response = worker.inference(
                        True,
                        batch=False,
                        prompt_batch=["first", "second"],
                        max_new_tokens=37,
                        headers={"x-request-id": "trace"},
                        **{
                            request_id_field_name: 700,
                            key: {
                                "max_new_tokens": 8,
                                "temperature": 0.5,
                                "role_addrs": [
                                    RoleAddr(
                                        role=r, ip="be", http_port=80, grpc_port=81
                                    )
                                    for r in assignments
                                ],
                            },
                        },
                    )
                    self.assertIs(
                        expected,
                        asyncio.run(
                            CompleteResponseAsyncGenerator.get_last_value(response)
                        ),
                    )
                    self.assertIs(
                        expected, asyncio.run(response.gen_complete_response_once())
                    )
                    self.assertEqual(
                        int(atomic), worker._yield_batch_generate.call_count
                    )
                    self.assertEqual(int(not atomic), worker._inference.call_count)
                    call = (
                        worker._yield_batch_generate if atomic else worker._inference
                    ).call_args
                    request = call.args[0]
                    configs = request.generate_configs
                    self.assertEqual(700, request.request_id)
                    self.assertFalse(request.is_streaming)
                    self.assertEqual([37, 37], [gc.max_new_tokens for gc in configs])
                    self.assertEqual([0.5, 0.5], [gc.temperature for gc in configs])
                    self.assertIsNot(configs[0], configs[1])
                    self.assertEqual({"x-request-id": "trace"}, call.kwargs["headers"])

    def test_batch_endpoint_rejects_streaming_and_root_preserves_it(self):
        worker = FrontendWorker.__new__(FrontendWorker)
        worker._yield_generate = MagicMock()
        worker._yield_batch_generate = MagicMock()
        worker._parallel_batch_async_generators = MagicMock(return_value="per-item")
        for items in ([], ["first", "second"]):
            for flags in (
                {"stream": True},
                {"yield_generator": True},
                {"is_streaming": True},
                {"generate_config": {"is_streaming": True}},
                {"generation_config": {"yield_generator": True}},
            ):
                args = {request_id_field_name: 700, "prompt_batch": items, **flags}
                with self.subTest(items=items, flags=flags), self.assertRaises(
                    FtRuntimeException
                ) as raised:
                    worker.inference(True, **args)
                self.assertEqual(
                    ExceptionType.UNSUPPORTED_OPERATION, raised.exception.exception_type
                )
        # Keyword batch=True is body data; the positional mode stays False,
        # so root inference must still accept this streaming request.
        worker.inference(batch=True, **args)
        worker._yield_batch_generate.assert_not_called()

    def test_root_inference_preserves_generate_config_alias_precedence(self):
        worker = FrontendWorker.__new__(FrontendWorker)
        worker._inference = MagicMock()
        for ignored in (None, 17, {"yield_generator": True}):
            with self.subTest(ignored=ignored):
                worker.inference(
                    prompt="hello",
                    generate_config={"max_new_tokens": 37},
                    generation_config=ignored,
                    **{request_id_field_name: 700},
                )
                request = worker._inference.call_args.args[0]
                self.assertFalse(request.is_streaming)
                self.assertEqual(37, request.generate_configs[0].max_new_tokens)

    def test_root_inference_preserves_null_stream_flag_with_incremental(self):
        worker = FrontendWorker.__new__(FrontendWorker)
        worker._inference = MagicMock()
        args = {
            request_id_field_name: 700,
            "prompt": "hello",
            "return_incremental": True,
        }
        worker.inference(yield_generator=None, **args)
        request = worker._inference.call_args.args[0]
        self.assertIsNone(request.is_streaming)
        self.assertTrue(request.incremental)
        with self.assertRaises(FtRuntimeException) as raised:
            worker.inference(yield_generator=False, **args)
        self.assertEqual(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR, raised.exception.exception_type
        )

    def test_prepared_batch_invokes_backend_once_with_group_identity(self):
        worker = FrontendWorker.__new__(FrontendWorker)
        worker.generate_env_config = None
        pipeline = Pipeline.__new__(Pipeline)
        worker.pipeline = pipeline
        pipeline._special_tokens = None
        pipeline.tokenizer = MagicMock()
        pipeline.tokenizer.encode.return_value = [1, 2]
        pipeline.create_generate_config = lambda config, *args, **kwargs: config
        visitor = pipeline.backend_rpc_server_visitor = MagicMock()
        worker.backend_rpc_server_visitor = visitor
        visitor.pd_sep_config.role_type = RoleType.PDFUSION
        visitor.host_service.service_available = False
        visitor.batch_enqueue = AsyncMock(
            return_value=[
                GenerateOutputs(generate_outputs=[GenerateOutput(finished=True)])
                for _ in range(2)
            ]
        )
        pipeline.decode_non_incremental_tokens = MagicMock(
            side_effect=[(["one"], [1], []), (["two"], [1], [])]
        )
        response = worker.inference(
            True,
            prompt_batch=["first", "second"],
            max_new_tokens=37,
            generation_config={"max_new_tokens": 8, "aux_info": False},
            headers={"X-Request-ID": "trace", "ignored": "value"},
            **{request_id_field_name: 700},
        )
        result = asyncio.run(CompleteResponseAsyncGenerator.get_last_value(response))
        self.assertEqual(
            ["one", "two"], [item.response for item in result.response_batch]
        )
        self.assertTrue(all(item.finished for item in result.response_batch))
        visitor.batch_enqueue.assert_awaited_once()
        inputs = visitor.batch_enqueue.call_args.args[0]
        self.assertEqual([700, 10_700], [item.request_id for item in inputs])
        self.assertEqual([2, 2], [item.group_size for item in inputs])
        self.assertEqual([700, 700], [item.group_id for item in inputs])
        self.assertEqual(
            [{"x-request-id": "trace"}] * 2, [item.headers for item in inputs]
        )
        self.assertEqual(
            [37, 37], [item.generate_config.max_new_tokens for item in inputs]
        )
        self.assertIsNot(inputs[0].generate_config, inputs[1].generate_config)

    def test_master_scheduled_batch_collects_all_final_responses(self):
        worker = FrontendWorker.__new__(FrontendWorker)
        worker.backend_rpc_server_visitor = SimpleNamespace(
            pd_sep_config=SimpleNamespace(role_type=RoleType.FRONTEND),
            host_service=SimpleNamespace(service_available=True),
        )

        async def generate(request_id, text, urls, **kwargs):
            yield PipelineResponse(response=text, finished=False)
            yield PipelineResponse(response=text + "-done", finished=True)

        worker._yield_generate = generate
        response = worker.inference(
            True, prompt_batch=["first", "second"], **{request_id_field_name: 700}
        )
        result = asyncio.run(CompleteResponseAsyncGenerator.get_last_value(response))
        self.assertEqual(
            ["first-done", "second-done"],
            [item.response for item in result.response_batch],
        )
        self.assertTrue(all(item.finished for item in result.response_batch))


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

    def test_root_and_batch_share_access_logs_metrics_and_concurrency(self):
        server = self.frontend_server
        for batch, body, expected in [
            (False, {"prompt": "hello"}, b'{"res":"hello"}'),
            (True, {"prompt_batch": []}, b'{"response_batch":[]}'),
        ]:
            for req in (body, json.dumps(body)):
                with self.subTest(batch=batch, req=req), patch.object(
                    server, "_access_logger"
                ) as logger, patch(
                    "rtp_llm.frontend.frontend_server.kmonitor.report"
                ) as report:
                    response = asyncio.run(
                        server.inference(req, FakeRawRequest(), batch=batch)
                    )
                    self.assertEqual(expected, response.body)
                    self.assertEqual(
                        0, server._global_controller.current_concurrency.value
                    )
                    logger.log_query_access.assert_called_once()
                    logger.log_success_access.assert_called_once()
                    metrics = [call.args[0] for call in report.call_args_list]
                    for metric in (
                        AccMetrics.QPS_METRIC,
                        AccMetrics.SUCCESS_QPS_METRIC,
                        GaugeMetrics.LANTENCY_METRIC,
                    ):
                        self.assertEqual(1, metrics.count(metric))

    def test_batch_checks_disconnect_before_and_after_execution(self):
        server = self.frontend_server
        for states in ([True], [False, True]):
            with self.subTest(states=states), patch.object(
                server, "_access_logger"
            ) as logger:
                server._frontend_worker.batch_calls.clear()
                raw = FakeRawRequest()
                raw.is_disconnected = AsyncMock(side_effect=states)
                response = asyncio.run(
                    server.inference({"prompt_batch": ["hello"]}, raw, batch=True)
                )
                self.assertEqual(500, response.status_code)
                self.assertEqual(
                    len(states) - 1, len(server._frontend_worker.batch_calls)
                )
                self.assertEqual(0, server._global_controller.current_concurrency.value)
                logger.log_exception_access.assert_called_once()
                logger.log_success_access.assert_not_called()

    def test_routing_credentials_are_required_only_on_configured_frontends(self):
        server = self.frontend_server
        for batch in (False, True):
            for key in ("generate_config", "generation_config"):
                for expected, provided in (
                    ("", ""),
                    ("trusted-secret", ""),
                    ("trusted-secret", "wrong"),
                    ("trusted-secret", "trusted-secret"),
                ):
                    with self.subTest(
                        batch=batch, key=key, expected=expected, provided=provided
                    ):
                        server._dispatcher_routing_token = expected
                        config = {
                            "role_addrs": [
                                {
                                    "role": "PDFUSION",
                                    "ip": "be",
                                    "http_port": 80,
                                    "grpc_port": 81,
                                }
                            ]
                        }
                        request = {
                            key: config,
                            **(
                                {"prompt_batch": ["hello"]}
                                if batch
                                else {"prompt": "hello"}
                            ),
                        }
                        headers = {
                            "x-rtp-llm-dispatcher-routing-token": provided,
                            "X-Request-ID": "trace",
                            "ignored": "secret",
                        }
                        with patch.object(
                            server._frontend_worker,
                            "inference",
                            wraps=server._frontend_worker.inference,
                        ) as infer:
                            response = asyncio.run(
                                server.inference(
                                    request, FakeRawRequest(headers), batch=batch
                                )
                            )
                            if not expected or provided == expected:
                                self.assertEqual(
                                    200, response.status_code, response.body
                                )
                                infer.assert_called_once()
                                self.assertEqual(batch, infer.call_args.args[0])
                                self.assertEqual(config, infer.call_args.kwargs[key])
                                self.assertEqual(
                                    {"x-request-id": "trace"},
                                    infer.call_args.kwargs["headers"],
                                )
                            else:
                                infer.assert_not_called()
                                self.assertEqual(
                                    ExceptionType.INVALID_PARAMS.value,
                                    json.loads(response.body)["error_code"],
                                )
                            self.assertEqual(
                                0, server._global_controller.current_concurrency.value
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
            with patch(
                "rtp_llm.frontend.frontend_server.generate_request_id",
                return_value=3540218608800727041,
            ):
                loop.run_until_complete(_run())
        finally:
            self.frontend_server._infer_impl = original_impl
            self.frontend_server._openai_endpoint = original_endpoint
            tracing.shutdown_telemetry()

        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        for span, expected_model in zip(spans, ("requested-model", "loaded-model")):
            self.assertEqual(
                span.attributes[trace_attrs.REQUEST_ID], "3540218608800727041"
            )
            self.assertIsInstance(span.attributes[trace_attrs.REQUEST_ID], str)
            self.assertNotIn("rtp_llm.request_id", span.attributes)
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
            self.assertEqual(
                client_span.attributes[trace_attrs.REQUEST_ID], str(request_id)
            )
            self.assertIsInstance(client_span.attributes[trace_attrs.REQUEST_ID], str)
            self.assertNotIn("rtp_llm.request_id", client_span.attributes)
            self.assertEqual(client_span.status.status_code, StatusCode.ERROR)
            self.assertEqual(client_span.attributes["error.type"], "RuntimeError")


if __name__ == "__main__":
    main()
