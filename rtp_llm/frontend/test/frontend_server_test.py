import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest import TestCase, main
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import anyio
import torch
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    ModelRpcClient,
    _selected_pd_separation,
)
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.metrics import AccMetrics, GaugeMetrics
from rtp_llm.openai.api_datatype import ChatCompletionRequest, FinisheReason
from rtp_llm.ops import RoleType
from rtp_llm.server.misc import format_client_exception, format_exception
from rtp_llm.structure.request_constants import request_id_field_name
from rtp_llm.utils.base_model_datatypes import GenerateInput
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import init_controller, set_global_controller


class FakePipelinResponse(BaseModel):
    res: str


class FakeAuxResponse(BaseModel):
    aux_info: dict[str, Any]


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
        self.last_inference_kwargs = {}

    async def close(self):
        self.close_called = True

    def inference(self, prompt: str, *args: Any, **kwargs: Any):
        self.last_inference_kwargs = kwargs
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
        with patch.dict("os.environ", {"RTP_LLM_FRONTEND_METRICS_ENABLE": "1"}):
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

    def tearDown(self):
        self.frontend_server._request_metrics.close()

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
        self.assertTrue(
            callable(
                self.frontend_server._frontend_worker.last_inference_kwargs[
                    "frontend_metric_observer"
                ]
            )
        )
        res = loop.run_until_complete(
            self._async_run(req='{"prompt": "hello"}', raw_request=FakeRawRequest())
        )
        self.assertEqual(
            res.body.decode("utf-8"), '{"res":"hello"}', res.body.decode("utf-8")
        )

    def test_qos_priority_reaches_frontend_metrics_and_backend(self):
        with patch(
            "rtp_llm.metrics.kmonitor_metric_reporter.MetricReporter.is_reporting_enabled",
            new_callable=PropertyMock,
            return_value=True,
        ), patch("rtp_llm.frontend.frontend_server.kmonitor.report") as report:
            asyncio.run(
                self._async_run(
                    req={"prompt": "hello", "source": "test"},
                    raw_request=FakeRawRequest({"X-DashScope-Inner-Qos-Level": "70"}),
                )
            )
        backend_tags = self.frontend_server._frontend_worker.last_inference_kwargs[
            "frontend_metric_tags"
        ]
        self.assertEqual(backend_tags["priority"], "70")
        for metric in (
            AccMetrics.QPS_METRIC,
            AccMetrics.SUCCESS_QPS_METRIC,
            GaugeMetrics.FRONTEND_REQUEST_RT_MS_METRIC,
        ):
            calls = [call for call in report.call_args_list if call.args[0] == metric]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].args[2]["priority"], "70")
            self.assertEqual(calls[0].args[2]["source"], "test")

    def _native_batch_fixture(self, api):
        from rtp_llm.frontend.frontend_request_metrics import FrontendRequestMetrics
        from rtp_llm.frontend.frontend_worker import FrontendWorker
        from rtp_llm.openai.api_datatype import BatchChatCompletionRequest
        from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
        from rtp_llm.ops import SpecialTokens
        from rtp_llm.pipeline.pipeline import Pipeline
        from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
        from rtp_llm.utils.base_model_datatypes import (
            BatchedTerminalOutputs,
            GenerateOutputs,
        )

        server = self.frontend_server
        server._request_metrics.close()
        clock = [10.0]
        sink = SimpleNamespace(report=MagicMock())
        server._request_metrics = FrontendRequestMetrics(
            metric_reporter=sink, clock=lambda: clock[0], enabled=True
        )
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 1024
        visitor.host_service = SimpleNamespace(service_available=False)
        visitor.fill_request_info = MagicMock()
        visitor.check_sp_supported = MagicMock()
        visitor.check_prefill_cp_supported = MagicMock()
        visitor._frontend_route_cache_keys = MagicMock(return_value=[[101]])
        visitor._report_frontend_cache_key_metrics = MagicMock()
        outputs = [
            GenerateOutputs(
                generate_outputs=BatchedTerminalOutputs(
                    torch.ones((2, 1, i + 2), dtype=torch.int32),
                    torch.tensor([[1, 2, 3]]),
                ),
                frontend_input_len=(i + 1) * 10,
                frontend_output_len=(i + 2) * 2,
                frontend_generate_token_num=8 if i == 0 else None,
                frontend_context_token_num=30 if i == 0 else None,
                frontend_context_token_num_with_cache=30 if i == 0 else None,
                frontend_context_execute_time_us=2_000_000 if i == 0 else None,
                frontend_context_execute_time_with_cache_us=(
                    2_000_000 if i == 0 else None
                ),
            )
            for i in range(2)
        ]

        async def rpc(inputs):
            self.assertEqual(server._global_controller.current_concurrency.value, 1)
            self.assertEqual(server._request_metrics._active_requests, 1)
            self.assertEqual(len(inputs), 2)
            for inp in inputs:
                self.assertTrue(inp.generate_config.frontend_metric_streaming)
                self.assertFalse(inp.generate_config.aux_info)
                self.assertEqual(inp.frontend_metric_tags["priority"], "70")
            clock[0] = 11.0
            return outputs

        visitor.model_rpc_client = SimpleNamespace(
            batch_enqueue=AsyncMock(side_effect=rpc)
        )
        raw_request = FakeRawRequest({"X-DashScope-Inner-Qos-Level": "70"})
        config = {"aux_info": False, "num_beams": 2, "num_return_sequences": 2}
        if api == "pipeline":
            pipeline = Pipeline.__new__(Pipeline)
            pipeline._special_tokens = SpecialTokens()
            pipeline.tokenizer = MagicMock()
            pipeline.tokenizer.__len__.return_value = 100
            pipeline.tokenizer.encode.return_value = [1, 2, 3]
            pipeline.tokenizer.batch_decode.side_effect = lambda ids, **_: [
                "answer"
            ] * len(ids)
            pipeline.create_generate_config = (
                lambda config, *args, **kwargs: GenerateConfig(**config)
            )
            pipeline.backend_rpc_server_visitor = visitor
            worker = FrontendWorker.__new__(FrontendWorker)
            worker.pipeline = pipeline
            worker.generate_env_config = None
            server._frontend_worker = worker
            request = {"prompt_batch": ["first", "second"], "generate_config": config}
            call = lambda: server.batch_infer(request, raw_request)
        else:
            endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
            endpoint.backend_rpc_server_visitor = visitor

            def prepare(request_id, _request):
                gc = GenerateConfig(**config)
                return (
                    GenerateInput(
                        request_id=request_id,
                        token_ids=torch.tensor([1, 2, 3]),
                        mm_inputs=[],
                        generate_config=gc,
                    ),
                    gc,
                )

            endpoint._prepare_chat_input = prepare
            endpoint._render_single_output = AsyncMock(
                return_value=FakeAuxResponse(aux_info={})
            )
            server._openai_endpoint = endpoint
            request = BatchChatCompletionRequest(
                requests=[
                    ChatCompletionRequest(messages=[], aux_info=False),
                    ChatCompletionRequest(messages=[], aux_info=False),
                ]
            )
            call = lambda: server.batch_chat_completion(request, raw_request)
        return call, visitor, sink, clock, outputs

    def test_native_batch_metrics_use_private_terminal_snapshots(self):
        from rtp_llm.utils.base_model_datatypes import BatchedTerminalOutputs

        for api in ("pipeline", "openai"):
            with self.subTest(api=api):
                call, visitor, sink, _, outputs = self._native_batch_fixture(api)
                original_getitem = BatchedTerminalOutputs.__getitem__
                rows = []

                def getitem(storage, index):
                    rows.append(index)
                    return original_getitem(storage, index)

                with patch.object(BatchedTerminalOutputs, "__getitem__", getitem):
                    response = asyncio.run(call())
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("frontend_", response.body.decode())
                # Worker serialization already reads one row per item; private
                # projection and batch decoding must not materialize any rows.
                self.assertEqual(rows, [0, 0] if api == "pipeline" else [])
                visitor.model_rpc_client.batch_enqueue.assert_awaited_once()
                self.assertEqual(
                    visitor._report_frontend_cache_key_metrics.call_count, 2
                )
                values = {}
                for metric_call in sink.report.call_args_list:
                    values.setdefault(metric_call.args[0], []).append(
                        metric_call.args[1]
                    )
                self.assertEqual(
                    values[GaugeMetrics.FRONTEND_CONCURRENCY_METRIC], [1, 0]
                )
                self.assertEqual(
                    values[GaugeMetrics.FRONTEND_REQUEST_RT_MS_METRIC], [1000.0]
                )
                self.assertEqual(values[GaugeMetrics.FRONTEND_TTFT_MS_METRIC], [1000.0])
                self.assertEqual(
                    values[GaugeMetrics.FRONTEND_INPUT_LENGTH_METRIC], [30]
                )
                self.assertEqual(
                    values[GaugeMetrics.FRONTEND_OUTPUT_LENGTH_METRIC], [10]
                )
                self.assertEqual(
                    values[GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC], [8.0]
                )
                self.assertEqual(
                    values[GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC], [15.0]
                )
                self.assertEqual(
                    self.frontend_server._request_metrics._active_requests, 0
                )
                self.assertEqual(
                    self.frontend_server._global_controller.current_concurrency.value, 0
                )
                self.assertEqual(
                    [output.frontend_output_len for output in outputs], [4, 6]
                )

    def test_native_batch_failure_and_cancellation_finish_once(self):
        for api in ("pipeline", "openai"):
            for failure in ("partial", "cancel", "render", "disconnect"):
                with self.subTest(api=api, failure=failure):
                    call, visitor, sink, _, _ = self._native_batch_fixture(api)
                    closed = []

                    async def run():
                        entered = asyncio.Event()
                        original_rpc = (
                            visitor.model_rpc_client.batch_enqueue.side_effect
                        )

                        async def rpc(inputs):
                            try:
                                await original_rpc(inputs)
                                if failure == "partial":
                                    raise RuntimeError("batch item 1 failed")
                                entered.set()
                                await asyncio.Event().wait()
                            finally:
                                closed.append(True)

                        if failure == "render":
                            if api == "pipeline":
                                self.frontend_server._frontend_worker.pipeline.tokenizer.batch_decode.side_effect = ValueError(
                                    "decode failed"
                                )
                            else:
                                self.frontend_server._openai_endpoint._render_single_output.side_effect = ValueError(
                                    "render failed"
                                )
                            with self.assertRaises(ValueError):
                                await call()
                        elif failure == "partial":
                            visitor.model_rpc_client.batch_enqueue.side_effect = rpc
                            with self.assertRaisesRegex(RuntimeError, "batch item"):
                                await call()
                        else:
                            visitor.model_rpc_client.batch_enqueue.side_effect = rpc
                            if failure == "disconnect":
                                with patch.object(
                                    FakeRawRequest,
                                    "is_disconnected",
                                    new=AsyncMock(side_effect=[False, True]),
                                ):
                                    with self.assertRaises(asyncio.CancelledError):
                                        await call()
                            else:
                                task = asyncio.create_task(call())
                                await asyncio.wait_for(entered.wait(), timeout=5)
                                task.cancel()
                                with self.assertRaises(asyncio.CancelledError):
                                    await task

                    asyncio.run(run())
                    self.assertEqual(closed, [] if failure == "render" else [True])
                    values = [
                        c.args[1]
                        for c in sink.report.call_args_list
                        if c.args[0] == GaugeMetrics.FRONTEND_CONCURRENCY_METRIC
                    ]
                    self.assertEqual(values, [1, 0])
                    self.assertEqual(
                        sum(
                            c.args[0] == GaugeMetrics.FRONTEND_REQUEST_RT_MS_METRIC
                            for c in sink.report.call_args_list
                        ),
                        1,
                    )
                    self.assertEqual(
                        self.frontend_server._request_metrics._active_requests, 0
                    )
                    self.assertEqual(
                        self.frontend_server._global_controller.current_concurrency.value,
                        0,
                    )

    def test_native_batch_atomic_rpc_errors_do_not_count_partial_success(self):
        from rtp_llm.config.exceptions import FtRuntimeException
        from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
            BatchGenerateOutputsPB,
            ErrorCodePB,
        )

        for api in ("pipeline", "openai"):
            for failure in ("cardinality", "partial"):
                with self.subTest(api=api, failure=failure):
                    call, visitor, sink, _, _ = self._native_batch_fixture(api)
                    client = ModelRpcClient.__new__(ModelRpcClient)
                    client._addresses = ["backend:1234"]
                    client._max_rpc_timeout_ms = 1000
                    client._channel_pool = SimpleNamespace(get=AsyncMock())
                    visitor.model_rpc_client = client
                    response = BatchGenerateOutputsPB()
                    if failure == "partial":
                        output = response.results.add().final_output
                        output.frontend_input_len.value = 10
                        output.frontend_output_len.value = 4
                        output.frontend_generate_token_num.value = 8
                        failed = response.results.add().error_info
                        failed.error_code = ErrorCodePB.INVALID_PARAMS
                        failed.error_message = "invalid batch item"
                    stub = SimpleNamespace(
                        BatchGenerateCall=AsyncMock(return_value=response)
                    )
                    with patch(
                        "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
                        return_value=stub,
                    ), self.assertRaises(FtRuntimeException):
                        asyncio.run(call())
                    stub.BatchGenerateCall.assert_awaited_once()
                    reported = [c.args[0] for c in sink.report.call_args_list]
                    self.assertEqual(
                        reported.count(GaugeMetrics.FRONTEND_REQUEST_RT_MS_METRIC), 1
                    )
                    self.assertNotIn(
                        GaugeMetrics.FRONTEND_INPUT_LENGTH_METRIC, reported
                    )
                    self.assertNotIn(
                        GaugeMetrics.FRONTEND_OUTPUT_LENGTH_METRIC, reported
                    )
                    self.assertEqual(
                        self.frontend_server._request_metrics._active_requests, 0
                    )
                    self.assertEqual(
                        self.frontend_server._global_controller.current_concurrency.value,
                        0,
                    )

    def test_malformed_request_preserves_error_metric_priority(self):
        with patch("rtp_llm.frontend.frontend_server.kmonitor.report") as report:
            response = asyncio.run(
                self._async_run(
                    req="not json",
                    raw_request=FakeRawRequest({"x-dashscope-inner-qos-level": "80"}),
                )
            )
        self.assertEqual(response.status_code, 500)
        calls = [
            call
            for call in report.call_args_list
            if call.args[0] == AccMetrics.ERROR_QPS_METRIC
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[2]["priority"], "80")

    def test_closing_report_wrapper_closes_backend_and_finishes_metrics_once(self):
        closed = []

        async def backend():
            try:
                yield FakePipelinResponse(res="first")
                yield FakePipelinResponse(res="second")
            finally:
                closed.append(True)

        async def run():
            response = CompleteResponseAsyncGenerator(
                backend(), CompleteResponseAsyncGenerator.get_last_value
            )
            metrics = self.frontend_server._request_metrics.begin(
                rank_id="0", server_id="0", source="test", streaming=True
            )
            wrapped = await self.frontend_server._call_generate_with_report(
                lambda _: response,
                True,
                request_metrics=metrics,
            )
            await wrapped.__anext__()
            await wrapped.aclose()
            await wrapped.aclose()
            self.assertEqual(self.frontend_server._request_metrics._active_requests, 0)

        asyncio.run(run())
        self.assertEqual(closed, [True])

    def test_streaming_asgi_lifecycle_finishes_metrics_once(self):
        # Exercise the actual response, including disconnect before the first
        # body iteration and failure while sending an already yielded chunk.
        for mode in ("disconnect_headers", "cancel_headers", "send_failure", "success"):
            with self.subTest(mode=mode):
                metrics = self.frontend_server._request_metrics

                async def run():
                    header_started = asyncio.Event()
                    hold_headers = asyncio.Event()
                    closed = []

                    async def backend():
                        try:
                            yield FakePipelinResponse(res="first")
                            yield FakePipelinResponse(res="second")
                        finally:
                            # Cleanup must survive an enclosing cancellation scope.
                            await asyncio.sleep(0)
                            closed.append(True)

                    def generate(_):
                        return CompleteResponseAsyncGenerator(
                            backend(), CompleteResponseAsyncGenerator.get_last_value
                        )

                    self.frontend_server._global_controller.increment()
                    response = await self.frontend_server._infer_impl(
                        {"prompt": "test", request_id_field_name: 1},
                        FakeRawRequest(),
                        generate,
                    )
                    self.assertEqual(metrics._active_requests, 1)

                    async def receive():
                        await header_started.wait()
                        if mode == "disconnect_headers":
                            return {"type": "http.disconnect"}
                        await asyncio.Event().wait()

                    async def send(message):
                        if message["type"] == "http.response.start":
                            header_started.set()
                            if mode == "cancel_headers":
                                cancel_scope.cancel()
                            if mode in ("disconnect_headers", "cancel_headers"):
                                await hold_headers.wait()
                        elif mode == "send_failure":
                            raise OSError("client disconnected")

                    with anyio.CancelScope() as cancel_scope:
                        try:
                            await response(
                                {"type": "http", "asgi": {"spec_version": "2.3"}},
                                receive,
                                send,
                            )
                        except Exception:
                            if mode != "send_failure":
                                raise
                    self.assertEqual(metrics._active_requests, 0)
                    self.assertEqual(
                        self.frontend_server._global_controller.current_concurrency.value,
                        0,
                    )
                    self.assertEqual(
                        closed, [True] if mode in ("success", "send_failure") else []
                    )

                with patch.object(
                    self.frontend_server._frontend_worker,
                    "is_streaming",
                    return_value=True,
                ), patch(
                    "rtp_llm.metrics.kmonitor_metric_reporter.MetricReporter.is_reporting_enabled",
                    new_callable=PropertyMock,
                    return_value=True,
                ), patch(
                    "rtp_llm.frontend.frontend_server.kmonitor.report"
                ) as report:
                    asyncio.run(run())
                rt_calls = [
                    c
                    for c in report.call_args_list
                    if c.args[0] == GaugeMetrics.FRONTEND_REQUEST_RT_MS_METRIC
                ]
                self.assertEqual(len(rt_calls), 1)

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
        close_order = []
        original_worker_close = self.frontend_server._frontend_worker.close
        original_metrics_close = self.frontend_server._request_metrics.close

        async def worker_close():
            close_order.append("worker")
            await original_worker_close()

        def metrics_close():
            close_order.append("metrics")
            original_metrics_close()

        self.frontend_server._frontend_worker.close = worker_close
        self.frontend_server._request_metrics.close = metrics_close
        asyncio.run(self.frontend_server.close())

        self.assertTrue(self.frontend_server._frontend_worker.close_called)
        self.assertEqual(close_order, ["worker", "metrics"])

    def test_internal_aux_info_is_forced_but_can_be_hidden_from_response(self):
        request = {"generate_config": {"aux_info": False}}
        self.assertFalse(FrontendServer._request_aux_info_enabled(request))

        FrontendServer._force_internal_aux_info(request)
        self.assertTrue(request["generate_config"]["aux_info"])

        response = FakeAuxResponse(aux_info={"input_len": 10})
        FrontendServer._hide_aux_info(response)
        self.assertEqual(response.aux_info, {})

    def test_error_response_preserves_existing_aux_info_contract(self):
        error = RuntimeError("backend failed")
        error.aux_info = {"input_len": 10}

        response = self.frontend_server._handle_exception(
            {
                "aux_info": False,
                "source": "test",
                request_id_field_name: 1,
            },
            error,
        )

        payload = json.loads(response.body)
        self.assertEqual(payload["aux_info"], {"input_len": 10})

    def test_stream_error_response_preserves_existing_aux_info_contract(self):
        async def failing_stream():
            error = RuntimeError("stream backend failed")
            error.aux_info = {"input_len": 10}
            raise error
            yield FakePipelinResponse(res="unreachable")

        async def collect_error_chunks():
            self.frontend_server._global_controller.increment()
            response = CompleteResponseAsyncGenerator(
                failing_stream(), CompleteResponseAsyncGenerator.get_last_value
            )
            return [
                chunk
                async for chunk in self.frontend_server.stream_response(
                    {
                        "stream": True,
                        "aux_info": False,
                        "source": "test",
                        request_id_field_name: 1,
                    },
                    response,
                )
            ]

        chunks = asyncio.run(collect_error_chunks())
        self.assertEqual(len(chunks), 1)
        payload = json.loads(chunks[0].removeprefix("data: ").strip())
        self.assertEqual(payload["aux_info"], {"input_len": 10})
        self.assertEqual(payload["message"], "UNKNOWN_ERROR")
        self.assertNotIn("stream backend failed", chunks[0])

    def test_client_exception_payload_never_reuses_internal_error_text(self):
        for error in (
            RuntimeError("secret/path.py:42"),
            FtRuntimeException(
                ExceptionType.EXECUTION_EXCEPTION, "Traceback: secret/path.py:42"
            ),
            FtRuntimeException(ExceptionType.INVALID_PARAMS, "secret/path.py:42"),
            asyncio.CancelledError("secret/path.py:42"),
        ):
            with self.subTest(error=type(error).__name__):
                error.aux_info = {"input_len": 10}
                payload = format_client_exception(error)
                detailed = format_exception(error)
                self.assertEqual(payload["error_code"], detailed["error_code"])
                self.assertEqual(payload["error_code_str"], detailed["error_code_str"])
                self.assertEqual(payload["aux_info"], {"input_len": 10})
                self.assertNotIn("secret", json.dumps(payload))
                self.assertIn("secret", detailed["message"])

    def test_http_error_keeps_traceback_only_in_access_log(self):
        try:
            raise RuntimeError("secret exception detail")
        except RuntimeError as error:
            with patch.object(
                self.frontend_server._access_logger, "log_exception_access"
            ) as log:
                response = self.frontend_server._handle_exception(
                    {request_id_field_name: 1}, error
                )
        self.assertNotIn("secret", response.body.decode())
        self.assertNotIn("Traceback", response.body.decode())
        self.assertIn("Traceback", log.call_args.args[2]["message"])
        self.assertIn("secret", log.call_args.args[2]["message"])

    def test_aux_info_intent_matches_pydantic_boolean_coercion(self):
        for value in (False, 0, "false", "0", None):
            with self.subTest(value=value):
                self.assertFalse(
                    FrontendServer._request_aux_info_enabled({"aux_info": value})
                )
        for value in (True, 1, "true", "1"):
            with self.subTest(value=value):
                self.assertTrue(
                    FrontendServer._request_aux_info_enabled({"aux_info": value})
                )

    def test_internal_aux_info_copy_does_not_deepcopy_payload(self):
        images = [object()]
        request = {
            "images": images,
            "generate_config": {"aux_info": False, "top_k": 1},
        }

        copied = FrontendServer._copy_for_internal_aux_info(request)

        self.assertIs(copied["images"], images)
        self.assertIsNot(copied["generate_config"], request["generate_config"])
        self.assertFalse(request["generate_config"]["aux_info"])
        self.assertTrue(copied["generate_config"]["aux_info"])

    def test_request_can_disable_speculative_metrics(self):
        self.assertTrue(
            FrontendServer._request_disables_speculative(
                {"generate_config": {"force_disable_sp_run": True}}
            )
        )

    def test_prompt_outputs_force_internal_request_nonstreaming(self):
        captured = {}

        class FakeOpenaiEndpoint:
            model_name = "test-model"

            async def chat_completion_async(
                self, request_id, request, raw_request, **kwargs
            ):
                captured["internal_request"] = request
                captured["metric_tags"] = kwargs["frontend_metric_tags"]

                async def response_generator():
                    if False:
                        yield None

                return CompleteResponseAsyncGenerator(
                    response_generator(),
                    CompleteResponseAsyncGenerator.get_last_value,
                )

        async def fake_infer_wrap(request_dict, raw_request, generate_call):
            captured["request_dict"] = request_dict
            captured["response"] = await generate_call(
                SimpleNamespace(observe_tps=lambda response: None)
            )
            return ORJSONResponse({"ok": True})

        self.frontend_server._openai_endpoint = FakeOpenaiEndpoint()
        self.frontend_server._infer_wrap = fake_infer_wrap
        for request in (
            ChatCompletionRequest(messages=[], stream=True, prompt_logprobs=5),
            ChatCompletionRequest(
                messages=[],
                stream=True,
                extra_configs={"return_prompt_logits": True},
            ),
        ):
            with self.subTest(request=request):
                asyncio.run(
                    self.frontend_server.chat_completion(
                        request,
                        FakeRawRequest({"X-DashScope-Inner-Qos-Level": "70"}),
                    )
                )

                self.assertFalse(request.stream)
                self.assertFalse(captured["internal_request"].stream)
                self.assertFalse(captured["request_dict"]["stream"])
                self.assertEqual(captured["metric_tags"]["priority"], "70")

    def test_hidden_internal_aux_info_does_not_change_iter_latency_metric(self):
        async def response_generator():
            yield FakeAuxResponse(aux_info={"step_output_len": 1})
            yield FakeAuxResponse(aux_info={"step_output_len": 4})

        response = CompleteResponseAsyncGenerator(
            response_generator(),
            CompleteResponseAsyncGenerator.get_last_value,
        )
        request_metrics = MagicMock()

        async def collect():
            wrapped = await self.frontend_server._call_generate_with_report(
                lambda _: response,
                True,
                request_metrics=request_metrics,
                expose_aux_info=False,
            )
            return [item async for item in wrapped]

        with patch(
            "rtp_llm.frontend.frontend_server.current_time_ms",
            side_effect=[0, 0, 10, 30, 30],
        ), patch("rtp_llm.frontend.frontend_server.kmonitor.report") as report:
            outputs = asyncio.run(collect())

        iter_values = [
            call.args[1]
            for call in report.call_args_list
            if call.args and call.args[0] == GaugeMetrics.RESPONSE_ITER_RT_METRIC
        ]
        self.assertEqual(iter_values, [20])
        self.assertEqual([output.aux_info for output in outputs], [{}, {}])
        request_metrics.finish.assert_called_once_with()

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
