import asyncio
import json
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import PropertyMock, patch

import anyio
from fastapi.testclient import TestClient

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import ThinkingMode
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.metrics import AccMetrics, kmonitor
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    FinisheReason,
    FunctionCall,
    RoleEnum,
    ToolCall,
    UsageInfo,
)
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    init_controller,
    set_global_controller,
)


class _DummyBackendVisitor:
    def is_backend_service_ready(self, refresh: bool = True) -> bool:
        return True


class _FakeFrontendWorker:
    def __init__(self):
        self.backend_rpc_server_visitor = _DummyBackendVisitor()

    def is_streaming(self, req):
        return bool(req.get("stream", False))

    def stop(self):
        pass


class _FakeOpenaiEndpoint:
    model_name = "test-dual-model"

    def __init__(self):
        self.mode = "ok"
        self.last_requests = []
        self.closed_streams = 0

    async def render_chat_async(self, chat_request: ChatCompletionRequest):
        self.last_requests.append(("render", chat_request))
        if self.mode == "render_bad_request":
            raise FtRuntimeException(ExceptionType.INVALID_PARAMS, "bad render")
        if self.mode == "render_internal_error":
            raise RuntimeError("secret render failure /internal/path")
        token_ids = list(range(100, 100 + 7 * len(chat_request.messages)))
        return SimpleNamespace(input_ids=token_ids)

    async def chat_completion_async(
        self,
        request_id: int,
        chat_request: ChatCompletionRequest,
        raw_request,
        frontend_metric_tags=None,
        frontend_metric_observer=None,
    ) -> CompleteResponseAsyncGenerator:
        self.last_requests.append(("chat", chat_request))
        mode = self.mode

        if mode == "pre_bad_request":
            raise FtRuntimeException(ExceptionType.INVALID_PARAMS, "bad render param")
        if mode == "pre_timeout":
            raise FtRuntimeException(
                ExceptionType.GENERATE_TIMEOUT, "secret 10.0.0.9 timeout"
            )
        if mode == "pre_capacity":
            raise FtRuntimeException(
                ExceptionType.RESOURCE_EXHAUSTED, "secret pool full"
            )
        if mode == "pre_internal_error":
            raise RuntimeError("secret crash /mnt/nas/internal")

        endpoint = self

        async def stream_gen():
            try:
                if mode == "slow":
                    await asyncio.sleep(5)
                    return
                if mode == "pre_first_chunk_error":
                    raise ValueError("secret error before first chunk")
                yield ChatCompletionStreamResponse(
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(role=RoleEnum.assistant, content=""),
                        )
                    ],
                    usage=None,
                )
                if mode == "mid_stream_error":
                    yield ChatCompletionStreamResponse(
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta=DeltaMessage(content="partial"),
                            )
                        ],
                        usage=UsageInfo(prompt_tokens=12, completion_tokens=2),
                    )
                    raise RuntimeError("backend failed mid-stream")
                if mode == "stop_sequence":
                    yield ChatCompletionStreamResponse(
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta=DeltaMessage(content="Answer"),
                                finish_reason=FinisheReason.stop,
                                matched_stop_sequence="<STOP>",
                            )
                        ],
                        usage=UsageInfo(prompt_tokens=10, completion_tokens=2),
                    )
                elif mode == "tool":
                    yield ChatCompletionStreamResponse(
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta=DeltaMessage(reasoning_content="Plan step."),
                            )
                        ],
                        usage=UsageInfo(prompt_tokens=15, completion_tokens=3),
                    )
                    yield ChatCompletionStreamResponse(
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta=DeltaMessage(
                                    tool_calls=[
                                        ToolCall(
                                            index=0,
                                            id="toolu_call_1",
                                            type="function",
                                            function=FunctionCall(
                                                name="get_weather",
                                                arguments='{"city": "Paris"}',
                                            ),
                                        )
                                    ]
                                ),
                                finish_reason=FinisheReason.tool_calls,
                            )
                        ],
                        usage=UsageInfo(prompt_tokens=15, completion_tokens=9),
                    )
                else:
                    yield ChatCompletionStreamResponse(
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta=DeltaMessage(content="Hello "),
                            )
                        ],
                        usage=UsageInfo(prompt_tokens=12, completion_tokens=2),
                    )
                    yield ChatCompletionStreamResponse(
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta=DeltaMessage(content="world!"),
                                finish_reason=FinisheReason.stop,
                            )
                        ],
                        usage=UsageInfo(prompt_tokens=12, completion_tokens=5),
                    )
            finally:
                endpoint.closed_streams += 1

        async def collect(gen):
            if mode == "collect_error":
                raise ValueError("secret collect decode failure")
            content = ""
            reasoning = None
            tool_calls = None
            finish_reason = None
            matched_stop_sequence = None
            usage = UsageInfo(prompt_tokens=0, completion_tokens=0)
            async for chunk in gen:
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content += delta.content
                    if delta.reasoning_content:
                        reasoning = (reasoning or "") + delta.reasoning_content
                    if delta.tool_calls:
                        tool_calls = delta.tool_calls
                    if chunk.choices[0].finish_reason is not None:
                        finish_reason = chunk.choices[0].finish_reason
                    if chunk.choices[0].matched_stop_sequence is not None:
                        matched_stop_sequence = chunk.choices[0].matched_stop_sequence
            return ChatCompletionResponse(
                model=self.model_name,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(
                            role=RoleEnum.assistant,
                            content=content or None,
                            reasoning_content=reasoning,
                            tool_calls=tool_calls,
                        ),
                        finish_reason=finish_reason,
                        matched_stop_sequence=matched_stop_sequence,
                    )
                ],
                usage=usage,
            )

        return CompleteResponseAsyncGenerator(stream_gen(), collect)


def _parse_anthropic_sse(raw_text: str):
    events = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = [line for line in block.split("\n") if line]
        ev = lines[0][len("event: ") :]
        data = json.loads(lines[1][len("data: ") :])
        events.append((ev, data))
    return events


def _parse_openai_sse(raw_text: str):
    chunks = []
    for block in raw_text.strip().split("\r\n\r\n"):
        if block.startswith("data: "):
            chunks.append(json.loads(block[len("data: ") :]))
    return chunks


class AnthropicFrontendAppTest(TestCase):
    def _build_app(self, *, frontend_metrics_enable: bool = False):
        kmonitor.init()
        py_env_configs = PyEnvConfigs()
        py_env_configs.server_config.start_port = 0
        py_env_configs.server_config.rank_id = 0
        py_env_configs.server_config.frontend_metrics_enable = frontend_metrics_enable
        set_global_controller(init_controller(py_env_configs.concurrency_config))

        frontend_app = FrontendApp(
            py_env_configs=py_env_configs,
            separated_frontend=True,
        )
        server = frontend_app.frontend_server
        server._frontend_worker = _FakeFrontendWorker()
        fake_endpoint = _FakeOpenaiEndpoint()
        server._openai_endpoint = fake_endpoint
        app = frontend_app.create_app()
        return frontend_app, server, fake_endpoint, app

    def setUp(self):
        super().setUp()
        self.frontend_app, self.server, self.fake_endpoint, app = self._build_app(
            frontend_metrics_enable=False
        )
        self.client = TestClient(app)

    def test_dual_protocol_openai_and_anthropic_end_to_end(self):
        # 1. OpenAI non-stream and streaming both work cleanly on the same app
        oai_resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "test-dual-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(oai_resp.status_code, 200)
        self.assertEqual(
            oai_resp.json()["choices"][0]["message"]["content"], "Hello world!"
        )

        oai_stream = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "test-dual-model",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(oai_stream.status_code, 200)
        oai_chunks = _parse_openai_sse(oai_stream.text)
        self.assertTrue(len(oai_chunks) >= 2)

        # 2. Anthropic non-stream works and explicitly preserves stop_sequence: null on the wire
        ant_resp = self.client.post(
            "/v1/messages",
            headers={
                "anthropic-version": "2023-06-01",
                "x-api-key": "test-key",
                "anthropic-beta": "interleaved-thinking-2025-05-14",
            },
            json={
                "model": "claude-sonnet",
                "max_tokens": 128,
                "system": "Be helpful.",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(ant_resp.status_code, 200)
        ant_body = ant_resp.json()
        self.assertIn("stop_sequence", ant_body)
        self.assertIsNone(ant_body["stop_sequence"])
        self.assertEqual(ant_body["stop_reason"], "end_turn")
        self.assertEqual(
            ant_body["content"], [{"type": "text", "text": "Hello world!"}]
        )
        self.assertEqual(ant_body["usage"]["input_tokens"], 12)
        self.assertEqual(ant_body["usage"]["output_tokens"], 5)

        kind, recorded_chat = self.fake_endpoint.last_requests[-1]
        self.assertEqual(kind, "chat")
        self.assertEqual(recorded_chat.max_tokens, 128)
        self.assertEqual(recorded_chat.messages[0].role, RoleEnum.system)

        # 3. Stop sequences work end-to-end for BOTH non-streaming and streaming without DebugInfo
        self.fake_endpoint.mode = "stop_sequence"
        stop_sync = self.client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "max_tokens": 32,
                "stop_sequences": ["<STOP>"],
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(stop_sync.status_code, 200)
        self.assertEqual(stop_sync.json()["stop_reason"], "stop_sequence")
        self.assertEqual(stop_sync.json()["stop_sequence"], "<STOP>")

        stop_stream = self.client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "max_tokens": 32,
                "stream": True,
                "stop_sequences": ["<STOP>"],
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(stop_stream.status_code, 200)
        stop_events = _parse_anthropic_sse(stop_stream.text)
        self.assertEqual(
            stop_events[-2][1]["delta"],
            {"stop_reason": "stop_sequence", "stop_sequence": "<STOP>"},
        )

        # 4. Anthropic streaming SSE with thinking + tools
        self.fake_endpoint.mode = "tool"
        stream_resp = self.client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet",
                "max_tokens": 256,
                "stream": True,
                "thinking": {"type": "enabled", "budget_tokens": 64},
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {"type": "object"},
                    }
                ],
                "messages": [{"role": "user", "content": "Paris weather?"}],
            },
        )
        self.assertEqual(stream_resp.status_code, 200)
        sse_events = _parse_anthropic_sse(stream_resp.text)
        self.assertEqual(sse_events[-2][1]["delta"]["stop_reason"], "tool_use")
        self.assertEqual(sse_events[-2][1]["usage"]["output_tokens"], 9)
        _, tool_req = self.fake_endpoint.last_requests[-1]
        self.assertTrue(tool_req.enable_thinking)
        self.assertEqual(tool_req.thinking_budget, 64)
        self.assertEqual(tool_req.chat_template_kwargs, {"enable_thinking": True})

        # 5. /v1/messages/count_tokens works with adaptive thinking
        self.fake_endpoint.mode = "ok"
        count_resp = self.client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "claude-sonnet",
                "system": "System instructions",
                "thinking": {"type": "adaptive"},
                "messages": [{"role": "user", "content": "How many tokens?"}],
            },
        )
        self.assertEqual(count_resp.status_code, 200)
        self.assertEqual(count_resp.json(), {"input_tokens": 14})
        count_kind, count_req = self.fake_endpoint.last_requests[-1]
        self.assertEqual(count_kind, "render")
        self.assertEqual(count_req.extra_configs.thinking_mode, ThinkingMode.ADAPTIVE)
        self.assertEqual(
            self.server._global_controller.get_available_concurrency(),
            self.server._global_controller.max_concurrency,
        )

    def test_anthropic_error_matrix_and_sanitization_and_openai_unchanged(self):
        before = self.server._global_controller.get_available_concurrency()

        # Validation 422 / malformed JSON mapped to 400 on /v1/messages*
        bad_ant = self.client.post(
            "/v1/messages",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(bad_ant.status_code, 400)
        self.assertEqual(bad_ant.json()["error"]["type"], "invalid_request_error")

        malformed_json = self.client.post(
            "/v1/messages",
            content=b"{not-valid-json",
            headers={"content-type": "application/json"},
        )
        self.assertEqual(malformed_json.status_code, 400)
        self.assertEqual(
            malformed_json.json()["error"]["type"], "invalid_request_error"
        )

        # OpenAI keeps 422
        bad_oai = self.client.post("/v1/chat/completions", json={"model": "m"})
        self.assertEqual(bad_oai.status_code, 422)

        # /v1/messages error status codes & sanitized messages
        for mode, expected_code, expected_type in (
            ("pre_bad_request", 400, "invalid_request_error"),
            ("pre_timeout", 504, "api_error"),
            ("pre_capacity", 529, "overloaded_error"),
            ("pre_internal_error", 500, "api_error"),
            ("collect_error", 500, "api_error"),
        ):
            with self.subTest(mode=mode):
                self.fake_endpoint.mode = mode
                resp = self.client.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 32,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                self.assertEqual(resp.status_code, expected_code)
                self.assertEqual(resp.json()["error"]["type"], expected_type)
                self.assertNotIn("secret", resp.json()["error"]["message"])
                self.assertEqual(
                    self.server._global_controller.get_available_concurrency(), before
                )

        # /v1/messages/count_tokens error paths
        for mode, expected_code, expected_type in (
            ("render_bad_request", 400, "invalid_request_error"),
            ("render_internal_error", 500, "api_error"),
        ):
            with self.subTest(count_mode=mode):
                self.fake_endpoint.mode = mode
                resp = self.client.post(
                    "/v1/messages/count_tokens",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                self.assertEqual(resp.status_code, expected_code)
                self.assertEqual(resp.json()["error"]["type"], expected_type)
                self.assertNotIn("secret", resp.json()["error"]["message"])

        # Mid-stream error & pre-first-chunk error & post-chunk collect_error (must NOT emit message_stop!)
        for stream_mode in (
            "pre_first_chunk_error",
            "mid_stream_error",
            "collect_error",
        ):
            with self.subTest(stream_mode=stream_mode):
                self.fake_endpoint.mode = stream_mode
                err_stream = self.client.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 32,
                        "stream": True,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                self.assertEqual(err_stream.status_code, 200)
                events = _parse_anthropic_sse(err_stream.text)
                self.assertEqual(events[-1][0], "error")
                self.assertEqual(events[-1][1]["error"]["type"], "api_error")
                self.assertNotIn("secret", events[-1][1]["error"]["message"])
                self.assertNotIn("message_stop", [ev for ev, _ in events])
                self.assertEqual(
                    self.server._global_controller.get_available_concurrency(), before
                )

    def test_streaming_and_non_streaming_early_disconnect_release_metrics_and_slots(
        self,
    ):
        frontend_app, server, fake_endpoint, app = self._build_app(
            frontend_metrics_enable=True
        )
        metrics = server._request_metrics
        shutdown = frontend_app.shutdown_manager

        async def run_asgi(body_bytes: bytes, mode: str):
            body_sent = False
            start_seen = asyncio.Event()
            hold_send = asyncio.Event()

            async def receive():
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {
                        "type": "http.request",
                        "body": body_bytes,
                        "more_body": False,
                    }
                if mode in ("disconnect_headers", "disconnect_nonstream"):
                    await start_seen.wait()
                    return {"type": "http.disconnect"}
                await asyncio.Event().wait()

            async def send(message):
                if message["type"] == "http.response.start":
                    start_seen.set()
                    if mode == "cancel_headers":
                        cancel_scope.cancel()
                    if mode in ("disconnect_headers", "cancel_headers"):
                        await hold_send.wait()
                elif message["type"] == "http.response.body" and mode == "send_failure":
                    raise OSError("client socket broken")

            if mode == "disconnect_nonstream":
                fake_endpoint.mode = "slow"
                # Trigger disconnect promptly once inference task awaits backend
                asyncio.get_running_loop().call_later(0.05, start_seen.set)
            else:
                fake_endpoint.mode = "ok"

            with anyio.CancelScope() as cancel_scope:
                try:
                    await app(
                        {
                            "type": "http",
                            "asgi": {"spec_version": "2.3"},
                            "method": "POST",
                            "path": "/v1/messages",
                            "raw_path": b"/v1/messages",
                            "query_string": b"",
                            "headers": [(b"content-type", b"application/json")],
                        },
                        receive,
                        send,
                    )
                except Exception:
                    if mode != "send_failure":
                        raise

            self.assertEqual(metrics._active_requests, 0, msg=mode)
            self.assertEqual(
                server._global_controller.current_concurrency.value, 0, msg=mode
            )
            self.assertEqual(shutdown.active_request_count(), 0, msg=mode)

        stream_body = json.dumps(
            {
                "model": "m",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode("utf-8")
        sync_body = json.dumps(
            {
                "model": "m",
                "max_tokens": 16,
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode("utf-8")

        for mode in (
            "success",
            "send_failure",
            "disconnect_headers",
            "cancel_headers",
        ):
            with self.subTest(mode=mode):
                with patch(
                    "rtp_llm.metrics.kmonitor_metric_reporter.MetricReporter.is_reporting_enabled",
                    new_callable=PropertyMock,
                    return_value=True,
                ):
                    asyncio.run(run_asgi(stream_body, mode))

        with self.subTest(mode="disconnect_nonstream"):
            before_closed = fake_endpoint.closed_streams
            asyncio.run(run_asgi(sync_body, "disconnect_nonstream"))
            self.assertGreaterEqual(fake_endpoint.closed_streams, before_closed + 1)

    def test_draining_and_concurrency_limit_and_qos_priority_tags(self):
        # Concurrency limit exhausted -> 429 overloaded_error (sanitized) & preserves priority tag
        self.server._global_controller = ConcurrencyController(max_concurrency=0)
        with patch("rtp_llm.frontend.frontend_server.kmonitor.report") as mock_report:
            limited = self.client.post(
                "/v1/messages",
                headers={"x-dashscope-inner-qos-level": "100"},
                json={
                    "model": "m",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(limited.status_code, 429)
            self.assertEqual(limited.json()["error"]["type"], "overloaded_error")
            self.assertEqual(
                limited.json()["error"]["message"], "Concurrency limit exceeded"
            )
            conflict_calls = [
                c
                for c in mock_report.call_args_list
                if c.args[0] == AccMetrics.CONFLICT_QPS_METRIC
            ]
            self.assertEqual(len(conflict_calls), 1)
            self.assertIn("priority", conflict_calls[0].args[2])

            limited_count = self.client.post(
                "/v1/messages/count_tokens",
                headers={"x-dashscope-inner-qos-level": "100"},
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(limited_count.status_code, 429)
            self.assertEqual(limited_count.json()["error"]["type"], "overloaded_error")

        limited_oai = self.client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(limited_oai.status_code, 429)

        # Draining -> 503 + Retry-After: 1 on /v1/messages*, legacy JSON on OpenAI
        self.frontend_app.shutdown_manager.start_draining("shutdown_test")
        for endpoint, payload in (
            (
                "/v1/messages",
                {
                    "model": "m",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ),
            (
                "/v1/messages/count_tokens",
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            ),
        ):
            drained_ant = self.client.post(endpoint, json=payload)
            self.assertEqual(drained_ant.status_code, 503)
            self.assertEqual(drained_ant.headers.get("retry-after"), "1")
            self.assertEqual(drained_ant.json()["error"]["type"], "overloaded_error")

        drained_oai = self.client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(drained_oai.status_code, 503)
        self.assertIn("active_requests", drained_oai.json())


if __name__ == "__main__":
    main()
