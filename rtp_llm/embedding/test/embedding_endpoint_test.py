import asyncio
import json
import threading
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import numpy as np

from rtp_llm.config.exceptions import (
    EmbeddingInputException,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.py_config_modules import EmbeddingConfig
from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.models.downstream_modules.common_input_generator import (
    CommonInputGenerator,
)
from rtp_llm.models.downstream_modules.reranker.qwen3_reranker import (
    Qwen3RerankerRenderer,
)
from rtp_llm.models.downstream_modules.reranker.reranker_module import RerankerRenderer


class FakeGrpcConfig:
    def get_client_config(self):
        return {"grpc.max_receive_message_length": 1024}


def _make_endpoint(**embedding_args) -> EmbeddingEndpoint:
    embedding_config = EmbeddingConfig()
    for key, value in embedding_args.items():
        setattr(embedding_config, key, value)
    with (
        patch(
            "rtp_llm.embedding.embedding_endpoint.create_custom_module",
            return_value=SimpleNamespace(renderer=MagicMock()),
        ),
        patch("rtp_llm.embedding.embedding_endpoint.HostServiceArgs"),
        patch("rtp_llm.embedding.embedding_endpoint.HostService"),
    ):
        return EmbeddingEndpoint(
            model_config=MagicMock(),
            grpc_config=FakeGrpcConfig(),
            server_config=SimpleNamespace(embedding_rpc_server_port=12345),
            tokenizer=MagicMock(),
            embedding_config=embedding_config,
        )


class EmbeddingEndpointChannelTest(TestCase):
    def test_shared_channel_defaults(self):
        endpoint = _make_endpoint()
        self.assertTrue(endpoint.share_channel)
        # The deadline is opt-in, and without it the call does not wait for ready.
        self.assertIsNone(endpoint.rpc_timeout)
        self.assertFalse(endpoint.wait_for_ready)
        options = dict(endpoint.options)
        self.assertEqual(options["grpc.max_receive_message_length"], 1024)
        self.assertFalse(any("keepalive" in key for key in options))
        self.assertNotIn("grpc.http2.max_pings_without_data", options)
        self.assertEqual(options["grpc.max_reconnect_backoff_ms"], 1000)

    def test_keepalive_is_opt_in(self):
        for share_channel in (True, False):
            with self.subTest(share_channel=share_channel):
                endpoint = _make_endpoint(
                    embedding_share_grpc_channel=share_channel,
                    embedding_grpc_keepalive=True,
                )
                options = dict(endpoint.options)
                self.assertEqual(options["grpc.keepalive_time_ms"], 10000)
                self.assertEqual(options["grpc.keepalive_timeout_ms"], 5000)

    def test_explicit_grpc_options_take_precedence(self):
        with patch.object(
            FakeGrpcConfig,
            "get_client_config",
            return_value={"grpc.keepalive_timeout_ms": 60000},
        ):
            endpoint = _make_endpoint(embedding_grpc_keepalive=True)
        self.assertEqual(dict(endpoint.options)["grpc.keepalive_timeout_ms"], 60000)

    def test_per_request_channel_keeps_legacy_behavior(self):
        endpoint = _make_endpoint(embedding_share_grpc_channel=False)
        self.assertFalse(endpoint.share_channel)
        self.assertIsNone(endpoint.rpc_timeout)
        self.assertFalse(endpoint.wait_for_ready)
        self.assertNotIn("grpc.keepalive_time_ms", dict(endpoint.options))

    def test_rpc_timeout_override(self):
        endpoint = _make_endpoint(embedding_rpc_timeout_s=5.0)
        self.assertEqual(endpoint.rpc_timeout, 5.0)
        self.assertTrue(endpoint.wait_for_ready)
        endpoint = _make_endpoint(
            embedding_share_grpc_channel=False, embedding_rpc_timeout_s=5.0
        )
        self.assertEqual(endpoint.rpc_timeout, 5.0)
        self.assertFalse(endpoint.wait_for_ready)
        self.assertIsNone(_make_endpoint(embedding_rpc_timeout_s=0.0).rpc_timeout)
        self.assertIsNone(_make_endpoint(embedding_rpc_timeout_s=-1.0).rpc_timeout)

    def test_shared_channel_is_reused_per_event_loop(self):
        endpoint = _make_endpoint()

        async def get_twice():
            return endpoint._get_shared_channel(), endpoint._get_shared_channel()

        first, second = asyncio.run(get_twice())
        self.assertIs(first, second)
        third, _ = asyncio.run(get_twice())
        self.assertIsNot(first, third)
        # Channels of closed loops are pruned when a new loop creates its channel.
        self.assertEqual(len(endpoint._channels), 1)

    def test_close_closes_and_clears_shared_channels(self):
        endpoint = _make_endpoint()

        async def use_then_close():
            channel = endpoint._get_shared_channel()
            await endpoint.close()
            return channel

        channel = asyncio.run(use_then_close())
        self.assertEqual(endpoint._channels, {})
        self.assertEqual(
            channel.get_state(try_to_connect=False),
            grpc.ChannelConnectivity.SHUTDOWN,
        )

    def test_close_without_channels_is_noop(self):
        endpoint = _make_endpoint(embedding_share_grpc_channel=False)
        asyncio.run(endpoint.close())
        self.assertEqual(endpoint._channels, {})

    def test_close_preserves_another_live_loops_channel(self):
        endpoint = _make_endpoint()
        first_loop = asyncio.new_event_loop()
        second_loop = asyncio.new_event_loop()
        closed_loop = asyncio.new_event_loop()
        closed_channel = MagicMock(close=AsyncMock())

        async def get_channel():
            return endpoint._get_shared_channel()

        try:
            first_channel = first_loop.run_until_complete(get_channel())
            second_channel = second_loop.run_until_complete(get_channel())
            closed_loop.close()
            endpoint._channels[id(closed_loop)] = (closed_loop, closed_channel)
            first_loop.run_until_complete(endpoint.close())
            self.assertEqual(
                endpoint._channels, {id(second_loop): (second_loop, second_channel)}
            )
            self.assertEqual(
                first_channel.get_state(), grpc.ChannelConnectivity.SHUTDOWN
            )
            self.assertNotEqual(
                second_channel.get_state(), grpc.ChannelConnectivity.SHUTDOWN
            )
            self.assertIs(second_loop.run_until_complete(get_channel()), second_channel)
            second_loop.run_until_complete(endpoint.close())
            self.assertEqual(endpoint._channels, {})
            self.assertEqual(
                second_channel.get_state(), grpc.ChannelConnectivity.SHUTDOWN
            )
            closed_channel.close.assert_not_awaited()
        finally:
            first_loop.run_until_complete(endpoint.close())
            second_loop.run_until_complete(endpoint.close())
            first_loop.close()
            second_loop.close()
            closed_loop.close()


class _CharTokenizer:
    """Maps every character to its code point."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def __call__(self, prompts, max_length, truncation, **kwargs):
        ids = [self.encode(query + doc) for query, doc in prompts]
        if truncation:
            ids = [tokens[:max_length] for tokens in ids]
        return {
            "input_ids": [np.array(tokens, dtype=np.int64) for tokens in ids],
            "length": np.array([len(tokens) for tokens in ids]),
        }


def _make_common_reranker_renderer(max_seq_len: int) -> RerankerRenderer:
    # The bge/BERT reranker path: RerankerRenderer + CommonInputGenerator.
    generator = CommonInputGenerator.__new__(CommonInputGenerator)
    generator.tokenizer_ = _CharTokenizer()
    generator.config_ = SimpleNamespace(max_seq_len=max_seq_len, position_ids_style=0)
    generator.lock = threading.Lock()
    renderer = RerankerRenderer.__new__(RerankerRenderer)
    renderer.generator = generator
    return renderer


def _make_qwen3_raw_renderer(max_seq_len: int) -> Qwen3RerankerRenderer:
    renderer = Qwen3RerankerRenderer.__new__(Qwen3RerankerRenderer)
    renderer.config_ = SimpleNamespace(
        max_seq_len=max_seq_len, qwen3_reranker_input_format="raw"
    )
    renderer.tokenizer_ = _CharTokenizer()
    renderer.lock = threading.Lock()
    return renderer


class EmbeddingEndpointErrorTest(TestCase):
    def setUp(self):
        self.endpoint = _make_endpoint()
        self.endpoint.generate_embeddings = AsyncMock()
        self.server = FrontendServer.__new__(FrontendServer)
        self.server.rank_id = 0
        self.server.server_id = 0
        self.server.py_env_configs = SimpleNamespace(
            server_config=SimpleNamespace(ip="127.0.0.1", server_port=12345)
        )
        self.server._embedding_endpoint = self.endpoint
        self.server._global_controller = MagicMock()
        self.server._global_controller.increment.return_value = 1
        self.server._access_logger = MagicMock()

    def _request(self, renderer, request):
        self.endpoint.renderer = renderer
        self.server._global_controller.reset_mock()
        response = asyncio.run(self.server.embedding(request, MagicMock()))
        self.server._global_controller.decrement.assert_called_once_with()
        self.endpoint.generate_embeddings.assert_not_awaited()
        return response.status_code, json.loads(response.body)

    def _assert_legacy_format_error(self, status_code, body, message):
        # Legacy contract: input failures are wrapped as 507 and returned as 500.
        self.assertEqual(status_code, 500)
        self.assertEqual(
            body["error_code"], int(ExceptionType.ERROR_INPUT_FORMAT_ERROR)
        )
        self.assertIn(message, body["message"])

    def test_qwen3_raw_input_errors_return_400(self):
        renderer = _make_qwen3_raw_renderer(max_seq_len=4)
        cases = (
            ({"query": "q", "documents": []}, ExceptionType.EMPTY_PROMPT_ERROR),
            ({"query": "", "documents": ["a", ""]}, ExceptionType.EMPTY_PROMPT_ERROR),
            (
                {"query": "qq", "documents": ["abcdef"], "truncation": False},
                ExceptionType.LONG_PROMPT_ERROR,
            ),
        )
        for request, error_type in cases:
            with self.subTest(request=request):
                status_code, body = self._request(renderer, request)
                self.assertEqual(status_code, 400)
                self.assertEqual(body["error_code"], int(error_type))
                self.assertEqual(
                    body["error_code_str"], f"{int(error_type)}_{error_type.name}"
                )

    def test_common_reranker_long_prompt_keeps_legacy_507_500(self):
        renderer = _make_common_reranker_renderer(max_seq_len=4)
        status_code, body = self._request(
            renderer, {"query": "qq", "documents": ["abcdef"], "truncation": False}
        )
        self._assert_legacy_format_error(status_code, body, "max_length: 4")

    def test_common_reranker_bad_request_keeps_legacy_507_500(self):
        renderer = _make_common_reranker_renderer(max_seq_len=4)
        status_code, body = self._request(renderer, {"query": "q"})
        self._assert_legacy_format_error(status_code, body, "documents")

    def test_typed_and_untyped_input_errors_keep_legacy_507_500(self):
        for error in (
            FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR, "too long"),
            FtRuntimeException(ExceptionType.EMPTY_PROMPT_ERROR, "empty"),
            ValueError("invalid format"),
        ):
            with self.subTest(error=repr(error)):
                renderer = MagicMock()
                renderer.create_input.side_effect = error
                status_code, body = self._request(renderer, {"query": "q"})
                self._assert_legacy_format_error(status_code, body, str(error))

    def test_execution_error_keeps_legacy_response(self):
        self.endpoint.generate_embeddings.side_effect = RuntimeError("backend failed")
        response = asyncio.run(self.server.embedding({"query": "q"}, MagicMock()))
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            json.loads(response.body)["error_code"],
            int(ExceptionType.EXECUTION_EXCEPTION),
        )
        self.server._global_controller.decrement.assert_called_once_with()

    def test_other_endpoints_keep_legacy_error_status(self):
        response = self.server._handle_exception(
            {},
            EmbeddingInputException(ExceptionType.LONG_PROMPT_ERROR, "too long"),
        )
        self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    main()
