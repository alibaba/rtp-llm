import asyncio
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import AsyncMock, MagicMock, patch

import grpc

from rtp_llm.config.py_config_modules import EmbeddingConfig
from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint


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


if __name__ == "__main__":
    main()
