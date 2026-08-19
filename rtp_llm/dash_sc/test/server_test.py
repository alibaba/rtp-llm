import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

import grpc

from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.dash_sc.proxy.servicer import DashScProxyServicer
from rtp_llm.dash_sc.server import _merge_server_options
from rtp_llm.server.server_args.grpc_group_args import (
    DEFAULT_DASH_SC_GRPC_MAX_MESSAGE_BYTES,
    default_dash_sc_grpc_config_json,
)


class DashScGrpcMessageLimitTest(TestCase):
    def test_server_allows_large_messages_by_default(self) -> None:
        options = dict(_merge_server_options([]))

        self.assertEqual(
            options["grpc.max_send_message_length"],
            DEFAULT_DASH_SC_GRPC_MAX_MESSAGE_BYTES,
        )
        self.assertEqual(
            options["grpc.max_receive_message_length"],
            DEFAULT_DASH_SC_GRPC_MAX_MESSAGE_BYTES,
        )

    def test_explicit_server_config_overrides_default(self) -> None:
        options = dict(
            _merge_server_options(
                [("grpc.max_receive_message_length", 16 * 1024 * 1024)]
            )
        )

        self.assertEqual(options["grpc.max_receive_message_length"], 16 * 1024 * 1024)

    def test_default_config_is_symmetric(self) -> None:
        config = json.loads(default_dash_sc_grpc_config_json())

        self.assertEqual(
            config["client_config"]["grpc.max_send_message_length"],
            DEFAULT_DASH_SC_GRPC_MAX_MESSAGE_BYTES,
        )
        self.assertEqual(
            config["server_config"]["grpc.max_send_message_length"],
            DEFAULT_DASH_SC_GRPC_MAX_MESSAGE_BYTES,
        )
        self.assertEqual(
            config["server_config"]["grpc.max_receive_message_length"],
            DEFAULT_DASH_SC_GRPC_MAX_MESSAGE_BYTES,
        )


class _LargeMessageEchoServicer(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self, payload_bytes: int):
        self._payload_bytes = payload_bytes

    async def ModelStreamInfer(self, request_iterator, context):
        async for request in request_iterator:
            if request.ByteSize() <= 4 * 1024 * 1024:
                await context.abort(
                    grpc.StatusCode.INTERNAL, "test request did not exceed 4 MiB"
                )
            response = predict_v2_pb2.ModelStreamInferResponse()
            response.infer_response.parameters["blob"].string_param = (
                "y" * self._payload_bytes
            )
            yield response


class DashScGrpcProxyLargeMessageTest(IsolatedAsyncioTestCase):
    async def test_eight_mib_request_and_response_cross_proxy(self) -> None:
        payload_bytes = 8 * 1024 * 1024
        backend_server = grpc.aio.server(options=_merge_server_options([]))
        predict_v2_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
            _LargeMessageEchoServicer(payload_bytes), backend_server
        )
        backend_port = backend_server.add_insecure_port("127.0.0.1:0")
        await backend_server.start()

        discovery = SimpleNamespace(
            resolve=lambda: SimpleNamespace(grpc_target=f"127.0.0.1:{backend_port}")
        )
        with patch(
            "rtp_llm.dash_sc.proxy.servicer.create_service_discovery_from_env",
            return_value=discovery,
        ):
            proxy_servicer = DashScProxyServicer()

        proxy_server = grpc.aio.server(options=_merge_server_options([]))
        predict_v2_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
            proxy_servicer, proxy_server
        )
        proxy_port = proxy_server.add_insecure_port("127.0.0.1:0")
        await proxy_server.start()

        client_options = [
            (
                "grpc.max_send_message_length",
                DEFAULT_DASH_SC_GRPC_MAX_MESSAGE_BYTES,
            ),
            (
                "grpc.max_receive_message_length",
                DEFAULT_DASH_SC_GRPC_MAX_MESSAGE_BYTES,
            ),
        ]
        channel = grpc.aio.insecure_channel(
            f"127.0.0.1:{proxy_port}", options=client_options
        )
        try:
            stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
            request = predict_v2_pb2.ModelInferRequest(id="large-message")
            request.parameters["payload"].string_param = "x" * payload_bytes

            async def requests():
                yield request

            responses = [
                response async for response in stub.ModelStreamInfer(requests())
            ]
            self.assertEqual(len(responses), 1)
            self.assertEqual(
                len(responses[0].infer_response.parameters["blob"].string_param),
                payload_bytes,
            )
        finally:
            await channel.close()
            await proxy_server.stop(0)
            await proxy_servicer.close()
            await backend_server.stop(0)


if __name__ == "__main__":
    main()
