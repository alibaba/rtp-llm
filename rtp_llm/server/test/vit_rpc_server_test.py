from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    MultimodalOutputPB,
    ReleaseEmbeddingPB,
)
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
from rtp_llm.server.vit_rpc_server import MultimodalRpcServer


class _FakeTransport:
    """Stands in for MMOutputTransport. Everything about *how* the output travels is covered by
    multimodal/test/mm_output_transport_test.py; here we only check the server forwards."""

    def __init__(self):
        self.transfer_calls = []
        self.released = []
        self.closed = False
        self.receipt = MultimodalOutputPB(split_size=[1])

    def transfer(self, request, res):
        self.transfer_calls.append((request, res))
        return self.receipt

    def release(self, request):
        self.released.append(request)

    def close(self):
        self.closed = True


class MultimodalRpcServerTest(TestCase):
    def setUp(self):
        self.server = MultimodalRpcServer.__new__(MultimodalRpcServer)
        self.server.engine = MagicMock()
        self.transport = _FakeTransport()
        self.server._transport = self.transport

    @patch("rtp_llm.server.vit_rpc_server.create_mm_output_transport")
    def test_constructor_passes_transport_config_to_factory(self, create_transport):
        transport_config = MagicMock()

        MultimodalRpcServer(MagicMock(), transport_config)

        create_transport.assert_called_once_with(transport_config)

    @patch("rtp_llm.server.vit_rpc_server.kmonitor.report")
    def test_request_and_result_are_handed_to_the_transport_verbatim(self, _report):
        # The server must not interpret support_rdma, pick a data plane, or touch the receipt --
        # it passes the request through so a backend can read whatever field it needs.
        res = MMEmbeddingRes([torch.ones(1, 4)])
        self.server.engine.mm_embedding_rpc.return_value = res
        request = MultimodalInputsPB(support_rdma=True)
        context = MagicMock()
        context.add_callback.return_value = True

        output = self.server.RemoteMultimodalEmbedding(request, context)

        self.assertEqual(len(self.transport.transfer_calls), 1)
        forwarded_request, forwarded_res = self.transport.transfer_calls[0]
        self.assertIs(forwarded_request, request)
        self.assertIs(forwarded_res, res)
        self.assertIs(output, self.transport.receipt)

    def test_release_forwards_the_whole_request(self):
        # The server does not even unpack the handles: it does not know what one means.
        request = ReleaseEmbeddingPB(handle=["one", "two"])

        self.server.ReleaseMultimodalEmbedding(request, MagicMock())

        self.assertEqual(self.transport.released, [request])

    def test_stop_closes_the_transport(self):
        self.server.stop()

        self.server.engine.stop.assert_called_once()
        self.assertTrue(self.transport.closed)


if __name__ == "__main__":
    main()
