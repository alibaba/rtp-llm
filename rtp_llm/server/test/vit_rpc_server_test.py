import unittest
from types import SimpleNamespace

import grpc
import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMRdmaDescPB,
    MultimodalInputPB,
    MultimodalInputsPB,
    ReleaseEmbeddingPB,
)
from rtp_llm.ops import get_multimodal_feature_hash
from rtp_llm.server.vit_rpc_server import MultimodalRpcServer, trans_output
from rtp_llm.utils.mm_process_engine import MMEmbeddingRes, MMProcessEngine


class Aborted(Exception):
    pass


class Context:
    def add_callback(self, callback):
        return True

    def time_remaining(self):
        return 5.0

    def set_trailing_metadata(self, metadata):
        self.metadata = dict(metadata)

    def abort(self, code, details):
        self.code = code
        raise Aborted(details)


class FakeRdmaEncoder:
    def __init__(self, descriptors):
        self.descriptors = iter(descriptors)
        self.released = []
        self.exports = 0

    def export_embedding(self, embedding):
        self.exports += 1
        return next(self.descriptors)

    def release(self, handles):
        self.released.extend(handles)


class VitRpcServerTest(unittest.TestCase):
    def test_positionless_images_roundtrip_without_synthetic_positions(self):
        result = MMEmbeddingRes([torch.ones((2, 4)), torch.zeros((3, 4))])
        output = trans_output(result)
        self.assertEqual(len(output.multimodal_outputs), 2)
        for item, length in zip(output.multimodal_outputs, (2, 3)):
            self.assertFalse(item.HasField("multimodal_pos_id"))
            self.assertEqual(list(item.multimodal_embedding.shape), [length, 4])

    def server(self, result):
        model = SimpleNamespace(
            model_config=SimpleNamespace(hidden_size=4, compute_dtype=torch.float32)
        )
        return MultimodalRpcServer(
            SimpleNamespace(
                model=model,
                submit=lambda *args, **kwargs: result,
                _check_request=MMProcessEngine._check_request,
            )
        )

    def request(self):
        return MultimodalInputsPB(
            multimodal_inputs=[MultimodalInputPB(multimodal_url="image")]
        )

    def test_returns_batch_metrics_without_proto_changes(self):
        server = self.server(
            MMEmbeddingRes([torch.ones((2, 4))], max_batch_size=8, gpu_forwards=1)
        )
        context = Context()
        server.RemoteMultimodalEmbedding(self.request(), context)
        self.assertEqual(
            context.metadata, {"vit-max-batch-images": "8", "vit-gpu-forwards": "1"}
        )

    def test_invalid_result_does_not_return_partial_embeddings(self):
        server = self.server(MMEmbeddingRes([torch.ones((2, 3))]))
        context = Context()
        with self.assertRaisesRegex(Aborted, "shape or dtype"):
            server.RemoteMultimodalEmbedding(self.request(), context)
        self.assertEqual(context.code, grpc.StatusCode.INVALID_ARGUMENT)

    def test_metadata_only_uses_feature_ids_without_transferring_features(self):
        features = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        server = self.server(MMEmbeddingRes([features]))
        request = self.request()
        request.metadata_only = True
        result = server.RemoteMultimodalEmbedding(request, Context())
        output = result.multimodal_outputs[0]
        self.assertFalse(output.HasField("multimodal_embedding"))
        self.assertFalse(output.HasField("multimodal_pos_id"))
        self.assertEqual(
            list(output.token_ids), get_multimodal_feature_hash(features).tolist()
        )
        self.assertEqual(server._active, 0)

    def test_rdma_opt_in_falls_back_per_image_and_releases_explicitly(self):
        descriptor = MMRdmaDescPB(handle="first").SerializeToString()
        encoder = FakeRdmaEncoder([descriptor, b""])
        result = MMEmbeddingRes([torch.ones((2, 4)), torch.ones((3, 4))])
        output = trans_output(result, rdma_encoder=encoder)
        self.assertTrue(output.multimodal_outputs[0].HasField("output_rdma"))
        self.assertFalse(output.multimodal_outputs[0].HasField("multimodal_embedding"))
        self.assertTrue(output.multimodal_outputs[1].HasField("multimodal_embedding"))
        server = self.server(result)
        server.rdma_encoder = encoder
        server.ReleaseEmbedding(ReleaseEmbeddingPB(handle=["first"]), Context())
        self.assertEqual(encoder.released, ["first"])

    def test_metadata_and_old_clients_do_not_export_rdma_slots(self):
        encoder = FakeRdmaEncoder([])
        server = self.server(MMEmbeddingRes([torch.ones((2, 4))]))
        server.rdma_encoder = encoder
        request = self.request()
        result = server.RemoteMultimodalEmbedding(request, Context())
        self.assertTrue(result.multimodal_outputs[0].HasField("multimodal_embedding"))
        request.metadata_only = True
        request.support_rdma = True
        server.RemoteMultimodalEmbedding(request, Context())
        self.assertEqual(encoder.exports, 0)

    def test_embedding_limit_keeps_release_handler_available(self):
        encoder = FakeRdmaEncoder([])
        server = self.server(MMEmbeddingRes([torch.ones((2, 4))]))
        server.rdma_encoder = encoder
        server._active = server.max_requests
        context = Context()
        with self.assertRaisesRegex(Aborted, "busy"):
            server.RemoteMultimodalEmbedding(self.request(), context)
        self.assertEqual(context.code, grpc.StatusCode.RESOURCE_EXHAUSTED)
        server.ReleaseEmbedding(ReleaseEmbeddingPB(handle=["done"]), Context())
        self.assertEqual(encoder.released, ["done"])
        self.assertEqual(server._active, server.max_requests)

    def test_expiration_after_export_releases_unpublished_slots(self):
        descriptor = MMRdmaDescPB(handle="first").SerializeToString()
        encoder = FakeRdmaEncoder([descriptor])
        server = self.server(MMEmbeddingRes([torch.ones((2, 4))]))
        server.rdma_encoder = encoder
        checks = iter([False, True])

        def check_request(deadline, cancelled):
            if next(checks):
                raise TimeoutError("expired after export")

        server.engine._check_request = check_request
        request = self.request()
        request.support_rdma = True
        context = Context()
        with self.assertRaisesRegex(Aborted, "expired"):
            server.RemoteMultimodalEmbedding(request, context)
        self.assertEqual(encoder.released, ["first"])
        self.assertEqual(context.code, grpc.StatusCode.DEADLINE_EXCEEDED)
        self.assertEqual(server._active, 0)

    def test_unpublished_slots_are_released_on_serialization_failure(self):
        descriptor = MMRdmaDescPB(handle="first").SerializeToString()
        encoder = FakeRdmaEncoder([descriptor, b"malformed protobuf"])
        with self.assertRaises(Exception):
            trans_output(
                MMEmbeddingRes([torch.ones((2, 4)), torch.ones((3, 4))]),
                rdma_encoder=encoder,
            )
        self.assertEqual(encoder.released, ["first"])


if __name__ == "__main__":
    unittest.main()
