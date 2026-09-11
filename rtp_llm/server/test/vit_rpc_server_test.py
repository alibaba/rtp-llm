import unittest
from types import SimpleNamespace

import grpc
import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputPB,
    MultimodalInputsPB,
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


if __name__ == "__main__":
    unittest.main()
