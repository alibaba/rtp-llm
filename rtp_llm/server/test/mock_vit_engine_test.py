import asyncio
import base64
import io
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import grpc
import torch
from PIL import Image

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import MultimodalRpcServiceStub
from rtp_llm.models.multimodal.deepseek_v41_processor import image_token_types
from rtp_llm.server.mock_vit_engine import cpu_feature_hash, make_service
from rtp_llm.server.vit_rpc_server import _create_rpc_server
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor


class LegacyMockVitTest(unittest.TestCase):
    def setUp(self):
        self.service = make_service(hidden_size=8, cache_bytes=0, delay_ms=1)
        self.server, self.executor = _create_rpc_server(self.service, 4)
        port = self.server.add_insecure_port('127.0.0.1:0')
        self.server.start()
        self.port = port
        self.channel = grpc.insecure_channel(f'127.0.0.1:{port}')
        self.stub = MultimodalRpcServiceStub(self.channel)

    def tearDown(self):
        self.channel.close()
        self.server.stop(0).wait()
        self.executor.shutdown(wait=True)

    def image(self):
        data = io.BytesIO()
        Image.new('RGB', (56, 56), (70, 140, 50)).save(data, format='PNG')
        config = pb.MMPreprocessConfigPB(width=-1, height=-1, min_pixels=-1, max_pixels=-1,
            fps=-1, min_frames=-1, max_frames=-1)
        return pb.MultimodalInputsPB(multimodal_inputs=[pb.MultimodalInputPB(
            multimodal_url='data:image/png;base64,' + base64.b64encode(data.getvalue()).decode(),
            multimodal_type=1, mm_preprocess_config=config)])

    def test_mock_hash_backend_does_not_replace_global_real_kernel(self):
        from rtp_llm.server import vit_rpc_server
        original = vit_rpc_server.get_multimodal_feature_hash
        make_service(hidden_size=8)
        self.assertIs(vit_rpc_server.get_multimodal_feature_hash, original)

    def test_metadata_only_then_feature_uses_release_envelope_and_stable_ids(self):
        request = self.image()
        request.metadata_only = True
        first = self.stub.RemoteMultimodalEmbedding(request, timeout=5)
        self.assertEqual(len(first.multimodal_outputs[0].token_ids), 4)
        self.assertFalse(first.multimodal_outputs[0].HasField('multimodal_embedding'))
        self.assertEqual(first, self.stub.RemoteMultimodalEmbedding(request, timeout=5))
        self.assertEqual(self.service.engine.forward_calls, 1)
        request.metadata_only = False
        features = self.stub.RemoteMultimodalEmbedding(request, timeout=5)
        tensor = trans_tensor(features.multimodal_outputs[0].multimodal_embedding)
        self.assertEqual(tuple(tensor.shape), (4, 8))
        self.assertEqual(cpu_feature_hash(tensor).tolist(), list(first.multimodal_outputs[0].token_ids))
        self.assertEqual(self.service.engine.forward_calls, 2)

    def test_release_frontend_metadata_client_uses_actual_old_rpc(self):
        from rtp_llm.config.generate_config import RoleAddr, RoleType
        from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
        from rtp_llm.utils.grpc_host_channel_pool import GrpcHostChannelPool

        async def run():
            visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
            visitor._vit_channel_pool = GrpcHostChannelPool()
            address = RoleAddr(role=RoleType.VIT, ip='127.0.0.1', http_port=self.port - 1, grpc_port=self.port)
            request = pb.GenerateInputPB()
            request.multimodal_inputs.extend(self.image().multimodal_inputs)
            try:
                first = await visitor._get_vit_token_ids(address, request, 2)
                second = await visitor._get_vit_token_ids(address, request, 2)
                self.assertEqual(first, second)
                self.assertEqual(len(first[0]), 4)
            finally:
                await visitor._vit_channel_pool.close()
        asyncio.run(run())

    def test_status_version_advances_and_directory_remains_empty(self):
        first = self.stub.GetWorkerStatus(pb.StatusVersionPB(), timeout=5)
        second = self.stub.GetWorkerStatus(pb.StatusVersionPB(), timeout=5)
        self.assertEqual(first.role, 'VIT')
        self.assertTrue(first.alive)
        self.assertGreater(second.status_version, first.status_version)
        self.assertEqual((first.tp_size, first.dp_size), (1, 1))
        self.assertEqual(self.stub.GetCacheStatus(pb.CacheVersionPB(), timeout=5).ByteSize(), 0)

    def test_admission_cap_and_status_survive_slow_forward(self):
        self.service.max_requests = 1
        self.service.engine.delay_ms = 250
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(self.stub.RemoteMultimodalEmbedding, self.image(), timeout=5)
            deadline = time.monotonic() + 2
            while self.stub.GetWorkerStatus(pb.StatusVersionPB(), timeout=2).running_query_len == 0:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.005)
            with self.assertRaises(grpc.RpcError) as error:
                self.stub.RemoteMultimodalEmbedding(self.image(), timeout=2)
            self.assertEqual(error.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)
            self.assertEqual(len(first.result().multimodal_outputs), 1)
        self.assertEqual(self.stub.GetWorkerStatus(pb.StatusVersionPB(), timeout=2).running_query_len, 0)
        self.assertEqual(self.service._active_bytes, 0)

    def test_bad_image_and_deadline_are_typed_errors(self):
        request = self.image()
        request.multimodal_inputs[0].multimodal_url = 'data:image/png;base64,AAAA'
        with self.assertRaises(grpc.RpcError) as error:
            self.stub.RemoteMultimodalEmbedding(request, timeout=2)
        self.assertEqual(error.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        self.service.engine.delay_ms = 200
        with self.assertRaises(grpc.RpcError) as error:
            self.stub.RemoteMultimodalEmbedding(self.image(), timeout=.02)
        self.assertEqual(error.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)

    def test_v41_prepared_features_and_processor_validation(self):
        config = self.service.engine.model.mm_part.processor_config
        types = image_token_types(1, 1).tolist()
        request = pb.MultimodalInputsPB()
        typed = request.v41_inputs
        typed.schema_version = 1
        typed.token_types.extend([-1] + types + [-1])
        typed.image_mask.extend([False] + [True] * len(types) + [False])
        image = typed.images.add(start=1, n_vit_h=3, n_vit_w=3, types=types,
            processor_identity=config.identity, content_sha256='mock-patches')
        image.patches.CopyFrom(trans_from_tensor(torch.zeros((9, 3, 14, 14), dtype=torch.bfloat16)))
        result = self.stub.RemoteMultimodalEmbedding(request, timeout=5)
        self.assertEqual(tuple(trans_tensor(result.multimodal_outputs[0].multimodal_embedding).shape), (len(types), 8))
        self.assertFalse(result.multimodal_outputs[0].token_ids)
        image.processor_identity = 'wrong-model'
        with self.assertRaises(grpc.RpcError) as error:
            self.stub.RemoteMultimodalEmbedding(request, timeout=2)
        self.assertEqual(error.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)

    def test_hash_matches_release_cpp_partial_word_vector(self):
        data = torch.tensor([(i * 31 + 17) % 256 for i in range(9)], dtype=torch.uint8).reshape(1, 9)
        self.assertEqual(cpu_feature_hash(data).tolist(), [-940810509])


if __name__ == '__main__':
    unittest.main()
