import os
import tempfile
import unittest
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import grpc
import torch
from PIL import Image, ImageDraw

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    iter_multimodal_inputs,
    multimodal_cache_keys,
    trans_input,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalInputsPB
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceStub,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import Qwen3_VLMixin
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.master_client import FlexlbResponse, MasterClient
from rtp_llm.server.vit_rpc_server import MultimodalRpcServer
from rtp_llm.utils.base_model_datatypes import GenerateInput, MMUrlType, VitParameters
from rtp_llm.utils.grpc_util import trans_tensor


class RealVitRoutingIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """Frontend -> loopback gRPC -> actual Qwen3-VL ViT, with FlexLB selection stubbed."""

    async def asyncSetUp(self):
        checkpoint = os.environ.get("RTP_LLM_TEST_VIT_CHECKPOINT")
        if not checkpoint:
            self.skipTest("set RTP_LLM_TEST_VIT_CHECKPOINT to real Qwen3-VL weights")
        if not Path(checkpoint).is_dir():
            self.fail(f"Qwen3-VL checkpoint is missing: {checkpoint}")

        torch.set_num_threads(min(8, torch.get_num_threads()))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.image_paths = []
        for index in range(2):
            path = Path(self.tmp.name) / f"image-{index}.png"
            image = Image.new("RGB", (224, 224), (220, 30, 30))
            draw = ImageDraw.Draw(image)
            draw.rectangle((32 + index * 40, 32, 160, 160), fill=(20, 70, 220))
            image.save(path)
            self.image_paths.append(path)

        params = VitParameters()
        params.config["ckpt_path"] = checkpoint
        mixin = Qwen3_VLMixin(
            torch.float32, "cpu", params, LoadMethod.AUTO, VitConfig(), checkpoint
        )
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        vit_config.vit_concurrency = 1
        vit_config.vit_max_queue_size = 4
        vit_config.gpu_max_batch_size = 1
        vit_config.gpu_batch_wait_ms = 0
        vit_config.mm_cache_gpu_max_bytes = 0
        vit_config.mm_cache_cpu_max_bytes = 64 * 1024 * 1024
        vit_config.mm_hash_key_cache_max_bytes = 1024 * 1024
        self.vit_config = vit_config
        model_config = SimpleNamespace(
            mm_model_config=SimpleNamespace(mm_position_ids_style=3),
            mm_related_params=params,
        )
        self.engine = MMProcessEngine(
            mixin.mm_part,
            model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
            device="cpu",
        )
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        add_MultimodalRpcServiceServicer_to_server(
            MultimodalRpcServer(self.engine), self.server
        )
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        self.stub = MultimodalRpcServiceStub(self.channel)
        self.address = RoleAddr(
            role=RoleType.VIT, ip="127.0.0.1", http_port=1, grpc_port=port
        )
        self.prefill = RoleAddr(
            role=RoleType.PREFILL, ip="127.0.0.2", http_port=8000, grpc_port=8001
        )
        self.status = {
            "role": "VIT",
            "server_ip": self.address.ip,
            "http_port": self.address.http_port,
            "grpc_port": self.address.grpc_port,
        }
        self.client = MasterClient()
        self.visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        self.visitor._mm_cache_routing = True
        self.visitor.mm_model_config = SimpleNamespace(
            mm_sep_tokens=[[99]], include_sep_tokens=False
        )
        self.visitor.seq_size_per_block = 2
        self.visitor._page_rr_route_cache_keys = False
        self.visitor._page_rr_cp_size = 1
        self.visitor.max_seq_len = 512
        self.visitor._report_recent_cache_key_metrics = lambda keys: None
        self.visitor.master_client = self.client

    async def asyncTearDown(self):
        if hasattr(self, "client"):
            await self.client.close()
        if hasattr(self, "channel"):
            await self.channel.close()
        if hasattr(self, "server"):
            self.server.stop(0).wait()
        if hasattr(self, "engine"):
            self.engine.stop()

    def request(self, image_path, request_id):
        config = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], -1)
        config.max_pixels = 65536
        return GenerateInput(
            request_id,
            torch.tensor([1, 99, 2]),
            [MultimodalInput(str(image_path), MMUrlType.IMAGE, torch.empty(0), config)],
            GenerateConfig(),
        )

    async def route(self, request):
        keys = multimodal_cache_keys(request)
        self.client.get_backend_role_addrs = AsyncMock(
            side_effect=[
                FlexlbResponse(
                    role_addrs=[self.address],
                    result={"server_status": [self.status]},
                ),
                FlexlbResponse.ok([self.prefill, self.address]),
            ]
        )
        self.assertIsNone(await self.visitor.get_master_route_addrs(request))
        calls = self.client.get_backend_role_addrs.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["media_keys"], keys)
        self.assertTrue(calls[0].kwargs["vit_only"])
        self.assertEqual(calls[1].kwargs["selected_vit"], self.status)
        wire = calls[1].kwargs["input_pb"]
        ids = list(wire.token_ids)
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[-1], 2)
        self.assertEqual(calls[1].kwargs["seq_len"], len(ids))
        self.assertEqual(
            [(span.offset, span.length) for span in wire.multimodal_token_layout.spans],
            [(1, len(ids) - 2)],
        )
        self.assertEqual(list(trans_input(request).token_ids), ids)
        self.assertEqual(request.token_ids.tolist(), [1, 99, 2])
        self.assertEqual(
            request.generate_config.role_addrs, [self.prefill, self.address]
        )
        return ids[1:-1]

    async def test_real_vit_hashes_match_frontend_tokens_and_rpc_embeddings(self):
        cold_request = self.request(self.image_paths[0], 1001)
        cold_hashes = await self.route(cold_request)
        self.assertGreater(len(cold_hashes), 1)
        self.assertNotEqual(set(cold_hashes), {0})

        hot_request = self.request(self.image_paths[0], 1002)
        self.assertEqual(await self.route(hot_request), cold_hashes)
        hot = await self.client.get_vit_cache_metadata(
            self.address, multimodal_cache_keys(hot_request), hot_request
        )
        self.assertTrue(hot["entries"][0]["hash_hit"])

        wire = MultimodalInputsPB(request_id=1003)
        wire.multimodal_inputs.extend(
            iter_multimodal_inputs(cold_request, cold_request.generate_config)
        )
        output = await self.stub.RemoteMultimodalEmbedding(wire, timeout=120)
        embedding = trans_tensor(output.multimodal_embedding)
        rpc_hashes = trans_tensor(output.multimodal_feature_hash)
        self.assertEqual(embedding.shape[0], len(cold_hashes))
        self.assertEqual(rpc_hashes.tolist(), cold_hashes)
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertGreater(embedding.std().item(), 0)
        self.assertEqual(len(output.multimodal_extra_input), 1)
        self.assertGreater(trans_tensor(output.multimodal_extra_input[0]).numel(), 0)

        other_hashes = await self.route(self.request(self.image_paths[1], 1004))
        self.assertNotEqual(other_hashes, cold_hashes)

        preprocessed = self.engine.mm_part.preprocess_input(
            [cold_request.mm_inputs[0]],
            self.vit_config,
            **self.engine.mm_part.get_preprocess_params(),
        )
        single_embedding, position_ids, deepstack = self.engine.mm_part.embedding(
            preprocessed
        )
        self.assertEqual(single_embedding.shape, embedding.shape)
        self.assertTrue(torch.isfinite(single_embedding).all())
        self.assertGreater(position_ids.numel(), 0)
        self.assertGreater(deepstack.numel(), 0)


if __name__ == "__main__":
    unittest.main()
