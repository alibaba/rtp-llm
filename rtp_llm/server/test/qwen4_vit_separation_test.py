"""Opt-in real-weight integrated / separated ViT transport equivalence."""

import asyncio
import base64
import gc
import hashlib
import json
import multiprocessing
import os
import socket
import tempfile
import threading
import time
import traceback
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
from rtp_llm.cpp.model_rpc.model_rpc_client import iter_multimodal_inputs
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalHashRequestPB,
    MultimodalInputsPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceStub,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp
from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput, PDSepConfig, VitSeparation
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.master_client import FlexlbResponse, MasterClient
from rtp_llm.server.vit_proxy_server import VitProxyServer
from rtp_llm.server.vit_rpc_server import MultimodalRpcServer
from rtp_llm.utils.base_model_datatypes import GenerateInput, MMUrlType
from rtp_llm.utils.grpc_util import trans_tensor


# Match the production ViT proxy's transport limits for full-size image tensors.
_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 1024 * 1024 * 1024),
    ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
]


def _engine(checkpoint, local_preprocess, server_id=0, proxy_mode=False):
    # Use the same config and registered mixin factory as vit_start_server.
    # Only visual weights are loaded; the LLM is deliberately outside this test.
    config = Qwen4Exp.create_config(checkpoint)
    config.data_type = "bf16"
    vit = VitConfig()
    vit.use_local_preprocess = local_preprocess
    vit.mm_preprocess_max_workers = 1
    vit.vit_concurrency = 2
    vit.vit_max_queue_size = 8
    vit.gpu_max_batch_size = 1
    vit.mm_cache_gpu_max_bytes = 0
    vit.mm_cache_cpu_max_bytes = 64 * 1024**2
    vit.mm_hash_key_cache_max_bytes = 1024**2
    engine_config = SimpleNamespace(
        load_config=SimpleNamespace(load_method=LoadMethod.AUTO),
        profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
        kv_cache_config=None,
    )
    return MultimodalMixinFactory.create_multimodal_process_engine(
        config, engine_config, vit, "cuda:0", server_id, proxy_mode
    )


def _serve(checkpoint, server_id, proxy_mode, port, ready, stop):
    engine = None
    server = None
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        engine = _engine(checkpoint, False, server_id, proxy_mode)
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4), options=_GRPC_OPTIONS
        )
        add_MultimodalRpcServiceServicer_to_server(MultimodalRpcServer(engine), server)
        if server.add_insecure_port(f"127.0.0.1:{port}") == 0:
            raise RuntimeError(f"cannot bind test worker on port {port}")
        server.start()
        ready.send((True, [p.pid for p in engine.preprocess_executor.pool._pool]))
        stop.wait()
    except BaseException:
        ready.send((False, traceback.format_exc()))
        raise
    finally:
        if server is not None:
            server.stop(0).wait()
        if engine is not None:
            engine.stop()
        ready.close()


def _free_port():
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


class Qwen4VitSeparationTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = os.environ.get("QWEN4_EXP_CHECKPOINT")
        if not self.checkpoint:
            self.skipTest("set QWEN4_EXP_CHECKPOINT to the complete Qwen4 checkpoint")
        self.assertTrue(Path(self.checkpoint, "config.json").is_file())
        self.assertTrue(torch.cuda.is_available(), "real ViT test requires a GPU")
        self.worker_count = int(os.environ.get("QWEN4_VIT_TEST_WORKERS", "1"))
        self.assertIn(self.worker_count, (1, 2))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = []
        for i, size in enumerate(((224, 224), (320, 192), (192, 256), (192, 320))):
            path = Path(self.tmp.name) / f"image-{i}.png"
            image = Image.new("RGB", size, (30, 40 + i * 60, 180))
            ImageDraw.Draw(image).rectangle((20, 30, 100, 110), fill=(220, 30, 30))
            image.save(path)
            self.paths.append(str(path))
        fixture = Path(__file__).parents[2] / "test/smoke/data/model/qwen_vl/1.jpeg"
        with Image.open(fixture) as image:
            self.assertEqual(image.size, (2048, 1365))
        self.large_image_index = len(self.paths)
        self.paths.append(
            "data:image/jpeg;base64," + base64.b64encode(fixture.read_bytes()).decode()
        )
        self.request_id = 0
        self.children = []
        self.addCleanup(self._stop_children)

    def _inputs(self, indices):
        result = []
        for index in indices:
            config = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], -1)
            if index != self.large_image_index:
                config.max_pixels = 65536
            result.append(
                MultimodalInput(
                    self.paths[index], MMUrlType.IMAGE, torch.empty(0), config
                )
            )
        return result

    def _request(self, indices):
        self.request_id += 1
        request = GenerateInput(
            self.request_id, torch.tensor([1]), self._inputs(indices), GenerateConfig()
        )
        wire = MultimodalInputsPB(request_id=self.request_id)
        wire.multimodal_inputs.extend(
            iter_multimodal_inputs(request, request.generate_config)
        )
        return wire

    def _stop_children(self):
        for process, stop, ready in self.children:
            stop.set()
        failures = []
        for process, stop, ready in self.children:
            process.join(timeout=30)
            # Only children owned by this test; cleanup failure fails the test.
            if process.is_alive():
                failures.append(f"worker {process.pid} failed graceful shutdown")
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            if process.exitcode != 0:
                failures.append(f"worker {process.pid}: exitcode={process.exitcode}")
            ready.close()
        self.children.clear()
        self.assertFalse(failures, failures)

    async def _check_frontend_hash_route(self, address, expected_hashes):
        host, port = address.split(":")
        vit = RoleAddr(role=RoleType.VIT, ip=host, http_port=1, grpc_port=int(port))
        prefill = RoleAddr(
            role=RoleType.PREFILL, ip="127.0.0.2", http_port=1, grpc_port=2
        )
        status = {
            "role": "VIT",
            "server_ip": host,
            "http_port": 1,
            "grpc_port": int(port),
        }
        client = MasterClient()
        client.get_backend_role_addrs = AsyncMock(
            side_effect=[
                FlexlbResponse(role_addrs=[vit], result={"server_status": [status]}),
                FlexlbResponse.ok([prefill, vit]),
            ]
        )
        pd = PDSepConfig()
        pd.role_type = RoleType.FRONTEND
        visitor = BackendRPCServerVisitor(
            max_seq_len=4096,
            seq_size_per_block=2,
            pd_sep_config=pd,
            addresses=[],
            vit_separation=VitSeparation.VIT_SEPARATION_REMOTE,
            mm_model_config=SimpleNamespace(
                mm_sep_tokens=[[99]], include_sep_tokens=False
            ),
        )
        visitor.master_client = client
        request = GenerateInput(
            10000, torch.tensor([1, 99, 2]), self._inputs([3]), GenerateConfig()
        )
        try:
            self.assertIsNone(await visitor.get_master_route_addrs(request))
            calls = client.get_backend_role_addrs.call_args_list
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1].kwargs["selected_vit"], status)
            self.assertEqual(
                list(calls[1].kwargs["input_pb"].token_ids), [1] + expected_hashes + [2]
            )
            self.assertIsNotNone(request.mm_token_expansion)
        finally:
            await client.close()

    def test_real_weights_match_integrated_path_with_spawn_preprocessing(self):
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        integrated = _engine(self.checkpoint, True)
        expected = []
        try:
            for index in range(len(self.paths)):
                result = integrated.get_embedding_result(self._inputs([index]))[0]
                expected.append(
                    (
                        torch.cat(result.embeddings).cpu(),
                        torch.cat(result.position_ids).cpu(),
                        torch.cat(result.feature_hashes).cpu(),
                    )
                )
            self.assertEqual(expected[self.large_image_index][0].shape[0], 2752)
        finally:
            integrated.stop()
            del integrated
            gc.collect()
            torch.cuda.empty_cache()

        context = multiprocessing.get_context("spawn")
        ports = []
        while len(ports) < self.worker_count:
            port = _free_port()
            if port not in ports:
                ports.append(port)
        addresses = [f"127.0.0.1:{port}" for port in ports]
        proxy = None
        if self.worker_count > 1:
            proxy = VitProxyServer(list(addresses), _free_port())
            proxy.start()
            self.addCleanup(proxy.stop)
            # No worker can be routed before a successful readiness probe.
            self.assertEqual(proxy.load_balancer.get_alive_worker_addresses(), [])
        for index, port in enumerate(ports):
            ready, child_ready = context.Pipe(duplex=False)
            stop = context.Event()
            process = context.Process(
                target=_serve,
                args=(
                    self.checkpoint,
                    index,
                    proxy is not None,
                    port,
                    child_ready,
                    stop,
                ),
            )
            process.start()
            child_ready.close()
            self.children.append((process, stop, ready))
        pool_pids = []
        for process, stop, ready in self.children:
            self.assertTrue(ready.poll(300), f"worker {process.pid} startup timed out")
            ok, detail = ready.recv()
            self.assertTrue(ok, detail)
            self.assertEqual(len(detail), 1, "CPU pool budget must be per worker")
            pool_pids.extend(detail)
        self.assertEqual(len(set(pool_pids)), self.worker_count)

        if proxy is not None:
            deadline = time.monotonic() + 15
            while (
                len(proxy.load_balancer.get_alive_worker_addresses())
                != self.worker_count
            ):
                self.assertLess(
                    time.monotonic(), deadline, "background recovery stalled"
                )
                threading.Event().wait(0.05)
            addresses.append(f"127.0.0.1:{proxy.external_grpc_port}")

        channels = [
            grpc.insecure_channel(address, options=_GRPC_OPTIONS)
            for address in addresses
        ]
        for channel in channels:
            self.addCleanup(channel.close)
        stubs = [MultimodalRpcServiceStub(channel) for channel in channels]

        def check(stub, indices, wire):
            output = stub.RemoteMultimodalEmbedding(wire, timeout=120)
            self.assertEqual(
                list(output.split_size), [expected[i][0].shape[0] for i in indices]
            )
            for field, offset in (
                ("multimodal_embedding", 0),
                ("multimodal_pos_id", 1),
                ("multimodal_feature_hash", 2),
            ):
                actual = trans_tensor(getattr(output, field))
                target = torch.cat([expected[i][offset] for i in indices])
                torch.testing.assert_close(actual, target, rtol=0, atol=0)
                self.assertTrue(
                    torch.equal(
                        actual.contiguous().view(torch.uint8),
                        target.contiguous().view(torch.uint8),
                    ),
                    f"{field}: tensor bytes differ",
                )
            self.assertEqual(len(output.multimodal_extra_input), 0)
            return output

        # The actual worker caches must distinguish cold, hot, and partial hits.
        keys = [item.cache_key() for item in self._inputs([0, 1, 2])]
        for index, stub in enumerate(stubs):
            if index < self.worker_count:
                cold = stub.GetMultimodalHashes(
                    MultimodalHashRequestPB(keys=keys), timeout=10
                )
                self.assertEqual(
                    [entry.hash_hit for entry in cold.entries], [False] * 3
                )
            check(stub, [0], self._request([0]))
            if index < self.worker_count:
                partial = stub.GetMultimodalHashes(
                    MultimodalHashRequestPB(keys=keys), timeout=10
                )
                self.assertEqual(
                    [entry.hash_hit for entry in partial.entries], [True, False, False]
                )
            for indices in ([0], [0, 1], [1, 2], [2, 0]):
                check(stub, indices, self._request(indices))
            hot = stub.GetMultimodalHashes(
                MultimodalHashRequestPB(keys=keys), timeout=10
            )
            self.assertTrue(
                all(entry.hash_hit and entry.embedding_hit for entry in hot.entries)
            )

        # Use the exact large JPEG and data URL used by the combined LLM smoke.
        # No max_pixels override: every 2752-row tensor and hash is compared,
        # including rows after the LLM's last reusable 64-token cache block.
        def digest(tensor):
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            return dict(
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )

        parity = dict(
            image="rtp_llm/test/smoke/data/model/qwen_vl/1.jpeg",
            image_size=[2048, 1365],
            image_tokens=2752,
            integrated=[digest(t) for t in expected[self.large_image_index]],
            workers=[],
        )
        key = self._inputs([self.large_image_index])[0].cache_key()
        for index, stub in enumerate(stubs[: self.worker_count]):
            probe = stub.GetMultimodalHashes(
                MultimodalHashRequestPB(keys=[key]), timeout=10
            )
            self.assertEqual(len(probe.entries), 1)
            self.assertFalse(probe.entries[0].hash_hit)
            self.assertFalse(probe.entries[0].embedding_hit)
            for stage in ("cold", "hot"):
                output = check(
                    stub,
                    [self.large_image_index],
                    self._request([self.large_image_index]),
                )
                parity["workers"].append(
                    dict(
                        worker=index,
                        stage=stage,
                        tensors=[
                            digest(trans_tensor(getattr(output, field)))
                            for field in (
                                "multimodal_embedding",
                                "multimodal_pos_id",
                                "multimodal_feature_hash",
                            )
                        ],
                    )
                )
                probe = stub.GetMultimodalHashes(
                    MultimodalHashRequestPB(keys=[key]), timeout=10
                )
                self.assertTrue(probe.entries[0].hash_hit)
                self.assertTrue(probe.entries[0].embedding_hit)
        output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", self.tmp.name))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "large_image_vit_parity.json").write_text(
            json.dumps(parity, indent=2)
        )
        # The fourth image is still cold: exercise the real two-phase hash
        # acquisition through the frontend, then repeat with a warm hash cache.
        for _ in range(2):
            asyncio.run(
                self._check_frontend_hash_route(addresses[-1], expected[3][2].tolist())
            )
        with futures.ThreadPoolExecutor(max_workers=4) as executor:
            pending = [
                executor.submit(check, stubs[-1], indices, self._request(indices))
                for indices in ([0, 2], [2, 1], [1, 0], [2, 0])
            ]
            for result in pending:
                result.result(timeout=150)


if __name__ == "__main__":
    unittest.main()
