"""Small control-plane contract tests; deliberately no testlib/model dependency."""

import threading
import unittest

import grpc

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    StatusVersionPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    MultimodalRpcServiceStub,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.server.vit_rpc_control import create_vit_rpc_server


class BlockingWorker(MultimodalRpcServiceServicer):
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cache_entered = threading.Event()
        self.status_thread = None
        self.cache_thread = None

    def RemoteMultimodalEmbedding(self, request, context):
        self.entered.set()
        if not self.release.wait(5):
            context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED, "test did not release work"
            )
        return MultimodalOutputPB()

    def GetWorkerStatus(self, request, context):
        self.status_thread = threading.current_thread().name
        status = WorkerStatusPB(role="VIT", alive=True, status_version=1)
        status.mm_inflight_inputs.value = 7
        return status

    def GetCacheStatus(self, request, context):
        self.cache_thread = threading.current_thread().name
        self.cache_entered.set()
        return CacheStatusPB()


class VitRpcControlTest(unittest.TestCase):
    def setUp(self):
        self.worker = BlockingWorker()
        self.server = create_vit_rpc_server(max_workers=1)
        add_MultimodalRpcServiceServicer_to_server(self.worker, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.assertGreater(port, 0)
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.stub = MultimodalRpcServiceStub(self.channel)

    def tearDown(self):
        self.worker.release.set()
        self.channel.close()
        self.assertTrue(self.server.stop(0).wait(5))

    def test_generated_proto_distinguishes_legacy_unknown_from_idle(self):
        status = WorkerStatusPB(role="VIT", alive=True, status_version=1)
        self.assertFalse(
            WorkerStatusPB.FromString(status.SerializeToString()).HasField(
                "mm_inflight_inputs"
            )
        )
        status.mm_inflight_inputs.value = 0
        restored = WorkerStatusPB.FromString(status.SerializeToString())
        self.assertTrue(restored.HasField("mm_inflight_inputs"))
        self.assertEqual(restored.mm_inflight_inputs.value, 0)

    def test_wire_load_contract_matches_java_fixture(self):
        # Shared with GrpcWorkerStatusRunnerTest: field 28, Int64Value(7).
        wire = bytes.fromhex("e201020807")
        restored = WorkerStatusPB.FromString(wire)
        self.assertTrue(restored.HasField("mm_inflight_inputs"))
        self.assertEqual(restored.mm_inflight_inputs.value, 7)
        self.assertEqual(restored.SerializeToString(), wire)

    def test_only_scalar_status_bypasses_saturated_business_executor(self):
        work = self.stub.RemoteMultimodalEmbedding.future(
            MultimodalInputsPB(), timeout=5
        )
        self.assertTrue(self.worker.entered.wait(2))
        cache = self.stub.GetCacheStatus.future(CacheVersionPB(), timeout=5)
        status = self.stub.GetWorkerStatus(StatusVersionPB(), timeout=1)
        self.assertTrue(status.alive)
        self.assertEqual(status.mm_inflight_inputs.value, 7)
        self.assertTrue(self.worker.status_thread.startswith("vit-status"))
        self.assertFalse(work.done())
        self.assertFalse(self.worker.cache_entered.is_set())
        self.worker.release.set()
        work.result(timeout=2)
        cache.result(timeout=2)
        self.assertTrue(self.worker.cache_thread.startswith("vit-rpc"))

    def test_graceful_shutdown_drains_accepted_business_rpc(self):
        work = self.stub.RemoteMultimodalEmbedding.future(
            MultimodalInputsPB(), timeout=5
        )
        self.assertTrue(self.worker.entered.wait(2))
        stopped = self.server.stop(5)
        self.assertFalse(stopped.is_set())
        self.worker.release.set()
        self.assertIsInstance(work.result(timeout=2), MultimodalOutputPB)
        self.assertTrue(stopped.wait(5))

    def test_unknown_rpc_keeps_unimplemented_status(self):
        unknown = self.channel.unary_unary("/MultimodalRpcService/DoesNotExist")
        with self.assertRaises(grpc.RpcError) as caught:
            unknown(b"", timeout=1)
        self.assertEqual(caught.exception.code(), grpc.StatusCode.UNIMPLEMENTED)


if __name__ == "__main__":
    unittest.main()
