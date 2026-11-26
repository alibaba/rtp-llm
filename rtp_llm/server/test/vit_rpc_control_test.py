"""Small control-plane contract tests; deliberately no testlib/model dependency."""

import threading
import unittest

import grpc

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    EmptyPB,
    GreenNetQueryPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    ReleaseLeasePB,
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
        self.inspection_entered = threading.Event()
        self.status_thread = None
        self.cache_thread = None
        self.lease_calls = []

    def RemoteMultimodalEmbedding(self, request, context):
        self.entered.set()
        if not self.release.wait(5):
            context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED, "test did not release work"
            )
        return MultimodalOutputPB()

    def GetGreenNetVerdict(self, request, context):
        self.inspection_entered.set()
        if not self.release.wait(5):
            context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED, "test inspection timed out"
            )
        return EmptyPB()

    def GetWorkerStatus(self, request, context):
        self.status_thread = threading.current_thread().name
        status = WorkerStatusPB(role="VIT", alive=True, status_version=1)
        status.mm_inflight_inputs.value = 7
        return status

    def ClaimRdmaLease(self, request, context):
        self.lease_calls.append(("claim", threading.current_thread().name))
        return EmptyPB()

    def ReleaseRdmaLease(self, request, context):
        self.lease_calls.append(("release", threading.current_thread().name))
        return EmptyPB()

    def GetCacheStatus(self, request, context):
        self.cache_thread = threading.current_thread().name
        self.cache_entered.set()
        return CacheStatusPB()


class VitRpcControlTest(unittest.TestCase):
    def setUp(self):
        self.worker = BlockingWorker()
        self.serialized_embeddings = []
        self.server = create_vit_rpc_server(
            max_workers=1, embedding_serializer=self._serialize_embedding
        )
        add_MultimodalRpcServiceServicer_to_server(self.worker, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.assertGreater(port, 0)
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.stub = MultimodalRpcServiceStub(self.channel)

    def _serialize_embedding(self, response):
        payload = response.SerializeToString()
        self.serialized_embeddings.append(payload)
        return payload

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

    def test_control_rpcs_bypass_saturated_business_executor(self):
        work = self.stub.RemoteMultimodalEmbedding.future(
            MultimodalInputsPB(), timeout=5
        )
        self.assertTrue(self.worker.entered.wait(2))
        cache = self.stub.GetCacheStatus.future(CacheVersionPB(), timeout=5)
        status = self.stub.GetWorkerStatus(StatusVersionPB(), timeout=1)
        self.assertTrue(status.alive)
        self.assertEqual(status.mm_inflight_inputs.value, 7)
        self.assertTrue(self.worker.status_thread.startswith("vit-status"))
        lease = ReleaseLeasePB(lease_id=["test"])
        self.stub.ClaimRdmaLease(lease, timeout=1)
        self.stub.ReleaseRdmaLease(lease, timeout=1)
        self.assertEqual(
            [op for op, _ in self.worker.lease_calls], ["claim", "release"]
        )
        for _, thread in self.worker.lease_calls:
            self.assertTrue(thread.startswith("vit-lease"), thread)
        self.assertFalse(work.done())
        self.assertFalse(self.worker.cache_entered.is_set())
        self.worker.release.set()
        response = work.result(timeout=2)
        cache.result(timeout=2)
        self.assertEqual(self.serialized_embeddings, [response.SerializeToString()])
        self.assertTrue(self.worker.cache_thread.startswith("vit-rpc"))

    def test_waiting_for_inspection_leaves_data_and_status_threads_available(self):
        query = self.stub.GetGreenNetVerdict.future(GreenNetQueryPB(), timeout=5)
        self.assertTrue(self.worker.inspection_entered.wait(2))
        self.stub.GetCacheStatus(CacheVersionPB(), timeout=1)
        self.assertTrue(self.worker.cache_thread.startswith("vit-rpc"))
        self.assertTrue(self.stub.GetWorkerStatus(StatusVersionPB(), timeout=1).alive)
        self.assertFalse(query.done())
        self.worker.release.set()
        query.result(timeout=2)

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
