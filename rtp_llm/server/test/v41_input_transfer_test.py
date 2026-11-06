import unittest
from concurrent import futures
from threading import Event

import grpc

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    MultimodalOutputPB,
    V41InputRefPB,
    V41InputUploadPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    MultimodalRpcServiceStub,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.server.v41_input_transfer import V41InputTransfer


class _EchoVit(V41InputTransfer, MultimodalRpcServiceServicer):
    """Real transfer handlers; echo patches in place of GPU inference."""

    def __init__(self):
        super().__init__(max_bytes=4 * 1024**2, max_entries=2)
        self.read_started = Event()
        self.continue_read = Event()
        self.continue_read.set()

    def RemoteMultimodalEmbedding(self, request, context):
        resolved = self.resolve_v41_inputs(request, context)
        self.read_started.set()
        self.continue_read.wait(5)
        output = MultimodalOutputPB()
        # Validate actual bytes reached the embedding boundary, without torch/CUDA.
        output.multimodal_embedding.CopyFrom(resolved.v41_inputs.images[0].patches)
        return output


class V41InputTransferTest(unittest.TestCase):
    def setUp(self):
        self.handler = _EchoVit()
        self.pool = futures.ThreadPoolExecutor(max_workers=4)
        self.server = grpc.server(self.pool)
        add_MultimodalRpcServiceServicer_to_server(self.handler, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.stub = MultimodalRpcServiceStub(self.channel)
        self.upload = V41InputUploadPB(
            request_id=123, payload_id="a" * 32, ttl_ms=10000
        )
        typed = self.upload.inputs
        typed.schema_version = 1
        typed.token_types.extend([-1, 0, 1, 2, 3])
        typed.image_mask.extend([False, True, True, True, True])
        image = typed.images.add(
            start=1,
            n_vit_h=1,
            n_vit_w=1,
            types=[0, 1, 2, 3],
            content_sha256="hash",
            processor_identity="processor",
        )
        image.patches.data_type = 3
        image.patches.shape.extend([1, 3, 14, 14])
        image.patches.bf16_data = b"xy" * (1024 * 512)

    def tearDown(self):
        self.handler.continue_read.set()
        for key, entry in list(self.handler._v41_uploads.items()):
            self.stub.ReleaseV41Inputs(
                V41InputRefPB(request_id=entry.request_id, payload_id=key)
            )
        self.channel.close()
        self.server.stop(0).wait()
        self.pool.shutdown()

    def reference(self):
        request = MultimodalInputsPB(request_id=self.upload.request_id, timeout_ms=1000)
        request.v41_inputs.CopyFrom(self.upload.inputs)
        request.v41_inputs.payload_id = self.upload.payload_id
        for image in request.v41_inputs.images:
            image.ClearField("patches")
        return request

    def assert_rpc_error(self, code, call, request):
        with self.assertRaises(grpc.RpcError) as failure:
            call(request, timeout=2)
        self.assertEqual(code, failure.exception.code())

    def test_direct_upload_and_repeated_read_preserve_patches(self):
        self.stub.UploadV41Inputs(self.upload)
        reference = self.reference()
        self.assertLess(reference.ByteSize(), 200)
        self.assertGreater(self.upload.ByteSize(), 1024**2)
        for _ in range(2):
            response = self.stub.RemoteMultimodalEmbedding(reference)
            self.assertEqual(
                response.multimodal_embedding, self.upload.inputs.images[0].patches
            )
        self.assertFalse(reference.v41_inputs.images[0].HasField("patches"))

    def test_release_is_idempotent_and_invalidates_reference(self):
        self.stub.UploadV41Inputs(self.upload)
        ref = V41InputRefPB(request_id=123, payload_id=self.upload.payload_id)
        for _ in range(2):
            self.stub.ReleaseV41Inputs(ref)
        self.assertEqual(0, self.handler._v41_bytes)
        self.assert_rpc_error(
            grpc.StatusCode.NOT_FOUND,
            self.stub.RemoteMultimodalEmbedding,
            self.reference(),
        )

    def test_expiry_frees_idle_upload_without_another_request(self):
        self.upload.ttl_ms = 20
        self.stub.UploadV41Inputs(self.upload)
        reaper = self.handler._v41_reaper
        if reaper:
            reaper.join(2)
            self.assertFalse(reaper.is_alive())
        self.assertEqual(0, self.handler._v41_bytes)
        self.assert_rpc_error(
            grpc.StatusCode.NOT_FOUND,
            self.stub.RemoteMultimodalEmbedding,
            self.reference(),
        )

    def test_request_and_metadata_mismatch_do_not_consume_upload(self):
        self.stub.UploadV41Inputs(self.upload)
        wrong = self.reference()
        wrong.request_id += 1
        self.assert_rpc_error(
            grpc.StatusCode.INVALID_ARGUMENT, self.stub.RemoteMultimodalEmbedding, wrong
        )
        wrong = self.reference()
        wrong.v41_inputs.images[0].content_sha256 = "other-image"
        self.assert_rpc_error(
            grpc.StatusCode.INVALID_ARGUMENT, self.stub.RemoteMultimodalEmbedding, wrong
        )
        self.assert_rpc_error(
            grpc.StatusCode.INVALID_ARGUMENT,
            self.stub.ReleaseV41Inputs,
            V41InputRefPB(request_id=999, payload_id=self.upload.payload_id),
        )
        self.stub.RemoteMultimodalEmbedding(self.reference())

    def test_live_uploads_are_not_evicted_at_capacity(self):
        self.stub.UploadV41Inputs(self.upload)
        second = V41InputUploadPB()
        second.CopyFrom(self.upload)
        second.payload_id = "b" * 32
        self.stub.UploadV41Inputs(second)
        second.payload_id = "c" * 32
        self.assert_rpc_error(
            grpc.StatusCode.RESOURCE_EXHAUSTED, self.stub.UploadV41Inputs, second
        )
        self.stub.RemoteMultimodalEmbedding(self.reference())

    def test_byte_budget_rejects_before_retaining_payload(self):
        self.handler._v41_max_bytes = 1
        self.assert_rpc_error(
            grpc.StatusCode.RESOURCE_EXHAUSTED, self.stub.UploadV41Inputs, self.upload
        )
        self.assertEqual(0, self.handler._v41_bytes)

    def test_duplicate_cannot_overwrite_existing_data(self):
        self.stub.UploadV41Inputs(self.upload)
        self.assert_rpc_error(
            grpc.StatusCode.ALREADY_EXISTS, self.stub.UploadV41Inputs, self.upload
        )
        self.stub.RemoteMultimodalEmbedding(self.reference())

    def test_release_during_read_keeps_admitted_data_valid(self):
        self.stub.UploadV41Inputs(self.upload)
        self.handler.continue_read.clear()
        read = self.stub.RemoteMultimodalEmbedding.future(self.reference(), timeout=5)
        self.assertTrue(self.handler.read_started.wait(2))
        self.stub.ReleaseV41Inputs(
            V41InputRefPB(request_id=123, payload_id=self.upload.payload_id)
        )
        self.handler.continue_read.set()
        self.assertEqual(
            read.result().multimodal_embedding, self.upload.inputs.images[0].patches
        )
        self.assertEqual(0, self.handler._v41_bytes)

    def test_reference_rejects_inline_patches(self):
        self.stub.UploadV41Inputs(self.upload)
        request = self.reference()
        request.v41_inputs.images[0].patches.CopyFrom(
            self.upload.inputs.images[0].patches
        )
        self.assert_rpc_error(
            grpc.StatusCode.INVALID_ARGUMENT,
            self.stub.RemoteMultimodalEmbedding,
            request,
        )


if __name__ == "__main__":
    unittest.main()
