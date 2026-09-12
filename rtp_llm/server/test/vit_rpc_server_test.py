import threading
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import grpc
import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheVersionPB,
    MMRdmaDescPB,
    MultimodalInputPB,
    MultimodalInputsPB,
    ReleaseEmbeddingPB,
    StatusVersionPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceStub,
)
from rtp_llm.ops import get_multimodal_feature_hash
from rtp_llm.server.vit_rpc_server import (
    MultimodalRpcServer,
    _create_rpc_server,
    trans_output,
)
from rtp_llm.server.vit_token_id_cache import MMTokenIdCache
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
    def test_token_cache_expiration_refresh_capacity_and_disabled_mode(self):
        with mock.patch(
            "rtp_llm.server.vit_token_id_cache.time.monotonic", return_value=0.0
        ) as clock:
            cache = MMTokenIdCache(max_items=2, time_window_ms=1000)
            cache.put(b"a", array("i", [1, -2]))
            cache.put(b"b", array("i", [3]))
            clock.return_value = 0.75
            self.assertEqual(cache.get(b"a"), array("i", [1, -2]))
            clock.return_value = 1.25
            self.assertEqual(cache.get(b"a"), array("i", [1, -2]))
            self.assertIsNone(cache.get(b"b"))
            clock.return_value = 2.5
            self.assertIsNone(cache.get(b"a"))

            # Capacity eviction happens while every item is still within its TTL.
            cache = MMTokenIdCache(max_items=2, time_window_ms=10000)
            cache.put(b"a", array("i", [1]))
            cache.put(b"b", array("i", [2]))
            cache.put(b"a", array("i", [4]))
            self.assertEqual(cache.get(b"b"), array("i", [2]))
            self.assertEqual(cache.get(b"a"), array("i", [4]))
            cache.put(b"c", array("i", [3]))
            self.assertIsNone(cache.get(b"b"))
            self.assertEqual(cache.get(b"a"), array("i", [4]))
            self.assertEqual(cache.get(b"c"), array("i", [3]))
            for max_items, window in ((0, 10000), (2, 0)):
                disabled = MMTokenIdCache(max_items=max_items, time_window_ms=window)
                disabled.put(b"a", array("i", [1]))
                self.assertIsNone(disabled.get(b"a"))

    def test_positionless_images_roundtrip_without_synthetic_positions(self):
        result = MMEmbeddingRes([torch.ones((2, 4)), torch.zeros((3, 4))])
        output = trans_output(result)
        self.assertEqual(len(output.multimodal_outputs), 2)
        for item, length in zip(output.multimodal_outputs, (2, 3)):
            self.assertFalse(item.HasField("multimodal_pos_id"))
            self.assertEqual(list(item.multimodal_embedding.shape), [length, 4])

    def server(self, result):
        model = SimpleNamespace(
            model_config=SimpleNamespace(
                hidden_size=4, compute_dtype=torch.float32, max_seq_len=8192
            )
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

    def test_worker_status_has_a_positive_increasing_version(self):
        server = self.server(MMEmbeddingRes([]))
        with mock.patch(
            "rtp_llm.server.vit_rpc_server.time.time_ns", return_value=123000
        ):
            first = server.GetWorkerStatus(None, None)
            second = server.GetWorkerStatus(None, None)
        self.assertTrue(first.alive)
        self.assertEqual(first.role, "VIT")
        self.assertEqual(first.max_seq_len, 8192)
        self.assertGreater(first.status_version, 0)
        self.assertGreater(second.status_version, first.status_version)

    def test_status_and_cache_rpc_remain_available_at_embedding_limit(self):
        entered = threading.Event()
        release = threading.Event()
        result = MMEmbeddingRes([torch.ones((2, 4))])
        service = self.server(result)
        self.assertIsNone(service.rdma_encoder)

        def blocked_submit(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release the embedding request")
            return result

        service.engine.submit = mock.Mock(side_effect=blocked_submit)
        # Legacy engines remain serial even if their configured RPC limit was larger.
        service.max_requests = 32
        server, executor = _create_rpc_server(service, concurrency=1)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = MultimodalRpcServiceStub(channel)
                embedding = stub.RemoteMultimodalEmbedding.future(
                    self.request(), timeout=10
                )
                self.assertTrue(entered.wait(5))
                status = stub.GetWorkerStatus(StatusVersionPB(), timeout=2)
                self.assertTrue(status.alive)
                self.assertGreater(status.status_version, 0)
                self.assertEqual(status.running_query_len, 1)
                cache = stub.GetCacheStatus(CacheVersionPB(), timeout=2)
                self.assertFalse(cache.cache_keys)
                # The reserved RPC slots do not increase embedding admission.
                with self.assertRaises(grpc.RpcError) as error:
                    stub.RemoteMultimodalEmbedding(self.request(), timeout=2)
                self.assertEqual(
                    error.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED
                )
                self.assertEqual(service.engine.submit.call_count, 1)
                release.set()
                output = embedding.result(timeout=5)
                self.assertEqual(
                    list(output.multimodal_outputs[0].multimodal_embedding.shape),
                    [2, 4],
                )
                self.assertEqual(
                    stub.GetWorkerStatus(
                        StatusVersionPB(), timeout=2
                    ).running_query_len,
                    0,
                )
        finally:
            release.set()
            server.stop(0).wait()
            executor.shutdown(wait=True)

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

        expected = list(output.token_ids)
        features.fill_(777)
        server.engine.submit = mock.Mock(
            side_effect=AssertionError("ID hit must not preprocess or run the engine")
        )
        with mock.patch(
            "rtp_llm.server.vit_rpc_server.get_multimodal_feature_hash",
            side_effect=AssertionError("ID hit must not read/hash an embedding"),
        ):
            context = Context()
            cached = server.RemoteMultimodalEmbedding(request, context)
        self.assertEqual(list(cached.multimodal_outputs[0].token_ids), expected)
        self.assertFalse(cached.multimodal_outputs[0].HasField("multimodal_embedding"))
        self.assertEqual(
            context.metadata, {"vit-max-batch-images": "0", "vit-gpu-forwards": "0"}
        )
        server.engine.submit.assert_not_called()

    def test_mixed_metadata_only_computes_missing_images_and_preserves_order(self):
        server = self.server(MMEmbeddingRes([torch.ones((2, 4))]))
        first = self.request()
        first.metadata_only = True
        with mock.patch(
            "rtp_llm.server.vit_rpc_server.get_multimodal_feature_hash",
            side_effect=[torch.tensor([11, 12]), torch.tensor([21, 22, 23])],
        ) as feature_hash:
            server.RemoteMultimodalEmbedding(first, Context())
            server.engine.submit = mock.Mock(
                return_value=MMEmbeddingRes(
                    [torch.zeros((3, 4))], max_batch_size=4, gpu_forwards=1
                )
            )
            mixed = MultimodalInputsPB(
                metadata_only=True,
                multimodal_inputs=[
                    first.multimodal_inputs[0],
                    MultimodalInputPB(multimodal_url="missing"),
                    first.multimodal_inputs[0],
                ],
            )
            context = Context()
            output = server.RemoteMultimodalEmbedding(mixed, context)
        server.engine.submit.assert_called_once()
        self.assertEqual(server.engine.submit.call_args.args[0], ["missing"])
        self.assertEqual(feature_hash.call_count, 2)
        self.assertEqual(
            [list(item.token_ids) for item in output.multimodal_outputs],
            [[11, 12], [21, 22, 23], [11, 12]],
        )
        self.assertEqual(
            context.metadata, {"vit-max-batch-images": "4", "vit-gpu-forwards": "1"}
        )

    def test_cached_metadata_still_checks_total_length_and_deadline(self):
        server = self.server(MMEmbeddingRes([torch.ones((2, 4))]))
        request = self.request()
        request.metadata_only = True
        server.RemoteMultimodalEmbedding(request, Context())
        server.engine.submit = mock.Mock(
            side_effect=AssertionError("cached image must not be recomputed")
        )
        server.engine.model.model_config.max_seq_len = 3
        repeated = MultimodalInputsPB(
            metadata_only=True,
            multimodal_inputs=[
                request.multimodal_inputs[0],
                request.multimodal_inputs[0],
            ],
        )
        context = Context()
        with self.assertRaisesRegex(Aborted, "sequence length"):
            server.RemoteMultimodalEmbedding(repeated, context)
        self.assertEqual(context.code, grpc.StatusCode.INVALID_ARGUMENT)
        expired = Context()
        expired.time_remaining = lambda: 0
        with self.assertRaises(Aborted):
            server.RemoteMultimodalEmbedding(request, expired)
        self.assertEqual(expired.code, grpc.StatusCode.DEADLINE_EXCEEDED)
        server.engine.submit.assert_not_called()
        self.assertEqual(server._active, 0)

    def test_metadata_cache_separates_phase_and_type_for_the_same_url(self):
        server = self.server(MMEmbeddingRes([torch.ones((2, 4))]))
        server.engine.submit = mock.Mock(
            return_value=MMEmbeddingRes([torch.ones((2, 4))])
        )
        first = MultimodalInputPB(multimodal_url="image", multimodal_type=1)
        first.mm_preprocess_config.image_block_start_mod4 = 0
        phase = MultimodalInputPB()
        phase.CopyFrom(first)
        phase.mm_preprocess_config.image_block_start_mod4 = 1
        kind = MultimodalInputPB()
        kind.CopyFrom(first)
        kind.multimodal_type = 0
        with mock.patch(
            "rtp_llm.server.vit_rpc_server.get_multimodal_feature_hash",
            side_effect=[
                torch.tensor([1, 2]),
                torch.tensor([3, 4]),
                torch.tensor([5, 6]),
            ],
        ):
            for item in (first, phase, kind):
                server.RemoteMultimodalEmbedding(
                    MultimodalInputsPB(metadata_only=True, multimodal_inputs=[item]),
                    Context(),
                )
        self.assertEqual(server.engine.submit.call_count, 3)
        server.engine.submit.side_effect = AssertionError(
            "all three variants should hit"
        )
        for item, ids in ((kind, [5, 6]), (phase, [3, 4]), (first, [1, 2])):
            result = server.RemoteMultimodalEmbedding(
                MultimodalInputsPB(metadata_only=True, multimodal_inputs=[item]),
                Context(),
            )
            self.assertEqual(list(result.multimodal_outputs[0].token_ids), ids)

    def test_failed_metadata_does_not_cache_partial_ids(self):
        request = MultimodalInputsPB(
            metadata_only=True,
            multimodal_inputs=[
                MultimodalInputPB(multimodal_url="first"),
                MultimodalInputPB(multimodal_url="second"),
            ],
        )
        valid = MMEmbeddingRes([torch.ones((2, 4)), torch.zeros((2, 4))])
        for failure in ("submit", "shape", "hash"):
            with self.subTest(failure=failure):
                server = self.server(valid)
                server.engine.submit = mock.Mock(return_value=valid)
                if failure == "submit":
                    server.engine.submit.side_effect = RuntimeError("submit failed")
                if failure == "shape":
                    server.engine.submit.return_value = MMEmbeddingRes(
                        [torch.ones((2, 4)), torch.zeros((2, 3))]
                    )
                with mock.patch(
                    "rtp_llm.server.vit_rpc_server.get_multimodal_feature_hash",
                    side_effect=[torch.tensor([1, 2]), RuntimeError("hash failed")],
                ):
                    with self.assertRaises(Aborted):
                        server.RemoteMultimodalEmbedding(request, Context())
                server.engine.submit.side_effect = None
                server.engine.submit.return_value = valid
                with mock.patch(
                    "rtp_llm.server.vit_rpc_server.get_multimodal_feature_hash",
                    side_effect=[torch.tensor([1, 2]), torch.tensor([3, 4])],
                ):
                    output = server.RemoteMultimodalEmbedding(request, Context())
                self.assertEqual(server.engine.submit.call_count, 2)
                self.assertEqual(
                    server.engine.submit.call_args.args[0], ["first", "second"]
                )
                self.assertEqual(
                    [list(item.token_ids) for item in output.multimodal_outputs],
                    [[1, 2], [3, 4]],
                )
                self.assertEqual(server._active, 0)

    def test_feature_rpc_ignores_cached_ids_and_still_exports_rdma(self):
        result = MMEmbeddingRes([torch.ones((2, 4))])
        server = self.server(result)
        request = self.request()
        request.metadata_only = True
        server.RemoteMultimodalEmbedding(request, Context())
        server.engine.submit = mock.Mock(return_value=result)
        server.rdma_encoder = FakeRdmaEncoder(
            [MMRdmaDescPB(handle="fresh-features").SerializeToString()]
        )
        server.require_rdma = True
        request.metadata_only = False
        request.support_rdma = True
        with mock.patch.object(
            server._token_id_cache,
            "get",
            side_effect=AssertionError("feature RPC cannot read ID cache"),
        ), mock.patch.object(
            server._token_id_cache,
            "put",
            side_effect=AssertionError("feature RPC cannot write ID cache"),
        ), mock.patch(
            "rtp_llm.server.vit_rpc_server.get_multimodal_feature_hash",
            side_effect=AssertionError("feature RPC cannot substitute IDs"),
        ):
            output = server.RemoteMultimodalEmbedding(request, Context())
        server.engine.submit.assert_called_once()
        self.assertEqual(server.rdma_encoder.exports, 1)
        self.assertEqual(
            output.multimodal_outputs[0].output_rdma.handle, "fresh-features"
        )
        self.assertFalse(output.multimodal_outputs[0].HasField("multimodal_embedding"))

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

    def test_strict_rdma_requires_opt_in_but_allows_metadata(self):
        server = self.server(MMEmbeddingRes([torch.ones((2, 4))]))
        server.require_rdma = True
        server.engine.submit = mock.Mock(
            return_value=MMEmbeddingRes([torch.ones((2, 4))])
        )
        context = Context()
        with self.assertRaisesRegex(Aborted, "support_rdma"):
            server.RemoteMultimodalEmbedding(self.request(), context)
        self.assertEqual(context.code, grpc.StatusCode.FAILED_PRECONDITION)
        server.engine.submit.assert_not_called()
        request = self.request()
        request.metadata_only = True
        output = server.RemoteMultimodalEmbedding(request, Context())
        self.assertTrue(output.multimodal_outputs[0].token_ids)
        self.assertFalse(output.multimodal_outputs[0].HasField("multimodal_embedding"))

    def test_strict_export_failure_releases_prior_slots_and_never_sends_bytes(self):
        descriptor = MMRdmaDescPB(handle="first").SerializeToString()
        encoder = FakeRdmaEncoder([descriptor, b""])
        with self.assertRaisesRegex(RuntimeError, "inline features are disabled"):
            trans_output(
                MMEmbeddingRes([torch.ones((2, 4)), torch.ones((3, 4))]),
                rdma_encoder=encoder,
                require_rdma=True,
            )
        self.assertEqual(encoder.released, ["first"])

    def test_strict_export_success_has_descriptors_only(self):
        descriptor = MMRdmaDescPB(handle="first").SerializeToString()
        output = trans_output(
            MMEmbeddingRes([torch.ones((2, 4))]),
            rdma_encoder=FakeRdmaEncoder([descriptor]),
            require_rdma=True,
        ).multimodal_outputs[0]
        self.assertTrue(output.HasField("output_rdma"))
        self.assertFalse(output.HasField("multimodal_embedding"))

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
