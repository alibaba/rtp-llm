import multiprocessing
import os
import signal
import threading
import time
import unittest
import urllib.error
import urllib.request
from array import array
from concurrent import futures
from contextlib import contextmanager
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
    _create_health_server,
    _serve_rpc_server,
    trans_output,
)
from rtp_llm.server.vit_token_id_cache import MMTokenIdCache
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor
from rtp_llm.utils.mm_process_engine import MMEmbeddingRes, MMProcessEngine


class Aborted(Exception):
    pass


class VitHealthServerTest(unittest.TestCase):
    def test_rpc_lifecycle_controls_health_and_closes_port(self):
        server, engine, executor = mock.Mock(), mock.Mock(), mock.Mock()
        http = []
        create = _create_health_server

        def create_http(port, ready):
            http.append(create(port, ready))
            return http[0]

        def running(timeout):
            server.start.assert_called_once()
            url = f"http://127.0.0.1:{http[0].server_port}/health"
            with urllib.request.urlopen(url, timeout=2) as response:
                self.assertEqual(response.status, 200)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(url, timeout=2)
            self.assertEqual(error.exception.code, 503)
            return False

        server.wait_for_termination.side_effect = running
        with mock.patch(
            "rtp_llm.server.vit_rpc_server._create_health_server",
            side_effect=create_http,
        ):
            _serve_rpc_server(server, engine, executor, 5, health_port=0)
        self.assertEqual(http[0].fileno(), -1)
        server.stop.assert_called_once()
        engine.stop.assert_called_once()
        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    def test_health_tracks_readiness_and_rejects_unknown_paths(self):
        ready = False
        server = _create_health_server(0, lambda: ready)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            for ready in (False, True, False):
                if ready:
                    with urllib.request.urlopen(url + "/health", timeout=2) as response:
                        self.assertEqual(response.status, 200)
                        self.assertEqual(response.read(), b"ok")
                else:
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(url + "/health", timeout=2)
                    self.assertEqual(error.exception.code, 503)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(url + "/unknown", timeout=2)
            self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()


class Context:
    def add_callback(self, callback):
        self.callback = callback
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


def _shutdown_test_worker(connection, stuck):
    """Real gRPC and OS signals in a disposable CPU-only process."""
    release = threading.Event()

    def allow_completion():
        connection.recv()
        release.set()

    threading.Thread(target=allow_completion, daemon=True).start()

    def handle(request, context):
        if request == b"block":
            connection.send(("active", None))
            release.wait()
        return b"complete"

    def stop_engine():
        connection.send(("engine_stopped", None))
        if stuck == "scheduler":
            threading.Event().wait()

    class Executor(futures.ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            super().shutdown(wait=wait, cancel_futures=cancel_futures)
            connection.send(("executor_stopped", None))

    executor = Executor(max_workers=2)
    server = grpc.server(executor)
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "shutdown.Test", {"Call": grpc.unary_unary_rpc_method_handler(handle)}
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    start, stop = server.start, server.stop

    def start_and_report():
        start()
        connection.send(("ready", port))

    def stop_and_report(grace):
        event = stop(grace)
        connection.send(("draining", grace))
        return event

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    with mock.patch.object(
        server, "start", side_effect=start_and_report
    ), mock.patch.object(server, "stop", side_effect=stop_and_report):
        _serve_rpc_server(
            server, SimpleNamespace(stop=stop_engine), executor, 1 if stuck else 5
        )
    connection.send(
        (
            "restored",
            all(signal.getsignal(sig) == old for sig, old in previous.items()),
        )
    )
    connection.close()


class VitRpcShutdownTest(unittest.TestCase):
    def receive(self, connection, expected):
        self.assertTrue(connection.poll(60), f"Timed out waiting for {expected}")
        event, value = connection.recv()
        self.assertEqual(event, expected)
        return value

    @contextmanager
    def worker(self, stuck=None):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_shutdown_test_worker, args=(child, stuck))
        process.start()
        child.close()
        try:
            port = self.receive(parent, "ready")
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                grpc.channel_ready_future(channel).result(timeout=5)
                yield process, parent, channel.unary_unary("/shutdown.Test/Call")
        finally:
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
            parent.close()
            process.close()

    def test_signals_stop_accepting_drain_rpc_and_close_resources(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=signum), self.worker() as (
                process,
                connection,
                rpc,
            ):
                response = rpc.future(b"block", timeout=10)
                self.receive(connection, "active")
                os.kill(process.pid, signum)
                grace = self.receive(connection, "draining")
                self.assertGreater(grace, 0)
                self.assertLessEqual(grace, 5)
                # Repeated signals must leave the ongoing drain intact.
                os.kill(process.pid, signum)
                self.assertFalse(response.done())
                with self.assertRaises(grpc.RpcError) as rejected:
                    rpc(b"new", timeout=1)
                self.assertEqual(rejected.exception.code(), grpc.StatusCode.UNAVAILABLE)
                connection.send("release")
                self.assertEqual(response.result(timeout=3), b"complete")
                self.receive(connection, "engine_stopped")
                self.receive(connection, "executor_stopped")
                self.assertTrue(self.receive(connection, "restored"))
                process.join(timeout=3)
                self.assertEqual(process.exitcode, 0)

    def test_shutdown_budget_bounds_stuck_handler_and_scheduler(self):
        for stuck in ("handler", "scheduler"):
            with self.subTest(stuck=stuck), self.worker(stuck) as (
                process,
                connection,
                rpc,
            ):
                if stuck == "handler":
                    response = rpc.future(b"block", timeout=10)
                    self.receive(connection, "active")
                started = time.monotonic()
                os.kill(process.pid, signal.SIGTERM)
                self.receive(connection, "draining")
                if stuck == "scheduler":
                    self.receive(connection, "engine_stopped")
                process.join(timeout=5)
                self.assertEqual(process.exitcode, 1)
                self.assertLess(time.monotonic() - started, 5)
                if stuck == "handler":
                    with self.assertRaises(grpc.RpcError):
                        response.result(timeout=1)


class VitRpcServerTest(unittest.TestCase):
    def v41_request(self):
        request = MultimodalInputsPB()
        typed = request.v41_inputs
        typed.schema_version = 1
        typed.token_types.extend([-1, 0, 1, 2, 3, -1, 0, 1, 2, 3, -1])
        typed.image_mask.extend(kind != -1 for kind in typed.token_types)
        for start in (1, 6):
            image = typed.images.add(
                start=start,
                n_vit_h=3,
                n_vit_w=3,
                types=[0, 1, 2, 3],
                content_sha256=str(start) * 64,
                processor_identity="processor-v41",
            )
            image.patches.CopyFrom(
                trans_from_tensor(
                    torch.full((9, 3, 14, 14), start, dtype=torch.bfloat16)
                )
            )
        return request

    def v41_server(self):
        server = self.server(MMEmbeddingRes([]))
        server.engine.model.model_config.compute_dtype = torch.bfloat16
        server.engine.model.mm_part = SimpleNamespace(
            processor_config=SimpleNamespace(
                vision_patch_size=14,
                vision_downsample_ratio=3,
                vision_max_n_token=1024,
                identity="processor-v41",
            )
        )
        server.engine.submit = mock.Mock(side_effect=AssertionError("generic submit"))
        server.engine.submit_v41 = mock.Mock(
            side_effect=lambda images: MMEmbeddingRes(
                [torch.full((4, 4), images[0]["start"], dtype=torch.bfloat16)]
            )
        )
        return server

    def test_v41_roundtrip_preserves_all_images_and_typed_payload(self):
        service = self.v41_server()
        request = self.v41_request()
        server, executor = _create_rpc_server(service, concurrency=1)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                output = MultimodalRpcServiceStub(channel).RemoteMultimodalEmbedding(
                    request, timeout=5
                )
        finally:
            server.stop(0).wait()
            executor.shutdown(wait=True)
        self.assertEqual(len(output.multimodal_outputs), 2)
        self.assertEqual(service.engine.submit_v41.call_count, 2)
        service.engine.submit.assert_not_called()
        for source, call, result in zip(
            request.v41_inputs.images,
            service.engine.submit_v41.call_args_list,
            output.multimodal_outputs,
        ):
            image = call.args[0][0]
            self.assertEqual(image["start"], source.start)
            self.assertEqual(image["content_sha256"], source.content_sha256)
            self.assertEqual(image["processor_identity"], source.processor_identity)
            self.assertEqual(image["types"].tolist(), list(source.types))
            torch.testing.assert_close(image["patches"], trans_tensor(source.patches))
            torch.testing.assert_close(
                trans_tensor(result.multimodal_embedding),
                torch.full((4, 4), source.start, dtype=torch.bfloat16),
            )
            self.assertFalse(result.HasField("multimodal_pos_id"))
            self.assertEqual(list(result.token_ids), [])

    def test_v41_rejects_mixed_metadata_and_malformed_typed_inputs(self):
        for invalid in (
            "mixed",
            "metadata",
            "schema",
            "mask",
            "types",
            "patches",
            "processor",
        ):
            with self.subTest(invalid=invalid):
                service = self.v41_server()
                request = self.v41_request()
                if invalid == "mixed":
                    request.multimodal_inputs.add(multimodal_url="image")
                elif invalid == "metadata":
                    request.metadata_only = True
                elif invalid == "schema":
                    request.v41_inputs.schema_version = 2
                elif invalid == "mask":
                    request.v41_inputs.image_mask[1] = False
                elif invalid == "types":
                    request.v41_inputs.images[1].types[0] = 1
                elif invalid == "patches":
                    request.v41_inputs.images[1].patches.bf16_data = b""
                else:
                    request.v41_inputs.images[1].processor_identity = "other"
                context = Context()
                with self.assertRaises(Aborted):
                    service.RemoteMultimodalEmbedding(request, context)
                self.assertEqual(context.code, grpc.StatusCode.INVALID_ARGUMENT)
                service.engine.submit_v41.assert_not_called()
                self.assertEqual(service._active, 0)

    def test_v41_rejects_image_above_processor_token_limit_before_encoding(self):
        service = self.v41_server()
        service.engine.model.mm_part.processor_config.vision_max_n_token = 3
        context = Context()
        with self.assertRaisesRegex(Aborted, "per-image token limit"):
            service.RemoteMultimodalEmbedding(self.v41_request(), context)
        self.assertEqual(context.code, grpc.StatusCode.INVALID_ARGUMENT)
        service.engine.submit_v41.assert_not_called()
        self.assertEqual(service._active, 0)

    def test_v41_checks_deadline_and_cancellation_between_images(self):
        for expired in (False, True):
            service = self.v41_server()
            context = Context()
            if expired:
                context.time_remaining = lambda: 0
            else:

                def cancel(images):
                    context.callback()
                    return MMEmbeddingRes([torch.ones((4, 4), dtype=torch.bfloat16)])

                service.engine.submit_v41.side_effect = cancel
            with self.assertRaises(Aborted):
                service.RemoteMultimodalEmbedding(self.v41_request(), context)
            self.assertEqual(
                context.code,
                grpc.StatusCode.DEADLINE_EXCEEDED
                if expired
                else grpc.StatusCode.CANCELLED,
            )
            self.assertEqual(service.engine.submit_v41.call_count, 0 if expired else 1)
            self.assertEqual(service._active, 0)

    def test_v41_rejects_missing_or_malformed_embeddings(self):
        for result in (
            MMEmbeddingRes([]),
            MMEmbeddingRes([torch.ones((3, 4), dtype=torch.bfloat16)]),
            MMEmbeddingRes([torch.ones((4, 4), dtype=torch.float32)]),
        ):
            service = self.v41_server()
            service.engine.submit_v41.side_effect = None
            service.engine.submit_v41.return_value = result
            context = Context()
            with self.assertRaises(Aborted):
                service.RemoteMultimodalEmbedding(self.v41_request(), context)
            self.assertEqual(context.code, grpc.StatusCode.INVALID_ARGUMENT)

    def test_v41_uses_rdma_without_metadata_cache_or_feature_hashing(self):
        service = self.v41_server()
        request = self.v41_request()
        request.support_rdma = True
        service.require_rdma = True
        service.rdma_encoder = FakeRdmaEncoder(
            [MMRdmaDescPB(handle=name).SerializeToString() for name in ("one", "two")]
        )
        with mock.patch.object(
            service._token_id_cache, "get", side_effect=AssertionError("metadata cache")
        ), mock.patch(
            "rtp_llm.server.vit_rpc_server.get_multimodal_feature_hash",
            side_effect=AssertionError("feature hash"),
        ):
            output = service.RemoteMultimodalEmbedding(request, Context())
        self.assertEqual(
            [item.output_rdma.handle for item in output.multimodal_outputs],
            ["one", "two"],
        )
        self.assertTrue(
            all(
                not item.HasField("multimodal_embedding")
                for item in output.multimodal_outputs
            )
        )

    def test_v41_cancellation_after_export_releases_every_slot(self):
        service = self.v41_server()
        request = self.v41_request()
        request.support_rdma = True
        context = Context()
        encoder = FakeRdmaEncoder(
            [MMRdmaDescPB(handle=name).SerializeToString() for name in ("one", "two")]
        )
        export = encoder.export_embedding

        def cancel_after_export(embedding):
            descriptor = export(embedding)
            context.callback()
            return descriptor

        encoder.export_embedding = cancel_after_export
        service.rdma_encoder = encoder
        with self.assertRaises(Aborted):
            service.RemoteMultimodalEmbedding(request, context)
        self.assertEqual(context.code, grpc.StatusCode.CANCELLED)
        self.assertEqual(encoder.released, ["one", "two"])
        self.assertEqual(service._active, 0)

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

    def test_metadata_cache_separates_padding_and_type_for_the_same_url(self):
        server = self.server(MMEmbeddingRes([torch.ones((2, 4))]))
        server.engine.submit = mock.Mock(
            return_value=MMEmbeddingRes([torch.ones((2, 4))])
        )
        first = MultimodalInputPB(multimodal_url="image", multimodal_type=1)
        first.mm_preprocess_config.mm_padding_size = 0
        phase = MultimodalInputPB()
        phase.CopyFrom(first)
        phase.mm_preprocess_config.mm_padding_size = 1
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
