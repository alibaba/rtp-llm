"""CPU-only real gRPC tests: propagation, terminal status, and thread reuse."""

import asyncio
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import grpc
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from rtp_llm.telemetry import attributes as attrs
from rtp_llm.telemetry import tracing, vit_media
from rtp_llm.telemetry.vit_rpc import (
    VitClientTraceInterceptor,
    VitServerTraceInterceptor,
    vit_client_span,
)

METHOD = "/MultimodalRpcService/RemoteMultimodalEmbedding"


def request(request_id=42):
    return SimpleNamespace(request_id=request_id, multimodal_inputs=["private-url"])


def rpc(channel, method=METHOD):
    return channel.unary_unary(
        method,
        request_serializer=lambda r: str(r.request_id).encode(),
        response_deserializer=lambda data: data,
    )


class VitRpcTraceTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(tracing.reset_telemetry_for_test())
        self.exporter = InMemorySpanExporter()
        self.assertTrue(tracing.init_telemetry_for_test(self.exporter, role="vit"))
        self.servers = []
        self.channels = []

    def tearDown(self):
        for channel in self.channels:
            channel.close()
        for server, executor in self.servers:
            server.stop(None).wait(3)
            executor.shutdown(wait=True)
        tracing.reset_telemetry_for_test()

    def server(self, handler):
        executor = ThreadPoolExecutor(max_workers=1)
        server = grpc.server(executor, interceptors=[VitServerTraceInterceptor()])
        handlers = {
            method: grpc.unary_unary_rpc_method_handler(
                handler,
                request_deserializer=lambda data: request(int(data)),
                response_serializer=lambda response: response,
            )
            for method in (
                "RemoteMultimodalEmbedding",
                "GetWorkerStatus",
                "GetCacheStatus",
                "AsyncSubmitEmbedding",
            )
        }
        server.add_generic_rpc_handlers(
            [grpc.method_handlers_generic_handler("MultimodalRpcService", handlers)]
        )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        self.servers.append((server, executor))
        target = f"127.0.0.1:{port}"
        channel = grpc.insecure_channel(target)
        self.channels.append(channel)
        return target, channel

    def spans(self, count):
        deadline = time.monotonic() + 3
        while True:
            tracing._provider.force_flush()
            spans = self.exporter.get_finished_spans()
            if len(spans) >= count or time.monotonic() >= deadline:
                self.assertEqual(len(spans), count)
                return spans
            time.sleep(0.01)

    def test_proxy_future_preserves_parentage_and_business_metadata(self):
        received = []

        def worker(req, ctx):
            received.append(dict(ctx.invocation_metadata()))
            return b"ok"

        worker_target, worker_channel = self.server(worker)
        worker_channel = grpc.intercept_channel(
            worker_channel, VitClientTraceInterceptor(worker_target)
        )

        def proxy(req, ctx):
            return (
                rpc(worker_channel)
                .future(req, timeout=3, metadata=[("x-dashscope-uid", "private-id")])
                .result()
            )

        proxy_target, proxy_channel = self.server(proxy)
        root = tracing.start_server_span("frontend", {})
        with vit_client_span(
            "RemoteMultimodalEmbedding", proxy_target, request()
        ) as metadata:
            self.assertEqual(
                rpc(proxy_channel)(request(), timeout=3, metadata=metadata), b"ok"
            )
        root.finish()
        spans = self.spans(5)
        by_id = {s.context.span_id: s for s in spans}
        parents = {s.parent.span_id for s in spans if s.parent}
        leaf = next(s for s in spans if s.context.span_id not in parents)
        kinds = []
        while leaf.parent:
            kinds.append(leaf.kind)
            self.assertEqual(leaf.status.status_code, StatusCode.OK)
            self.assertEqual(leaf.attributes[attrs.REQUEST_ID], "42")
            self.assertEqual(leaf.attributes[attrs.RTP_LLM_VIT_INPUT_COUNT], 1)
            leaf = by_id[leaf.parent.span_id]
        self.assertEqual(
            kinds, [SpanKind.SERVER, SpanKind.CLIENT, SpanKind.SERVER, SpanKind.CLIENT]
        )
        self.assertEqual(received[0]["x-dashscope-uid"], "private-id")
        self.assertIn("traceparent", received[0])
        self.assertNotIn("private", str([dict(s.attributes) for s in spans]))

    def test_media_and_stages_keep_ingress_trace_id_through_proxy(self):
        @vit_media.request
        def process(mm_inputs, request_id):
            with vit_media.stage("preprocess"):
                pass

        def worker(req, _ctx):
            process(
                [SimpleNamespace(url=url, mm_type=1) for url in req.multimodal_inputs],
                request_id=req.request_id,
            )
            return b"ok"

        target, worker_channel = self.server(worker)
        worker_channel = grpc.intercept_channel(
            worker_channel, VitClientTraceInterceptor(target)
        )

        def proxy(req, _ctx):
            return rpc(worker_channel).future(req, timeout=3).result()

        target, channel = self.server(proxy)
        trace_id = "0af7651916cd43dd8448eb211c80319c"
        root = tracing.start_server_span(
            "frontend", {"traceparent": f"00-{trace_id}-b7ad6b7169203331-01"}
        )
        try:
            with vit_client_span(
                "RemoteMultimodalEmbedding", target, request()
            ) as metadata:
                self.assertEqual(
                    rpc(channel)(request(), timeout=3, metadata=metadata), b"ok"
                )
        finally:
            root.finish()
        spans = self.spans(7)
        self.assertEqual(
            {format(s.context.trace_id, "032x") for s in spans}, {trace_id}
        )
        media = next(s for s in spans if s.name == vit_media.PREFIX + "media")
        stage = next(s for s in spans if s.name == vit_media.PREFIX + "preprocess")
        self.assertEqual(stage.parent.span_id, media.context.span_id)
        for span in (media, stage):
            self.assertEqual(span.attributes[attrs.RTP_LLM_REQUEST_TRACE_ID], trace_id)

    def test_abort_and_set_code_are_errors_and_threads_do_not_leak_parent(self):
        def handler(req, ctx):
            if req.request_id == 1:
                ctx.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "private-error")
            if req.request_id == 2:
                ctx.set_code(grpc.StatusCode.PERMISSION_DENIED)
            return b"ok"

        _, channel = self.server(handler)
        for value, expected in [
            (1, grpc.StatusCode.RESOURCE_EXHAUSTED),
            (2, grpc.StatusCode.PERMISSION_DENIED),
        ]:
            with self.assertRaises(grpc.RpcError) as caught:
                rpc(channel)(request(value), timeout=3)
            self.assertEqual(caught.exception.code(), expected)
        rpc(channel)(request(3), timeout=3)
        spans = self.spans(3)
        self.assertEqual(len({s.context.trace_id for s in spans}), 3)
        self.assertTrue(all(s.parent is None for s in spans))
        self.assertEqual(
            [s.status.status_code for s in spans],
            [StatusCode.ERROR, StatusCode.ERROR, StatusCode.OK],
        )
        self.assertNotIn("private-error", str(spans))

    def test_polling_and_unsampled_parent_do_not_export(self):
        _, channel = self.server(lambda req, ctx: b"ok")
        for method in ("GetWorkerStatus", "GetCacheStatus"):
            rpc(channel, "/MultimodalRpcService/" + method)(request(), timeout=3)
        rpc(channel)(
            request(),
            timeout=3,
            metadata=[
                (
                    "traceparent",
                    "00-12345678901234567890123456789012-1234567890123456-00",
                )
            ],
        )
        self.spans(0)

    def test_async_client_cancellation_and_metadata_preservation(self):
        root = tracing.start_server_span("frontend", {})

        async def cancelled():
            with vit_client_span(
                "GetMultimodalHashes",
                "localhost:1234",
                request(),
                [("x-dashscope-uid", "private-id")],
            ) as metadata:
                self.assertIn(("x-dashscope-uid", "private-id"), metadata)
                raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(cancelled())
        root.finish()
        child = next(s for s in self.spans(2) if s.kind == SpanKind.CLIENT)
        self.assertEqual(child.attributes[attrs.RPC_RESPONSE_STATUS_CODE], "CANCELLED")
        self.assertEqual(child.status.status_code, StatusCode.ERROR)

    def test_deadline_finishes_both_client_and_server_as_errors(self):
        entered = threading.Event()
        cancelled = threading.Event()

        def worker(req, ctx):
            ctx.add_callback(cancelled.set)
            entered.set()
            cancelled.wait(3)
            return b"ok"

        target, channel = self.server(worker)
        channel = grpc.intercept_channel(channel, VitClientTraceInterceptor(target))
        root = tracing.start_server_span("frontend", {})
        future = rpc(channel).future(request(), timeout=0.2)
        self.assertTrue(entered.wait(2))
        with self.assertRaises(grpc.RpcError) as caught:
            future.result(timeout=3)
        self.assertEqual(caught.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)
        root.finish()
        for span in self.spans(3):
            if span.name != "frontend":
                self.assertEqual(
                    span.attributes[attrs.RPC_RESPONSE_STATUS_CODE], "DEADLINE_EXCEEDED"
                )
                self.assertEqual(span.status.status_code, StatusCode.ERROR)

    def test_cache_probe_has_no_fabricated_request_id_or_media_attributes(self):
        root = tracing.start_server_span("frontend", {})
        probe = SimpleNamespace(
            keys=["private-key"], inputs=request(0), HasField=lambda name: False
        )
        with vit_client_span("GetMultimodalHashes", "localhost:1234", probe):
            pass
        root.finish()
        child = next(s for s in self.spans(2) if s.kind == SpanKind.CLIENT)
        self.assertNotIn(attrs.REQUEST_ID, child.attributes)
        self.assertNotIn(attrs.RTP_LLM_VIT_INPUT_COUNT, child.attributes)
        self.assertTrue(child.attributes[attrs.RTP_LLM_VIT_CACHE_PROBE])
        self.assertEqual(child.attributes[attrs.RTP_LLM_VIT_KEY_COUNT], 1)

    def test_disabled_path_does_not_load_sdk_or_change_rpc(self):
        tracing.reset_telemetry_for_test()
        _, channel = self.server(lambda req, ctx: b"ok")
        with mock.patch.object(
            tracing, "_load_otel_sdk", side_effect=AssertionError("SDK imported")
        ):
            with vit_client_span(
                "RemoteMultimodalEmbedding",
                "localhost:1234",
                request(),
                [("business", "value")],
            ) as metadata:
                self.assertEqual(metadata, [("business", "value")])
                self.assertEqual(rpc(channel)(request(), timeout=3), b"ok")


if __name__ == "__main__":
    unittest.main()
