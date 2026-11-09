import concurrent.futures
import gc
import pickle
import queue
import threading
import time
import unittest
import weakref
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import requests
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from rtp_llm.telemetry import attributes as attrs
from rtp_llm.telemetry import tracing, vit_media


class VitMediaTraceTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(tracing.reset_telemetry_for_test())
        self.exporter = InMemorySpanExporter()
        self.assertTrue(tracing.init_telemetry_for_test(self.exporter, role="vit"))
        self.root = tracing.start_server_span("rpc", {})

    def tearDown(self):
        self.root.finish()
        tracing.reset_telemetry_for_test()

    def spans(self):
        tracing._provider.force_flush()
        return self.exporter.get_finished_spans()

    def inputs(self):
        return [
            SimpleNamespace(
                url="https://media.invalid/a.jpg?signature=raw#fragment", mm_type=1
            )
            for _ in range(2)
        ]

    def test_raw_urls_duplicate_occurrences_and_worker_intervals(self):
        values = self.inputs()
        media = vit_media._new_media(values, 42)
        with vit_media.collect_worker(
            vit_media.worker_spec(media), preprocess=True
        ) as recorder:
            with vit_media.input_scope(0):

                @vit_media.download
                def load():
                    return b"image"

                self.assertEqual(load(), b"image")
            with vit_media.input_scope(1):
                load()
        payload = pickle.loads(pickle.dumps(recorder))
        vit_media.consume(media, payload)
        for item in media:
            item.finish()
        spans = self.spans()
        roots = [s for s in spans if s.name == "rtp_llm.vit.media"]
        self.assertEqual(
            [s.attributes[vit_media.MEDIA + "index"] for s in roots], [0, 1]
        )
        for root in roots:
            self.assertEqual(root.attributes[vit_media.MEDIA + "url"], values[0].url)
            self.assertFalse(any("url_hash" in key for key in root.attributes))
            children = [s for s in spans if s.parent.span_id == root.context.span_id]
            self.assertTrue(
                {
                    "rtp_llm.vit.preprocess.queue",
                    "rtp_llm.vit.download",
                    "rtp_llm.vit.preprocess",
                }.issubset({s.name for s in children})
            )
            downloads = [s for s in children if s.name.endswith(".download")]
            cpu = [s for s in children if s.name.endswith(".preprocess")]
            self.assertEqual(len(downloads), 1)
            for interval in cpu:
                self.assertTrue(
                    interval.end_time <= downloads[0].start_time
                    or interval.start_time >= downloads[0].end_time
                )

    def test_async_lifetime_and_cache_consumer_link(self):
        entry = SimpleNamespace(error=None)
        captured = []

        @vit_media.request
        def submit(mm_inputs, request_id=0):
            media = vit_media.current()[0]
            vit_media.claim(media, entry, "miss")
            media.begin("compute.queue")
            captured.append(media)

        submit(self.inputs()[:1], request_id=3)
        self.assertFalse(any(s.name.endswith(".media") for s in self.spans()))
        with concurrent.futures.ThreadPoolExecutor(1) as pool:

            @vit_media.compute_task
            def run(_self, _inputs, _key, _entry):
                self.assertEqual(vit_media.current(), (captured[0],))
                with vit_media.stage("embedding"):
                    pass
                vit_media.finish_producer(entry)

            pool.submit(run, None, [], "key", entry).result()
            self.assertEqual(pool.submit(vit_media.current).result(), ())
        observer = vit_media._new_media(self.inputs()[:1], 4)[0]
        vit_media.claim(observer, entry, "complete")
        observer.finish()
        roots = [s for s in self.spans() if s.name.endswith(".media")]
        self.assertEqual(len(roots), 2)
        self.assertEqual(roots[1].links[0].context.span_id, roots[0].context.span_id)
        self.assertEqual(vit_media.current(), ())

    def test_finished_handle_does_not_retain_sdk_span_or_its_context(self):
        for error in (None, ValueError("failed")):
            with self.subTest(error=type(error).__name__):
                media = vit_media._new_media(self.inputs()[:1], 5)[0]
                original_context = media.span.get_span_context()
                sdk_span_ref = weakref.ref(media.span)
                media.begin("compute.queue")
                media.finish(error)
                media.finish(error)
                spans = self.spans()
                gc.collect()
                self.assertIsNone(sdk_span_ref())
                self.assertFalse(media.span.is_recording())
                self.assertEqual(media.span.get_span_context(), original_context)
                self.assertEqual(
                    tracing.trace.get_current_span(media.context).get_span_context(),
                    original_context,
                )
                finished = [span for span in spans if span.context == original_context]
                self.assertEqual(len(finished), 1)
                self.assertEqual(
                    finished[0].status.status_code,
                    StatusCode.ERROR if error else StatusCode.OK,
                )
                self.assertEqual(
                    len(
                        [
                            span
                            for span in spans
                            if span.parent
                            and span.parent.span_id == original_context.span_id
                        ]
                    ),
                    1,
                )

    def test_otlp_upload_includes_request_trace_id_on_media_and_stages(self):
        payloads = queue.Queue()

        class Collector(BaseHTTPRequestHandler):
            def do_POST(self):
                payloads.put(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Collector)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        session = requests.Session()
        session.trust_env = False
        exporter = OTLPSpanExporter(
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/traces",
            session=session,
            timeout=3,
        )
        try:
            self.root.finish()
            tracing.reset_telemetry_for_test()
            self.assertTrue(tracing.init_telemetry_for_test(exporter, role="vit"))
            trace_id = "0af7651916cd43dd8448eb211c80319c"
            self.root = tracing.start_server_span(
                "rpc", {"traceparent": f"00-{trace_id}-b7ad6b7169203331-01"}
            )
            media = vit_media._new_media(self.inputs(), 6)
            with vit_media.stage("download", media):
                pass
            for member in media:
                member.finish()
            self.root.finish()
            self.assertTrue(tracing._provider.force_flush())
            uploaded = []
            while not payloads.empty():
                message = ExportTraceServiceRequest.FromString(payloads.get_nowait())
                uploaded.extend(
                    span
                    for resource in message.resource_spans
                    for scope in resource.scope_spans
                    for span in scope.spans
                )
            spans = [s for s in uploaded if s.name.startswith(vit_media.PREFIX)]
            self.assertEqual(len(spans), 4)
            for span in spans:
                self.assertEqual(span.trace_id.hex(), trace_id)
                attributes = {a.key: a.value for a in span.attributes}
                self.assertEqual(
                    attributes[attrs.RTP_LLM_REQUEST_TRACE_ID].string_value, trace_id
                )
            roots = [s for s in spans if s.name.endswith(".media")]
            self.assertEqual(len(roots), 2)
            for span in roots:
                attributes = {a.key: a.value for a in span.attributes}
                self.assertEqual(attributes[attrs.REQUEST_ID].string_value, "6")
                self.assertEqual(
                    attributes[vit_media.MEDIA + "url"].string_value,
                    self.inputs()[0].url,
                )
        finally:
            tracing.reset_telemetry_for_test()
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()
            session.close()

    def test_batch_links_preserve_distinct_request_traces(self):
        first = vit_media._new_media(self.inputs()[:1], 1)[0]
        other_root = tracing.start_server_span("other_rpc", {})
        second = vit_media._new_media(self.inputs()[:1], 2)[0]
        items = [
            SimpleNamespace(mm_inputs=[1], media_traces=(member,))
            for member in (first, second)
        ]
        vit_media.enqueue(items)

        @vit_media.embedding_batch
        def embed(_model, _items):
            return "unchanged"

        self.assertEqual(embed(None, items), "unchanged")
        first.finish()
        second.finish()
        other_root.finish()
        spans = self.spans()
        batch = next(s for s in spans if s.name.endswith("embedding.batch"))
        self.assertEqual(
            {link.context.trace_id for link in batch.links},
            {
                first.span.get_span_context().trace_id,
                second.span.get_span_context().trace_id,
            },
        )
        self.assertEqual(batch.attributes[vit_media.PREFIX + "batch.media_count"], 2)
        self.assertEqual(len([s for s in spans if s.name.endswith(".embedding")]), 2)
        for span in spans:
            if span.name.startswith(vit_media.PREFIX):
                self.assertEqual(
                    span.attributes[attrs.RTP_LLM_REQUEST_TRACE_ID],
                    format(span.context.trace_id, "032x"),
                )

    def test_failures_are_bounded_and_do_not_export_exception_messages(self):
        media = vit_media._new_media(self.inputs()[:1], 1)
        with vit_media.collect_worker(vit_media.worker_spec(media)) as recorder:
            with self.assertRaises(ValueError):
                with vit_media.stage("download"):
                    raise ValueError("not-an-attribute")
            for _ in range(300):
                with vit_media.stage("preprocess"):
                    pass
        self.assertEqual(len(recorder.records), 256)
        self.assertEqual(recorder.dropped, 45)
        vit_media.consume(media, recorder)
        media[0].finish(ValueError())
        download = next(s for s in self.spans() if s.name.endswith(".download"))
        self.assertEqual(download.status.status_code, StatusCode.ERROR)
        self.assertEqual(download.attributes["error.type"], "ValueError")
        self.assertNotIn("not-an-attribute", str(download.attributes))

    def test_worker_verdict_and_download_cache_hit(self):
        media = vit_media._new_media(self.inputs()[:1], 1)
        with vit_media.collect_worker(vit_media.worker_spec(media)) as recorder:
            with vit_media.stage("greennet"):
                vit_media.record_verdict(SimpleNamespace(passed=False, code=2))

            @vit_media.download
            def cached():
                vit_media.download_cache_hit()

            cached()
        vit_media.consume(media, recorder)
        media[0].finish()
        spans = self.spans()
        green = next(s for s in spans if s.name.endswith(".greennet"))
        self.assertEqual(green.status.status_code, StatusCode.ERROR)
        self.assertEqual(green.attributes["error.type"], "GREENNET_REJECTED")
        self.assertTrue(any(s.name.endswith("download.cache") for s in spans))
        self.assertFalse(any(s.name.endswith(".download") for s in spans))

    def test_unsampled_and_disabled_paths_do_not_read_urls(self):
        self.root.finish()
        tracing._provider.force_flush()
        self.exporter.clear()
        parent = tracing.start_server_span(
            "unsampled",
            {"traceparent": "00-1234567890abcdef1234567890abcdef-1234567890abcdef-00"},
        )
        self.assertEqual(vit_media._new_media([object()], 1), ())
        parent.finish()
        self.assertEqual(len(self.spans()), 0)
        tracing.reset_telemetry_for_test()
        self.assertEqual(vit_media._new_media([object()], 1), ())


if __name__ == "__main__":
    unittest.main()
