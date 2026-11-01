"""CPU-only protocol fixtures; never model or performance evidence."""
import io
import json
import socket
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

from tools.engram_sidecar.framework import http_decode_benchmark as bench


def chunk(text=None, *, count=None, finish=None, aux=None, role=None):
    delta = {}
    if text is not None:
        delta["content"] = text
    if role:
        delta["role"] = role
    result = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if count is not None:
        result["usage"] = {"prompt_tokens": 12, "completion_tokens": count, "total_tokens": 12 + count}
    if aux is not None:
        result["aux_info"] = aux
    return result


def auxiliary(count=5, **fields):
    return {"cost_time": 18, "first_token_cost_time": 10, "output_len": count,
            "iter_count": 3, "step_output_len": 3, "reuse_len": 0, **fields}


def body(events, done=False):
    data = b": heartbeat\r\n\r\n" + b"".join(
        ("data: " + json.dumps(event, ensure_ascii=False) + "\r\n\r\n").encode() for event in events)
    return data + (b"data: [DONE]\n\n" if done else b"")


def valid_events():
    return [chunk(role="assistant"), chunk("你好", count=2), chunk("世界", count=5),
            chunk("", count=5, finish="length", aux=auxiliary())]


class Response(io.BytesIO):
    def __init__(self, data, *, status=200, content_type="text/event-stream; charset=utf-8"):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Type": content_type}


class ProtocolTest(unittest.TestCase):
    def record(self, events=None, **kwargs):
        response = Response(body(valid_events() if events is None else events), **kwargs)
        return bench.measure("http://localhost/v1/chat/completions", bench.payload("test", 11, 5),
                             expected_prompt_tokens=12, opener=SimpleNamespace(open=lambda *a, **k: response))

    def test_main_sse_without_done_uses_final_cumulative_count_not_events(self):
        result = self.record()
        self.assertEqual(result["status"], "ok")
        metrics = result["metrics"]
        self.assertEqual(metrics["usage"]["completion_tokens"], 5)
        self.assertEqual(metrics["text_event_count"], 2)
        self.assertEqual(metrics["engine_stream_average_tpot_ms"], 2)
        self.assertEqual(metrics["aux_info"]["step_output_len"], 3)
        self.assertEqual(metrics["content"], "你好世界")
        self.assertEqual(len(result["events"]), 4)
        self.assertGreater(metrics["client_visible_text_ttft_ms"], metrics["first_sse_event_ms"])

    def test_done_crlf_comments_and_multiline_data(self):
        raw = b':ignore\r\nevent: message\r\nid: ignored\r\ndata: {"choices": [],\r\ndata: "usage":null}\r\n\r\n'
        events = list(bench.sse_events(io.BytesIO(raw), started=0, clock=lambda: 1))
        self.assertEqual(json.loads(events[0]["data"])["choices"], [])
        observer = bench.Observation()
        for event in bench.sse_events(io.BytesIO(body(valid_events(), done=True)), started=0, clock=lambda: 1):
            observer.accept(event)
        self.assertTrue(observer.done)
        self.assertEqual(observer.summary(eof_ms=1001, expected_prompt_tokens=12, requested_tokens=5)["content"], "你好世界")

    def test_missing_final_or_counts_is_invalid_and_all_events_retained(self):
        result = self.record(valid_events()[:-1])
        self.assertEqual(result["error_code"], "eof_without_finish_reason")
        self.assertEqual(len(result["events"]), 3)
        events = [chunk("hello"), chunk("", finish="stop", aux=auxiliary())]
        self.assertEqual(self.record(events)["error_code"], "missing_or_insufficient_final_usage")
        events = [chunk("hello", count=5, aux=auxiliary()), chunk("", finish="stop")]
        self.assertEqual(self.record(events)["error_code"], "missing_or_insufficient_final_usage")
        events[-1] = chunk("", count=5, finish="stop")
        result = self.record(events)
        self.assertIsNone(result["metrics"]["engine_stream_average_tpot_ms"])

    def test_error_in_200_stream_is_failure_and_preserved(self):
        error = {"error_code": 8500, "message": "test routing error"}
        record = self.record([chunk("partial", count=1), error])
        self.assertEqual(record["error_code"], "server_inference_error")
        self.assertEqual(json.loads(record["events"][-1]["data"]), error)

    def test_http_error_and_non_sse_body_preserved(self):
        def failed(*args, **kwargs):
            raise urllib.error.HTTPError("http://localhost", 503, "unavailable", {}, io.BytesIO(b'failure body'))
        record = bench.measure("http://localhost", bench.payload("test", 11, 5), expected_prompt_tokens=12,
                               opener=SimpleNamespace(open=failed))
        self.assertEqual(record["http_status"], 503)
        self.assertEqual(record["error_body"], "failure body")
        record = self.record(content_type="application/json")
        self.assertEqual(record["error_code"], "unexpected_http_response")
        self.assertIn("data:", record["unexpected_body"])

    def test_http_error_body_timeout_preserves_record_without_exception_text(self):
        class BrokenBody(io.BytesIO):
            def read(self, *args):
                raise socket.timeout("private-transport-detail")
        stream = BrokenBody()
        def failed(*args, **kwargs):
            raise urllib.error.HTTPError("http://localhost", 503, "unavailable", {}, stream)
        record = bench.measure("http://localhost", bench.payload("test", 11, 5), expected_prompt_tokens=12,
                               opener=SimpleNamespace(open=failed))
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["http_status"], 503)
        self.assertEqual(record["error_body_read_code"], "body_read_failed")
        self.assertTrue(stream.closed)
        self.assertNotIn("private-transport-detail", json.dumps(record))

    def test_response_close_failure_invalidates_previously_complete_measurement(self):
        class BrokenClose(Response):
            def __exit__(self, *args):
                super().__exit__(*args)
                raise socket.timeout("private-close-detail")
        response = BrokenClose(body(valid_events()))
        record = bench.measure("http://localhost", bench.payload("test", 11, 5), expected_prompt_tokens=12,
                               opener=SimpleNamespace(open=lambda *a, **k: response))
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error_code"], "transport_or_client_error")
        self.assertEqual(len(record["events"]), 4)
        self.assertNotIn("private-close-detail", json.dumps(record))

    def test_content_filter_or_tool_finish_is_not_successful_decode(self):
        events = valid_events()
        events[-1]["choices"][0]["finish_reason"] = "content_filter"
        self.assertEqual(self.record(events)["error_code"], "unexpected_finish_reason")

    def test_missing_bad_engine_aux_does_not_fabricate_step_statistics(self):
        for aux in (None, auxiliary(cost_time=9), auxiliary(output_len=4), auxiliary(first_token_cost_time=0)):
            events = valid_events()
            events[-1] = chunk("", count=5, finish="length", aux=aux)
            record = self.record(events)
            self.assertEqual(record["status"], "ok")
            self.assertIsNone(record["metrics"]["engine_stream_average_tpot_ms"])
            self.assertIn("engine_average_tpot_unavailable", record["metrics"]["warnings"])

    def test_wrong_prompt_or_regressed_cumulative_count_rejected(self):
        events = valid_events()
        events[-1]["usage"].update(prompt_tokens=13, total_tokens=18)
        self.assertEqual(self.record(events)["error_code"], "prompt_token_count_changed")
        events = valid_events()
        events[-1]["usage"].update(completion_tokens=4, total_tokens=16)
        self.assertEqual(self.record(events)["error_code"], "nonmonotonic_cumulative_usage")

    def test_truncated_utf8_invalid_json_done_and_boundaries(self):
        for raw, code in ((b'data: {}\n', "truncated_sse_event"),
                          (b'data: \xff\n\n', "invalid_sse_utf8"),
                          (b'data: {}', "truncated_sse_line")):
            with self.subTest(code=code), self.assertRaisesRegex(bench.StreamError, code):
                list(bench.sse_events(io.BytesIO(raw), started=0))
        for data, code in (("[DONE]", "done_without_finish_reason"),
                           ('{"choices":[],"usage":NaN}', "invalid_sse_json"),
                           ('not-json', "invalid_sse_json")):
            observer = bench.Observation()
            with self.assertRaisesRegex(bench.StreamError, code):
                observer.accept({"received_ms": 1, "data": data})
            self.assertEqual(observer.events[0]["data"], data)
        with patch.object(bench, "MAX_EVENT_BYTES", 8), self.assertRaisesRegex(bench.StreamError, "size_limit"):
            list(bench.sse_events(io.BytesIO(b"data: 123456789\n\n"), started=0))

    def test_empty_role_content_does_not_count_as_visible_ttft(self):
        observer = bench.Observation()
        for index, event in enumerate(valid_events()):
            observer.accept({"received_ms": index * 10, "data": json.dumps(event)})
        result = observer.summary(eof_ms=40, expected_prompt_tokens=12, requested_tokens=5)
        self.assertEqual(result["first_sse_event_ms"], 0)
        self.assertEqual(result["client_visible_text_ttft_ms"], 10)
        self.assertEqual(result["text_event_intervals_ms"], [10])
        self.assertEqual(result["client_text_window_per_final_token_ms"], 2.5)


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [self.root / label for label in bench.LABEL_MODES]
        self.template = bench.payload("same prompt", 0, 5)
        self.report = {"storage_mode": "local-pinned", "model_revision": "unit-test-not-a-model",
                       "checkpoint": {"config_sha256": "config", "tokenizer_sha256": "tokenizer", "index_sha256": "index",
                                      "shard_header_sha256": {"fixture.safetensors": "header"},
                                      "full_layer_count": 40, "engram_layers": [1, 14], "consumer_has_table_entries": True},
                       "runtime": {"native_sha256": {"engine": "sha"}},
                       "experiment_environment": {"ENABLE_CUDA_GRAPH": "1"}}
        self.records = []
        for run_index, (path, label) in enumerate(zip(self.paths, bench.LABEL_MODES)):
            path.mkdir()
            identity = bench.public_identity(self.report)
            identity.update(storage_mode=bench.LABEL_MODES[label], consumer_has_table_entries=label.startswith("A"))
            manifest = {"cycle": "fixture", "label": label, "identity": identity, "seeds": [11], "warmups": 1,
                        "expected_prompt_tokens": 12, "request_template_sha256": bench.fingerprint(self.template),
                        "started_unix_ns": 10 * run_index}
            bench.write_new(path / "manifest.json", manifest)
            bench.write_new(path / "complete.json", {"finished_unix_ns": 10 * run_index + 5, "requests": 2})
            for index in (0, 1):
                record = ProtocolTest().record()
                record.update(index=index, seed=11, warmup=index == 0, label=label, cycle="fixture",
                              request=bench.payload("same prompt", 11, 5))
                if label.startswith("B"):
                    record["metrics"]["engine_stream_average_tpot_ms"] += 0.2
                bench.write_new(path / f"request-{index:05d}.json", record)

    def replace(self, path, function):
        data = json.loads(path.read_text())
        function(data)
        path.write_text(json.dumps(data))

    def test_abba_means_and_outlier_retained_without_full_slo_pass(self):
        result = bench.compare_runs(self.paths)
        self.assertTrue(result["comparable"])
        self.assertEqual(len(result["rows"]), 2)
        row = result["rows"][1]
        self.assertAlmostEqual(row["block_mean_delta_us"], 200)
        self.assertEqual(row["paired_request_mean_delta_exceeds_100us"], [True, True])
        self.assertIn("not_established", result["full_decode_step_100us_acceptance"])

    def test_request_failure_invalidates_quartet_does_not_discard(self):
        self.replace(self.paths[1] / "request-00001.json", lambda x: x.update(status="error", error_code="timeout"))
        result = bench.compare_runs(self.paths)
        self.assertFalse(result["comparable"])
        self.assertIn("request_failure", result["rows"][1]["invalid_reasons"])
        self.assertEqual(len(result["rows"]), 2)

    def test_different_generated_text_invalidates_even_same_token_count(self):
        self.replace(self.paths[1] / "request-00001.json", lambda x: x["metrics"].update(content="other output"))
        result = bench.compare_runs(self.paths)
        self.assertFalse(result["comparable"])
        self.assertIn("different_generated_text", result["rows"][1]["invalid_reasons"])

    def test_missing_cache_count_or_different_cache_invalidates(self):
        self.replace(self.paths[1] / "request-00001.json", lambda x: x["metrics"]["aux_info"].update(reuse_len=3))
        result = bench.compare_runs(self.paths)
        self.assertIn("different_reuse_length", result["rows"][1]["invalid_reasons"])
        self.replace(self.paths[1] / "request-00001.json", lambda x: x["metrics"]["aux_info"].pop("reuse_len"))
        self.assertIn("missing_reuse_length", bench.compare_runs(self.paths)["rows"][1]["invalid_reasons"])

    def test_order_and_runtime_mutation_rejected(self):
        with self.assertRaisesRegex(ValueError, "ordered A1"):
            bench.compare_runs(self.paths[::-1])
        self.replace(self.paths[1] / "manifest.json", lambda x: x.update(started_unix_ns=2))
        with self.assertRaisesRegex(ValueError, "overlap"):
            bench.compare_runs(self.paths)
        self.replace(self.paths[1] / "manifest.json", lambda x: x.update(started_unix_ns=10))
        self.replace(self.paths[1] / "manifest.json", lambda x: x["identity"]["experiment_environment"].update(ENABLE_CUDA_GRAPH="0"))
        with self.assertRaisesRegex(ValueError, "settings differ"):
            bench.compare_runs(self.paths)

    def test_same_mode_index_and_headers_must_match(self):
        for key in ("index_sha256", "shard_header_sha256"):
            path = self.paths[3] / "manifest.json"
            original = path.read_text()
            self.replace(path, lambda x: x["identity"].update({key: "changed"}))
            with self.assertRaisesRegex(ValueError, "same-mode checkpoint"):
                bench.compare_runs(self.paths)
            path.write_text(original)
    def test_request_body_mutation_rejected(self):
        self.replace(self.paths[1] / "request-00001.json", lambda x: x["request"].update(temperature=0))
        with self.assertRaisesRegex(ValueError, "request body differs"):
            bench.compare_runs(self.paths)

    def test_outputs_exclusive_and_private(self):
        path = self.root / "private.json"
        bench.write_new(path, {"a": 1})
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            bench.write_new(path, {"a": 2})

    def test_public_report_never_copies_private_bundle(self):
        self.report["bundle"] = {"capability": "secret"}
        self.assertNotIn("secret", json.dumps(bench.public_identity(self.report)))

    def test_run_records_each_failure_and_continues_serially(self):
        report = self.root / "launch.json"
        report.write_text(json.dumps(self.report))
        prompt = self.root / "prompt.txt"
        prompt.write_text("same prompt")
        args = SimpleNamespace(endpoint="http://localhost/v1/chat/completions", tokens=5, expected_prompt_tokens=12,
                               warmups=1, timeout=10, seeds="11,23", launch_report=report, label="A1", cycle="ut",
                               prompt_file=prompt, temperature=0.7, output=self.root / "new-run")
        def result(endpoint, request, **kwargs):
            return {"status": "error", "request": request, "events": [], "error_code": "fixture_error"}
        with patch.object(bench, "measure", side_effect=result) as called, patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(bench.run(args), 1)
        self.assertEqual(called.call_count, 3)
        files = sorted(args.output.glob("request-*.json"))
        self.assertEqual(len(files), 3)
        self.assertEqual(json.loads(files[-1].read_text())["seed"], 23)
        self.assertEqual(json.loads((args.output / "complete.json").read_text())["failures"], 3)


if __name__ == "__main__":
    unittest.main()
