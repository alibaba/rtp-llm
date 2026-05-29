"""Unit tests for compact dash_sc proxy forward access summaries."""

from __future__ import annotations

import asyncio
import json
import struct
import time
import unittest

import grpc

from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.dash_sc.proxy.access_record import (
    ForwardAccessRecord,
    extract_forward_frame_stats,
)


def _make_request() -> predict_v2_pb2.ModelInferRequest:
    req = predict_v2_pb2.ModelInferRequest()
    req.id = "req-1"
    req.model_name = "m"

    inp = req.inputs.add()
    inp.name = "input_ids"
    inp.datatype = "INT32"
    inp.shape.extend([1, 4])
    req.raw_input_contents.append(struct.pack("<4i", 424242, 424243, 424244, 424245))

    inp = req.inputs.add()
    inp.name = "max_new_tokens"
    inp.datatype = "INT32"
    inp.shape.append(1)
    req.raw_input_contents.append(struct.pack("<i", 32))

    inp = req.inputs.add()
    inp.name = "top_p"
    inp.datatype = "FP32"
    inp.shape.append(1)
    req.raw_input_contents.append(struct.pack("<f", 0.8))

    inp = req.inputs.add()
    inp.name = "return_input_ids"
    inp.datatype = "BOOL"
    inp.shape.append(1)
    req.raw_input_contents.append(b"\x01")

    inp = req.inputs.add()
    inp.name = "enable_thinking"
    inp.datatype = "FP32"
    inp.shape.append(1)
    req.raw_input_contents.append(struct.pack("<f", 1.0))

    inp = req.inputs.add()
    inp.name = "stop_words_list"
    inp.datatype = "INT32"
    inp.shape.extend([2, 2])
    req.raw_input_contents.append(struct.pack("<4i", 515151, 515152, 515153, 515154))

    req.parameters["top_k"].int64_param = 50
    return req


def _make_response(
    *,
    generated_len: int = 0,
    finish_reason: int | None = None,
    finished: bool | None = None,
    prompt_token_num: int | None = None,
    prompt_cached_token_num: int | None = None,
) -> predict_v2_pb2.ModelStreamInferResponse:
    resp = predict_v2_pb2.ModelStreamInferResponse()
    infer = resp.infer_response

    out = infer.outputs.add()
    out.name = "generated_ids"
    out.datatype = "INT32"
    out.shape.extend([1, generated_len])
    raw = (
        struct.pack("<%di" % generated_len, *range(1000, 1000 + generated_len))
        if generated_len
        else struct.pack("<i", 0)
    )
    infer.raw_output_contents.append(raw)

    if finish_reason is not None:
        out = infer.outputs.add()
        out.name = "finish_reason"
        out.datatype = "INT64"
        out.shape.append(1)
        infer.raw_output_contents.append(struct.pack("<q", finish_reason))
    if finished is not None:
        out = infer.outputs.add()
        out.name = "finished"
        out.datatype = "BOOL"
        out.shape.append(1)
        infer.raw_output_contents.append(b"\x01" if finished else b"\x00")
    if prompt_token_num is not None:
        out = infer.outputs.add()
        out.name = "prompt_token_num"
        out.datatype = "INT32"
        out.shape.append(1)
        infer.raw_output_contents.append(struct.pack("<i", prompt_token_num))
    if prompt_cached_token_num is not None:
        out = infer.outputs.add()
        out.name = "prompt_cached_token_num"
        out.datatype = "INT32"
        out.shape.append(1)
        infer.raw_output_contents.append(struct.pack("<i", prompt_cached_token_num))
    return resp


class ForwardAccessRecordTest(unittest.TestCase):
    def test_request_summary_uses_shape_and_whitelisted_config(self) -> None:
        record = ForwardAccessRecord(
            method="/m",
            stream_type="bidi_stream",
            peer="peer",
            start_ts=time.time(),
        )
        record.capture_request(_make_request())

        self.assertEqual(record.request_id, "req-1")
        self.assertEqual(record.model_name, "m")
        self.assertEqual(record.input_token_len, 4)
        self.assertEqual(record.generate_config["max_new_tokens"], 32)
        self.assertAlmostEqual(record.generate_config["top_p"], 0.8, places=5)
        self.assertEqual(record.generate_config["top_k"], 50)
        self.assertEqual(record.generate_config["return_input_ids"], True)
        self.assertEqual(record.generate_config["enable_thinking"], True)
        self.assertEqual(record.generate_config["stop_words_group_count"], 2)
        self.assertEqual(record.generate_config["stop_words_token_count"], 4)

        out = record.build_record(server_id=1, rank_id=0)
        self.assertEqual(out["capture_mode"], "forward_summary")
        self.assertEqual(out["legacy_capture_mode"], "raw")
        self.assertIsNone(out["input_len"])
        self.assertIsNone(out["output_len"])
        self.assertIsNone(out["input_ids"])
        self.assertIsNone(out["generated_ids"])
        rendered = json.dumps(out)
        self.assertNotIn("515151", rendered)
        self.assertNotIn("424242", rendered)
        self.assertFalse(hasattr(record, "generated_ids"))

    def test_response_stats_count_tokens_without_returning_ids(self) -> None:
        stats = extract_forward_frame_stats(
            _make_response(generated_len=3, finish_reason=2, finished=False)
        )
        self.assertEqual(stats.output_token_len, 3)
        self.assertFalse(stats.terminal_seen)

        terminal = extract_forward_frame_stats(
            _make_response(generated_len=0, finish_reason=0, finished=True)
        )
        self.assertEqual(terminal.output_token_len, 0)
        self.assertTrue(terminal.terminal_seen)

    def test_record_tracks_terminal_and_backend_aux(self) -> None:
        record = ForwardAccessRecord(
            method="/m",
            stream_type="bidi_stream",
            peer="peer",
            start_ts=time.time(),
        )
        record.mark_backend_call_start("127.0.0.1:9000", 0)
        record.capture_backend_response_chunk(
            _make_response(
                generated_len=2,
                prompt_token_num=7,
                prompt_cached_token_num=3,
            )
        )
        record.mark_first_resp()
        record.resp_count += 1
        record.capture_response_chunk(_make_response(generated_len=2))
        record.mark_first_resp()
        record.resp_count += 1
        record.capture_response_chunk(
            _make_response(generated_len=0, finish_reason=0, finished=True)
        )
        record.mark_backend_done()
        record.mark_stream_close(_FakeContext(), None)

        out = record.build_record(server_id=1, rank_id=0)
        self.assertEqual(out["backend_addr"], "127.0.0.1:9000")
        self.assertEqual(out["backend_resp_count"], 1)
        self.assertEqual(out["backend_input_token_len"], 7)
        self.assertEqual(out["backend_prompt_cached_token_num"], 3)
        self.assertEqual(
            out["backend_aux_info"],
            {"prompt_token_num": 7, "prompt_cached_token_num": 3},
        )
        self.assertEqual(out["output_token_len"], 2)
        self.assertNotIn("iteration_count", out)
        self.assertNotIn("response_iteration_count", out)
        self.assertEqual(out["finished"], True)
        self.assertTrue(out["terminal_seen"])
        self.assertEqual(out["finished_only_frame_count"], 1)
        self.assertIsNotNone(out["finish_to_close_ms"])

    def test_tpot_boundaries(self) -> None:
        record = ForwardAccessRecord(
            method="/m",
            stream_type="bidi_stream",
            peer="peer",
            start_ts=10.0,
        )
        record.stream_close_ts = 11.0

        out = record.build_record(server_id=1, rank_id=0)
        self.assertIsNone(out["latency_tpot_ms"])

        record.first_token_ts = 10.2
        record.last_token_ts = 10.2
        record.output_token_len = 1
        out = record.build_record(server_id=1, rank_id=0)
        self.assertIsNone(out["latency_tpot_ms"])

        record.last_token_ts = 10.5
        record.output_token_len = 4
        out = record.build_record(server_id=1, rank_id=0)
        self.assertEqual(out["latency_tpot_ms"], 100.0)

    def test_close_reason_classification_paths(self) -> None:
        record = ForwardAccessRecord("/m", "bidi_stream", "peer", time.time())
        record.error_message = "ValueError: bad frame"
        record.mark_stream_close(_FakeContext(), None)
        self.assertEqual(record.close_reason, "backend_protocol_error")

        record = ForwardAccessRecord("/m", "bidi_stream", "peer", time.time())
        record.mark_stream_close(_FakeContext(), asyncio.CancelledError())
        self.assertEqual(record.close_reason, "client_cancel_before_terminal")

        record = ForwardAccessRecord("/m", "bidi_stream", "peer", time.time())
        record.terminal_seen = True
        record.mark_stream_close(_FakeContext(), asyncio.CancelledError())
        self.assertEqual(record.close_reason, "client_cancel_after_terminal")

        class AbortError(RuntimeError):
            pass

        record = ForwardAccessRecord("/m", "bidi_stream", "peer", time.time())
        record.mark_stream_close(_FakeContext(), AbortError())
        self.assertEqual(record.close_reason, "proxy_context_abort")

        record = ForwardAccessRecord("/m", "bidi_stream", "peer", time.time())
        record.backend_exc_type = "AioRpcError"
        record.mark_stream_close(_FakeContext(), None)
        self.assertEqual(record.close_reason, "backend_transport_fail")

        record = ForwardAccessRecord("/m", "bidi_stream", "peer", time.time())
        record.mark_stream_close(_FakeContext(grpc.StatusCode.CANCELLED), None)
        self.assertEqual(record.close_reason, "client_cancel_before_terminal")

        record = ForwardAccessRecord("/m", "bidi_stream", "peer", time.time())
        record.mark_stream_close(_FakeContext(grpc.StatusCode.INTERNAL), None)
        self.assertEqual(record.close_reason, "grpc_context_non_ok")

        record = ForwardAccessRecord("/m", "bidi_stream", "peer", time.time())
        record.mark_stream_close(_FakeContext(grpc.StatusCode.OK), None)
        self.assertEqual(record.close_reason, "eof")

    def test_mark_backend_error_extracts_grpc_code_and_detail(self) -> None:
        class BackendError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

            def details(self):
                return "backend unavailable"

        record = ForwardAccessRecord("/m", "bidi_stream", "peer", time.time())
        record.mark_backend_error(BackendError())

        self.assertEqual(record.backend_exc_type, "BackendError")
        self.assertEqual(record.backend_rpc_code, "UNAVAILABLE")
        self.assertEqual(record.backend_rpc_detail, "backend unavailable")


class _FakeContext:
    def __init__(self, code=None):
        self._code = code

    def code(self):
        return self._code

    def is_active(self):
        return False


if __name__ == "__main__":
    unittest.main()
