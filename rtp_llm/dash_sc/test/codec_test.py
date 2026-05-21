"""Unit tests for ``rtp_llm.dash_sc.codec`` (request parsing + response builders)."""

from __future__ import annotations

import json
import struct
from unittest import TestCase, main

import torch

from rtp_llm.dash_sc.codec import (
    OtherParams,
    SamplingParams,
    build_stream_response_from_generate_outputs,
    parse_dash_sc_grpc_request,
    parse_input_ids_from_request,
    parse_other_params,
    parse_sampling_params,
    prepend_to_generated_ids_tensor,
)
from rtp_llm.dash_sc.inference.servicer import stream_log_tag
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


def _unpack_int32_le(raw: bytes) -> list[int]:
    return list(struct.unpack("<%di" % (len(raw) // 4), raw))


def _unpack_int64_le(raw: bytes) -> list[int]:
    return [int(x) for x in struct.unpack("<%dq" % (len(raw) // 8), raw)]


def _add_tensor(
    req: predict_v2_pb2.ModelInferRequest,
    name: str,
    datatype: str,
    shape: list[int],
    raw: bytes,
) -> None:
    inp = req.inputs.add()
    inp.name = name
    inp.datatype = datatype
    inp.shape[:] = shape
    req.raw_input_contents.append(raw)


class DashScGrpcRequestTest(TestCase):
    def test_parse_input_ids_int32(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        raw = struct.pack("<3i", 10, 20, 30)
        _add_tensor(req, "input_ids", "INT32", [3], raw)
        self.assertEqual(parse_input_ids_from_request(req), [10, 20, 30])

    def test_parse_input_ids_int64(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        raw = struct.pack("<2q", 7, -1)
        _add_tensor(req, "input_ids", "INT64", [2], raw)
        self.assertEqual(parse_input_ids_from_request(req), [7, -1])

    def test_parse_input_ids_missing_tensor(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        self.assertIsNone(parse_input_ids_from_request(req))

    def test_parse_input_ids_raw_index_mismatch(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        inp = req.inputs.add()
        inp.name = "input_ids"
        inp.datatype = "INT32"
        inp.shape.append(1)
        self.assertIsNone(parse_input_ids_from_request(req))

    def test_parse_input_ids_bad_length(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "input_ids", "INT32", [2], b"\x01\x02\x03")
        self.assertIsNone(parse_input_ids_from_request(req))

    def test_parse_sampling_defaults(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        sp = parse_sampling_params(req)
        self.assertIsInstance(sp, SamplingParams)
        self.assertEqual(sp.max_new_tokens, 32000)
        self.assertEqual(sp.top_k, 0)
        self.assertEqual(sp.top_p, 1.0)
        self.assertEqual(sp.stop_words_list, ())

    def test_parse_sampling_scalars(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", 128))
        _add_tensor(req, "num_return_sequences", "INT32", [1], struct.pack("<i", 2))
        _add_tensor(req, "top_p", "FP32", [1], struct.pack("<f", 0.9))
        _add_tensor(req, "top_k", "INT32", [1], struct.pack("<i", 50))
        _add_tensor(req, "temperature", "FP32", [1], struct.pack("<f", 0.7))
        _add_tensor(req, "min_new_tokens", "INT32", [1], struct.pack("<i", 3))
        _add_tensor(req, "seed", "INT64", [1], struct.pack("<q", 999))
        _add_tensor(req, "repetition_penalty", "FP32", [1], struct.pack("<f", 1.1))
        _add_tensor(req, "frequency_penalty", "FP32", [1], struct.pack("<f", 0.2))
        _add_tensor(req, "presence_penalty", "FP32", [1], struct.pack("<f", 0.3))
        sp = parse_sampling_params(req)
        self.assertEqual(sp.max_new_tokens, 128)
        self.assertEqual(sp.num_return_sequences, 2)
        self.assertAlmostEqual(sp.top_p, 0.9)
        self.assertEqual(sp.top_k, 50)
        self.assertAlmostEqual(sp.temperature, 0.7)
        self.assertEqual(sp.min_new_tokens, 3)
        self.assertEqual(sp.random_seed, 999)
        self.assertAlmostEqual(sp.repetition_penalty, 1.1)
        self.assertAlmostEqual(sp.frequency_penalty, 0.2)
        self.assertAlmostEqual(sp.presence_penalty, 0.3)

    def test_parse_sampling_dashscope_aliases(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "n", "INT32", [1], struct.pack("<i", 1))
        _add_tensor(req, "min_length", "INT32", [1], struct.pack("<i", 2))
        sp = parse_sampling_params(req)
        self.assertEqual(sp.num_return_sequences, 1)
        self.assertEqual(sp.min_new_tokens, 2)

    def test_parse_sampling_max_completion_tokens_parameter_alias(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["max_completion_tokens"].int64_param = 100
        sp = parse_sampling_params(req)
        self.assertEqual(sp.max_new_tokens, 100)

    def test_parse_sampling_top_p_as_int32(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "top_p", "INT32", [1], struct.pack("<i", 1))
        sp = parse_sampling_params(req)
        self.assertEqual(sp.top_p, 1.0)

    def test_parse_sampling_legacy_top_k_parameter(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        p = predict_v2_pb2.InferParameter()
        p.int64_param = 88
        req.parameters["top_k"].CopyFrom(p)
        sp = parse_sampling_params(req)
        self.assertEqual(sp.top_k, 88)

    def test_parse_sampling_stop_words_list_2d(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        flat = [1, 2, 3, 4]
        raw = struct.pack("<%di" % len(flat), *flat)
        _add_tensor(req, "stop_words_list", "INT32", [2, 2], raw)
        sp = parse_sampling_params(req)
        self.assertEqual(sp.stop_words_list, ((1, 2), (3, 4)))
        self.assertEqual(sp.stop_words_list_py(), [[1, 2], [3, 4]])

    def test_sampling_params_n_alias(self) -> None:
        sp = SamplingParams(num_return_sequences=5)
        self.assertEqual(sp.n, 5)

    def test_parse_other_params_default(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        op = parse_other_params(req)
        self.assertEqual(op, OtherParams(return_input_ids=False))

    def test_parse_other_params_bool(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "return_input_ids", "BOOL", [1], b"\x01")
        op = parse_other_params(req)
        self.assertTrue(op.return_input_ids)

    def test_parse_other_params_int32(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "return_input_ids", "INT32", [1], struct.pack("<i", 0))
        op = parse_other_params(req)
        self.assertFalse(op.return_input_ids)

    def test_parse_other_params_thinking_controls(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {
                "x-ds-llm-thinking": "false",
                "x-dashscope-inner-timeout": 1800,
                "x-ds-request-priority": "10",
                "user_id": "u1",
                "x-dashscope-apikeyid": "ak1",
            }
        )
        _add_tensor(req, "max_new_think_tokens", "INT32", [1], struct.pack("<i", 0))
        op = parse_other_params(req)
        self.assertFalse(op.return_input_ids)
        self.assertIs(op.enable_thinking, False)
        self.assertEqual(op.max_new_think_tokens, 0)
        self.assertEqual(op.timeout_ms, 1_800_000)
        self.assertEqual(op.traffic_reject_priority, 10)
        self.assertEqual(
            op.request_headers, {"user_id": "u1", "x-dashscope-apikeyid": "ak1"}
        )

    def test_parse_other_params_dashscope_body_thinking_aliases(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["enable_thinking"].bool_param = False
        req.parameters["thinking_budget"].int64_param = 100
        op = parse_other_params(req)
        self.assertIs(op.enable_thinking, False)
        self.assertEqual(op.max_new_think_tokens, 100)

    def test_parse_dash_sc_grpc_request_ok(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "input_ids", "INT32", [2], struct.pack("<2i", 1, 2))
        _add_tensor(req, "top_k", "INT32", [1], struct.pack("<i", 10))
        _add_tensor(req, "return_input_ids", "BOOL", [1], b"\x01")
        ids, sp, op = parse_dash_sc_grpc_request(req)
        self.assertEqual(ids, [1, 2])
        self.assertIsNotNone(sp)
        self.assertIsNotNone(op)
        assert sp is not None and op is not None
        self.assertEqual(sp.top_k, 10)
        self.assertTrue(op.return_input_ids)

    def test_parse_dash_sc_grpc_request_no_input_ids(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        ids, sp, op = parse_dash_sc_grpc_request(req)
        self.assertIsNone(ids)
        self.assertIsNone(sp)
        self.assertIsNone(op)

    def test_sampling_to_generate_config(self) -> None:
        sp = SamplingParams(max_new_tokens=64, top_k=1, stop_words_list=((42,),))
        gc = sp.to_generate_config(other=OtherParams(return_input_ids=True))
        self.assertEqual(gc.max_new_tokens, 64)
        self.assertEqual(gc.top_k, 1)
        self.assertEqual(gc.stop_words_list, [[42]])
        self.assertTrue(gc.return_input_ids)


class BuildStreamResponseFromGenerateOutputsTest(TestCase):
    def test_empty_generate_outputs_raises(self) -> None:
        go = GenerateOutputs(generate_outputs=[])
        with self.assertRaises(ValueError) as ctx:
            build_stream_response_from_generate_outputs(
                dash_sc_request_id="r1",
                model_name="m",
                go=go,
                request_log_tag=stream_log_tag(request_id_numeric=1, trace_id="r1"),
            )
        self.assertIn("non-empty", str(ctx.exception))

    def test_basic_generated_ids_finish_aux(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([7, 8, 9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=10, reuse_len=4),
        )
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="req-a",
            model_name="mdl",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=99, trace_id="req-a"),
            return_input_ids=False,
        )
        self.assertFalse(resp.error_message)
        infer = resp.infer_response
        self.assertEqual(infer.id, "req-a")
        self.assertEqual(infer.model_name, "mdl")
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [7, 8, 9])
        self.assertEqual(list(infer.outputs[0].shape), [1, 3])
        self.assertEqual(_unpack_int64_le(by_name["finish_reason"]), [0])
        self.assertEqual(_unpack_int32_le(by_name["prompt_token_num"]), [10])
        self.assertEqual(_unpack_int32_le(by_name["prompt_cached_token_num"]), [4])

    def test_not_finished_finish_reason_two(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([1], dtype=torch.int32),
            finished=False,
            aux_info=None,
        )
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=0, trace_id="r"),
        )
        infer = resp.infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int64_le(by_name["finish_reason"]), [2])
        self.assertEqual(_unpack_int32_le(by_name["prompt_token_num"]), [0])
        self.assertEqual(_unpack_int32_le(by_name["prompt_cached_token_num"]), [0])

    def test_output_ids_2d_uses_first_row(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([[100, 101]], dtype=torch.int32),
            finished=True,
        )
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=1, trace_id="r"),
        )
        infer = resp.infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [100, 101])

    def test_return_input_ids_prepends_prompt_tensor(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([5], dtype=torch.int32),
            finished=True,
        )
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=1, trace_id="r"),
            request_input_ids=[1, 2, 3],
            return_input_ids=True,
        )
        infer = resp.infer_response
        names = [infer.outputs[i].name for i in range(len(infer.outputs))]
        self.assertEqual(
            names,
            [
                "prompt_token_ids",
                "generated_ids",
                "finish_reason",
                "finished",
                "prompt_token_num",
                "prompt_cached_token_num",
            ],
        )
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["prompt_token_ids"]), [1, 2, 3])
        self.assertEqual(list(infer.outputs[0].shape), [1, 3])

    def test_missing_output_ids_empty_generated(self) -> None:
        out = GenerateOutput(output_ids=None, finished=False, aux_info=AuxInfo())
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=1, trace_id="r"),
        )
        infer = resp.infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(by_name["generated_ids"], struct.pack("<i", 0))
        self.assertEqual(list(infer.outputs[0].shape), [1, 0])


class PrependToGeneratedIdsTensorTest(TestCase):
    def _build_infer_with_generated_ids(self, ids: list[int]):
        out = GenerateOutput(
            output_ids=torch.tensor(ids, dtype=torch.int32) if ids else None,
            finished=True,
            aux_info=AuxInfo(input_len=0, reuse_len=0),
        )
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=1, trace_id="r"),
        )
        return resp.infer_response

    def test_prepend_list_success(self) -> None:
        infer = self._build_infer_with_generated_ids([7, 8, 9])
        self.assertTrue(prepend_to_generated_ids_tensor(infer, [100, 200]))
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(
            _unpack_int32_le(by_name["generated_ids"]), [100, 200, 7, 8, 9]
        )
        gen_out = next(o for o in infer.outputs if o.name == "generated_ids")
        self.assertEqual(list(gen_out.shape), [1, 5])

    def test_prepend_empty_list_noop(self) -> None:
        infer = self._build_infer_with_generated_ids([7, 8, 9])
        self.assertFalse(prepend_to_generated_ids_tensor(infer, []))
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [7, 8, 9])

    def test_prepend_to_empty_tensor_noop(self) -> None:
        infer = self._build_infer_with_generated_ids([])
        self.assertFalse(prepend_to_generated_ids_tensor(infer, [100]))
        gen_out = next(o for o in infer.outputs if o.name == "generated_ids")
        self.assertEqual(list(gen_out.shape), [1, 0])

    def test_prepend_without_generated_ids_output_returns_false(self) -> None:
        resp = predict_v2_pb2.ModelStreamInferResponse()
        self.assertFalse(prepend_to_generated_ids_tensor(resp.infer_response, [100]))


class StreamLogTagTest(TestCase):
    def test_stream_log_tag_format(self) -> None:
        self.assertEqual(
            stream_log_tag(request_id_numeric=-1, trace_id="tid"),
            "request_id=-1 trace_id=tid",
        )


if __name__ == "__main__":
    main()
