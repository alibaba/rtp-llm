"""Unit tests for ``rtp_llm.bailian.bailian_grpc_request``."""

from __future__ import annotations

import struct
from unittest import TestCase, main

from rtp_llm.bailian.bailian_grpc_request import (
    OtherParams,
    SamplingParams,
    parse_bailian_grpc_request,
    parse_input_ids_from_request,
    parse_other_params,
    parse_sampling_params,
)
from rtp_llm.bailian.proto import predict_v2_pb2


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


class BailianGrpcRequestTest(TestCase):
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

    def test_parse_bailian_grpc_request_ok(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "input_ids", "INT32", [2], struct.pack("<2i", 1, 2))
        _add_tensor(req, "top_k", "INT32", [1], struct.pack("<i", 10))
        _add_tensor(req, "return_input_ids", "BOOL", [1], b"\x01")
        ids, sp, op = parse_bailian_grpc_request(req)
        self.assertEqual(ids, [1, 2])
        self.assertIsNotNone(sp)
        self.assertIsNotNone(op)
        assert sp is not None and op is not None
        self.assertEqual(sp.top_k, 10)
        self.assertTrue(op.return_input_ids)

    def test_parse_bailian_grpc_request_no_input_ids(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        ids, sp, op = parse_bailian_grpc_request(req)
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


if __name__ == "__main__":
    main()
