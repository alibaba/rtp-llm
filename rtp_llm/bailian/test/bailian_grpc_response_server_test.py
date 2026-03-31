"""Tests for Bailian gRPC response build, real infer stream, and servicer."""

from __future__ import annotations

import struct
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.bailian import bailian_grpc_server as bgs
from rtp_llm.bailian import bailian_grpc_service as bg_svc
from rtp_llm.bailian.bailian_grpc_real_infer import (
    _derive_rtp_llm_request_id,
    iter_real_model_stream_infer,
    stream_log_tag,
)
from rtp_llm.bailian.bailian_grpc_request import OtherParams, SamplingParams
from rtp_llm.bailian.bailian_grpc_response_real import (
    build_stream_response_from_generate_outputs,
)
from rtp_llm.bailian.proto import predict_v2_pb2
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


def _add_input_tensor(
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


def _unpack_int32_le(raw: bytes) -> list[int]:
    return list(struct.unpack("<%di" % (len(raw) // 4), raw))


def _unpack_int64_le(raw: bytes) -> list[int]:
    return [int(x) for x in struct.unpack("<%dq" % (len(raw) // 8), raw)]


class BuildStreamResponseFromGenerateOutputsTest(TestCase):
    def test_empty_generate_outputs_raises(self) -> None:
        go = GenerateOutputs(generate_outputs=[])
        with self.assertRaises(ValueError) as ctx:
            build_stream_response_from_generate_outputs(
                bailian_request_id="r1",
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
            bailian_request_id="req-a",
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
            bailian_request_id="r",
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
            bailian_request_id="r",
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
            bailian_request_id="r",
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
            bailian_request_id="r",
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


class DeriveRequestIdAndTagTest(TestCase):
    def test_derive_stable(self) -> None:
        a = _derive_rtp_llm_request_id("same")
        b = _derive_rtp_llm_request_id("same")
        self.assertEqual(a, b)
        self.assertNotEqual(_derive_rtp_llm_request_id("other"), a)

    def test_stream_log_tag_format(self) -> None:
        self.assertEqual(
            stream_log_tag(request_id_numeric=-1, trace_id="tid"),
            "request_id=-1 trace_id=tid",
        )


class IterRealModelStreamInferTest(TestCase):
    def _minimal_request(self) -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = "trace-real"
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [2], struct.pack("<2i", 1, 2))
        return req

    def test_yields_one_chunk_from_mock_enqueue(self) -> None:
        req = self._minimal_request()
        sampling = SamplingParams()
        other = OtherParams()

        def fake_sync(_visitor, _gi):
            out = GenerateOutput(
                output_ids=torch.tensor([3, 4], dtype=torch.int32),
                finished=True,
                aux_info=AuxInfo(input_len=2, reuse_len=0),
            )
            return [GenerateOutputs(generate_outputs=[out])]

        chunks = list(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                sampling,
                other,
                MagicMock(),
                run_enqueue_sync=fake_sync,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].error_message)
        infer = chunks[0].infer_response
        self.assertEqual(infer.id, "trace-real")
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [3, 4])

    def test_empty_list_yields_error_response(self) -> None:
        req = self._minimal_request()

        def empty_sync(_v, _g):
            return []

        chunks = list(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                OtherParams(),
                MagicMock(),
                run_enqueue_sync=empty_sync,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("empty outputs_list", chunks[0].error_message)

    def test_enqueue_exception_yields_error_message(self) -> None:
        req = self._minimal_request()

        def boom(_v, _g):
            raise RuntimeError("backend down")

        chunks = list(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                OtherParams(),
                MagicMock(),
                run_enqueue_sync=boom,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("backend down", chunks[0].error_message)


class BailianGrpcInferenceServicerTest(TestCase):
    def _valid_infer_request(self) -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = "srv-1"
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [1], struct.pack("<i", 42))
        return req

    def test_fake_mode_returns_incremented_ids(self) -> None:
        servicer = bgs.BailianGrpcInferenceServicer(backend_visitor=None)
        req = self._valid_infer_request()
        responses = list(servicer.ModelStreamInfer(iter([req]), MagicMock()))
        self.assertEqual(len(responses), 1)
        infer = responses[0].infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [142])

    def test_missing_input_ids_error(self) -> None:
        servicer = bgs.BailianGrpcInferenceServicer(backend_visitor=None)
        bad = predict_v2_pb2.ModelInferRequest()
        bad.id = "x"
        bad.model_name = "m"
        responses = list(servicer.ModelStreamInfer(iter([bad]), MagicMock()))
        self.assertEqual(len(responses), 1)
        self.assertIn("input_ids", responses[0].error_message)

    @patch.object(bg_svc, "_iter_enqueue_sync")
    def test_real_mode_uses_enqueue(self, mock_iter: MagicMock) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=1, reuse_len=0),
        )
        mock_iter.side_effect = lambda *a, **k: iter(
            [GenerateOutputs(generate_outputs=[out])]
        )

        servicer = bgs.BailianGrpcInferenceServicer(backend_visitor=MagicMock())
        req = self._valid_infer_request()
        responses = list(servicer.ModelStreamInfer(iter([req]), MagicMock()))
        self.assertEqual(len(responses), 1)
        mock_iter.assert_called_once()
        infer = responses[0].infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [9])


if __name__ == "__main__":
    main()
