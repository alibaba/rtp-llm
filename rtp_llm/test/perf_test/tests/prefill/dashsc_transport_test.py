"""Performance framework behavior and failure regressions."""

import struct
import unittest
from unittest.mock import Mock

import requests

from rtp_llm.test.perf_test.execution.runtime import HttpPrefillClient


class DashScTransportTest(unittest.TestCase):

    def _client(self, responses):
        from rtp_llm.dash_sc.proto import predict_v2_pb2
        from rtp_llm.test.perf_test.prefill.dashsc_transport import DashScPrefillClient

        captured = []

        class Stub:

            def ModelStreamInfer(self, requests, **kwargs):
                captured.extend(
                    (
                        predict_v2_pb2.ModelInferRequest.FromString(
                            r.SerializeToString()
                        )
                        for r in requests
                    )
                )
                return iter(responses)

        channel = Mock()
        http = HttpPrefillClient("http://localhost:18000", post=Mock())
        return (
            DashScPrefillClient(http, stub=Stub(), channel=channel),
            captured,
            channel,
        )

    def _request(self, client):
        return client.request(
            phase="measure",
            case_id="case-a",
            request_index=2,
            run_index=1,
            prompt="must never be tokenized",
            input_ids=[7, 123456, 9],
            output_length=1,
        )

    def test_exact_int32_wire_ids_and_http_response_parity(self):
        from rtp_llm.dash_sc.proto import predict_v2_pb2

        response = predict_v2_pb2.ModelStreamInferResponse()
        infer = response.infer_response
        infer.parameters["engine_first_token_cost_time_us"].int64_param = 12500
        infer.parameters["engine_wait_time_us"].int64_param = 1250
        infer.parameters["error_no"].int64_param = 0
        infer.parameters["error_msg"].string_param = ""
        infer.parameters["status_code"].int64_param = 200
        for name, shape, values in (
            ("prompt_token_num", [1], [3]),
            ("prompt_cached_token_num", [1], [2]),
            ("generated_ids", [1, 1], [42]),
        ):
            output = infer.outputs.add(name=name, datatype="INT32")
            output.shape.extend(shape)
            infer.raw_output_contents.append(
                struct.pack("<" + "i" * len(values), *values)
            )
        (client, captured, channel) = self._client([response])
        self.assertEqual(
            {
                "success": True,
                "input_len": 3,
                "output_len": 1,
                "reuse_len": 2,
                "latency_ms": 12.5,
                "ttft_ms": 12.5,
                "wait_time_ms": 1.25,
            },
            self._request(client),
        )
        request = captured[0]
        tensors = {
            tensor.name: (tensor, raw)
            for (tensor, raw) in zip(request.inputs, request.raw_input_contents)
        }
        (tensor, raw) = tensors["input_ids"]
        self.assertEqual("INT32", tensor.datatype)
        self.assertEqual([3], list(tensor.shape))
        self.assertEqual((7, 123456, 9), struct.unpack("<3i", raw))
        self.assertEqual((1,), struct.unpack("<i", tensors["max_new_tokens"][1]))
        self.assertEqual((1,), struct.unpack("<i", tensors["min_new_tokens"][1]))
        self.assertTrue(request.parameters["force_sp_accept"].bool_param)
        client.close()
        client.close()
        channel.close.assert_called_once()

    def _assert_business_error_stream(self, *, valid_first):
        from rtp_llm.dash_sc.proto import predict_v2_pb2

        for fields in (
            {"error_no": 5},
            {"error_msg": "model unavailable"},
            {"status_code": 503},
            {"error_no": 5, "error_msg": "model unavailable", "status_code": 503},
        ):
            with self.subTest(fields=fields):
                responses = []
                if valid_first:
                    valid = predict_v2_pb2.ModelStreamInferResponse()
                    infer = valid.infer_response
                    infer.parameters["engine_first_token_cost_time_us"].int64_param = (
                        12500
                    )
                    for name, value in (
                        ("prompt_token_num", 3),
                        ("prompt_cached_token_num", 2),
                        ("generated_ids", 42),
                    ):
                        output = infer.outputs.add(name=name, datatype="INT32")
                        output.shape.append(1)
                        infer.raw_output_contents.append(struct.pack("<i", value))
                    responses.append(valid)
                error = predict_v2_pb2.ModelStreamInferResponse()
                for name, value in fields.items():
                    parameter = error.infer_response.parameters[name]
                    if isinstance(value, str):
                        parameter.string_param = value
                    else:
                        parameter.int64_param = value
                responses.append(error)
                (client, _, _) = self._client(responses)
                result = self._request(client)
                self.assertFalse(result["success"])
                self.assertIn("DashSC", result["error"])
                if "error_msg" in fields:
                    self.assertIn("model unavailable", result["error"])

    def test_business_error_after_valid_frame_invalidates_request(self):
        self._assert_business_error_stream(valid_first=True)


if __name__ == "__main__":
    unittest.main()
