"""Core performance behavior contracts."""

import struct
import unittest
from unittest.mock import Mock

import requests

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.execution.runtime import (
    HttpPrefillClient,
    build_engine_launch_spec,
)
from rtp_llm.test.perf_test.execution.scheduler import configure_scheduler
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.test_support import (
    fixture_engine,
    fixture_model,
    fixture_workload,
)


def runtime_adapter_plan(backend="local", transport="http_text"):
    source = {
        "schema_version": 1,
        "model": fixture_model(
            label="model",
            type="qwen_2",
            checkpoint="/models/qwen",
            tokenizer="/models/qwen-tokenizer",
            topology={"tp": 2, "dp": 1, "pp": 1, "ep": 2},
        ),
        "engine": fixture_engine(
            concurrency_limit=4,
            args={"reuse_cache": 1, "enable_cuda_graph": 0},
            env={"TOKENIZERS_PARALLELISM": "false"},
        ),
        "workload": fixture_workload(
            transport=transport,
            max_input_len=1024,
            compute_steps=[256],
            cache_lengths=[0],
        ),
        "execution": (
            {"backend": "local", "devices": [3, 5]}
            if backend == "local"
            else {
                "backend": "aone",
                "resource": "daily",
                "gpu_type": "H20_CU13",
                "max_parallel": 1,
                "shard_count": 1,
            }
        ),
        "output": {"result_root": "/tmp/results"},
        "regression": {},
        "publish": {},
    }
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(source),
        cases=[
            {"input_len": 1024, "cache_len": 0, "compute_len": 1024, "batch_size": 1}
        ],
        result_directory="/tmp/results/run",
    )


class decode_transport_Response:

    def __init__(self, payload=None, *, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload


class RuntimeAdapterTest(unittest.TestCase):

    def test_launch_spec_is_derived_only_from_resolved_plan(self):
        spec = build_engine_launch_spec(runtime_adapter_plan())
        self.assertEqual((3, 5), spec.device_ids)
        self.assertEqual("false", spec.environment["TOKENIZERS_PARALLELISM"])
        self.assertEqual("2", spec.environment["WORLD_SIZE"])
        self.assertEqual("1", spec.environment["USE_BATCH_DECODE_SCHEDULER"])
        self.assertIn("--model_type=qwen_2", spec.argv)
        self.assertIn("--checkpoint_path=/models/qwen", spec.argv)
        self.assertIn("--tokenizer_path=/models/qwen-tokenizer", spec.argv)
        self.assertIn("--tp_size=2", spec.argv)
        self.assertIn("--ep_size=2", spec.argv)
        self.assertNotIn("--pp_size=1", spec.argv)
        self.assertIn("--max_batch_tokens_size=8192", spec.argv)
        self.assertIn("--enable_cuda_graph=0", spec.argv)

    def test_http_prefill_client_projects_server_timing_and_reuse(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "aux_info": {
                "input_len": 128,
                "output_len": 1,
                "reuse_len": 64,
                "first_token_cost_time": 12.5,
            }
        }
        post = Mock(return_value=response)
        client = HttpPrefillClient(
            "http://127.0.0.1:18000", post=post, timeout_seconds=30
        )
        result = client.request(
            phase="measure",
            case_id="case-a",
            request_index=0,
            run_index=1,
            prompt="hello",
            input_ids=[1, 2],
            output_length=1,
        )
        self.assertEqual("http://127.0.0.1:18000", post.call_args.args[0])
        self.assertEqual(
            {
                "success": True,
                "input_len": 128,
                "output_len": 1,
                "reuse_len": 64,
                "latency_ms": 12.5,
            },
            result,
        )
        request = post.call_args.kwargs
        self.assertEqual("hello", request["json"]["prompt"])
        self.assertEqual(1, request["json"]["generate_config"]["max_new_tokens"])
        self.assertNotIn("input_ids", request["json"])


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


class ExecutionSchedulerTest(unittest.TestCase):

    def test_requires_explicit_success_acknowledgement(self):
        response = Mock()
        response.json.return_value = {"status": "error", "error": "bad mode"}
        post = Mock(return_value=response)
        with self.assertRaisesRegex(RuntimeError, "scheduler rejected"):
            configure_scheduler(1234, batch_size=2, mode="decode", post=post)
        response.raise_for_status.assert_called_once_with()


class DecodeHttpTransportTest(unittest.TestCase):

    def test_round_forces_planned_output_after_user_generate_config(self):
        post = Mock(
            return_value=decode_transport_Response(
                {
                    "aux_info": {
                        "input_len": 128,
                        "output_len": 16,
                        "wait_time": 1.0,
                        "first_token_cost_time": 3.0,
                        "cost_time": 18.0,
                    }
                }
            )
        )
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000/",
            prompt_for_length=lambda length: f"prompt-{length}",
            post=post,
        )
        config = {
            "min_new_tokens": 2,
            "max_new_tokens": 2,
            "top_p": 0.8,
            "top_k": 3,
            "force_sp_accept": False,
            "nested": {"value": 1},
        }
        outcomes = transport.run_round(
            input_lengths=(128,), output_length=16, generate_config=config
        )
        request = post.call_args.kwargs["json"]
        self.assertEqual("http://127.0.0.1:18000", post.call_args.args[0])
        self.assertEqual("prompt-128", request["prompt"])
        self.assertEqual(16, request["generate_config"]["min_new_tokens"])
        self.assertEqual(16, request["generate_config"]["max_new_tokens"])
        self.assertEqual(0.8, request["generate_config"]["top_p"])
        self.assertEqual(3, request["generate_config"]["top_k"])
        self.assertTrue(request["generate_config"]["force_sp_accept"])
        request["generate_config"]["nested"]["value"] = 2
        self.assertEqual(1, config["nested"]["value"])
        self.assertEqual(2, config["max_new_tokens"])
        self.assertTrue(outcomes[0]["success"])
        self.assertEqual(
            {"wait": 1.0, "prefill": 2.0, "decode": 15.0, "tpot": 1.0, "total": 17.0},
            outcomes[0]["latency_ms"],
        )

    def test_round_preserves_http_failure_as_request_outcome(self):
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000",
            prompt_for_length=lambda _length: "prompt",
            post=Mock(
                return_value=decode_transport_Response(
                    status_code=503, text="not ready"
                )
            ),
        )
        outcome = transport.run_round(
            input_lengths=(128,), output_length=8, generate_config={}
        )[0]
        self.assertFalse(outcome["success"])
        self.assertIn("HTTP 503", outcome["error"])
        self.assertEqual(
            (0, 128, 8),
            tuple(
                (
                    outcome[key]
                    for key in ("request_index", "input_length", "output_length")
                )
            ),
        )

    def test_scheduler_retries_http_200_error_body(self):
        post = Mock(
            side_effect=[
                decode_transport_Response(
                    {"error": "Failed on some addresses: worker unavailable"}
                ),
                decode_transport_Response({"status": "ok"}),
            ]
        )
        sleep = Mock()
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000",
            prompt_for_length=str,
            post=post,
            sleep=sleep,
            scheduler_attempts=2,
        )
        transport.configure_scheduler(({"local_batch_size": 2},))
        self.assertEqual(2, post.call_count)
        sleep.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
