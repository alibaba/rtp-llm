import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.metrics import GaugeMetrics
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.structure.request_constants import request_id_field_name
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import init_controller, set_global_controller


class FakePipelinResponse(BaseModel):
    res: str


class FakeAuxResponse(BaseModel):
    aux_info: dict[str, Any]


class FakeFrontendWorker(object):
    class FakeBackendRpcServerVisitor:
        def __init__(self):
            self.refresh_calls = []

        def is_backend_service_ready(self, refresh: bool = False):
            self.refresh_calls.append(refresh)
            return True

    def __init__(self):
        self.backend_rpc_server_visitor = self.FakeBackendRpcServerVisitor()
        self.close_called = False
        self.last_inference_kwargs = {}

    async def close(self):
        self.close_called = True

    def inference(self, prompt: str, *args: Any, **kwargs: Any):
        self.last_inference_kwargs = kwargs
        response_generator = self._inference(prompt, *args, **kwargs)
        return CompleteResponseAsyncGenerator(
            response_generator, CompleteResponseAsyncGenerator.get_last_value
        )

    def tokenizer_encode(self, prompt: str):
        return [1, 2, 3, 4], ["b", "c", "d", "e"]

    async def _inference(self, prompt: str, *args: Any, **kwargs: Any):
        yield FakePipelinResponse(res=prompt)

    def is_streaming(self, *args: Any, **kwargs: Any):
        return False


class FakeRawRequest(object):
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}

    async def is_disconnected(self):
        return False


class FrontendServerTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Create PyEnvConfigs with default values for testing
        py_env_configs = PyEnvConfigs()
        set_global_controller(init_controller(py_env_configs.concurrency_config))
        py_env_configs.server_config.start_port = 0
        py_env_configs.server_config.rank_id = 0
        with patch.dict("os.environ", {"RTP_LLM_FRONTEND_METRICS_ENABLE": "1"}):
            self.frontend_server = FrontendServer(
                rank_id=0,
                server_id=0,
                py_env_configs=py_env_configs,
            )
        self.frontend_server._frontend_worker = FakeFrontendWorker()

    def tearDown(self):
        self.frontend_server._request_metrics.close()

    async def _async_run(self, *args: Any, **kwargs: Any):
        res = await self.frontend_server.inference(*args, **kwargs)
        return res

    def test_simple(self):
        loop = asyncio.new_event_loop()
        res = loop.run_until_complete(
            self._async_run(req={"prompt": "hello"}, raw_request=FakeRawRequest())
        )
        self.assertEqual(
            res.body.decode("utf-8"), '{"res":"hello"}', res.body.decode("utf-8")
        )
        self.assertTrue(
            callable(
                self.frontend_server._frontend_worker.last_inference_kwargs[
                    "frontend_metric_observer"
                ]
            )
        )
        res = loop.run_until_complete(
            self._async_run(req='{"prompt": "hello"}', raw_request=FakeRawRequest())
        )
        self.assertEqual(
            res.body.decode("utf-8"), '{"res":"hello"}', res.body.decode("utf-8")
        )

    def test_encode(self):
        res = self.frontend_server.tokenizer_encode('{"prompt": "b c d e"}')
        self.assertEqual(
            res.body.decode("utf-8"),
            '{"token_ids":[1,2,3,4],"tokens":["b","c","d","e"],"error":""}',
        )
        # test error input
        res = self.frontend_server.tokenizer_encode('{"text": "b c d e"}')
        self.assertEqual(json.loads(res.body.decode("utf-8"))["error_code"], 514)

    def test_check_health_uses_cached_service_discovery(self):
        self.assertTrue(self.frontend_server.check_health())
        visitor = self.frontend_server._frontend_worker.backend_rpc_server_visitor
        self.assertEqual(visitor.refresh_calls, [False])

    def test_close_uses_production_frontend_server_contract(self):
        close_order = []
        original_worker_close = self.frontend_server._frontend_worker.close
        original_metrics_close = self.frontend_server._request_metrics.close

        async def worker_close():
            close_order.append("worker")
            await original_worker_close()

        def metrics_close():
            close_order.append("metrics")
            original_metrics_close()

        self.frontend_server._frontend_worker.close = worker_close
        self.frontend_server._request_metrics.close = metrics_close
        asyncio.run(self.frontend_server.close())

        self.assertTrue(self.frontend_server._frontend_worker.close_called)
        self.assertEqual(close_order, ["worker", "metrics"])

    def test_internal_aux_info_is_forced_but_can_be_hidden_from_response(self):
        request = {"generate_config": {"aux_info": False}}
        self.assertFalse(FrontendServer._request_aux_info_enabled(request))

        FrontendServer._force_internal_aux_info(request)
        self.assertTrue(request["generate_config"]["aux_info"])

        response = FakeAuxResponse(aux_info={"input_len": 10})
        FrontendServer._hide_aux_info(response)
        self.assertEqual(response.aux_info, {})

    def test_error_response_preserves_existing_aux_info_contract(self):
        error = RuntimeError("backend failed")
        error.aux_info = {"input_len": 10}

        response = self.frontend_server._handle_exception(
            {
                "aux_info": False,
                "source": "test",
                request_id_field_name: 1,
            },
            error,
        )

        payload = json.loads(response.body)
        self.assertEqual(payload["aux_info"], {"input_len": 10})

    def test_stream_error_response_preserves_existing_aux_info_contract(self):
        async def failing_stream():
            error = RuntimeError("stream backend failed")
            error.aux_info = {"input_len": 10}
            raise error
            yield FakePipelinResponse(res="unreachable")

        async def collect_error_chunks():
            self.frontend_server._global_controller.increment()
            response = CompleteResponseAsyncGenerator(
                failing_stream(), CompleteResponseAsyncGenerator.get_last_value
            )
            return [
                chunk
                async for chunk in self.frontend_server.stream_response(
                    {
                        "stream": True,
                        "aux_info": False,
                        "source": "test",
                        request_id_field_name: 1,
                    },
                    response,
                )
            ]

        chunks = asyncio.run(collect_error_chunks())
        self.assertEqual(len(chunks), 1)
        payload = json.loads(chunks[0].removeprefix("data: ").strip())
        self.assertEqual(payload["aux_info"], {"input_len": 10})

    def test_client_exception_payload_strips_traceback_locations(self):
        payload = FrontendServer._client_exception_payload(
            {"aux_info": False},
            {
                "message": "ErrorMsg: failed\n Traceback: /secret/path.py:42",
                "aux_info": {"input_len": 10},
            },
        )
        self.assertEqual(payload["message"], "ErrorMsg: failed")
        self.assertEqual(payload["aux_info"], {"input_len": 10})

    def test_aux_info_intent_matches_pydantic_boolean_coercion(self):
        for value in (False, 0, "false", "0", None):
            with self.subTest(value=value):
                self.assertFalse(
                    FrontendServer._request_aux_info_enabled({"aux_info": value})
                )
        for value in (True, 1, "true", "1"):
            with self.subTest(value=value):
                self.assertTrue(
                    FrontendServer._request_aux_info_enabled({"aux_info": value})
                )

    def test_internal_aux_info_copy_does_not_deepcopy_payload(self):
        images = [object()]
        request = {
            "images": images,
            "generate_config": {"aux_info": False, "top_k": 1},
        }

        copied = FrontendServer._copy_for_internal_aux_info(request)

        self.assertIs(copied["images"], images)
        self.assertIsNot(copied["generate_config"], request["generate_config"])
        self.assertFalse(request["generate_config"]["aux_info"])
        self.assertTrue(copied["generate_config"]["aux_info"])

    def test_request_can_disable_speculative_metrics(self):
        self.assertTrue(
            FrontendServer._request_disables_speculative(
                {"generate_config": {"force_disable_sp_run": True}}
            )
        )

    def test_prompt_outputs_force_internal_request_nonstreaming(self):
        captured = {}

        class FakeOpenaiEndpoint:
            def chat_completion(self, request_id, request, raw_request, **kwargs):
                captured["internal_request"] = request

                async def response_generator():
                    if False:
                        yield None

                return CompleteResponseAsyncGenerator(
                    response_generator(),
                    CompleteResponseAsyncGenerator.get_last_value,
                )

        async def fake_infer_wrap(request_dict, raw_request, generate_call):
            captured["request_dict"] = request_dict
            captured["response"] = generate_call(
                SimpleNamespace(observe_tps=lambda response: None)
            )
            return ORJSONResponse({"ok": True})

        self.frontend_server._openai_endpoint = FakeOpenaiEndpoint()
        self.frontend_server._infer_wrap = fake_infer_wrap
        for request in (
            ChatCompletionRequest(messages=[], stream=True, prompt_logprobs=5),
            ChatCompletionRequest(
                messages=[],
                stream=True,
                extra_configs={"return_prompt_logits": True},
            ),
        ):
            with self.subTest(request=request):
                asyncio.run(
                    self.frontend_server.chat_completion(request, FakeRawRequest())
                )

                self.assertFalse(request.stream)
                self.assertFalse(captured["internal_request"].stream)
                self.assertFalse(captured["request_dict"]["stream"])

    def test_hidden_internal_aux_info_does_not_change_iter_latency_metric(self):
        async def response_generator():
            yield FakeAuxResponse(aux_info={"step_output_len": 1})
            yield FakeAuxResponse(aux_info={"step_output_len": 4})

        response = CompleteResponseAsyncGenerator(
            response_generator(),
            CompleteResponseAsyncGenerator.get_last_value,
        )
        request_metrics = MagicMock()

        async def collect():
            wrapped = await self.frontend_server._call_generate_with_report(
                lambda _: response,
                request_metrics,
                expose_aux_info=False,
            )
            return [item async for item in wrapped]

        with patch(
            "rtp_llm.frontend.frontend_server.current_time_ms",
            side_effect=[0, 0, 10, 30, 30],
        ), patch("rtp_llm.frontend.frontend_server.kmonitor.report") as report:
            outputs = asyncio.run(collect())

        iter_values = [
            call.args[1]
            for call in report.call_args_list
            if call.args and call.args[0] == GaugeMetrics.RESPONSE_ITER_RT_METRIC
        ]
        self.assertEqual(iter_values, [20])
        self.assertEqual([output.aux_info for output in outputs], [{}, {}])
        request_metrics.finish.assert_called_once_with()


main()
