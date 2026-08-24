import asyncio
import json
from typing import Any
from unittest import TestCase, main
from unittest.mock import patch

from pydantic import BaseModel

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.metrics import AccMetrics
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
        self.last_inference_kwargs = {}

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
    def __init__(self, headers=None):
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

    def test_qos_priority_reaches_frontend_metrics_and_backend(self):
        raw_request = FakeRawRequest(
            {"x-dashscope-inner-qos-level": "70"}
        )
        with patch("rtp_llm.frontend.frontend_server.kmonitor.report") as report:
            asyncio.run(
                self._async_run(req={"prompt": "hello"}, raw_request=raw_request)
            )

        backend_tags = self.frontend_server._frontend_worker.last_inference_kwargs[
            "frontend_metric_tags"
        ]
        self.assertEqual(backend_tags["priority"], "70")
        qps_calls = [
            call
            for call in report.call_args_list
            if call.args[0] == AccMetrics.QPS_METRIC
        ]
        self.assertEqual(qps_calls[0].args[2]["priority"], "70")

    def test_router_queue_full_error_metric_keeps_priority(self):
        error = FtRuntimeException(
            ExceptionType.ROUTER_QUEUE_FULL,
            "router queue is full",
        )
        metric_tags = {
            "rank_id": "0",
            "server_id": "0",
            "source": "test",
            "priority": "70",
        }
        with patch("rtp_llm.frontend.frontend_server.kmonitor.report") as report:
            response = self.frontend_server._handle_exception(
                {
                    "source": "test",
                    request_id_field_name: 1,
                },
                error,
                metric_tags,
            )

        self.assertEqual(json.loads(response.body)["error_code"], 8502)
        report.assert_called_once_with(
            AccMetrics.ERROR_QPS_METRIC,
            1,
            {**metric_tags, "error_code": "8502_ROUTER_QUEUE_FULL"},
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

    def test_internal_aux_info_is_forced_but_can_be_hidden_from_response(self):
        request = {"generate_config": {"aux_info": False}}
        self.assertFalse(FrontendServer._request_aux_info_enabled(request))

        FrontendServer._force_internal_aux_info(request)
        self.assertTrue(request["generate_config"]["aux_info"])

        response = FakeAuxResponse(aux_info={"input_len": 10})
        FrontendServer._hide_aux_info(response)
        self.assertEqual(response.aux_info, {})

    def test_internal_aux_info_is_hidden_from_nonstream_error_response(self):
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
        self.assertNotIn("aux_info", payload)

    def test_internal_aux_info_is_hidden_from_stream_error_response(self):
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
        self.assertNotIn("aux_info", payload)

    def test_request_can_disable_speculative_metrics(self):
        self.assertTrue(
            FrontendServer._request_disables_speculative(
                {"generate_config": {"force_disable_sp_run": True}}
            )
        )


main()
