"""Exercise JEV HTTP/worker wiring with the real protocol and a mocked engine RPC."""

import asyncio
import copy
import json
import math
import string
import unittest
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from fastapi.testclient import TestClient

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException

from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.frontend.frontend_worker import FrontendWorker
from rtp_llm.frontend.jev import (
    JevExecutor,
    JevHeadSpec,
    build_row,
    parse_decide_request,
)
from rtp_llm.frontend.shutdown_manager import FrontendShutdownManager
from rtp_llm.metrics import AccMetrics, GaugeMetrics
from rtp_llm.structure.request_constants import request_id_field_name
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateOutput,
    GenerateOutputs,
    MMUrlType,
)
from rtp_llm.utils.concurrency_controller import ConcurrencyController

IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"


def head_config():
    choices = list(string.ascii_uppercase) + ["AA", "AB", "AC", "AD", "AE", "AF"]
    labels = ["false", "true"] + list("012345") + choices
    kinds = {}
    for kind, offset, size in (("noul", 0, 2), ("score", 2, 6), ("choice", 8, 32)):
        bias = [0.0] * size
        bias[0], bias[1] = 1.0, -1.0
        kinds[kind] = {
            "row_indices": list(range(offset, offset + size)),
            "label_names": labels[offset : offset + size],
            "bias": bias,
            "temperature": 2.0,
        }
    return {
        "schema_version": 1,
        "mode": "system1",
        "model_type": "jev",
        "num_labels": len(labels),
        "kinds": kinds,
        "default_strategy": "single",
    }


class BareTokenizer:
    """Each image marker token occupies one position, as in Qwen's tokenizer."""

    special = {
        "<|vision_start|>": 151652,
        "<|image_pad|>": 151655,
        "<|vision_end|>": 151653,
    }

    def __init__(self):
        self.calls = []

    def get_real_tokenizer(self):
        return self

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError(
                "System1 must use the bare prompt without BOS/chat wrapping"
            )
        self.calls.append(text)
        ids = []
        while text:
            for marker, token in self.special.items():
                if text.startswith(marker):
                    ids.append(token)
                    text = text[len(marker) :]
                    break
            else:
                ids.append(ord(text[0]) + 1000)
                text = text[1:]
        return SimpleNamespace(input_ids=ids)

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("System1 must not invoke a chat template")


class JevDecideTest(unittest.TestCase):
    def setUp(self):
        reporter = patch("rtp_llm.frontend.frontend_server.kmonitor.report")
        self.report = reporter.start()
        self.addCleanup(reporter.stop)
        self.worker = object.__new__(FrontendWorker)
        self.worker.tokenizer = BareTokenizer()
        self.worker.kev_config = None
        self.worker.kev_max_row_tokens = 32768
        self.worker.kev_reuse_cache = True
        self.worker.kev_tokens_per_block = 64
        self.worker.jev_config = head_config()
        self.worker._jev_request_context = ContextVar("test_jev_request_context")
        self.worker.jev_executor = JevExecutor(
            JevHeadSpec.from_config(self.worker.jev_config),
            self.worker._jev_row,
            max_concurrency=4,
            timeout_seconds=10,
        )
        self.enqueue = AsyncMock()
        self.worker.backend_rpc_server_visitor = SimpleNamespace(
            batch_enqueue=self.enqueue
        )
        self.request = {
            "kind": "choice",
            "state": "context",
            "question": "Which one?",
            "options": ["red", "blue"],
        }
        self.logits = torch.zeros((1, 40), dtype=torch.float32)
        self.logits[0, 0:2] = torch.tensor([-2.0, 4.0])
        self.logits[0, 8:10] = torch.tensor([-2.0, 4.0])
        # Huge irrelevant slots must be excluded before normalization.
        self.logits[0, 2:8] = torch.tensor([50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        self.logits[0, 10:] = 1000

    def output(self, *, logits=None, finished=True, input_len=300, reuse_len=64):
        return GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    finished=finished,
                    custom_output=self.logits.clone() if logits is None else logits,
                    aux_info=AuxInfo(input_len=input_len, reuse_len=reuse_len),
                )
            ]
        )

    @staticmethod
    def systemone_request():
        return {
            "model": "jev-business",
            "state": {"text": "context", "number": 7},
            "questions": {
                "selection": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"red": "warm", "blue": "cool"},
                },
                "check": {"type": "noul", "instructions": "Does it match?"},
            },
        }

    def server(self):
        server = object.__new__(FrontendServer)
        server._frontend_worker = self.worker
        server._global_controller = MagicMock()
        server._global_controller.max_concurrency = 4
        server._access_logger = MagicMock()
        server.rank_id = "0"
        server.server_id = "0"
        server.is_embedding = False
        return server

    def client(self, server):
        owner = object.__new__(FrontendApp)
        owner.frontend_server = server
        owner.shutdown_manager = FrontendShutdownManager()
        owner.separated_frontend = True
        owner.server_config = SimpleNamespace(http_port=0)
        owner.grpc_client = None
        return TestClient(owner.create_app()), owner

    def assert_metrics(self, *expected):
        self.assertCountEqual(
            [call.args[0] for call in self.report.call_args_list], expected
        )
        for call in self.report.call_args_list:
            self.assertEqual(call.args[2]["rank_id"], "0")
            self.assertEqual(call.args[2]["server_id"], "0")

    def test_choice_uses_last_text_bare_prompt_and_calibration_exactly_once(self):
        self.enqueue.return_value = [self.output()]
        original = copy.deepcopy(self.request)
        result = asyncio.run(self.worker.decide(self.request))
        self.assertEqual(
            self.worker.tokenizer.calls,
            [
                "[kind] choice\n[state] context\n[question] Which one?\n[options]\nA) red\nB) blue\n[decision]:"
            ],
        )
        item = self.enqueue.await_args.args[0][0]
        self.assertEqual(item.token_ids[-1].item(), ord(":") + 1000)
        self.assertEqual(item.token_ids.dtype, torch.int32)
        self.assertEqual(item.token_ids.device.type, "cpu")
        self.assertEqual(item.generate_config.max_new_tokens, 0)
        self.assertFalse(item.generate_config.is_streaming)
        self.assertEqual(item.mm_inputs, [])
        self.assertEqual(item.prefix_length, 0)
        self.assertIs(item.tokenizer, self.worker.tokenizer)
        # raw [-2,4] + bias [1,-1], divided once by T=2 -> [-0.5,1.5].
        expected_true = 1.0 / (1.0 + math.exp(-2.0))
        self.assertAlmostEqual(result["probabilities"][1], expected_true, places=12)
        self.assertAlmostEqual(result["probabilities"][0], 1 - expected_true, places=12)
        self.assertEqual(result["choice"], "blue")
        self.assertEqual(result["num_model_requests"], 1)
        self.assertEqual(
            result["usage"],
            {
                "prompt_tokens": 300,
                "completion_tokens": 0,
                "total_tokens": 300,
                "vision_tokens": 0,
                "cache_reuse_tokens": 64,
            },
        )
        self.assertEqual(self.request, original)
        self.assertIsNone(self.worker._jev_request_context.get(None))

    def test_interleaved_images_preserve_marker_url_order_and_expanded_usage(self):
        request = dict(
            self.request,
            state=[
                "before",
                {"image": "https://example.invalid/first.png"},
                "between",
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,c2Vjb25k"},
                },
                "after",
            ],
        )
        captured = {}

        async def respond(inputs, *, routing_key):
            item = inputs[0]
            # Image pad placeholders are replaced by 5 and 8 feature rows.
            captured["expanded"] = len(item.token_ids) - 2 + 5 + 8
            return [self.output(input_len=captured["expanded"], reuse_len=64)]

        self.enqueue.side_effect = respond
        result = asyncio.run(self.worker.decide(request))
        item = self.enqueue.await_args.args[0][0]
        prompt = self.worker.tokenizer.calls[0]
        self.assertIn(
            "before" + IMAGE_MARKER + "between" + IMAGE_MARKER + "after", prompt
        )
        self.assertTrue(prompt.endswith("\n[decision]:"))
        self.assertEqual(
            [image.url for image in item.mm_inputs],
            ["https://example.invalid/first.png", "data:image/png;base64,c2Vjb25k"],
        )
        self.assertTrue(
            all(
                image.mm_type == MMUrlType.IMAGE and image.tensor.numel() == 0
                for image in item.mm_inputs
            )
        )
        self.assertEqual(item.token_ids.tolist().count(151655), 2)
        self.assertEqual(result["usage"]["prompt_tokens"], captured["expanded"])
        self.assertNotEqual(result["usage"]["prompt_tokens"], len(item.token_ids))

        async def direct_row():
            token = self.worker._jev_request_context.set(({}, 1))
            try:
                parsed = parse_decide_request(request, self.worker.jev_executor.spec)
                row = build_row(parsed, self.worker.jev_executor.spec, parsed.options)
                return await self.worker._jev_row(row)
            finally:
                self.worker._jev_request_context.reset(token)

        row = asyncio.run(direct_row())
        self.assertEqual(row.vision_tokens, 13)
        self.assertEqual(row.cache_reuse_tokens, 64)
        self.assertEqual(row.prompt_tokens, captured["expanded"])

    def test_systemone_counts_actual_rows_and_keeps_answer_order(self):
        self.enqueue.return_value = [self.output(input_len=300)]
        result = asyncio.run(self.worker.systemone(self.systemone_request()))
        self.assertEqual(self.enqueue.await_count, 2)
        self.assertEqual(list(result["answers"]), ["selection", "check"])
        self.assertEqual(result["answers"]["selection"]["choice"], "blue")
        self.assertAlmostEqual(
            result["answers"]["check"]["noul"], round(1 / (1 + math.exp(-2)), 4)
        )
        self.assertEqual(result["model"], "jev-business")
        self.assertEqual(result["num_model_requests"], 2)
        self.assertEqual(
            result["usage"],
            {
                "input_tokens": 600,
                "output_tokens": 0,
                "prompt_tokens": 600,
                "completion_tokens": 0,
                "vision_tokens": 0,
                "cache_reuse_tokens": 128,
                "total_tokens": 600,
            },
        )
        self.assertEqual(
            len({call.kwargs["routing_key"] for call in self.enqueue.await_args_list}),
            1,
        )

    def test_concurrent_requests_keep_headers_routing_and_context_separate(self):
        request = self.systemone_request()
        first_headers = {"x-request-id": "first", "traceparent": "trace-first"}
        second_headers = {"x-request-id": "second", "traceparent": "trace-second"}

        async def run():
            entered = 0
            all_entered = asyncio.Event()

            async def respond(inputs, *, routing_key):
                nonlocal entered
                entered += 1
                if entered == 4:
                    all_entered.set()
                await asyncio.wait_for(all_entered.wait(), 1)
                return [self.output()]

            self.enqueue.side_effect = respond
            results = await asyncio.gather(
                self.worker.systemone(request, headers=first_headers),
                self.worker.systemone(request, headers=second_headers),
            )
            self.assertIsNone(self.worker._jev_request_context.get(None))
            return results

        self.assertEqual(len(asyncio.run(run())), 2)
        routes = {"first": set(), "second": set()}
        ids = set()
        for call in self.enqueue.await_args_list:
            item = call.args[0][0]
            key = item.headers["x-request-id"]
            self.assertEqual(
                item.headers, first_headers if key == "first" else second_headers
            )
            routes[key].add(call.kwargs["routing_key"])
            ids.add(item.request_id)
        self.assertEqual(len(ids), 4)
        self.assertEqual(len(routes["first"]), 1)
        self.assertEqual(len(routes["second"]), 1)
        self.assertTrue(routes["first"].isdisjoint(routes["second"]))
        self.assertTrue(all(0 <= value < (1 << 63) for value in ids))

    def test_invalid_requests_have_client_status_and_do_not_call_rpc(self):
        for method, request, status, field in (
            ("decide", [], 400, "error"),
            ("decide", {}, 400, "error"),
            ("decide", dict(self.request, thinking="on"), 400, "error"),
            ("systemone", [], 422, "detail"),
            ("systemone", {}, 422, "detail"),
        ):
            with self.subTest(method=method, request=request):
                server = self.server()
                self.report.reset_mock()
                response = asyncio.run(getattr(server, method)(request))
                self.assertEqual(response.status_code, status)
                self.assertIn(field, json.loads(response.body))
                self.enqueue.assert_not_awaited()
                self.assert_metrics(AccMetrics.QPS_METRIC, AccMetrics.ERROR_QPS_METRIC)
                self.assertEqual(
                    self.report.call_args_list[-1].args[2]["error_code"],
                    "INVALID_PARAMS",
                )
                server._global_controller.increment.assert_called_once_with()
                server._global_controller.decrement.assert_called_once_with()

    def test_malformed_backend_results_are_server_errors_and_release_admission(self):
        malformed = [
            [],
            [GenerateOutputs(generate_outputs=[])],
            [self.output(finished=False)],
            [self.output(logits=torch.zeros(40))],
            [self.output(logits=torch.zeros((1, 39)))],
            [self.output(logits=torch.full((1, 40), float("nan")))],
            [self.output(logits=torch.full((1, 40), float("inf")))],
            [
                GenerateOutputs(
                    generate_outputs=[GenerateOutput(finished=True, custom_output=None)]
                )
            ],
            [self.output(), self.output()],
        ]
        for result in malformed:
            with self.subTest(result=result):
                self.report.reset_mock()
                server = self.server()
                self.enqueue.return_value = result
                response = asyncio.run(server.decide(self.request))
                self.assertEqual(response.status_code, 500)
                self.assert_metrics(AccMetrics.QPS_METRIC, AccMetrics.ERROR_QPS_METRIC)
                server._global_controller.decrement.assert_called_once_with()
                server._access_logger.log_exception_access.assert_called_once()

    def test_expanded_image_context_limit_is_a_client_error(self):
        for code in (
            ExceptionType.LONG_PROMPT_ERROR,
            ExceptionType.MM_LONG_PROMPT_ERROR,
        ):
            for method, status in (("decide", 400), ("systemone", 422)):
                with self.subTest(code=code, method=method):
                    server = self.server()
                    self.enqueue.side_effect = FtRuntimeException(
                        code, "expanded image exceeds context"
                    )
                    request = (
                        self.request if method == "decide" else self.systemone_request()
                    )
                    response = asyncio.run(getattr(server, method)(request))
                    self.assertEqual(response.status_code, status)
                    server._global_controller.decrement.assert_called_once_with()

    def test_internal_value_error_is_not_misclassified_as_client_error(self):
        for location in ("backend", "tokenizer"):
            with self.subTest(location=location):
                self.report.reset_mock()
                server = self.server()
                self.enqueue.side_effect = ValueError("internal failure")
                if location == "tokenizer":
                    self.worker.tokenizer = MagicMock(
                        side_effect=ValueError("internal failure")
                    )
                    self.worker.tokenizer.get_real_tokenizer.return_value = (
                        self.worker.tokenizer
                    )
                response = asyncio.run(server.decide(self.request))
                self.assertEqual(response.status_code, 500)
                self.assertIn("internal failure", json.loads(response.body)["message"])
                self.assert_metrics(AccMetrics.QPS_METRIC, AccMetrics.ERROR_QPS_METRIC)
                server._global_controller.decrement.assert_called_once_with()

    def test_wide_choice_reports_one_service_operation_for_four_model_rows(self):
        self.enqueue.return_value = [self.output(input_len=300)]
        request = dict(
            self.request, options=[f"option-{i}" for i in range(20)], strategy="permute"
        )
        server = self.server()
        response = asyncio.run(server.decide(request))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.enqueue.await_count, 4)
        body = json.loads(response.body)
        self.assertEqual(body["num_model_requests"], 4)
        self.assertEqual(body["usage"]["prompt_tokens"], 1200)
        self.assert_metrics(
            AccMetrics.QPS_METRIC,
            GaugeMetrics.LANTENCY_METRIC,
            AccMetrics.SUCCESS_QPS_METRIC,
        )
        server._global_controller.increment.assert_called_once_with()
        server._global_controller.decrement.assert_called_once_with()
        server._access_logger.log_query_access.assert_called_once()
        server._access_logger.log_success_access.assert_called_once()
        logged = server._access_logger.log_query_access.call_args.args[0]
        self.assertIsInstance(logged[request_id_field_name], int)
        for call in self.enqueue.await_args_list:
            self.assertEqual(
                call.args[0][0].headers["x-request-id"],
                logged["correlation_request_id"],
            )

    def test_cancellation_drains_all_rpc_rows_and_releases_one_service_slot(self):
        request = dict(
            self.request, options=[f"option-{i}" for i in range(20)], strategy="permute"
        )
        server = self.server()

        async def run():
            entered = 0
            stopped = 0
            all_entered = asyncio.Event()

            async def blocked(inputs, *, routing_key):
                nonlocal entered, stopped
                entered += 1
                if entered == 4:
                    all_entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped += 1

            self.enqueue.side_effect = blocked
            task = asyncio.create_task(server.decide(request))
            await asyncio.wait_for(all_entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(stopped, 4)
            self.assertEqual(
                [
                    task
                    for task in asyncio.all_tasks()
                    if task is not asyncio.current_task()
                ],
                [],
            )

        asyncio.run(run())
        self.assert_metrics(AccMetrics.QPS_METRIC, AccMetrics.CANCEL_QPS_METRIC)
        server._global_controller.increment.assert_called_once_with()
        server._global_controller.decrement.assert_called_once_with()

    def test_overload_keeps_the_existing_requests_admission_slot(self):
        server = self.server()
        server._global_controller = ConcurrencyController(1)
        server._global_controller.increment()
        response = asyncio.run(server.decide(self.request))
        self.assertEqual(response.status_code, 429)
        self.enqueue.assert_not_awaited()
        self.assertEqual(server._global_controller.get_available_concurrency(), 0)
        self.assert_metrics(AccMetrics.QPS_METRIC, AccMetrics.CONFLICT_QPS_METRIC)
        server._global_controller.decrement()

    def test_http_routes_info_headers_and_shutdown_tracking(self):
        self.enqueue.return_value = [self.output()]
        server = self.server()
        client, owner = self.client(server)
        info = client.get("/v1/decide/info")
        self.assertEqual(info.status_code, 200)
        self.assertEqual(info.json()["max_options"], 32)
        self.assertEqual(info.json()["native_choice_options"], 16)
        self.assertEqual(info.json()["defaults"]["thinking"], "off")
        self.assertEqual(info.json()["capabilities"]["systems"], [1])
        self.assertTrue(info.json()["capabilities"]["images"])
        self.enqueue.assert_not_awaited()
        server._global_controller.increment.assert_not_called()
        response = client.post("/v1/decide", json=self.request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choice"], "blue")
        response = client.post(
            "/v1/systemone",
            json=self.systemone_request(),
            headers={"x-typesafe-request-id": "jev-http-correlation"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["x-typesafe-request-id"], "jev-http-correlation"
        )
        self.assertRegex(response.headers["server-timing"], r"^app;dur=\d+\.\d$")
        self.assertEqual(response.json()["answers"]["selection"]["choice"], "blue")
        self.assertEqual(owner.shutdown_manager.active_request_count(), 0)
        self.assertEqual(server._global_controller.increment.call_count, 2)
        self.assertEqual(server._global_controller.decrement.call_count, 2)
        self.worker.jev_executor = None
        self.assertEqual(client.get("/v1/decide/info").status_code, 404)


if __name__ == "__main__":
    unittest.main()
