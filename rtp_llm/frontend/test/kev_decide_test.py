import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from fastapi import Request
from fastapi.testclient import TestClient

from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.frontend.frontend_worker import FrontendWorker
from rtp_llm.frontend.shutdown_manager import FrontendShutdownManager
from rtp_llm.metrics import AccMetrics, GaugeMetrics
from rtp_llm.models.downstream_modules.kev import SPECIAL
from rtp_llm.structure.request_constants import request_id_field_name
from rtp_llm.utils.base_model_datatypes import GenerateOutput, GenerateOutputs
from rtp_llm.utils.concurrency_controller import ConcurrencyController


class FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        return SPECIAL.index(token) + 1

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("Kev segments must not acquire additional tokens")
        return SimpleNamespace(input_ids=[ord(character) + 1000 for character in text])


class KevDecideTest(unittest.TestCase):
    def setUp(self):
        reporter = patch("rtp_llm.frontend.frontend_server.kmonitor.report")
        self.report = reporter.start()
        self.addCleanup(reporter.stop)
        self.worker = object.__new__(FrontendWorker)
        self.worker.tokenizer = FakeTokenizer()
        self.worker.kev_config = {
            "max_options": 4,
            "max_state_tokens": 65536,
            "max_row_tokens": 73728,
            "special_token_ids": [1, 2, 3, 4, 5],
        }
        self.worker.kev_max_row_tokens = 32768
        self.worker.kev_reuse_cache = False
        self.worker.kev_tokens_per_block = 64
        self.enqueue = AsyncMock()
        self.worker.backend_rpc_server_visitor = SimpleNamespace(
            batch_enqueue=self.enqueue
        )
        self.request = {
            "state": "ab",
            "questions": [
                {"instr": "x", "options": ["a", "b"]},
                {"instr": "Y", "options": ["no"]},
            ],
        }

    def output(self, probabilities, *, finished=True):
        values = torch.zeros((1, 4), dtype=torch.float32)
        values[0, : len(probabilities)] = torch.tensor(probabilities)
        return GenerateOutputs(
            generate_outputs=[GenerateOutput(finished=finished, custom_output=values)]
        )

    def decide(self, request=None):
        return asyncio.run(
            self.worker.decide(self.request if request is None else request)
        )

    def test_exact_rows_prefill_only_and_probability_response(self):
        self.enqueue.return_value = [self.output([0.25, 0.75]), self.output([1.0])]
        result = self.decide()
        self.enqueue.assert_awaited_once()
        inputs = self.enqueue.await_args.args[0]
        self.assertEqual(
            [item.token_ids.tolist() for item in inputs],
            [
                [1, 1097, 1098, 2, 1120, 3, 1097, 4, 3, 1098, 4, 5],
                [1, 1097, 1098, 2, 1089, 3, 1110, 1111, 4, 5],
            ],
        )
        for item in inputs:
            self.assertEqual(item.token_ids.dtype, torch.int32)
            self.assertEqual(item.token_ids.device.type, "cpu")
            self.assertEqual(item.generate_config.max_new_tokens, 0)
            self.assertFalse(item.generate_config.is_streaming)
            self.assertEqual(item.mm_inputs, [])
            self.assertEqual(item.prefix_length, 0)
            self.assertIs(item.tokenizer, self.worker.tokenizer)
        self.assertEqual(
            result,
            {
                "results": [
                    {"probabilities": [0.25, 0.75], "choice": 1, "option": "b"},
                    {"probabilities": [1.0], "choice": 0, "option": "no"},
                ]
            },
        )

    def test_ids_are_distinct_across_concurrent_multi_question_requests(self):

        async def respond(inputs, *, routing_key):
            return [self.output([0.5, 0.5]), self.output([1.0])]

        self.enqueue.side_effect = respond

        async def concurrent_requests():
            return await asyncio.gather(
                self.worker.decide(self.request), self.worker.decide(self.request)
            )

        results = asyncio.run(concurrent_requests())
        self.assertEqual(len(results), 2)
        ids = [
            item.request_id
            for call in self.enqueue.await_args_list
            for item in call.args[0]
        ]
        self.assertEqual(len(ids), 4)
        self.assertEqual(len(set(ids)), 4)
        self.assertTrue(all(0 <= request_id < (1 << 63) for request_id in ids))
        routing_keys = [
            call.kwargs["routing_key"] for call in self.enqueue.await_args_list
        ]
        self.assertEqual(len(set(routing_keys)), 2)
        self.assertTrue(all(0 <= key < (1 << 63) for key in routing_keys))

    def test_token_budget_splits_batches_preserving_question_order(self):
        request = {
            "state": "s" * 5000,
            "questions": [
                {"instr": "a", "options": ["one"]},
                {"instr": "b", "options": ["left", "right"]},
                {"instr": "c", "options": ["first", "middle", "last"]},
                {"instr": "d", "options": ["yes", "no"]},
            ],
        }
        self.enqueue.side_effect = [
            [self.output([1]), self.output([0, 1]), self.output([0, 0, 1])],
            [self.output([1, 0])],
        ]
        result = self.decide(request)
        batches = [call.args[0] for call in self.enqueue.await_args_list]
        self.assertEqual(list(map(len, batches)), [3, 1])
        self.assertTrue(
            all(sum(item.input_length for item in batch) <= 16384 for batch in batches)
        )
        self.assertEqual(
            [item["option"] for item in result["results"]],
            ["one", "right", "last", "yes"],
        )
        self.assertEqual(
            [item["probabilities"] for item in result["results"]],
            [[1.0], [0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0]],
        )
        # Independent rows contain the entire state in every batch.
        expected_prefix = [1] + [ord("s") + 1000] * 5000
        for batch in batches:
            for item in batch:
                self.assertEqual(item.token_ids[:5001].tolist(), expected_prefix)
        self.assertEqual(
            len({call.kwargs["routing_key"] for call in self.enqueue.await_args_list}),
            1,
        )

    def test_cache_waits_for_leader_before_sending_followers(self):
        self.worker.kev_reuse_cache = True
        request = {
            # The state delimiter plus 63 text tokens is exactly one full block.
            "state": "s" * 63,
            "questions": [
                {"instr": str(index), "options": [f"answer-{index}"]}
                for index in range(3)
            ],
        }

        async def run():
            leader_started = asyncio.Event()
            release_leader = asyncio.Event()
            leader_finished = False

            async def respond(inputs, *, routing_key):
                nonlocal leader_finished
                if self.enqueue.await_count == 1:
                    self.assertEqual(len(inputs), 1)
                    leader_started.set()
                    await release_leader.wait()
                    leader_finished = True
                else:
                    self.assertTrue(leader_finished)
                    self.assertEqual(len(inputs), 2)
                return [self.output([1.0]) for _ in inputs]

            self.enqueue.side_effect = respond
            task = asyncio.create_task(self.worker.decide(request))
            await asyncio.wait_for(leader_started.wait(), timeout=1)
            await asyncio.sleep(0)
            self.assertEqual(self.enqueue.await_count, 1)
            self.assertFalse(task.done())
            release_leader.set()
            return await task

        result = asyncio.run(run())
        self.assertEqual(
            [item["option"] for item in result["results"]],
            ["answer-0", "answer-1", "answer-2"],
        )
        calls = self.enqueue.await_args_list
        self.assertEqual(calls[0].kwargs["routing_key"], calls[1].kwargs["routing_key"])
        inputs = [item for call in calls for item in call.args[0]]
        self.assertEqual(len({item.request_id for item in inputs}), 3)
        for item in inputs:
            self.assertEqual(item.token_ids[:64].tolist(), [1] + [1115] * 63)
            self.assertTrue(item.generate_config.reuse_cache)
            self.assertEqual(item.prefix_length, 0)

    def test_cache_disabled_short_state_or_single_question_keeps_one_batch(self):
        for reuse_cache, state_length, question_count in (
            (False, 63, 3),
            (True, 62, 3),
            (True, 0, 3),
            (True, 63, 1),
        ):
            with self.subTest(
                reuse_cache=reuse_cache,
                state_length=state_length,
                question_count=question_count,
            ):
                self.enqueue.reset_mock()
                self.worker.kev_reuse_cache = reuse_cache
                self.enqueue.return_value = [self.output([1])] * question_count
                request = {
                    "state": "s" * state_length,
                    "questions": [{"instr": "x", "options": ["yes"]}] * question_count,
                }
                self.decide(request)
                self.enqueue.assert_awaited_once()
                self.assertEqual(len(self.enqueue.await_args.args[0]), question_count)

    def test_cache_followers_keep_token_budget_for_64_questions(self):
        self.worker.kev_reuse_cache = True
        request = {
            "state": "s" * 5000,
            "questions": [
                {"instr": str(index), "options": [f"answer-{index}"]}
                for index in range(64)
            ],
        }

        async def respond(inputs, *, routing_key):
            return [self.output([1]) for _ in inputs]

        self.enqueue.side_effect = respond
        result = self.decide(request)
        batches = [call.args[0] for call in self.enqueue.await_args_list]
        self.assertEqual(list(map(len, batches)), [1] + [3] * 21)
        self.assertTrue(
            all(sum(item.input_length for item in batch) <= 16384 for batch in batches)
        )
        self.assertEqual(
            len({call.kwargs["routing_key"] for call in self.enqueue.await_args_list}),
            1,
        )
        self.assertEqual(
            len({item.request_id for batch in batches for item in batch}), 64
        )
        self.assertEqual(
            [item["option"] for item in result["results"]],
            [f"answer-{index}" for index in range(64)],
        )

    def test_cache_leader_failure_does_not_submit_followers(self):
        self.worker.kev_reuse_cache = True
        self.enqueue.side_effect = RuntimeError("leader failed")
        request = dict(self.request, state="s" * 63)
        with self.assertRaisesRegex(RuntimeError, "leader failed"):
            self.decide(request)
        self.enqueue.assert_awaited_once()
        self.assertEqual(len(self.enqueue.await_args.args[0]), 1)

    def test_long_row_is_admitted_as_its_own_batch(self):
        request = {
            "state": "s" * 17000,
            "questions": [
                {"instr": "a", "options": ["one"]},
                {"instr": "b", "options": ["two"]},
            ],
        }
        self.enqueue.side_effect = [[self.output([1])], [self.output([1])]]
        result = self.decide(request)
        self.assertEqual(self.enqueue.await_count, 2)
        for call in self.enqueue.await_args_list:
            self.assertEqual(len(call.args[0]), 1)
            self.assertGreater(call.args[0][0].input_length, 16384)
        self.assertEqual([item["option"] for item in result["results"]], ["one", "two"])

    def test_invalid_requests_never_enqueue(self):
        invalid_requests = [
            None,
            [],
            {},
            dict(self.request, max_new_tokens=1),
            {"state": 123, "questions": self.request["questions"]},
            {"state": "", "questions": []},
            {"state": "", "questions": "question"},
            {"state": "", "questions": self.request["questions"] * 33},
            {"state": "", "questions": [{"instr": "", "options": []}]},
            {"state": "", "questions": [{"instr": "", "options": ["x"] * 5}]},
            {"state": "", "questions": [{"instr": "", "options": [1]}]},
            {"state": "", "questions": [{"instr": 1, "options": ["x"]}]},
        ]
        for request in invalid_requests:
            with self.subTest(request=request), self.assertRaises(ValueError):
                asyncio.run(self.worker.decide(request))
        self.enqueue.assert_not_awaited()

    def test_overflow_in_later_question_rejects_whole_request_before_enqueue(self):
        self.worker.kev_max_row_tokens = 20
        request = {
            "state": "s",
            "questions": [
                {"instr": "a", "options": ["ok"]},
                {"instr": "b", "options": ["x" * 30]},
            ],
        }
        with self.assertRaisesRegex(ValueError, "exceeding"):
            self.decide(request)
        self.enqueue.assert_not_awaited()
        self.worker.kev_config["max_state_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "state"):
            self.decide(request)
        self.enqueue.assert_not_awaited()

    def test_backend_row_count_and_result_count_are_validated(self):
        cases = [
            [],
            [self.output([0.5, 0.5])],
            [GenerateOutputs(generate_outputs=[]), self.output([1])],
            [
                GenerateOutputs(
                    generate_outputs=self.output([0.5, 0.5]).generate_outputs * 2
                ),
                self.output([1]),
            ],
        ]
        for outputs in cases:
            with self.subTest(outputs=outputs), self.assertRaises(RuntimeError):
                self.enqueue.return_value = outputs
                self.decide()

    def test_backend_invalid_decisions_and_probabilities_are_rejected(self):
        invalid_values = [
            None,
            torch.tensor([0.5, 0.5, 0.0, 0.0]),
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[float("nan"), 0.5, 0.0, 0.0]]),
            torch.tensor([[0.5, 0.5, float("inf"), 0.0]]),
            torch.tensor([[-0.25, 1.25, 0.0, 0.0]]),
            torch.tensor([[0.25, 0.25, 0.0, 0.0]]),
        ]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                bad_output = GenerateOutputs(
                    generate_outputs=[
                        GenerateOutput(finished=True, custom_output=value)
                    ]
                )
                self.enqueue.return_value = [bad_output, self.output([1])]
                self.decide()
        self.enqueue.return_value = [
            self.output([0.5, 0.5], finished=False),
            self.output([1]),
        ]
        with self.assertRaises(RuntimeError):
            self.decide()

    def test_normal_model_is_rejected_and_server_releases_admission_slot(self):
        self.worker.kev_config = None
        server = self.systemone_server()
        response = asyncio.run(server.decide(self.request))
        self.assertEqual(response.status_code, 400)
        self.assertIn("Kev checkpoint", json.loads(response.body)["error"])
        self.enqueue.assert_not_awaited()
        server._global_controller.increment.assert_called_once_with()
        server._global_controller.decrement.assert_called_once_with()

    def test_server_releases_admission_slot_after_backend_failure(self):
        server = self.systemone_server()
        self.enqueue.side_effect = RuntimeError("backend unavailable")
        response = asyncio.run(server.decide(self.request))
        self.assertEqual(response.status_code, 500)
        self.assertIn("backend unavailable", json.loads(response.body)["message"])
        self.assert_metrics(AccMetrics.QPS_METRIC, AccMetrics.ERROR_QPS_METRIC)
        server._global_controller.increment.assert_called_once_with()
        server._global_controller.decrement.assert_called_once_with()

    @staticmethod
    def systemone_request():
        return {
            "state": {"text": "synthetic", "tags": ["fresh", "safe"]},
            "questions": {
                "selection": {
                    "type": "choice",
                    "instructions": "Choose a color",
                    "criteria": {"red": "warm", "蓝": {"tone": "cool"}},
                },
                "rating": {
                    "type": "score",
                    "instructions": {"task": "Rate"},
                    "criteria": ["low", {"level": "high"}],
                },
                "check": {
                    "type": "noul",
                    "criteria": {"true": "matches", "false": "misses"},
                },
            },
        }

    def systemone_outputs(self):
        return [
            self.output([0.25, 0.75]),
            self.output([0.25, 0.75]),
            self.output([0.125, 0.875]),
        ]

    def systemone_server(self):
        server = object.__new__(FrontendServer)
        server._frontend_worker = self.worker
        server._global_controller = MagicMock()
        server._global_controller.max_concurrency = 4
        server._access_logger = MagicMock()
        server.rank_id = "0"
        server.server_id = "0"
        server.is_embedding = False
        return server

    def test_systemone_three_types_match_raw_rows_and_logical_usage(self):
        request = self.systemone_request()
        original = copy.deepcopy(request)
        raw_request = {
            "state": "text: synthetic\ntags:\n  - fresh\n  - safe",
            "questions": [
                {"instr": "Choose a color", "options": ["red: warm", "蓝: tone: cool"]},
                {"instr": "task: Rate", "options": ["low", "level: high"]},
                {"instr": "", "options": ["no: misses", "yes: matches"]},
            ],
        }
        self.enqueue.return_value = self.systemone_outputs()
        raw_result = self.decide(raw_request)
        raw_rows = [item.token_ids.tolist() for item in self.enqueue.await_args.args[0]]
        self.assertEqual(set(raw_result), {"results"})
        self.assertEqual(
            [item["option"] for item in raw_result["results"]],
            ["蓝: tone: cool", "level: high", "yes: matches"],
        )
        self.enqueue.reset_mock()
        result = asyncio.run(self.worker.systemone(request))
        self.enqueue.assert_awaited_once()
        self.assertEqual(
            [item.token_ids.tolist() for item in self.enqueue.await_args.args[0]],
            raw_rows,
        )
        expected_answers = {
            "selection": {
                "type": "choice",
                "choice": "蓝",
                "confidence": 0.5,
                "probabilities": {"red": 0.25, "蓝": 0.75},
            },
            "rating": {
                "type": "score",
                "score": 0.75,
                "legend": {"0": "low", "1": "level: high"},
                "probabilities": {"0": 0.25, "1": 0.75},
                "confidence": 0.5,
            },
            "check": {"type": "noul", "noul": 0.875},
        }
        self.assertEqual(result["model"], "kev-latest")
        self.assertEqual(result["answers"], expected_answers)
        state_tokens = 1 + len(raw_request["state"])
        expected_input_tokens = state_tokens + sum(
            len(row) - state_tokens for row in raw_rows
        )
        self.assertEqual(
            result["usage"],
            {
                "input_tokens": expected_input_tokens,
                "output_tokens": len(json.dumps(expected_answers)),
            },
        )
        self.assertLess(result["usage"]["input_tokens"], sum(map(len, raw_rows)))
        self.assertGreater(result["usage"]["output_tokens"], 0)
        self.assertGreaterEqual(result["latency_ms"], 0)
        self.assertEqual(request, original)
        for item in self.enqueue.await_args.args[0]:
            self.assertEqual(item.generate_config.max_new_tokens, 0)
            self.assertFalse(item.generate_config.is_streaming)

    def test_systemone_shared_prefix_keeps_leader_order_and_routing_key(self):
        request = self.systemone_request()
        request["state"] = "s" * 63
        request["model"] = "kev-4b"
        self.worker.kev_reuse_cache = True
        outputs = self.systemone_outputs()
        self.enqueue.side_effect = [outputs[:1], outputs[1:]]
        result = asyncio.run(self.worker.systemone(request))
        calls = self.enqueue.await_args_list
        self.assertEqual([len(call.args[0]) for call in calls], [1, 2])
        self.assertEqual(calls[0].kwargs["routing_key"], calls[1].kwargs["routing_key"])
        inputs = [item for call in calls for item in call.args[0]]
        self.assertEqual(len({item.request_id for item in inputs}), 3)
        for item in inputs:
            self.assertEqual(item.token_ids[:64].tolist(), [1] + [1115] * 63)
            self.assertEqual(item.generate_config.max_new_tokens, 0)
            self.assertEqual(item.prefix_length, 0)
        self.assertEqual(result["model"], "kev-4b")
        self.assertEqual(list(result["answers"]), ["selection", "rating", "check"])
        self.assertEqual(
            result["usage"]["input_tokens"],
            64 + sum(item.input_length - 64 for item in inputs),
        )

    def test_systemone_limits_reject_all_rows_before_any_rpc(self):
        too_many_questions = {
            "state": "",
            "questions": {str(index): {"type": "noul"} for index in range(65)},
        }
        too_many_options = {
            "state": "",
            "questions": {
                "pick": {"type": "choice", "criteria": {str(i): "" for i in range(5)}}
            },
        }
        overflowing_follower = {
            "state": "s" * 63,
            "questions": {
                "first": {"type": "choice", "criteria": {"a": ""}},
                "second": {"type": "choice", "criteria": {"b": "x" * 100}},
            },
        }
        self.worker.kev_reuse_cache = True
        self.worker.kev_max_row_tokens = 80
        for request in (too_many_questions, too_many_options, overflowing_follower):
            with self.subTest(request=request), self.assertRaises(ValueError):
                asyncio.run(self.worker.systemone(request))
        self.worker.kev_config["max_state_tokens"] = 63
        with self.assertRaisesRegex(ValueError, "state"):
            asyncio.run(self.worker.systemone(overflowing_follower))
        self.enqueue.assert_not_awaited()

    def test_systemone_non_kev_model_returns_422_without_rpc(self):
        self.worker.kev_config = None
        server = self.systemone_server()
        response = asyncio.run(server.systemone(self.systemone_request()))
        self.assertEqual(response.status_code, 422)
        self.assertIn("Kev checkpoint", json.loads(response.body)["detail"])
        self.enqueue.assert_not_awaited()
        server._global_controller.increment.assert_called_once_with()
        server._global_controller.decrement.assert_called_once_with()

    def test_systemone_server_releases_admission_after_success_or_failure(self):
        for failure in (
            None,
            ValueError("invalid request"),
            RuntimeError("backend unavailable"),
        ):
            with self.subTest(failure=failure):
                server = self.systemone_server()
                self.enqueue.side_effect = failure
                self.enqueue.return_value = self.systemone_outputs()
                response = asyncio.run(server.systemone(self.systemone_request()))
                if failure is None:
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(json.loads(response.body)["model"], "kev-latest")
                else:
                    self.assertEqual(response.status_code, 500)
                    self.assertIn(str(failure), json.loads(response.body)["message"])
                server._global_controller.increment.assert_called_once_with()
                server._global_controller.decrement.assert_called_once_with()

    def test_systemone_server_releases_admission_after_cancellation(self):
        server = self.systemone_server()

        async def run():
            entered = asyncio.Event()

            async def blocked(inputs, *, routing_key):
                entered.set()
                await asyncio.Event().wait()

            self.enqueue.side_effect = blocked
            task = asyncio.create_task(server.systemone(self.systemone_request()))
            await asyncio.wait_for(entered.wait(), timeout=1)
            server._global_controller.decrement.assert_not_called()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        server._global_controller.increment.assert_called_once_with()
        server._global_controller.decrement.assert_called_once_with()

    def test_systemone_http_route_preserves_request_headers_and_validation(self):
        self.enqueue.return_value = self.systemone_outputs()
        server = self.systemone_server()
        owner = FrontendApp.__new__(FrontendApp)
        owner.frontend_server = server
        owner.shutdown_manager = FrontendShutdownManager()
        owner.separated_frontend = True
        owner.server_config = SimpleNamespace(http_port=0)
        owner.grpc_client = None
        client = TestClient(owner.create_app())

        for supplied_id in ("synthetic-request-id", None):
            headers = {"x-typesafe-request-id": supplied_id} if supplied_id else {}
            response = client.post(
                "/v1/systemone", json=self.systemone_request(), headers=headers
            )
            self.assertEqual(response.status_code, 200)
            if supplied_id:
                self.assertEqual(response.headers["x-typesafe-request-id"], supplied_id)
            else:
                self.assertRegex(
                    response.headers["x-typesafe-request-id"], r"^[0-9a-f]{32}$"
                )
            self.assertRegex(response.headers["server-timing"], r"^app;dur=\d+\.\d$")
            self.assertEqual(
                response.json()["answers"]["check"], {"type": "noul", "noul": 0.875}
            )
            self.assertEqual(owner.shutdown_manager.active_request_count(), 0)

        self.enqueue.reset_mock()
        response = client.post(
            "/v1/systemone", json=[], headers={"x-typesafe-request-id": "invalid-body"}
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("detail", response.json())
        self.assertEqual(response.headers["x-typesafe-request-id"], "invalid-body")
        self.assertIn("server-timing", response.headers)
        self.enqueue.assert_not_awaited()
        self.assertEqual(owner.shutdown_manager.active_request_count(), 0)
        self.assertEqual(server._global_controller.increment.call_count, 3)
        self.assertEqual(server._global_controller.decrement.call_count, 3)

    def assert_metrics(self, *expected):
        self.assertCountEqual(
            [call.args[0] for call in self.report.call_args_list], expected
        )
        for call in self.report.call_args_list:
            self.assertEqual(call.args[2]["rank_id"], "0")
            self.assertEqual(call.args[2]["server_id"], "0")
            if call.args[0] != GaugeMetrics.LANTENCY_METRIC:
                self.assertEqual(call.args[1], 1)

    def test_service_metrics_count_request_once_across_question_batches(self):
        self.worker.kev_reuse_cache = True
        for method in ("decide", "systemone"):
            with self.subTest(method=method):
                server = self.systemone_server()
                request = (
                    copy.deepcopy(self.request)
                    if method == "decide"
                    else self.systemone_request()
                )
                request["state"] = "s" * 63
                original = copy.deepcopy(request)
                outputs = (
                    [self.output([0.25, 0.75]), self.output([1.0])]
                    if method == "decide"
                    else self.systemone_outputs()
                )
                self.enqueue.reset_mock()
                self.enqueue.side_effect = [outputs[:1], outputs[1:]]
                self.report.reset_mock()
                response = asyncio.run(getattr(server, method)(request))
                self.assertEqual(response.status_code, 200)
                self.assertEqual(self.enqueue.await_count, 2)
                self.assert_metrics(
                    AccMetrics.QPS_METRIC,
                    GaugeMetrics.LANTENCY_METRIC,
                    AccMetrics.SUCCESS_QPS_METRIC,
                )
                latency = self.report.call_args_list[1]
                self.assertGreaterEqual(latency.args[1], 0)
                self.assertEqual(latency.args[2]["endpoint"], method)
                server._access_logger.log_query_access.assert_called_once()
                server._access_logger.log_success_access.assert_called_once()
                log_request = server._access_logger.log_query_access.call_args.args[0]
                self.assertIsInstance(log_request[request_id_field_name], int)
                for call in self.enqueue.await_args_list:
                    for row in call.args[0]:
                        self.assertEqual(
                            row.headers["x-request-id"],
                            log_request["correlation_request_id"],
                        )
                self.assertEqual(request, original)

    def test_validation_errors_are_reported_without_rpc(self):
        for method, status, field in (
            ("decide", 400, "error"),
            ("systemone", 422, "detail"),
        ):
            for request in ([], {}, {"state": "", "questions": []}):
                with self.subTest(method=method, request=request):
                    server = self.systemone_server()
                    self.report.reset_mock()
                    response = asyncio.run(getattr(server, method)(request))
                    self.assertEqual(response.status_code, status)
                    self.assertIn(field, json.loads(response.body))
                    self.assert_metrics(
                        AccMetrics.QPS_METRIC, AccMetrics.ERROR_QPS_METRIC
                    )
                    self.assertEqual(
                        self.report.call_args_list[-1].args[2]["error_code"],
                        "INVALID_PARAMS",
                    )
                    self.enqueue.assert_not_awaited()
                    server._global_controller.decrement.assert_called_once_with()
                    server._access_logger.log_exception_access.assert_called_once()

    def test_backend_and_tokenizer_value_errors_are_server_errors(self):
        class BrokenTokenizer(FakeTokenizer):
            def __call__(self, text, *, add_special_tokens):
                raise ValueError("internal failure")

        for method in ("decide", "systemone"):
            for broken_part in ("backend", "tokenizer"):
                with self.subTest(method=method, broken_part=broken_part):
                    server = self.systemone_server()
                    self.report.reset_mock()
                    self.enqueue.side_effect = ValueError("internal failure")
                    self.worker.tokenizer = (
                        BrokenTokenizer()
                        if broken_part == "tokenizer"
                        else FakeTokenizer()
                    )
                    request = (
                        self.request if method == "decide" else self.systemone_request()
                    )
                    response = asyncio.run(getattr(server, method)(request))
                    self.assertEqual(response.status_code, 500)
                    self.assert_metrics(
                        AccMetrics.QPS_METRIC, AccMetrics.ERROR_QPS_METRIC
                    )
                    server._access_logger.log_exception_access.assert_called_once()

    def test_service_latency_includes_the_complete_worker_operation(self):
        server = self.systemone_server()
        with patch(
            "rtp_llm.frontend.frontend_server.time.perf_counter", return_value=1
        ) as clock:

            async def completed(request, *, headers):
                clock.return_value = 1.25
                return {"results": []}

            server._frontend_worker = SimpleNamespace(decide=completed)
            response = asyncio.run(server.decide(self.request))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.report.call_args_list[1].args[1], 250)
        self.assert_metrics(
            AccMetrics.QPS_METRIC,
            GaugeMetrics.LANTENCY_METRIC,
            AccMetrics.SUCCESS_QPS_METRIC,
        )

    def test_concurrency_rejection_does_not_release_another_requests_slot(self):
        for method in ("decide", "systemone"):
            with self.subTest(method=method):
                server = self.systemone_server()
                server._global_controller = ConcurrencyController(1)
                server._global_controller.increment()
                self.report.reset_mock()
                response = asyncio.run(getattr(server, method)({}))
                self.assertEqual(response.status_code, 429)
                self.assertEqual(
                    server._global_controller.get_available_concurrency(), 0
                )
                self.assert_metrics(
                    AccMetrics.QPS_METRIC, AccMetrics.CONFLICT_QPS_METRIC
                )
                self.enqueue.assert_not_awaited()
                server._global_controller.decrement()

    def test_disconnect_and_task_cancellation_stop_followers_and_release_admission(
        self,
    ):
        for method in ("decide", "systemone"):
            for cancellation in ("before", "disconnect", "task"):
                with self.subTest(method=method, cancellation=cancellation):
                    server = self.systemone_server()
                    self.worker.kev_reuse_cache = True
                    request = (
                        copy.deepcopy(self.request)
                        if method == "decide"
                        else self.systemone_request()
                    )
                    request["state"] = "s" * 63
                    self.enqueue.reset_mock()
                    self.report.reset_mock()

                    async def run():
                        received = asyncio.Queue()
                        entered = asyncio.Event()
                        cleaned_up = asyncio.Event()

                        async def blocked(inputs, *, routing_key):
                            entered.set()
                            try:
                                await asyncio.Event().wait()
                            finally:
                                cleaned_up.set()

                        self.enqueue.side_effect = blocked
                        raw = Request(
                            {"type": "http", "headers": []}, receive=received.get
                        )
                        if cancellation == "before":
                            received.put_nowait({"type": "http.disconnect"})
                        task = asyncio.create_task(
                            getattr(server, method)(request, raw)
                        )
                        if cancellation != "before":
                            await asyncio.wait_for(entered.wait(), 1)
                            if cancellation == "disconnect":
                                received.put_nowait({"type": "http.disconnect"})
                            else:
                                task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await asyncio.wait_for(task, 1)
                        if cancellation != "before":
                            self.assertTrue(cleaned_up.is_set())
                        self.assertEqual(
                            [
                                t
                                for t in asyncio.all_tasks()
                                if t is not asyncio.current_task()
                            ],
                            [],
                        )

                    asyncio.run(run())
                    self.assertEqual(
                        self.enqueue.await_count, int(cancellation != "before")
                    )
                    self.assert_metrics(
                        AccMetrics.QPS_METRIC, AccMetrics.CANCEL_QPS_METRIC
                    )
                    server._global_controller.decrement.assert_called_once_with()
                    server._access_logger.log_exception_access.assert_called_once()

    def test_systemone_http_errors_keep_headers_and_shutdown_accounting(self):
        server = self.systemone_server()
        owner = FrontendApp.__new__(FrontendApp)
        owner.frontend_server = server
        owner.shutdown_manager = FrontendShutdownManager()
        owner.separated_frontend = True
        owner.server_config = SimpleNamespace(http_port=0)
        owner.grpc_client = None
        client = TestClient(owner.create_app())
        traceparent = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
        self.enqueue.side_effect = RuntimeError("backend unavailable")
        response = client.post(
            "/v1/systemone",
            json=self.systemone_request(),
            headers={
                "x-typesafe-request-id": "correlated-id",
                "traceparent": traceparent,
                "authorization": "must-not-reach-engine",
            },
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.headers["x-typesafe-request-id"], "correlated-id")
        self.assertIn("server-timing", response.headers)
        for row in self.enqueue.await_args.args[0]:
            self.assertEqual(
                row.headers,
                {"x-request-id": "correlated-id", "traceparent": traceparent},
            )
        self.assertEqual(owner.shutdown_manager.active_request_count(), 0)
        self.assert_metrics(AccMetrics.QPS_METRIC, AccMetrics.ERROR_QPS_METRIC)

    def test_log_correlation_uses_the_same_header_precedence_as_backend(self):
        server = self.systemone_server()
        self.enqueue.return_value = [self.output([0.25, 0.75]), self.output([1.0])]
        raw = MagicMock()
        raw.headers = {
            "x-request-id": "standard-id",
            "x-dashscope-request-id": "gateway-id",
        }
        raw.is_disconnected = AsyncMock(return_value=False)

        async def wait_for_disconnect():
            await asyncio.Event().wait()

        raw.receive = wait_for_disconnect
        response = asyncio.run(server.decide(self.request, raw))
        self.assertEqual(response.status_code, 200)
        log_request = server._access_logger.log_success_access.call_args.args[0]
        self.assertEqual(log_request["correlation_request_id"], "gateway-id")
        self.assertEqual(
            self.enqueue.await_args.args[0][0].headers,
            raw.headers,
        )


if __name__ == "__main__":
    unittest.main()
