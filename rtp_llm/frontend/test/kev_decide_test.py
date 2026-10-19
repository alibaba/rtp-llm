import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import torch

from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.frontend.frontend_worker import FrontendWorker
from rtp_llm.models.downstream_modules.kev import SPECIAL
from rtp_llm.utils.base_model_datatypes import GenerateOutput, GenerateOutputs


class FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        return SPECIAL.index(token) + 1

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("Kev segments must not acquire additional tokens")
        return SimpleNamespace(input_ids=[ord(character) + 1000 for character in text])


class KevDecideTest(unittest.TestCase):
    def setUp(self):
        self.worker = object.__new__(FrontendWorker)
        self.worker.tokenizer = FakeTokenizer()
        self.worker.kev_config = {
            "max_options": 4,
            "max_state_tokens": 65536,
            "max_row_tokens": 73728,
        }
        self.worker.kev_max_row_tokens = 32768
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
        async def respond(inputs):
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
        server = object.__new__(FrontendServer)
        server._frontend_worker = self.worker
        server._global_controller = MagicMock()
        response = asyncio.run(server.decide(self.request))
        self.assertEqual(response.status_code, 400)
        self.assertIn("Kev checkpoint", json.loads(response.body)["error"])
        self.enqueue.assert_not_awaited()
        server._global_controller.increment.assert_called_once_with()
        server._global_controller.decrement.assert_called_once_with()

    def test_server_releases_admission_slot_after_backend_failure(self):
        server = object.__new__(FrontendServer)
        server._frontend_worker = self.worker
        server._global_controller = MagicMock()
        self.enqueue.side_effect = RuntimeError("backend unavailable")
        with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
            asyncio.run(server.decide(self.request))
        server._global_controller.increment.assert_called_once_with()
        server._global_controller.decrement.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
