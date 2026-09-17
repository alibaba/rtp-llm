import copy
import json
import unittest

from rtp_llm.test.smoke.common_def import QueryStatus, SmokeException, Tracer
from rtp_llm.test.smoke.normal_comparer import NormalComparer, QueryInfo

GRAPH_STATUS = "generation_prefill_cuda_graph_status"
REPLAYED = "replayed"
FALLBACK = "input_tokens_exceed_capture_limit"


class NormalComparerGraphStatusTest(unittest.TestCase):
    def setUp(self):
        # Comparing responses does not require constructing a model server.
        self.comparer = NormalComparer(None, "", {}, Tracer(), False)
        self.query = QueryInfo(prompt="test")

    def _response(self, status):
        return {"response": "same output", "aux_info": {GRAPH_STATUS: status}}

    def _parse(self, payload):
        return self.comparer.format_result(
            self.comparer.curl_response_to_json(self.query, json.dumps(payload))
        )

    def _assert_compare_failed(self, expected_payload, actual_payload, diff_field):
        expected = self._parse(expected_payload)
        actual = self._parse(actual_payload)
        with self.assertRaises(SmokeException) as raised:
            self.comparer.compare_result(expected, actual)
        self.assertEqual(raised.exception.error_status, QueryStatus.COMPARE_FAILED)
        self.assertIn(diff_field, raised.exception.message)
        return raised.exception.message

    def test_graph_status_survives_json_and_pydantic_parsing(self):
        for status in (REPLAYED, FALLBACK):
            with self.subTest(status=status):
                result = self._parse(self._response(status))
                self.assertEqual(result.model_dump()["aux_info"][GRAPH_STATUS], status)

    def test_matching_graph_status_is_accepted(self):
        for status in (REPLAYED, FALLBACK):
            with self.subTest(status=status):
                self.comparer.compare_result(
                    self._parse(self._response(status)),
                    self._parse(self._response(status)),
                )

    def test_iter_count_remains_exact_even_with_tolerance_metadata(self):
        expected = {
            "response": "same output",
            "aux_info": {"iter_count": 44, "iter_count_tolerance": 1},
        }
        self.comparer.compare_result(self._parse(expected), self._parse(expected))
        for actual_count in (43, 45, None):
            with self.subTest(actual_count=actual_count):
                self._assert_compare_failed(
                    expected,
                    {"response": "same output", "aux_info": {"iter_count": actual_count}},
                    "iter_count",
                )

    def test_mismatched_graph_status_is_rejected(self):
        for expected, actual in (
            (REPLAYED, FALLBACK),
            (FALLBACK, REPLAYED),
            (REPLAYED, "not_requested"),
            (FALLBACK, "not_requested"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self._assert_compare_failed(
                    self._response(expected), self._response(actual), GRAPH_STATUS
                )

    def test_missing_or_null_graph_status_is_rejected(self):
        for expected in (REPLAYED, FALLBACK):
            for aux_info in ({}, {GRAPH_STATUS: None}):
                with self.subTest(expected=expected, aux_info=aux_info):
                    self._assert_compare_failed(
                        self._response(expected),
                        {"response": "same output", "aux_info": aux_info},
                        GRAPH_STATUS,
                    )

    def test_missing_or_null_aux_info_is_rejected(self):
        for expected in (REPLAYED, FALLBACK):
            for actual in (
                {"response": "same output"},
                {"response": "same output", "aux_info": None},
            ):
                with self.subTest(expected=expected, actual=actual):
                    self._assert_compare_failed(
                        self._response(expected), actual, "aux_info"
                    )

    def test_non_first_batch_item_is_checked(self):
        self.query = QueryInfo(prompt_batch=["test", "test"])
        for actual_second in (
            self._response(FALLBACK),
            {"response": "same output", "aux_info": {}},
            {"response": "same output"},
        ):
            with self.subTest(actual_second=actual_second):
                message = self._assert_compare_failed(
                    {
                        "response_batch": [
                            self._response(REPLAYED),
                            self._response(REPLAYED),
                        ]
                    },
                    {"response_batch": [self._response(REPLAYED), actual_second]},
                    "aux_info",
                )
                self.assertIn("[batch_idx=1]", message)
                self.assertNotIn("[batch_idx=0]", message)

    def test_matching_batch_graph_statuses_are_accepted(self):
        self.query = QueryInfo(prompt_batch=["test", "test"])
        payload = {
            "response_batch": [self._response(REPLAYED), self._response(FALLBACK)]
        }
        self.comparer.compare_result(self._parse(payload), self._parse(payload))

    def test_legacy_golden_without_graph_status_remains_compatible(self):
        for expected in (
            {"response": "same output"},
            {"response": "same output", "aux_info": {}},
        ):
            for status in (REPLAYED, FALLBACK):
                with self.subTest(expected=expected, actual_status=status):
                    self.comparer.compare_result(
                        self._parse(expected), self._parse(self._response(status))
                    )


class NormalComparerPromptScoringTest(unittest.TestCase):
    def setUp(self):
        # Value comparison needs neither server setup nor tensor golden files.
        self.comparer = object.__new__(NormalComparer)
        self.comparer.test_logits = self.comparer.test_hidden_states = False
        self.payload = {
            "response": "same output",
            "prompt_logprobs": {
                "start_pos": 0,
                "end_pos": 2,
                "topk_token_ids_head": [[11, 12], [21, 22]],
                "topk_logprobs_head": [[-0.5, -1.5], [-0.6, -1.6]],
                "target_logprobs": [-0.5, -0.6],
            },
        }

    def _parse(self, payload):
        return self.comparer.format_result(copy.deepcopy(payload))

    def test_prompt_scoring_survives_response_parsing(self):
        parsed = self._parse(self.payload)
        scoring = parsed.model_dump()["prompt_logprobs"]
        for field, expected in self.payload["prompt_logprobs"].items():
            self.assertEqual(scoring[field], expected)

    def test_full_prompt_arrays_match_golden_heads(self):
        self.comparer.compare_result(self._parse(self.payload), self._parse(self.payload))
        actual = copy.deepcopy(self.payload)
        scoring = actual["prompt_logprobs"]
        scoring["topk_token_ids"] = scoring.pop("topk_token_ids_head")
        scoring["topk_logprobs"] = scoring.pop("topk_logprobs_head")
        self.comparer.compare_result(self._parse(self.payload), self._parse(actual))

    def test_prompt_scoring_mismatches_are_rejected(self):
        invalid = []
        missing = copy.deepcopy(self.payload)
        missing.pop("prompt_logprobs")
        invalid.append(("missing", missing))
        for field, value in (
            ("start_pos", 1),
            ("end_pos", 3),
            ("topk_token_ids_head", [[99, 12], [21, 22]]),
            ("topk_logprobs_head", [[-5.0, -1.5], [-0.6, -1.6]]),
            ("topk_logprobs_head", None),
            ("target_logprobs", [-5.0, -0.6]),
            ("target_logprobs", None),
        ):
            actual = copy.deepcopy(self.payload)
            actual["prompt_logprobs"][field] = value
            invalid.append((field, actual))
        for field, actual in invalid:
            with self.subTest(field=field), self.assertRaises(SmokeException) as raised:
                self.comparer.compare_result(self._parse(self.payload), self._parse(actual))
            self.assertEqual(raised.exception.error_status, QueryStatus.COMPARE_FAILED)
            self.assertIn("prompt_logits", raised.exception.message)

    def test_unrequested_prompt_scoring_preserves_legacy_comparison(self):
        self.comparer.compare_result(
            self._parse({"response": "same output"}), self._parse(self.payload)
        )


if __name__ == "__main__":
    unittest.main()
