import copy
import json
import unittest

from smoke.common_def import QueryStatus, SmokeException, Tracer
from smoke.openai_comparer import OpenaiComparer

GRAPH_STATUS = "generation_prefill_cuda_graph_status"
REPLAYED = "replayed"
FALLBACK = "input_tokens_exceed_capture_limit"


class OpenaiComparerGraphStatusTest(unittest.TestCase):
    def setUp(self):
        # The OpenAI graph smoke goldens require the status via compare_config,
        # without specifying result.aux_info (which contains runtime metrics).
        self.golden = {
            "id": "chat-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "same output"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        qr_info = {
            "query": {
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
            "result": self.golden,
            "compare_config": {"required_aux_info": {GRAPH_STATUS: REPLAYED}},
        }
        self.comparer = OpenaiComparer(None, "", qr_info, Tracer(), False)
        self.query = self.comparer.format_query(qr_info["query"])

    def _response(self, aux_info):
        payload = copy.deepcopy(self.golden)
        payload["aux_info"] = aux_info
        return payload

    def _parse(self, payload):
        return self.comparer.format_result(
            self.comparer.curl_response_to_json(self.query, json.dumps(payload))
        )

    def _compare(self, expected_status, actual_payload):
        self.comparer.qr_info["compare_config"]["required_aux_info"] = {
            GRAPH_STATUS: expected_status
        }
        expected = self._parse(self.golden)
        self.assertIsNone(expected.aux_info)
        self.comparer.compare_result(expected, self._parse(actual_payload))

    def _assert_compare_failed(self, expected_status, actual_payload, diff_field):
        with self.assertRaises(SmokeException) as raised:
            self._compare(expected_status, actual_payload)
        self.assertEqual(raised.exception.error_status, QueryStatus.COMPARE_FAILED)
        self.assertIn("required aux_info", raised.exception.message)
        self.assertIn(diff_field, raised.exception.message)

    def test_graph_status_survives_json_and_pydantic_parsing(self):
        for status in (REPLAYED, FALLBACK):
            with self.subTest(status=status):
                result = self._parse(self._response({GRAPH_STATUS: status}))
                self.assertEqual(result.model_dump()["aux_info"][GRAPH_STATUS], status)

    def test_matching_required_status_is_accepted_without_golden_aux_info(self):
        for status in (REPLAYED, FALLBACK):
            with self.subTest(status=status):
                self._compare(status, self._response({GRAPH_STATUS: status}))

    def test_mismatched_required_status_is_rejected_without_golden_aux_info(self):
        for expected, actual in (
            (REPLAYED, FALLBACK),
            (FALLBACK, REPLAYED),
            (REPLAYED, "not_requested"),
            (FALLBACK, "not_requested"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self._assert_compare_failed(
                    expected, self._response({GRAPH_STATUS: actual}), GRAPH_STATUS
                )

    def test_missing_required_status_is_rejected(self):
        for expected in (REPLAYED, FALLBACK):
            with self.subTest(expected=expected):
                self._assert_compare_failed(expected, self._response({}), GRAPH_STATUS)

    def test_missing_or_null_aux_info_is_rejected(self):
        for expected in (REPLAYED, FALLBACK):
            for actual in (self.golden, self._response(None)):
                with self.subTest(expected=expected, actual=actual):
                    self._assert_compare_failed(expected, actual, "missing")

    def test_legacy_golden_without_required_status_remains_compatible(self):
        for compare_config in ({}, {"required_aux_info": {}}):
            self.comparer.qr_info["compare_config"] = compare_config
            for status in (REPLAYED, FALLBACK):
                with self.subTest(compare_config=compare_config, actual_status=status):
                    self.comparer.compare_result(
                        self._parse(self.golden),
                        self._parse(self._response({GRAPH_STATUS: status})),
                    )

    def test_legacy_golden_without_compare_config_remains_compatible(self):
        del self.comparer.qr_info["compare_config"]
        self.comparer.compare_result(
            self._parse(self.golden),
            self._parse(self._response({GRAPH_STATUS: REPLAYED})),
        )


class OpenaiComparerHealthTest(unittest.TestCase):
    def setUp(self):
        helper = OpenaiComparerGraphStatusTest()
        helper.setUp()
        self.comparer = helper.comparer
        self.golden = helper.golden
        self.config = {
            "skip_choices": True,
            "skip_usage": True,
            "output_health_check": {"min_content_chars": 5},
        }
        self.comparer.qr_info["compare_config"] = self.config

    def check_response(self, payload, stream=False):
        self.comparer.is_stream = stream
        if stream:
            payload = copy.deepcopy(payload)
            payload["object"] = "chat.completion.chunk"
            for choice in payload["choices"]:
                choice["delta"] = choice.pop("message")
        self.comparer.compare_result(
            self.comparer.format_result({}),
            self.comparer.format_result(payload),
        )

    def test_empty_expected_result_accepts_healthy_response(self):
        self.check_response(copy.deepcopy(self.golden))

    def test_skip_choices_does_not_bypass_health_checks(self):
        for content in ("", "bad", "abc\x01def"):
            with self.subTest(content=content):
                payload = copy.deepcopy(self.golden)
                payload["choices"][0]["message"]["content"] = content
                with self.assertRaises(SmokeException):
                    self.check_response(payload)

    def test_usage_threshold_is_checked_even_with_skip_usage(self):
        self.config["output_health_check"]["min_completion_tokens"] = 2
        with self.assertRaises(SmokeException):
            self.check_response(copy.deepcopy(self.golden))

    def test_exact_repetition_boundary_is_rejected(self):
        self.config["output_health_check"]["repeat_window"] = 3
        payload = copy.deepcopy(self.golden)
        payload["choices"][0]["message"]["content"] = "abcdefghijklmnop" * 3
        with self.assertRaises(SmokeException):
            self.check_response(payload)

    def test_reasoning_only_response_is_checked(self):
        self.config["output_health_check"].update(
            min_unique_non_ws_chars=5,
            required_substrings=["valid"],
            required_substrings_any=["reasoning", "alternative"],
            forbidden_substrings=["forbidden"],
        )
        for stream in (False, True):
            for content in (None, "", " \n\t"):
                with self.subTest(stream=stream, content=content):
                    payload = copy.deepcopy(self.golden)
                    payload["choices"][0]["message"].update(
                        content=content, reasoning_content="valid reasoning"
                    )
                    self.check_response(payload, stream=stream)
                    payload["choices"][0]["message"]["reasoning_content"] = "bad"
                    with self.assertRaises(SmokeException) as raised:
                        self.check_response(payload, stream=stream)
                    self.assertIn("content length", raised.exception.message)
                    self.assertIn("content unique chars", raised.exception.message)
                    self.assertIn("required substrings", raised.exception.message)
                    self.assertIn("required any substring", raised.exception.message)

    def test_both_fields_are_checked_for_corruption(self):
        self.config["output_health_check"]["repeat_window"] = 3
        for stream in (False, True):
            for field in ("content", "reasoning_content"):
                for text, failure in (
                    ("\x00" * 64, "control chars"),
                    ("abcdefghijklmnop" * 3, "repetition"),
                ):
                    with self.subTest(stream=stream, field=field, failure=failure):
                        payload = copy.deepcopy(self.golden)
                        message = payload["choices"][0]["message"]
                        message.update(
                            content="ABCDE", reasoning_content="valid reasoning"
                        )
                        message[field] = text
                        with self.assertRaises(SmokeException) as raised:
                            self.check_response(payload, stream=stream)
                        self.assertEqual(
                            raised.exception.error_status, QueryStatus.COMPARE_FAILED
                        )
                        self.assertIn(f"{field} {failure}", raised.exception.message)

    def test_healthy_reasoning_and_answer_are_accepted(self):
        self.config["output_health_check"].update(
            min_unique_non_ws_chars=5,
            repeat_window=3,
            required_substrings=["ABCDE"],
            required_substrings_any=["ABCDE", "alternative"],
            forbidden_substrings=["reasoning"],
        )
        for stream in (False, True):
            for reasoning in (None, "", "x", "valid reasoning"):
                with self.subTest(stream=stream, reasoning=reasoning):
                    payload = copy.deepcopy(self.golden)
                    payload["choices"][0]["message"].update(
                        content="ABCDE", reasoning_content=reasoning
                    )
                    self.check_response(payload, stream=stream)

    def test_answer_requirements_cannot_be_satisfied_by_reasoning(self):
        for stream in (False, True):
            for requirement, value, failure in (
                ("min_content_chars", 10, "content length"),
                ("min_unique_non_ws_chars", 6, "content unique chars"),
                ("required_substrings", ["reasoning"], "required substrings"),
                ("required_substrings_any", ["reasoning"], "required any substring"),
            ):
                with self.subTest(stream=stream, requirement=requirement):
                    self.config["output_health_check"] = {
                        "min_content_chars": 5,
                        requirement: value,
                    }
                    payload = copy.deepcopy(self.golden)
                    payload["choices"][0]["message"].update(
                        content="ABCDE", reasoning_content="valid reasoning"
                    )
                    with self.assertRaises(SmokeException) as raised:
                        self.check_response(payload, stream=stream)
                    self.assertIn(failure, raised.exception.message)

    def test_repetition_does_not_span_reasoning_and_answer(self):
        self.config["output_health_check"]["repeat_window"] = 3
        for stream in (False, True):
            with self.subTest(stream=stream):
                payload = copy.deepcopy(self.golden)
                payload["choices"][0]["message"].update(
                    content="abcdefghijklmnop" * 2,
                    reasoning_content="abcdefghijklmnop" * 2,
                )
                self.check_response(payload, stream=stream)

    def test_required_and_forbidden_substrings(self):
        health = self.config["output_health_check"]
        health["required_substrings"] = ["same"]
        health["required_substrings_any"] = ["output", "alternative"]
        self.check_response(copy.deepcopy(self.golden))
        health["forbidden_substrings"] = ["output"]
        with self.assertRaises(SmokeException):
            self.check_response(copy.deepcopy(self.golden))

    def test_cache_reuse_and_mtp_aux_metrics(self):
        self.config["aux_info_health_check"] = {
            "min_reuse_len": 64,
            "min_prefill_total_reuse_len": 64,
            "min_decode_total_reuse_len": 64,
            "min_tokens_per_iter": 2,
            "required_pd_sep": True,
        }
        payload = copy.deepcopy(self.golden)
        payload["aux_info"] = {
            "reuse_len": 64,
            "prefill_total_reuse_len": 64,
            "decode_total_reuse_len": 64,
            "output_len": 64,
            "iter_count": 16,
            "pd_sep": True,
        }
        self.check_response(payload)
        for field, value in (("reuse_len", 0), ("iter_count", 64), ("pd_sep", False)):
            bad = copy.deepcopy(payload)
            bad["aux_info"][field] = value
            with self.subTest(field=field), self.assertRaises(SmokeException):
                self.check_response(bad)
        payload.pop("aux_info")
        with self.assertRaises(SmokeException):
            self.check_response(payload)


if __name__ == "__main__":
    unittest.main()
