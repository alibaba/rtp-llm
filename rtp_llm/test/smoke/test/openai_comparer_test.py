import copy
import json
import unittest
from unittest.mock import patch

from smoke.common_def import QueryStatus, SmokeException, Tracer
from smoke.openai_comparer import OpenaiComparer
from smoke.task_info import TaskInfo

from rtp_llm.config.generate_config import RoleAddr
from rtp_llm.utils.base_model_datatypes import AuxInfo

GRAPH_STATUS = "generation_prefill_cuda_graph_status"
REPLAYED = "replayed"
FALLBACK = "input_tokens_exceed_capture_limit"


class OpenaiComparerTest(unittest.TestCase):
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

    def test_missing_or_null_required_aux_info_is_rejected(self):
        for expected in (REPLAYED, FALLBACK):
            for actual, field in (
                (self._response({}), GRAPH_STATUS),
                (self.golden, "missing"),
                (self._response(None), "missing"),
            ):
                with self.subTest(expected=expected, actual=actual):
                    self._assert_compare_failed(expected, actual, field)

    def test_legacy_golden_without_required_status_remains_compatible(self):
        for compare_config in (None, {}, {"required_aux_info": {}}):
            if compare_config is None:
                self.comparer.qr_info.pop("compare_config", None)
            else:
                self.comparer.qr_info["compare_config"] = compare_config
            for status in (REPLAYED, FALLBACK):
                with self.subTest(compare_config=compare_config, actual_status=status):
                    self.comparer.compare_result(
                        self._parse(self.golden),
                        self._parse(self._response({GRAPH_STATUS: status})),
                    )

    def test_aux_step_and_address_differences_are_checked_by_default(self):
        expected = AuxInfo(
            step_output_len=1,
            role_addrs=[RoleAddr(role="DECODE", ip="192.0.2.1", http_port=80, grpc_port=81)],
        )
        actual = AuxInfo(
            step_output_len=2,
            role_addrs=[RoleAddr(role="DECODE", ip="192.0.2.2", http_port=80, grpc_port=81)],
        )
        for config in (
            {},
            {"ignore_aux_info_fields": ["step_output_len", "role_addrs.ip"]},
            {},
        ):
            with self.subTest(config=config):
                self.comparer.qr_info["compare_config"] = config
                diffs = []
                self.comparer._compare_aux_info(expected, actual, diffs)
                if config:
                    self.assertEqual(diffs, [])
                else:
                    self.assertTrue(any("step_output_len" in diff for diff in diffs))
                    self.assertTrue(any("role_addrs" in diff for diff in diffs))

    def test_query_aux_exceptions_do_not_hide_cache_reuse_differences(self):
        self.comparer.qr_info["compare_config"] = {
            "ignore_aux_info_fields": ["step_output_len", "role_addrs.ip"]
        }
        diffs = []
        self.comparer._compare_aux_info(AuxInfo(reuse_len=0), AuxInfo(reuse_len=8), diffs)
        self.assertTrue(any("reuse_len" in diff for diff in diffs))

    def test_health_checks_image_tokens_without_comparing_other_usage_fields(self):
        expected = self._parse(self.golden)
        for image_tokens, expected_tokens, success in (
            (None, 0, True),
            (None, 486, False),
            (485, 486, False),
            (486, 486, True),
            (486, 0, False),
        ):
            with self.subTest(image_tokens=image_tokens, expected=expected_tokens):
                self.comparer.qr_info["compare_config"] = {
                    "skip_usage": True,
                    "output_health_check": {"expected_image_tokens": expected_tokens},
                }
                payload = copy.deepcopy(self.golden)
                if image_tokens is not None:
                    payload["usage"]["prompt_tokens_details"] = {
                        "image_tokens": image_tokens
                    }
                if success:
                    self.comparer.compare_result(expected, self._parse(payload))
                else:
                    with self.assertRaisesRegex(SmokeException, "health image token count"):
                        self.comparer.compare_result(expected, self._parse(payload))


class TaskInfoCheckpointPathTest(unittest.TestCase):
    def test_resolves_exact_environment_placeholder(self):
        with patch.dict("os.environ", {"TEST_MODEL_CHECKPOINT": "/fixture/model"}):
            task = TaskInfo(
                model_type="glm_5",
                model_path="${TEST_MODEL_CHECKPOINT}",
                query_result=[],
            )
        self.assertEqual(task.model_path, "/fixture/model")

    def test_rejects_unset_placeholder_and_preserves_literal_paths(self):
        with patch.dict("os.environ", {"TEST_MODEL_CHECKPOINT": ""}):
            with self.assertRaisesRegex(ValueError, "Set TEST_MODEL_CHECKPOINT"):
                TaskInfo(
                    model_type="glm_5",
                    model_path="${TEST_MODEL_CHECKPOINT}",
                    query_result=[],
                )
        task = TaskInfo(model_type="glm_5", model_path="oss://model", query_result=[])
        self.assertEqual(task.model_path, "oss://model")

if __name__ == "__main__":
    unittest.main()
