import json
import unittest

from smoke.base_comparer import _asserts_cold_cache
from smoke.common_def import QueryStatus, SmokeException, Tracer
from smoke.normal_comparer import NormalComparer, QueryInfo

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


class NormalComparerDiskReuseTest(unittest.TestCase):
    DISK_FIELDS = (
        "disk_reuse_len",
        "prefill_disk_reuse_len",
        "decode_disk_reuse_len",
    )

    def setUp(self):
        self.comparer = NormalComparer(None, "", {}, Tracer(), False)

    def _response(self, aux):
        return self.comparer.format_result({"response": "same output", "aux_info": aux})

    def test_explicit_disk_expectations_survive_parsing_and_reject_mismatch(self):
        for field in self.DISK_FIELDS:
            for expected_value, actual_value in ((512, 0), (0, 512), (512, None)):
                with self.subTest(
                    field=field, expected=expected_value, actual=actual_value
                ):
                    expected = self._response({field: expected_value})
                    self.assertEqual(getattr(expected.aux_info, field), expected_value)
                    with self.assertRaises(SmokeException) as raised:
                        self.comparer.compare_result(
                            expected, self._response({field: actual_value})
                        )
                    self.assertEqual(
                        raised.exception.error_status, QueryStatus.COMPARE_FAILED
                    )
                    self.assertIn(field, raised.exception.message)

    def test_matching_disk_expectations_pass(self):
        for field in self.DISK_FIELDS:
            for value in (0, 512):
                with self.subTest(field=field, value=value):
                    self.comparer.compare_result(
                        self._response({field: value}), self._response({field: value})
                    )

    def test_undeclared_or_null_disk_expectation_does_not_add_constraint(self):
        for field in self.DISK_FIELDS:
            for expected in ({}, {field: None}):
                with self.subTest(field=field, expected=expected):
                    self.comparer.compare_result(
                        self._response(expected), self._response({field: 512})
                    )

    def test_disk_only_cold_expectation_controls_retry_classification(self):
        for field in self.DISK_FIELDS:
            for reuse_enabled in (False, True):
                for value in (0, 512):
                    with self.subTest(field=field, reuse=reuse_enabled, value=value):
                        self.assertEqual(
                            _asserts_cold_cache(
                                {
                                    "_reuse_cache_enabled": reuse_enabled,
                                    "result": {"aux_info": {field: value}},
                                }
                            ),
                            reuse_enabled and value == 0,
                        )

    def test_mixed_tier_expectations_are_cold_only_when_all_fields_are_zero(self):
        for disk_field in self.DISK_FIELDS:
            prefix = disk_field.removesuffix("disk_reuse_len")
            for tier in ("local", "memory", "remote"):
                other_field = f"{prefix}{tier}_reuse_len"
                for disk_value, other_value in ((0, 0), (0, 512), (512, 0), (512, 512)):
                    for reuse_enabled in (False, True):
                        with self.subTest(
                            disk_field=disk_field,
                            other_field=other_field,
                            disk_value=disk_value,
                            other_value=other_value,
                            reuse_enabled=reuse_enabled,
                        ):
                            self.assertEqual(
                                _asserts_cold_cache(
                                    {
                                        "_reuse_cache_enabled": reuse_enabled,
                                        "result": {
                                            "aux_info": {
                                                disk_field: disk_value,
                                                other_field: other_value,
                                            }
                                        },
                                    }
                                ),
                                reuse_enabled and disk_value == 0 and other_value == 0,
                            )

    def test_warm_disk_expectation_is_not_misclassified_by_other_zero_fields(self):
        for field in self.DISK_FIELDS:
            with self.subTest(field=field):
                self.assertFalse(
                    _asserts_cold_cache(
                        {
                            "_reuse_cache_enabled": True,
                            "result": {"aux_info": {"reuse_len": 0, field: 512}},
                        }
                    )
                )


if __name__ == "__main__":
    unittest.main()
