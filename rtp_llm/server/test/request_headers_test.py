"""
Unit tests for request_headers.py, focusing on the resolve_priority pure function.
Covers default/valid/invalid/case-insensitive/whitespace scenarios.
"""

import unittest

from rtp_llm.server.request_headers import (
    DEFAULT_PRIORITY,
    PRIORITY_HEADER_NAME,
    REQUEST_HEADER_NAMES,
    VALID_PRIORITIES,
    apply_request_priority,
    extract_request_headers,
    resolve_priority,
)


class TestResolvePriority(unittest.TestCase):
    """Test resolve_priority: valid values {30,40,50,60,70} kept, everything else -> 50."""

    def test_none_headers_returns_default(self):
        self.assertEqual(resolve_priority(None), DEFAULT_PRIORITY)

    def test_empty_headers_returns_default(self):
        self.assertEqual(resolve_priority({}), DEFAULT_PRIORITY)

    def test_missing_header_returns_default(self):
        self.assertEqual(resolve_priority({"x-request-id": "abc"}), DEFAULT_PRIORITY)

    def test_all_valid_values_kept(self):
        for priority in sorted(VALID_PRIORITIES):
            self.assertEqual(
                resolve_priority({PRIORITY_HEADER_NAME: str(priority)}), priority
            )

    def test_valid_int_value_kept(self):
        self.assertEqual(resolve_priority({PRIORITY_HEADER_NAME: 70}), 70)

    def test_invalid_numeric_values_return_default(self):
        for raw in ("0", "45", "99", "-1", "31", "71"):
            self.assertEqual(
                resolve_priority({PRIORITY_HEADER_NAME: raw}), DEFAULT_PRIORITY
            )

    def test_non_numeric_values_return_default(self):
        for raw in ("high", "", "  ", "50.0", "0x30", "3O", None):
            self.assertEqual(
                resolve_priority({PRIORITY_HEADER_NAME: raw}), DEFAULT_PRIORITY
            )

    def test_case_insensitive_header_lookup(self):
        for name in (
            "X-DashScope-Inner-QoS-Level",
            "X-DASHSCOPE-INNER-QOS-LEVEL",
            PRIORITY_HEADER_NAME,
        ):
            self.assertEqual(resolve_priority({name: "60"}), 60)

    def test_whitespace_value_stripped(self):
        self.assertEqual(resolve_priority({PRIORITY_HEADER_NAME: "  30  "}), 30)
        self.assertEqual(resolve_priority({PRIORITY_HEADER_NAME: "\t70\n"}), 70)

    def test_non_mapping_like_headers_return_default(self):
        self.assertEqual(resolve_priority("not-a-mapping"), DEFAULT_PRIORITY)

    def test_valid_value_among_other_headers(self):
        headers = {"x-request-id": "abc", PRIORITY_HEADER_NAME: "40", "user_id": "u1"}
        self.assertEqual(resolve_priority(headers), 40)


class TestPriorityHeaderWhitelist(unittest.TestCase):
    """The qos-level header must pass through the request header whitelist."""

    def test_header_in_whitelist(self):
        self.assertIn(PRIORITY_HEADER_NAME, REQUEST_HEADER_NAMES)

    def test_extract_keeps_qos_header(self):
        extracted = extract_request_headers({"X-DashScope-Inner-Qos-Level": "70"})
        self.assertEqual(extracted.get(PRIORITY_HEADER_NAME), "70")


class _NewPbStub:
    """Mimics a freshly generated pb2 message that has the priority field."""

    def __init__(self):
        self.priority = 0


class _StalePbStub:
    """Mimics a stale pb2 message without the priority field (hasattr is False)."""

    __slots__ = ("request_id",)

    def __init__(self):
        self.request_id = 0


class TestApplyRequestPriority(unittest.TestCase):
    """apply_request_priority must set the field only when the PB supports it."""

    def test_new_pb_field_assigned(self):
        pb = _NewPbStub()
        result = apply_request_priority(pb, {PRIORITY_HEADER_NAME: "70"})
        self.assertEqual(result, 70)
        self.assertEqual(pb.priority, 70)

    def test_stale_pb_no_exception_no_assignment(self):
        pb = _StalePbStub()
        result = apply_request_priority(pb, {PRIORITY_HEADER_NAME: "70"})
        self.assertEqual(result, 70)
        self.assertFalse(hasattr(pb, "priority"))
        self.assertEqual(pb.request_id, 0)

    def test_missing_headers_assign_default(self):
        pb = _NewPbStub()
        result = apply_request_priority(pb, None)
        self.assertEqual(result, DEFAULT_PRIORITY)
        self.assertEqual(pb.priority, DEFAULT_PRIORITY)


if __name__ == "__main__":
    unittest.main()
