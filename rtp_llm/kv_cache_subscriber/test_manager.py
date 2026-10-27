from __future__ import annotations

import unittest

from rtp_llm.kv_cache_subscriber.manager import (
    KvcmHttpManagerClient,
    KvcmResponseError,
)


class KvcmHttpManagerClientTest(unittest.TestCase):
    def test_static_endpoint_list_is_parsed_and_validated(self) -> None:
        endpoints = KvcmHttpManagerClient._parse_static(
            "10.0.0.1:6382, 10.0.0.2:6383"
        )

        self.assertEqual(
            [endpoint.base_url for endpoint in endpoints],
            ["http://10.0.0.1:6382", "http://10.0.0.2:6383"],
        )
        for invalid in ("", "10.0.0.1", "10.0.0.1:not-a-port", "10.0.0.1:70000"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    KvcmHttpManagerClient._parse_static(invalid)

    def test_ok_payload_accepts_string_and_numeric_ok_item_results(self) -> None:
        KvcmHttpManagerClient._check_payload(
            "/api/reportEvent",
            {
                "header": {"status": {"code": "OK"}},
                "item_results": ["OK", 1],
            },
        )

    def test_top_level_kvcm_error_preserves_response_code(self) -> None:
        with self.assertRaises(KvcmResponseError) as context:
            KvcmHttpManagerClient._check_payload(
                "/api/reportEvent",
                {
                    "header": {
                        "status": {
                            "code": "NODE_NOT_REGISTERED",
                            "message": "register first",
                        }
                    }
                },
            )

        self.assertEqual(context.exception.code, "NODE_NOT_REGISTERED")
        self.assertIn("register first", str(context.exception))

    def test_partial_item_failure_is_not_treated_as_acknowledged(self) -> None:
        with self.assertRaises(KvcmResponseError) as context:
            KvcmHttpManagerClient._check_payload(
                "/api/reportEvent",
                {
                    "header": {"status": {"code": "OK"}},
                    "item_results": ["OK", "INTERNAL_ERROR"],
                },
            )

        self.assertEqual(context.exception.code, "PARTIAL_FAILURE")


if __name__ == "__main__":
    unittest.main()
