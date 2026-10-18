"""Offline request-correlation contract; no server/model dependencies."""

import importlib.util
import unittest
from pathlib import Path

_PATH = Path(__file__).parents[1] / "smoke_trace.py"
_SPEC = importlib.util.spec_from_file_location("frontend_smoke_trace", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
frontend_request_fields = _MODULE.frontend_request_fields


class SmokeTraceTest(unittest.TestCase):
    def test_generic_and_historical_headers_preserve_internal_id(self):
        for header in ("x-rtp-smoke-case", "x-k3-smoke-case"):
            self.assertEqual(
                frontend_request_fields({header: "cache.hit-1"}, 73),
                {"case": "cache.hit-1", "request_id": 73},
            )

    def test_generic_label_takes_precedence_without_invalid_fallback(self):
        headers = {"x-rtp-smoke-case": "current", "x-k3-smoke-case": "historical"}
        self.assertEqual(frontend_request_fields(headers, 9)["case"], "current")
        headers["x-rtp-smoke-case"] = "bad\nlabel"
        self.assertIsNone(frontend_request_fields(headers, 9))

    def test_rejects_unbounded_or_injectable_case_labels(self):
        for value in ("", "a" * 101, "bad label", "bad\nlabel", 'bad"label', None, 7):
            self.assertIsNone(frontend_request_fields({"x-rtp-smoke-case": value}, 3))
        self.assertIsNone(frontend_request_fields({}, 3))
        self.assertEqual(
            frontend_request_fields({"x-rtp-smoke-case": "a" * 100}, 3)["case"],
            "a" * 100,
        )

    def test_payload_and_other_headers_are_never_part_of_evidence(self):
        headers = {
            "x-rtp-smoke-case": "case",
            "authorization": "private",
            "body": "prompt",
        }
        self.assertEqual(
            set(frontend_request_fields(headers, 5)), {"case", "request_id"}
        )
        self.assertEqual(headers["authorization"], "private")


if __name__ == "__main__":
    unittest.main()
