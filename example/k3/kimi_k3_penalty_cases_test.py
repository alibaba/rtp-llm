"""Request adaptation must never silently lose a penalty or token budget."""

import copy
import unittest

from kimi_k3_penalty_cases import adapt_request


class PenaltyCaseAdapterTest(unittest.TestCase):
    def adapt(self, request, case_type="openai", protocol="http_sync"):
        return adapt_request(
            request, case_type=case_type, protocol=protocol, model="kimi-k3"
        )

    def request(self, **params):
        return {
            "model": "online-alias",
            "messages": [{"role": "user", "content": "你好"}],
            "repetition_penalty": 1e-12,
            **params,
        }

    def test_rtp_penalty_uses_supported_nested_field_without_mutating_source(self):
        request = self.request(max_length=100)
        original = copy.deepcopy(request)
        result = self.adapt(request)
        self.assertEqual(request, original)
        self.assertNotIn("repetition_penalty", result["rtp_request"])
        self.assertEqual(
            result["rtp_request"]["extra_configs"]["repetition_penalty"], 1e-12
        )
        self.assertEqual(result["vllm_request"]["repetition_penalty"], 1e-12)
        self.assertEqual(result["rtp_request"]["max_tokens"], 100)
        self.assertFalse(result["ready_for_comparable_replay"])

    def test_missing_budget_remains_unresolved(self):
        result = self.adapt(self.request(repetition_penalty=10))
        for name in ("rtp_request", "vllm_request"):
            self.assertNotIn("max_tokens", result[name])
        self.assertIn("max_tokens", result["unresolved_defaults"])

    def test_dash_streaming_comes_from_protocol_not_absent_payload_field(self):
        request = {
            "model": "online-alias",
            "input": {"messages": self.request()["messages"]},
            "parameters": {
                "repetition_penalty": 2,
                "max_tokens": 200,
                "result_format": "message",
            },
        }
        result = self.adapt(request, "dash", "http_sse")
        self.assertTrue(result["rtp_request"]["stream"])
        self.assertEqual(
            result["vllm_request"]["stream_options"], {"include_usage": True}
        )
        self.assertNotIn("stream_options", result["rtp_request"])

    def test_unmapped_and_conflicting_fields_are_rejected(self):
        for extra in (
            {"max_length": 100, "max_tokens": 200},
            {"stream": True},
            {"unrecognized_penalty": 2},
            {"repetition_penalty": 0},
            {"repetition_penalty": float("nan")},
            {"max_tokens": True},
            {"enable_thinking": False},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.adapt(self.request(**extra))


if __name__ == "__main__":
    unittest.main()
