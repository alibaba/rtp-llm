import unittest

from rtp_llm.server.request_headers import (
    extract_request_headers,
    is_valid_inference_session_id,
)


class RequestHeadersTest(unittest.TestCase):
    def test_preserves_session_routing_headers_case_insensitively(self):
        headers = extract_request_headers(
            {
                "X-DS-Inference-Session-ID": " isess_v1_example ",
                "X-DS-Inference-Session-State": " established ",
            }
        )

        self.assertEqual(headers["x-ds-inference-session-id"], "isess_v1_example")
        self.assertEqual(headers["x-ds-inference-session-state"], "established")

    def test_session_id_contract_is_printable_ascii(self):
        self.assertTrue(is_valid_inference_session_id("isess_v1_example-._:"))
        self.assertTrue(is_valid_inference_session_id("x" * 256))
        self.assertFalse(is_valid_inference_session_id(""))
        self.assertFalse(is_valid_inference_session_id("contains space"))
        self.assertFalse(is_valid_inference_session_id("emoji_😀"))
        self.assertFalse(is_valid_inference_session_id("x" * 257))


if __name__ == "__main__":
    unittest.main()
