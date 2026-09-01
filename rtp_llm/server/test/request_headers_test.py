import unittest

from rtp_llm.server.request_headers import extract_request_headers


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


if __name__ == "__main__":
    unittest.main()
