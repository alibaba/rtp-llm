import unittest

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr
from rtp_llm.structure.request_extractor import RequestExtractor


class RequestExtractorTest(unittest.TestCase):
    def test_role_addrs_dicts_are_converted_without_dynamic_fields(self) -> None:
        extractor = RequestExtractor(GenerateConfig())
        config, remaining = extractor._format_request(
            {
                "prompt": "hello",
                "generate_config": {
                    "role_addrs": [
                        {
                            "role": "DECODE",
                            "ip": "127.0.0.1",
                            "http_port": 18080,
                            "grpc_port": 18081,
                        }
                    ]
                },
            }
        )

        self.assertEqual(remaining["prompt"], "hello")
        self.assertEqual(len(config.role_addrs), 1)
        self.assertIsInstance(config.role_addrs[0], RoleAddr)
        self.assertEqual(config.role_addrs[0].role.name, "DECODE")
        self.assertFalse(hasattr(config, "original_role_addrs"))


if __name__ == "__main__":
    unittest.main()
