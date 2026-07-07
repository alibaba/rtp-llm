import unittest

from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint


class TestExtractProfileConfig(unittest.TestCase):
    def test_no_profile_config(self):
        request = {"input": "hello"}
        config = EmbeddingEndpoint._extract_profile_config(request)
        self.assertEqual(config["gen_timeline"], False)
        self.assertEqual(config["profile_step"], 1)
        self.assertEqual(config["profile_trace_name"], "")

    def test_generate_config(self):
        request = {
            "input": "hello",
            "generate_config": {
                "gen_timeline": True,
                "profile_step": 3,
                "profile_trace_name": "my_trace",
            },
        }
        config = EmbeddingEndpoint._extract_profile_config(request)
        self.assertEqual(config["gen_timeline"], True)
        self.assertEqual(config["profile_step"], 3)
        self.assertEqual(config["profile_trace_name"], "my_trace")

    def test_extra_configs(self):
        request = {
            "input": "hello",
            "extra_configs": {"gen_timeline": True, "profile_step": 2},
        }
        config = EmbeddingEndpoint._extract_profile_config(request)
        self.assertEqual(config["gen_timeline"], True)
        self.assertEqual(config["profile_step"], 2)

    def test_generate_config_overrides_extra_configs(self):
        request = {
            "extra_configs": {"gen_timeline": False, "profile_step": 5},
            "generate_config": {"gen_timeline": True, "profile_step": 1},
        }
        config = EmbeddingEndpoint._extract_profile_config(request)
        self.assertEqual(config["gen_timeline"], True)
        self.assertEqual(config["profile_step"], 1)

    def test_type_coercion(self):
        request = {
            "generate_config": {
                "gen_timeline": 1,
                "profile_step": "3",
                "profile_trace_name": 123,
            },
        }
        config = EmbeddingEndpoint._extract_profile_config(request)
        self.assertEqual(config["gen_timeline"], True)
        self.assertEqual(config["profile_step"], 3)
        self.assertEqual(config["profile_trace_name"], "123")

    def test_invalid_extra_configs_type(self):
        request = {"extra_configs": "not_a_dict", "generate_config": None}
        config = EmbeddingEndpoint._extract_profile_config(request)
        self.assertEqual(config["gen_timeline"], False)
        self.assertEqual(config["profile_step"], 1)

    def test_string_false_is_false(self):
        request = {"generate_config": {"gen_timeline": "false"}}
        config = EmbeddingEndpoint._extract_profile_config(request)
        self.assertEqual(config["gen_timeline"], False)

    def test_string_true_is_true(self):
        request = {"generate_config": {"gen_timeline": "True"}}
        config = EmbeddingEndpoint._extract_profile_config(request)
        self.assertEqual(config["gen_timeline"], True)

    def test_invalid_profile_step_falls_back(self):
        request = {"generate_config": {"gen_timeline": True, "profile_step": "abc"}}
        config = EmbeddingEndpoint._extract_profile_config(request)
        self.assertEqual(config["gen_timeline"], True)
        self.assertEqual(config["profile_step"], 1)


if __name__ == "__main__":
    unittest.main()
