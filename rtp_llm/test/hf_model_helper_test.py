import sys
import importlib.util
import unittest
from unittest.mock import MagicMock

# Load model_factory_register directly to avoid __init__.py's .so dependency
_spec = importlib.util.spec_from_file_location(
    "rtp_llm.model_factory_register", "rtp_llm/model_factory_register.py"
)
_mfr = importlib.util.module_from_spec(_spec)
sys.modules["rtp_llm.model_factory_register"] = _mfr
_spec.loader.exec_module(_mfr)

# Register architectures to simulate production state
_mfr.register_model("deepseek2", object, ["DeepseekV2ForCausalLM"])
_mfr.register_model("deepseek3", object, ["DeepseekV3ForCausalLM"])
_mfr.register_model("qwen35_moe", object, ["Qwen3_5MoeForConditionalGeneration"])
_mfr.register_model("qwen35_dense", object, ["Qwen3_5ForConditionalGeneration"])

# Mock huggingface_hub before importing hf_model_helper
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["huggingface_hub.hf_api"] = MagicMock()

_spec2 = importlib.util.spec_from_file_location(
    "rtp_llm.tools.api.hf_model_helper", "rtp_llm/tools/api/hf_model_helper.py"
)
_hfh = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_hfh)

HfStyleModelInfo = _hfh.HfStyleModelInfo


class TestResolveFtModelType(unittest.TestCase):
    def test_direct_match(self):
        config = {"architectures": ["DeepseekV3ForCausalLM"]}
        self.assertEqual(HfStyleModelInfo.resolve_ft_model_type(config), "deepseek3")

    def test_fuzzy_match_causal_to_conditional(self):
        config = {"architectures": ["Qwen3_5MoeForCausalLM"]}
        self.assertEqual(HfStyleModelInfo.resolve_ft_model_type(config), "qwen35_moe")

    def test_fuzzy_match_conditional_to_causal(self):
        _mfr.register_model("test_causal", object, ["TestModelForCausalLM"])
        config = {"architectures": ["TestModelForConditionalGeneration"]}
        self.assertEqual(HfStyleModelInfo.resolve_ft_model_type(config), "test_causal")

    def test_model_type_fallback_deepseek_v4(self):
        config = {"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"}
        self.assertEqual(HfStyleModelInfo.resolve_ft_model_type(config), "deepseek3")

    def test_model_type_fallback_qwen3_5(self):
        config = {"architectures": ["UnknownArch"], "model_type": "qwen3_5"}
        self.assertEqual(HfStyleModelInfo.resolve_ft_model_type(config), "qwen35_dense")

    def test_all_miss_returns_none(self):
        config = {"architectures": ["CompletelyUnknownArch"], "model_type": "unknown"}
        self.assertIsNone(HfStyleModelInfo.resolve_ft_model_type(config))

    def test_empty_architectures(self):
        config = {"architectures": [], "model_type": "deepseek_v3"}
        self.assertEqual(HfStyleModelInfo.resolve_ft_model_type(config), "deepseek3")

    def test_no_architectures_key(self):
        config = {"model_type": "qwen3_5_moe"}
        self.assertEqual(HfStyleModelInfo.resolve_ft_model_type(config), "qwen35_moe")


class TestFuzzyMatchArchitecture(unittest.TestCase):
    def test_empty_architectures_returns_none(self):
        self.assertIsNone(HfStyleModelInfo._fuzzy_match_architecture({"architectures": []}))

    def test_no_architectures_key_returns_none(self):
        self.assertIsNone(HfStyleModelInfo._fuzzy_match_architecture({}))

    def test_no_suffix_match_returns_none(self):
        config = {"architectures": ["SomeRandomModel"]}
        self.assertIsNone(HfStyleModelInfo._fuzzy_match_architecture(config))

    def test_does_not_match_nextn_suffix(self):
        _mfr.register_model("mtp", object, ["DeepseekV3ForCausalLMNextN"])
        config = {"architectures": ["DeepseekV3ForCausalLMNextN"]}
        self.assertIsNone(HfStyleModelInfo._fuzzy_match_architecture(config))


class TestDtypeBytes(unittest.TestCase):
    def test_f32_is_4_bytes(self):
        self.assertEqual(HfStyleModelInfo._DTYPE_BYTES["F32"], 4)

    def test_bf16_is_2_bytes(self):
        self.assertEqual(HfStyleModelInfo._DTYPE_BYTES["BF16"], 2)

    def test_i8_is_1_byte(self):
        self.assertEqual(HfStyleModelInfo._DTYPE_BYTES["I8"], 1)

    def test_f8_e4m3_is_1_byte(self):
        self.assertEqual(HfStyleModelInfo._DTYPE_BYTES["F8_E4M3"], 1)

    def test_unknown_dtype_defaults_to_2(self):
        self.assertEqual(HfStyleModelInfo._DTYPE_BYTES.get("UNKNOWN_TYPE", 2), 2)

    def test_bool_is_1_byte(self):
        self.assertEqual(HfStyleModelInfo._DTYPE_BYTES["BOOL"], 1)


if __name__ == "__main__":
    unittest.main()
