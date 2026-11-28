"""V4.1 target and draft must reach distinct real service runtime owners."""

import tempfile
import unittest

from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v41.language import DeepSeekV41ForCausalLM
from rtp_llm.models_py.new_models.deepseek_v41_dspark.checkpoint_test import values
from rtp_llm.models_py.new_models.deepseek_v41_dspark.language import (
    DeepSeekV41DSparkForCausalLM,
)
from rtp_llm.models_py.new_models.deepseek_v41_dspark.service_test import (
    config_and_load,
)
from rtp_llm.models_py.registry import get_model_class


class V41ServiceRouteTest(unittest.TestCase):
    def test_target_and_draft_are_distinct_production_owners(self):
        self.assertIs(get_model_class("deepseek_v41"), DeepSeekV41ForCausalLM)
        self.assertIs(
            get_model_class("deepseek_v41_dspark"), DeepSeekV41DSparkForCausalLM
        )
        self.assertTrue(DeepSeekV41ForCausalLM._captures_aux_hidden)
        self.assertFalse(DeepSeekV41DSparkForCausalLM._captures_aux_hidden)

    def test_registry_draft_load_does_not_publish_success_on_cpu(self):
        with tempfile.TemporaryDirectory() as root:
            save_file(values(False), root + "/model.safetensors")
            config, load = config_and_load(root, rank=0, commit=False)
            loader = NewModelLoader(config, load, model_path=root)
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                loader.load()
            self.assertIsNone(loader.loaded_method)


if __name__ == "__main__":
    unittest.main()
