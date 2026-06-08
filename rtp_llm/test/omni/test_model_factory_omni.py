import unittest

from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)
from rtp_llm.omni.engine.omni_engine import OmniEngine


class TestModelFactoryOmniDetection(unittest.TestCase):
    def setUp(self):
        OmniPipelineRegistry._registry.clear()
        OmniPipelineRegistry._arch_registry.clear()

    def test_registry_lookup_for_omni_model(self):
        pipeline = OmniPipelineConfig(
            model_type="test_omni_model",
            model_arch="TestOmniArch",
            stages=(
                OmniStageConfig(
                    name="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestThinker",
                    terminal=True,
                ),
            ),
        )
        OmniPipelineRegistry.register(pipeline)

        result = OmniPipelineRegistry.get("test_omni_model")
        self.assertIsNotNone(result)
        self.assertEqual(result.model_type, "test_omni_model")

    def test_registry_returns_none_for_regular_model(self):
        result = OmniPipelineRegistry.get("qwen_2")
        self.assertIsNone(result)

    def test_omni_engine_from_pipeline_config(self):
        pipeline = OmniPipelineConfig(
            model_type="test_omni_model",
            model_arch="TestOmniArch",
            stages=(
                OmniStageConfig(
                    name="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestThinker",
                    terminal=True,
                ),
            ),
        )
        engine = OmniEngine.from_pipeline_config(pipeline)
        self.assertIsInstance(engine, OmniEngine)
        self.assertEqual(engine.num_stages, 1)


if __name__ == "__main__":
    unittest.main()
