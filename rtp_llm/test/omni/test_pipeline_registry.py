import unittest

from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)


class TestOmniPipelineRegistry(unittest.TestCase):
    def setUp(self):
        OmniPipelineRegistry._registry.clear()
        OmniPipelineRegistry._arch_registry.clear()

    def _make_pipeline(self, model_type="test_omni"):
        return OmniPipelineConfig(
            model_type=model_type,
            model_arch=f"TestOmniArch_{model_type}",
            stages=(
                OmniStageConfig(
                    stage_id=0,
                    model_stage="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestThinker",
                    input_sources=(),
                ),
            ),
        )

    def test_register_and_get(self):
        pipeline = self._make_pipeline()
        OmniPipelineRegistry.register(pipeline)
        result = OmniPipelineRegistry.get("test_omni")
        self.assertIs(result, pipeline)

    def test_get_returns_none_for_unknown(self):
        result = OmniPipelineRegistry.get("nonexistent")
        self.assertIsNone(result)

    def test_get_by_arch(self):
        pipeline = self._make_pipeline()
        OmniPipelineRegistry.register(pipeline)
        result = OmniPipelineRegistry.get_by_arch("TestOmniArch_test_omni")
        self.assertIs(result, pipeline)

    def test_get_by_arch_returns_none_for_unknown(self):
        result = OmniPipelineRegistry.get_by_arch("NonexistentArch")
        self.assertIsNone(result)

    def test_register_duplicate_raises(self):
        pipeline1 = self._make_pipeline()
        pipeline2 = self._make_pipeline()
        OmniPipelineRegistry.register(pipeline1)
        with self.assertRaises(ValueError):
            OmniPipelineRegistry.register(pipeline2)

    def test_register_duplicate_arch_raises(self):
        p1 = OmniPipelineConfig(
            model_type="type_a",
            model_arch="SharedArch",
            stages=(
                OmniStageConfig(
                    stage_id=0,
                    model_stage="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestThinker",
                    input_sources=(),
                ),
            ),
        )
        p2 = OmniPipelineConfig(
            model_type="type_b",
            model_arch="SharedArch",
            stages=(
                OmniStageConfig(
                    stage_id=0,
                    model_stage="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestThinker",
                    input_sources=(),
                ),
            ),
        )
        OmniPipelineRegistry.register(p1)
        with self.assertRaises(ValueError):
            OmniPipelineRegistry.register(p2)

    def test_list_all(self):
        p1 = self._make_pipeline("a")
        p2 = self._make_pipeline("b")
        OmniPipelineRegistry.register(p1)
        OmniPipelineRegistry.register(p2)
        all_pipelines = OmniPipelineRegistry.list_all()
        self.assertEqual(len(all_pipelines), 2)


if __name__ == "__main__":
    unittest.main()
