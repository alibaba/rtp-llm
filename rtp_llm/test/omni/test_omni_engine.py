import unittest

from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)
from rtp_llm.omni.engine.omni_engine import OmniEngine


class TestOmniEngine(unittest.TestCase):
    def _make_pipeline_config(self):
        return OmniPipelineConfig(
            model_type="test_omni",
            model_arch="TestOmniArch",
            stages=(
                OmniStageConfig(
                    stage_id=0,
                    model_stage="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestThinker",
                    input_sources=(),
                    final_output=True,
                    final_output_type="text",
                ),
                OmniStageConfig(
                    stage_id=1,
                    model_stage="talker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestTalker",
                    input_sources=(0,),
                    final_output=True,
                    final_output_type="audio",
                ),
            ),
        )

    def test_create_omni_engine(self):
        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        self.assertEqual(engine.pipeline_config.model_type, "test_omni")
        self.assertEqual(engine.num_stages, 2)

    def test_get_final_output_types(self):
        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        output_types = engine.get_final_output_types()
        self.assertEqual(output_types, {"text": 0, "audio": 1})

    def test_stage_pools_initialized(self):
        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        self.assertEqual(len(engine.stage_pools), 2)
        self.assertIn(0, engine.stage_pools)
        self.assertIn(1, engine.stage_pools)

    def test_orchestrator_initialized(self):
        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        self.assertIsNotNone(engine.orchestrator)

    def test_connector_initialized(self):
        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        self.assertIsNotNone(engine.connector)


if __name__ == "__main__":
    unittest.main()
