import unittest

from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)
from rtp_llm.omni.engine.omni_engine import OmniEngine


class TestOmniEngine(unittest.TestCase):
    def _make_config(self):
        return OmniPipelineConfig(
            model_type="test_omni",
            model_arch="TestOmniArch",
            stages=(
                OmniStageConfig(
                    name="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestThinker",
                    next="talker",
                    stage_id=0,
                    final_output=True,
                    final_output_type="text",
                ),
                OmniStageConfig(
                    name="talker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestTalker",
                    terminal=True,
                    stage_id=1,
                    final_output=True,
                    final_output_type="audio",
                ),
            ),
        )

    def test_create_engine(self):
        engine = OmniEngine(pipeline_config=self._make_config())
        self.assertEqual(engine.pipeline_config.model_type, "test_omni")
        self.assertEqual(engine.num_stages, 2)

    def test_stage_pools_by_name(self):
        engine = OmniEngine(pipeline_config=self._make_config())
        self.assertEqual(len(engine.stage_pools), 2)
        self.assertIn("thinker", engine.stage_pools)
        self.assertIn("talker", engine.stage_pools)

    def test_orchestrator_initialized(self):
        engine = OmniEngine(pipeline_config=self._make_config())
        self.assertIsNotNone(engine.orchestrator)

    def test_connector_initialized(self):
        engine = OmniEngine(pipeline_config=self._make_config())
        self.assertIsNotNone(engine.connector)

    def test_get_final_output_types(self):
        engine = OmniEngine(pipeline_config=self._make_config())
        output_types = engine.get_final_output_types()
        self.assertEqual(output_types, {"text": "thinker", "audio": "talker"})

    def test_execution_order(self):
        engine = OmniEngine(pipeline_config=self._make_config())
        order = engine.orchestrator.get_execution_order()
        self.assertEqual(order, ["thinker", "talker"])


if __name__ == "__main__":
    unittest.main()
