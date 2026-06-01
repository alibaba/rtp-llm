import unittest

from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)
from rtp_llm.omni.engine.orchestrator import OmniOrchestrator, OmniRequestState
from rtp_llm.omni.engine.stage_connector import SharedMemoryConnector


class TestOmniRequestState(unittest.TestCase):
    def test_create_request_state(self):
        state = OmniRequestState(
            request_id="req_1",
            num_stages=3,
        )
        self.assertEqual(state.request_id, "req_1")
        self.assertEqual(state.current_stage, 0)
        self.assertFalse(state.is_complete)
        self.assertEqual(len(state.stage_status), 3)
        self.assertTrue(all(s == "pending" for s in state.stage_status))

    def test_advance_stage(self):
        state = OmniRequestState(request_id="req_1", num_stages=3)
        state.advance()
        self.assertEqual(state.current_stage, 1)
        self.assertEqual(state.stage_status[0], "completed")

    def test_complete_on_last_stage(self):
        state = OmniRequestState(request_id="req_1", num_stages=2)
        state.advance()
        state.advance()
        self.assertTrue(state.is_complete)


class TestOmniOrchestrator(unittest.TestCase):
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

    def test_create_orchestrator(self):
        config = self._make_pipeline_config()
        connector = SharedMemoryConnector()
        orchestrator = OmniOrchestrator(
            pipeline_config=config,
            connector=connector,
            stage_pools={},
        )
        self.assertIsNotNone(orchestrator)

    def test_submit_creates_request_state(self):
        config = self._make_pipeline_config()
        connector = SharedMemoryConnector()
        orchestrator = OmniOrchestrator(
            pipeline_config=config,
            connector=connector,
            stage_pools={},
        )
        state = orchestrator.submit("req_1")
        self.assertIsInstance(state, OmniRequestState)
        self.assertEqual(state.request_id, "req_1")

    def test_get_execution_order(self):
        config = self._make_pipeline_config()
        connector = SharedMemoryConnector()
        orchestrator = OmniOrchestrator(
            pipeline_config=config,
            connector=connector,
            stage_pools={},
        )
        order = orchestrator.get_execution_order()
        self.assertEqual(order, [0, 1])

    def test_cleanup_request(self):
        config = self._make_pipeline_config()
        connector = SharedMemoryConnector()
        orchestrator = OmniOrchestrator(
            pipeline_config=config,
            connector=connector,
            stage_pools={},
        )
        orchestrator.submit("req_1")
        orchestrator.cleanup("req_1")
        self.assertNotIn("req_1", orchestrator._requests)


if __name__ == "__main__":
    unittest.main()
