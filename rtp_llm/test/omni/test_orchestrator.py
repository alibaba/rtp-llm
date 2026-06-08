import unittest

from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)
from rtp_llm.omni.engine.orchestrator import OmniOrchestrator, OmniRequestState
from rtp_llm.omni.engine.stage_connector import SharedMemoryConnector
from rtp_llm.omni.engine.stage_pool import OmniStagePool


def _make_config():
    return OmniPipelineConfig(
        model_type="test",
        model_arch="Test",
        stages=(
            OmniStageConfig(
                name="thinker",
                execution_type=StageExecutionType.LLM_AR,
                model_cls="T",
                next=("talker", "decode"),
            ),
            OmniStageConfig(
                name="talker",
                execution_type=StageExecutionType.LLM_AR,
                model_cls="T",
                next="vocoder",
            ),
            OmniStageConfig(
                name="decode",
                execution_type=StageExecutionType.LLM_AR,
                model_cls="T",
                terminal=True,
            ),
            OmniStageConfig(
                name="vocoder",
                execution_type=StageExecutionType.LLM_GENERATION,
                model_cls="T",
                terminal=True,
            ),
        ),
    )


class TestOmniRequestState(unittest.TestCase):
    def test_create(self):
        state = OmniRequestState("r1", ["thinker", "talker", "vocoder"])
        self.assertEqual(state.request_id, "r1")
        self.assertFalse(state.is_complete)

    def test_mark_stage_complete(self):
        state = OmniRequestState("r1", ["a", "b"])
        state.mark_complete("a")
        self.assertTrue(state.is_stage_complete("a"))
        self.assertFalse(state.is_stage_complete("b"))

    def test_all_complete(self):
        state = OmniRequestState("r1", ["a", "b"])
        state.mark_complete("a")
        state.mark_complete("b")
        self.assertTrue(state.is_complete)


class TestOmniOrchestrator(unittest.TestCase):
    def _make_orchestrator(self):
        config = _make_config()
        connector = SharedMemoryConnector()
        pools = {s.name: OmniStagePool(s) for s in config.stages}
        return OmniOrchestrator(config, connector, pools)

    def test_create(self):
        orch = self._make_orchestrator()
        self.assertIsNotNone(orch)

    def test_get_execution_order(self):
        orch = self._make_orchestrator()
        order = orch.get_execution_order()
        self.assertIn("thinker", order)
        self.assertIn("talker", order)
        thinker_idx = order.index("thinker")
        talker_idx = order.index("talker")
        vocoder_idx = order.index("vocoder")
        self.assertLess(thinker_idx, talker_idx)
        self.assertLess(talker_idx, vocoder_idx)

    def test_submit(self):
        orch = self._make_orchestrator()
        state = orch.submit("req_1")
        self.assertFalse(state.is_complete)

    def test_submit_duplicate_raises(self):
        orch = self._make_orchestrator()
        orch.submit("req_1")
        with self.assertRaises(ValueError):
            orch.submit("req_1")

    def test_cleanup(self):
        orch = self._make_orchestrator()
        orch.submit("req_1")
        orch.cleanup("req_1")
        self.assertIsNone(orch.get_request_state("req_1"))

    def test_get_downstream(self):
        orch = self._make_orchestrator()
        downstream = orch.get_downstream("thinker")
        self.assertEqual(set(downstream), {"talker", "decode"})

    def test_get_downstream_terminal(self):
        orch = self._make_orchestrator()
        downstream = orch.get_downstream("decode")
        self.assertEqual(downstream, [])


if __name__ == "__main__":
    unittest.main()
