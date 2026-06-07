import unittest
from unittest.mock import MagicMock

from rtp_llm.omni.config.stage_config import OmniStageConfig, StageExecutionType
from rtp_llm.omni.engine.stage_pool import OmniStagePool


class TestOmniStagePool(unittest.TestCase):
    def _make_stage_config(self, stage_id=0):
        return OmniStageConfig(
            stage_id=stage_id,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="TestThinker",
            input_sources=(),
        )

    def test_create_pool(self):
        config = self._make_stage_config()
        pool = OmniStagePool(stage_config=config)
        self.assertEqual(pool.stage_config, config)
        self.assertEqual(pool.stage_id, 0)
        self.assertEqual(pool.num_replicas, 0)

    def test_add_replica(self):
        config = self._make_stage_config()
        pool = OmniStagePool(stage_config=config)
        mock_pipeline = MagicMock()
        pool.add_replica(mock_pipeline)
        self.assertEqual(pool.num_replicas, 1)

    def test_get_replica_round_robin(self):
        config = self._make_stage_config()
        pool = OmniStagePool(stage_config=config)
        p1 = MagicMock(name="pipeline_1")
        p2 = MagicMock(name="pipeline_2")
        pool.add_replica(p1)
        pool.add_replica(p2)
        self.assertIs(pool.get_replica(), p1)
        self.assertIs(pool.get_replica(), p2)
        self.assertIs(pool.get_replica(), p1)

    def test_get_replica_empty_raises(self):
        config = self._make_stage_config()
        pool = OmniStagePool(stage_config=config)
        with self.assertRaises(RuntimeError):
            pool.get_replica()


if __name__ == "__main__":
    unittest.main()
