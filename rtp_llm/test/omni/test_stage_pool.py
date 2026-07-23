import unittest

from rtp_llm.omni.config.stage_config import OmniStageConfig, StageExecutionType
from rtp_llm.omni.engine.stage_pool import OmniStagePool


class TestOmniStagePool(unittest.TestCase):
    def _make_stage(self, name="thinker"):
        return OmniStageConfig(
            name=name,
            execution_type=StageExecutionType.LLM_AR,
            model_cls="TestModel",
        )

    def test_create_pool(self):
        pool = OmniStagePool(stage_config=self._make_stage())
        self.assertEqual(pool.stage_name, "thinker")
        self.assertEqual(pool.num_replicas, 0)

    def test_add_replica(self):
        pool = OmniStagePool(stage_config=self._make_stage())
        pool.add_replica("engine_1")
        self.assertEqual(pool.num_replicas, 1)

    def test_get_replica_empty_raises(self):
        pool = OmniStagePool(stage_config=self._make_stage())
        with self.assertRaises(RuntimeError):
            pool.get_replica()

    def test_get_replica_round_robin(self):
        pool = OmniStagePool(stage_config=self._make_stage())
        pool.add_replica("a")
        pool.add_replica("b")
        self.assertEqual(pool.get_replica(), "a")
        self.assertEqual(pool.get_replica(), "b")
        self.assertEqual(pool.get_replica(), "a")


if __name__ == "__main__":
    unittest.main()
