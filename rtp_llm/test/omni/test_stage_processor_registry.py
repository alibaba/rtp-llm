import unittest

from rtp_llm.omni.engine.stage_connector import StageOutput
from rtp_llm.omni.engine.stage_processor_base import StageProcessorBase
from rtp_llm.omni.engine.stage_processor_registry import StageProcessorRegistry


class MockProcessor(StageProcessorBase):
    def process(self, source_output: StageOutput) -> StageOutput:
        return StageOutput(metadata={"processed": True})


class TestStageProcessorRegistry(unittest.TestCase):
    def setUp(self):
        self._saved_registry = dict(StageProcessorRegistry._registry)
        StageProcessorRegistry._registry.clear()

    def tearDown(self):
        StageProcessorRegistry._registry.clear()
        StageProcessorRegistry._registry.update(self._saved_registry)

    def test_register_and_get(self):
        StageProcessorRegistry.register("test.mock", MockProcessor)
        cls = StageProcessorRegistry.get("test.mock")
        self.assertIs(cls, MockProcessor)

    def test_get_returns_none_for_unknown(self):
        result = StageProcessorRegistry.get("nonexistent")
        self.assertIsNone(result)

    def test_register_duplicate_raises(self):
        StageProcessorRegistry.register("test.mock", MockProcessor)
        with self.assertRaises(ValueError):
            StageProcessorRegistry.register("test.mock", MockProcessor)

    def test_create_instance(self):
        StageProcessorRegistry.register("test.mock", MockProcessor)
        instance = StageProcessorRegistry.create("test.mock")
        self.assertIsInstance(instance, MockProcessor)
        result = instance.process(StageOutput())
        self.assertEqual(result.metadata["processed"], True)

    def test_create_unknown_raises(self):
        with self.assertRaises(KeyError):
            StageProcessorRegistry.create("nonexistent")


if __name__ == "__main__":
    unittest.main()
