import unittest

from rtp_llm.omni.engine.stage_connector import StageOutput
from rtp_llm.omni.engine.stage_processor_base import StageProcessorBase


class MockProcessor(StageProcessorBase):
    def process(self, source_output: StageOutput) -> StageOutput:
        new_ids = [x + 100 for x in source_output.token_ids] if source_output.token_ids else None
        return StageOutput(token_ids=new_ids, metadata={"transformed": True})


class TestStageProcessorBase(unittest.TestCase):
    def test_is_abstract(self):
        with self.assertRaises(TypeError):
            StageProcessorBase()

    def test_concrete_processor(self):
        proc = MockProcessor()
        source = StageOutput(token_ids=[1, 2, 3])
        result = proc.process(source)
        self.assertEqual(result.token_ids, [101, 102, 103])
        self.assertTrue(result.metadata["transformed"])


if __name__ == "__main__":
    unittest.main()
