import unittest

import torch

from rtp_llm.omni.engine.stage_connector import StageOutput


class TestThinker2TalkerProcessor(unittest.TestCase):
    def test_processor_exists(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            Thinker2TalkerProcessor,
        )

        self.assertTrue(callable(Thinker2TalkerProcessor))

    def test_process_extracts_embeddings(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            Thinker2TalkerProcessor,
        )

        proc = Thinker2TalkerProcessor()
        thinker_output = StageOutput(
            token_ids=[1, 2, 3],
            embeddings=torch.randn(1, 10, 3584),
            metadata={"text": "hello"},
        )
        talker_input = proc.process(thinker_output)
        self.assertIsNotNone(talker_input.embeddings)
        self.assertEqual(talker_input.embeddings.shape[-1], 3584)
        self.assertEqual(talker_input.metadata["source_text"], "hello")

    def test_process_preserves_source_token_ids(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            Thinker2TalkerProcessor,
        )

        proc = Thinker2TalkerProcessor()
        thinker_output = StageOutput(
            token_ids=[10, 20, 30],
            embeddings=torch.randn(1, 5, 3584),
            metadata={},
        )
        talker_input = proc.process(thinker_output)
        self.assertEqual(talker_input.metadata["source_token_ids"], [10, 20, 30])


class TestTalker2Code2WavProcessor(unittest.TestCase):
    def test_processor_exists(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            Talker2Code2WavProcessor,
        )

        self.assertTrue(callable(Talker2Code2WavProcessor))

    def test_process_converts_tokens(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            Talker2Code2WavProcessor,
        )

        proc = Talker2Code2WavProcessor()
        talker_output = StageOutput(
            token_ids=[10, 20, 30, 40, 50],
            metadata={"codec_tokens": True},
        )
        c2w_input = proc.process(talker_output)
        self.assertIsNotNone(c2w_input.token_ids)
        self.assertEqual(c2w_input.token_ids, [10, 20, 30, 40, 50])
        self.assertTrue(c2w_input.metadata["from_talker"])


if __name__ == "__main__":
    unittest.main()
