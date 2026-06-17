import unittest

import torch

from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.config.stage_config import StageExecutionType
from rtp_llm.omni.engine.stage_connector import StageOutput


class TestQwen25OmniPipeline(unittest.TestCase):
    def setUp(self):
        from rtp_llm.omni.models.qwen2_5_omni.pipeline import QWEN2_5_OMNI_PIPELINE
        self.config = QWEN2_5_OMNI_PIPELINE
        if OmniPipelineRegistry.get("qwen2_5_omni") is None:
            OmniPipelineRegistry.register(self.config)

    def test_pipeline_registered(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertIsNotNone(config)
        self.assertEqual(config.model_type, "qwen2_5_omni")

    def test_pipeline_has_three_stages(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertEqual(len(config.stages), 3)

    def test_thinker_stage(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        thinker = config.get_stage_by_name("thinker")
        self.assertEqual(thinker.model_cls, "Qwen2_5OmniThinker")
        self.assertEqual(thinker.next, "talker")
        self.assertFalse(thinker.terminal)

    def test_talker_stage(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        talker = config.get_stage_by_name("talker")
        self.assertEqual(talker.next, "token2wav")
        self.assertFalse(talker.terminal)

    def test_token2wav_stage(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        t2w = config.get_stage_by_name("token2wav")
        self.assertTrue(t2w.terminal)

    def test_pipeline_validates(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        config.validate()

    def test_legacy_stage_id_access(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        stage = config.get_stage(0)
        self.assertEqual(stage.name, "thinker")

    def test_legacy_final_output_stages(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        finals = config.get_final_output_stages()
        self.assertEqual(len(finals), 2)

    def test_entry_stages(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        entries = config.get_entry_stages()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].name, "thinker")

    def test_terminal_stages(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        terminals = config.get_terminal_stages()
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0].name, "token2wav")


class TestThinker2TalkerFunction(unittest.TestCase):
    def test_process_extracts_embeddings(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import thinker2talker

        embeddings = torch.randn(10, 3584)
        source = StageOutput(
            token_ids=[1, 2, 3],
            embeddings=embeddings,
            metadata={"text": "Hello"},
        )
        result = thinker2talker(source)
        self.assertIsNotNone(result.embeddings)
        self.assertEqual(result.embeddings.shape, embeddings.shape)
        self.assertIn("source_token_ids", result.metadata)

    def test_process_without_embeddings(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import thinker2talker

        source = StageOutput(token_ids=[1, 2, 3], metadata={"text": "Hello"})
        result = thinker2talker(source)
        self.assertIsNone(result.embeddings)
        self.assertEqual(result.metadata["source_token_ids"], [1, 2, 3])


class TestTalker2Code2WavFunction(unittest.TestCase):
    def test_process_filters_codec_end(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import talker2code2wav

        source = StageOutput(
            token_ids=[100, 200, 300, 8294],
            metadata={"stage": "talker"},
        )
        result = talker2code2wav(source)
        self.assertEqual(result.token_ids, [100, 200, 300])
        self.assertNotIn(8294, result.token_ids)

    def test_process_filters_all_codec_special_tokens(self):
        """All codec special tokens (pad=8292, bos=8293, eos=8294, etc.) must be filtered."""
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
            talker2code2wav,
            CODEC_SPECIAL_TOKEN_MIN,
        )

        self.assertEqual(CODEC_SPECIAL_TOKEN_MIN, 8292)

        source = StageOutput(
            token_ids=[100, 8292, 200, 8293, 300, 8294, 400, 8296, 500],
            metadata={"stage": "talker"},
        )
        result = talker2code2wav(source)
        self.assertEqual(result.token_ids, [100, 200, 300, 400, 500])
        self.assertEqual(result.metadata["codec_token_count"], 5)
        for special in (8292, 8293, 8294, 8296):
            self.assertNotIn(special, result.token_ids)

    def test_process_empty_token_ids(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import talker2code2wav

        source = StageOutput(token_ids=None, metadata={})
        result = talker2code2wav(source)
        self.assertEqual(result.token_ids, [])
        self.assertEqual(result.metadata["codec_token_count"], 0)


if __name__ == "__main__":
    unittest.main()
