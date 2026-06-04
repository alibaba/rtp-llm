import unittest

from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.config.stage_config import StageExecutionType


class TestQwen25OmniPipelineConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rtp_llm.omni.models.qwen2_5_omni  # noqa: F401

    def test_pipeline_registered(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertIsNotNone(config)
        self.assertEqual(config.model_type, "qwen2_5_omni")
        self.assertEqual(config.model_arch, "Qwen2_5OmniModel")

    def test_pipeline_has_three_stages(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertEqual(len(config.stages), 3)

    def test_thinker_stage(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        thinker = config.get_stage(0)
        self.assertEqual(thinker.model_stage, "thinker")
        self.assertEqual(thinker.execution_type, StageExecutionType.LLM_AR)
        self.assertEqual(thinker.model_cls, "qwen2_5_omni_thinker")
        self.assertTrue(thinker.final_output)
        self.assertEqual(thinker.final_output_type, "text")
        self.assertTrue(thinker.requires_multimodal_data)

    def test_talker_stage(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        talker = config.get_stage(1)
        self.assertEqual(talker.model_stage, "talker")
        self.assertEqual(talker.execution_type, StageExecutionType.LLM_AR)
        self.assertEqual(talker.model_cls, "qwen2_5_omni_talker")
        self.assertEqual(talker.input_sources, (0,))

    def test_code2wav_stage(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        c2w = config.get_stage(2)
        self.assertEqual(c2w.model_stage, "code2wav")
        self.assertEqual(c2w.execution_type, StageExecutionType.LLM_GENERATION)
        self.assertEqual(c2w.model_cls, "qwen2_5_omni_token2wav")
        self.assertTrue(c2w.final_output)
        self.assertEqual(c2w.final_output_type, "audio")

    def test_final_output_stages(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        finals = config.get_final_output_stages()
        types = {s.final_output_type for s in finals}
        self.assertEqual(types, {"text", "audio"})


if __name__ == "__main__":
    unittest.main()
