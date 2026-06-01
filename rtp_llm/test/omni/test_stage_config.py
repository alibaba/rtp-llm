import unittest

from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)


class TestStageExecutionType(unittest.TestCase):
    def test_enum_values_exist(self):
        self.assertEqual(StageExecutionType.LLM_AR.value, "llm_ar")
        self.assertEqual(StageExecutionType.LLM_GENERATION.value, "llm_generation")
        self.assertEqual(StageExecutionType.DIFFUSION.value, "diffusion")

    def test_enum_members_count(self):
        self.assertEqual(len(StageExecutionType), 3)


class TestOmniStageConfig(unittest.TestCase):
    def test_create_stage_config(self):
        stage = OmniStageConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Qwen2_5OmniThinker",
            input_sources=(),
        )
        self.assertEqual(stage.stage_id, 0)
        self.assertEqual(stage.model_stage, "thinker")
        self.assertEqual(stage.execution_type, StageExecutionType.LLM_AR)
        self.assertEqual(stage.model_cls, "Qwen2_5OmniThinker")
        self.assertEqual(stage.input_sources, ())
        self.assertFalse(stage.final_output)
        self.assertIsNone(stage.final_output_type)
        self.assertFalse(stage.requires_multimodal_data)
        self.assertIsNone(stage.engine_output_type)
        self.assertIsNone(stage.stage_processor)

    def test_stage_config_with_all_fields(self):
        stage = OmniStageConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Qwen2_5OmniThinker",
            input_sources=(),
            final_output=True,
            final_output_type="text",
            requires_multimodal_data=True,
            engine_output_type="latent",
            stage_processor=None,
        )
        self.assertTrue(stage.final_output)
        self.assertEqual(stage.final_output_type, "text")
        self.assertTrue(stage.requires_multimodal_data)
        self.assertEqual(stage.engine_output_type, "latent")

    def test_stage_config_is_frozen(self):
        stage = OmniStageConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Qwen2_5OmniThinker",
            input_sources=(),
        )
        with self.assertRaises(AttributeError):
            stage.stage_id = 1


class TestOmniPipelineConfig(unittest.TestCase):
    def _make_pipeline(self):
        return OmniPipelineConfig(
            model_type="qwen2_5_omni",
            model_arch="Qwen2_5OmniForConditionalGeneration",
            stages=(
                OmniStageConfig(
                    stage_id=0,
                    model_stage="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="Qwen2_5OmniThinker",
                    input_sources=(),
                    final_output=True,
                    final_output_type="text",
                    requires_multimodal_data=True,
                    engine_output_type="latent",
                ),
                OmniStageConfig(
                    stage_id=1,
                    model_stage="talker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="Qwen2_5OmniTalker",
                    input_sources=(0,),
                    engine_output_type="latent",
                    stage_processor="qwen2_5_omni.thinker2talker",
                ),
                OmniStageConfig(
                    stage_id=2,
                    model_stage="code2wav",
                    execution_type=StageExecutionType.LLM_GENERATION,
                    model_cls="Qwen2_5OmniToken2Wav",
                    input_sources=(1,),
                    final_output=True,
                    final_output_type="audio",
                    stage_processor="qwen2_5_omni.talker2code2wav",
                ),
            ),
        )

    def test_pipeline_config_fields(self):
        pipeline = self._make_pipeline()
        self.assertEqual(pipeline.model_type, "qwen2_5_omni")
        self.assertEqual(pipeline.model_arch, "Qwen2_5OmniForConditionalGeneration")
        self.assertEqual(len(pipeline.stages), 3)

    def test_pipeline_config_is_frozen(self):
        pipeline = self._make_pipeline()
        with self.assertRaises(AttributeError):
            pipeline.model_type = "other"

    def test_pipeline_stage_access(self):
        pipeline = self._make_pipeline()
        self.assertEqual(pipeline.stages[0].model_stage, "thinker")
        self.assertEqual(pipeline.stages[1].model_stage, "talker")
        self.assertEqual(pipeline.stages[2].model_stage, "code2wav")

    def test_pipeline_get_final_output_stages(self):
        pipeline = self._make_pipeline()
        final_stages = pipeline.get_final_output_stages()
        self.assertEqual(len(final_stages), 2)
        self.assertEqual(final_stages[0].stage_id, 0)
        self.assertEqual(final_stages[1].stage_id, 2)

    def test_pipeline_get_stage_by_id(self):
        pipeline = self._make_pipeline()
        stage = pipeline.get_stage(1)
        self.assertEqual(stage.model_stage, "talker")

    def test_pipeline_get_stage_by_id_not_found(self):
        pipeline = self._make_pipeline()
        with self.assertRaises(KeyError):
            pipeline.get_stage(99)


if __name__ == "__main__":
    unittest.main()
