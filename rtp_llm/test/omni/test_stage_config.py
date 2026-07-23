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
        self.assertEqual(StageExecutionType.CPU_EXECUTOR.value, "cpu_executor")

    def test_enum_members_count(self):
        self.assertEqual(len(StageExecutionType), 4)


class TestOmniStageConfig(unittest.TestCase):
    def test_create_named_stage(self):
        stage = OmniStageConfig(
            name="thinker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Qwen2_5OmniThinker",
        )
        self.assertEqual(stage.name, "thinker")
        self.assertIsNone(stage.gpu)
        self.assertEqual(stage.tp_size, 1)
        self.assertEqual(stage.process, "pipeline")
        self.assertIsNone(stage.next)
        self.assertEqual(stage.stream_to, ())
        self.assertEqual(stage.wait_for, ())
        self.assertFalse(stage.terminal)
        self.assertFalse(stage.can_accept_stream_before_payload)

    def test_stage_with_dag_fields(self):
        stage = OmniStageConfig(
            name="talker_ar",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Qwen3OmniTalker",
            gpu=1,
            tp_size=2,
            process="talker",
            next="code2wav",
            stream_to=("code2wav",),
            can_accept_stream_before_payload=True,
            terminal=False,
        )
        self.assertEqual(stage.gpu, 1)
        self.assertEqual(stage.tp_size, 2)
        self.assertEqual(stage.process, "talker")
        self.assertEqual(stage.next, "code2wav")
        self.assertEqual(stage.stream_to, ("code2wav",))

    def test_stage_with_fan_out(self):
        stage = OmniStageConfig(
            name="thinker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Thinker",
            next=("talker_ar", "decode"),
        )
        self.assertEqual(stage.next, ("talker_ar", "decode"))

    def test_stage_with_wait_for(self):
        stage = OmniStageConfig(
            name="mm_aggregate",
            execution_type=StageExecutionType.CPU_EXECUTOR,
            model_cls="Aggregator",
            wait_for=("preprocessing", "image_encoder", "audio_encoder"),
            merge_fn="some.module.merge_for_thinker",
        )
        self.assertEqual(len(stage.wait_for), 3)
        self.assertEqual(stage.merge_fn, "some.module.merge_for_thinker")

    def test_cpu_executor_type(self):
        stage = OmniStageConfig(
            name="preprocessing",
            execution_type=StageExecutionType.CPU_EXECUTOR,
            model_cls="Preprocessor",
        )
        self.assertEqual(stage.execution_type, StageExecutionType.CPU_EXECUTOR)
        self.assertIsNone(stage.gpu)

    def test_stage_is_frozen(self):
        stage = OmniStageConfig(
            name="thinker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Thinker",
        )
        with self.assertRaises(AttributeError):
            stage.name = "other"

    def test_legacy_fields_still_work(self):
        stage = OmniStageConfig(
            name="thinker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Thinker",
            stage_id=0,
            model_type="qwen2_5_omni_thinker",
            input_sources=(0,),
            final_output=True,
            final_output_type="text",
            requires_multimodal_data=True,
            engine_output_type="latent",
            stage_processor="some.proc",
        )
        self.assertEqual(stage.stage_id, 0)
        self.assertTrue(stage.final_output)


class TestOmniPipelineConfig(unittest.TestCase):
    def _make_pipeline(self):
        return OmniPipelineConfig(
            model_type="test_omni",
            model_arch="TestOmniArch",
            stages=(
                OmniStageConfig(
                    name="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestThinker",
                    next="talker",
                    terminal=False,
                ),
                OmniStageConfig(
                    name="talker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestTalker",
                    next="vocoder",
                    terminal=False,
                ),
                OmniStageConfig(
                    name="vocoder",
                    execution_type=StageExecutionType.LLM_GENERATION,
                    model_cls="TestVocoder",
                    terminal=True,
                ),
            ),
        )

    def test_pipeline_fields(self):
        pipeline = self._make_pipeline()
        self.assertEqual(pipeline.model_type, "test_omni")
        self.assertEqual(len(pipeline.stages), 3)

    def test_pipeline_is_frozen(self):
        pipeline = self._make_pipeline()
        with self.assertRaises(AttributeError):
            pipeline.model_type = "other"

    def test_get_stage_by_name(self):
        pipeline = self._make_pipeline()
        stage = pipeline.get_stage_by_name("talker")
        self.assertEqual(stage.model_cls, "TestTalker")

    def test_get_stage_by_name_not_found(self):
        pipeline = self._make_pipeline()
        with self.assertRaises(KeyError):
            pipeline.get_stage_by_name("nonexistent")

    def test_get_terminal_stages(self):
        pipeline = self._make_pipeline()
        terminals = pipeline.get_terminal_stages()
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0].name, "vocoder")

    def test_get_entry_stages(self):
        pipeline = self._make_pipeline()
        entries = pipeline.get_entry_stages()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].name, "thinker")

    def test_validate_valid_pipeline(self):
        pipeline = self._make_pipeline()
        pipeline.validate()

    def test_validate_duplicate_names(self):
        pipeline = OmniPipelineConfig(
            model_type="test",
            model_arch="Test",
            stages=(
                OmniStageConfig(name="a", execution_type=StageExecutionType.LLM_AR, model_cls="A", terminal=True),
                OmniStageConfig(name="a", execution_type=StageExecutionType.LLM_AR, model_cls="B", terminal=True),
            ),
        )
        with self.assertRaises(ValueError) as ctx:
            pipeline.validate()
        self.assertIn("Duplicate stage name", str(ctx.exception))

    def test_validate_dangling_next(self):
        pipeline = OmniPipelineConfig(
            model_type="test",
            model_arch="Test",
            stages=(
                OmniStageConfig(name="a", execution_type=StageExecutionType.LLM_AR, model_cls="A", next="nonexistent"),
            ),
        )
        with self.assertRaises(ValueError) as ctx:
            pipeline.validate()
        self.assertIn("nonexistent", str(ctx.exception))

    def test_validate_self_reference(self):
        pipeline = OmniPipelineConfig(
            model_type="test",
            model_arch="Test",
            stages=(
                OmniStageConfig(name="a", execution_type=StageExecutionType.LLM_AR, model_cls="A", next="a", terminal=True),
            ),
        )
        with self.assertRaises(ValueError) as ctx:
            pipeline.validate()
        self.assertIn("self-reference", str(ctx.exception))

    def test_validate_no_terminal(self):
        pipeline = OmniPipelineConfig(
            model_type="test",
            model_arch="Test",
            stages=(
                OmniStageConfig(name="a", execution_type=StageExecutionType.LLM_AR, model_cls="A"),
            ),
        )
        with self.assertRaises(ValueError) as ctx:
            pipeline.validate()
        self.assertIn("terminal", str(ctx.exception))

    def test_validate_no_entry_point(self):
        pipeline = OmniPipelineConfig(
            model_type="test",
            model_arch="Test",
            stages=(
                OmniStageConfig(name="a", execution_type=StageExecutionType.LLM_AR, model_cls="A", next="b"),
                OmniStageConfig(name="b", execution_type=StageExecutionType.LLM_AR, model_cls="B", next="a", terminal=True),
            ),
        )
        with self.assertRaises(ValueError) as ctx:
            pipeline.validate()
        self.assertIn("entry point", str(ctx.exception))

    def test_validate_dangling_stream_to(self):
        pipeline = OmniPipelineConfig(
            model_type="test",
            model_arch="Test",
            stages=(
                OmniStageConfig(
                    name="a",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="A",
                    stream_to=("ghost",),
                    terminal=True,
                ),
            ),
        )
        with self.assertRaises(ValueError) as ctx:
            pipeline.validate()
        self.assertIn("ghost", str(ctx.exception))

    def test_validate_dangling_wait_for(self):
        pipeline = OmniPipelineConfig(
            model_type="test",
            model_arch="Test",
            stages=(
                OmniStageConfig(
                    name="a",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="A",
                    wait_for=("ghost",),
                    terminal=True,
                ),
            ),
        )
        with self.assertRaises(ValueError) as ctx:
            pipeline.validate()
        self.assertIn("ghost", str(ctx.exception))

    # Legacy compat: get_stage by int id still works
    def test_get_stage_by_id_legacy(self):
        pipeline = OmniPipelineConfig(
            model_type="test",
            model_arch="Test",
            stages=(
                OmniStageConfig(
                    name="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="A",
                    stage_id=0,
                    terminal=True,
                ),
            ),
        )
        stage = pipeline.get_stage(0)
        self.assertEqual(stage.name, "thinker")

    def test_get_final_output_stages_legacy(self):
        pipeline = OmniPipelineConfig(
            model_type="test",
            model_arch="Test",
            stages=(
                OmniStageConfig(
                    name="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="A",
                    final_output=True,
                    final_output_type="text",
                    terminal=True,
                ),
            ),
        )
        finals = pipeline.get_final_output_stages()
        self.assertEqual(len(finals), 1)


if __name__ == "__main__":
    unittest.main()
