import unittest

from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry


class TestOmniEngineStageInstantiation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rtp_llm.omni.models.qwen2_5_omni  # noqa: F401
        import rtp_llm.omni.models.qwen2_5_omni.thinker  # noqa: F401
        import rtp_llm.omni.models.qwen2_5_omni.talker  # noqa: F401
        import rtp_llm.omni.models.qwen2_5_omni.token2wav  # noqa: F401

    def test_engine_resolves_stage_model_classes(self):
        from rtp_llm.model_factory_register import _model_factory

        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertIsNotNone(config)

        for stage in config.stages:
            self.assertIn(
                stage.model_cls,
                _model_factory,
                f"Stage model_cls '{stage.model_cls}' not registered",
            )

    def test_engine_from_pipeline_config_validates(self):
        from rtp_llm.omni.engine.omni_engine import OmniEngine

        config = OmniPipelineRegistry.get("qwen2_5_omni")
        engine = OmniEngine.from_pipeline_config(config)
        self.assertEqual(engine.num_stages, 3)

    def test_engine_final_output_types(self):
        from rtp_llm.omni.engine.omni_engine import OmniEngine

        config = OmniPipelineRegistry.get("qwen2_5_omni")
        engine = OmniEngine.from_pipeline_config(config)
        output_types = engine.get_final_output_types()
        self.assertIn("text", output_types)
        self.assertIn("audio", output_types)


class TestOmniImportChain(unittest.TestCase):
    def test_importing_omni_models_registers_pipeline(self):
        import rtp_llm.omni.models.qwen2_5_omni  # noqa: F401

        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertIsNotNone(config)

    def test_all_stage_models_registered(self):
        import rtp_llm.omni.models.qwen2_5_omni  # noqa: F401
        import rtp_llm.omni.models.qwen2_5_omni.thinker  # noqa: F401
        import rtp_llm.omni.models.qwen2_5_omni.talker  # noqa: F401
        import rtp_llm.omni.models.qwen2_5_omni.token2wav  # noqa: F401

        from rtp_llm.model_factory_register import _model_factory

        for model_name in [
            "qwen2_5_omni_thinker",
            "qwen2_5_omni_talker",
            "qwen2_5_omni_token2wav",
        ]:
            self.assertIn(
                model_name, _model_factory, f"{model_name} not registered"
            )


if __name__ == "__main__":
    unittest.main()
