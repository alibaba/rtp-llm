import unittest
from unittest.mock import MagicMock, patch

from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)
from rtp_llm.omni.engine.omni_engine import OmniEngine


class TestOmniEngine(unittest.TestCase):
    def _make_config(self):
        return OmniPipelineConfig(
            model_type="test_omni",
            model_arch="TestOmniArch",
            stages=(
                OmniStageConfig(
                    name="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestThinker",
                    next="talker",
                    stage_id=0,
                    final_output=True,
                    final_output_type="text",
                ),
                OmniStageConfig(
                    name="talker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestTalker",
                    terminal=True,
                    stage_id=1,
                    final_output=True,
                    final_output_type="audio",
                ),
            ),
        )

    def test_create_engine(self):
        engine = OmniEngine(pipeline_config=self._make_config())
        self.assertEqual(engine.pipeline_config.model_type, "test_omni")
        self.assertEqual(engine.num_stages, 2)

    def test_stage_pools_by_name(self):
        engine = OmniEngine(pipeline_config=self._make_config())
        self.assertEqual(len(engine.stage_pools), 2)
        self.assertIn("thinker", engine.stage_pools)
        self.assertIn("talker", engine.stage_pools)

    def test_orchestrator_initialized(self):
        engine = OmniEngine(pipeline_config=self._make_config())
        self.assertIsNotNone(engine.orchestrator)

    def test_connector_initialized(self):
        engine = OmniEngine(pipeline_config=self._make_config())
        self.assertIsNotNone(engine.connector)

    def test_get_final_output_types(self):
        engine = OmniEngine(pipeline_config=self._make_config())
        output_types = engine.get_final_output_types()
        self.assertEqual(output_types, {"text": "thinker", "audio": "talker"})

    def test_execution_order(self):
        engine = OmniEngine(pipeline_config=self._make_config())
        order = engine.orchestrator.get_execution_order()
        self.assertEqual(order, ["thinker", "talker"])

    def _make_engine_config(self):
        engine_config = MagicMock()
        engine_config.parallelism_config = MagicMock()
        engine_config.hw_kernel_config = MagicMock()
        engine_config.kv_cache_config = MagicMock()
        engine_config.fmha_config = MagicMock()
        engine_config.moe_config = MagicMock()
        engine_config.load_config.load_method = "fake"
        engine_config.load_config.force_cpu_load_weights = False
        engine_config.runtime_config.max_generate_batch_size = 4
        engine_config.device_resource_config = MagicMock()
        engine_config.profiling_debug_logging_config.ft_alog_conf_path = ""
        return engine_config

    @patch("rtp_llm.omni.engine.omni_engine.ModelFactory")
    @patch("rtp_llm.omni.engine.omni_engine.create_engine")
    def test_only_primary_stage_gets_world_info(self, mock_create_engine, mock_model_factory):
        """Non-primary stages must get world_info=None to avoid RPC port conflicts."""
        mock_model_cls = MagicMock()
        mock_model_factory.get_model_cls.return_value = mock_model_cls

        mock_thinker_sub = MagicMock()
        mock_talker_sub = MagicMock()
        mock_create_engine.side_effect = [mock_thinker_sub, mock_talker_sub]

        engine = OmniEngine(pipeline_config=self._make_config())

        model_config = MagicMock()
        model_config.max_seq_len = 8192
        model_config.task_type = MagicMock()
        model_config.attn_config.tokens_per_block = 64
        model_config.attn_config.kernel_tokens_per_block = 64
        engine_config = self._make_engine_config()

        sentinel_world_info = object()
        engine.initialize_stages(
            model_config=model_config,
            engine_config=engine_config,
            world_info=sentinel_world_info,
        )

        self.assertEqual(mock_create_engine.call_count, 2)
        # Primary stage (thinker) gets the real world_info
        first_call_kwargs = mock_create_engine.call_args_list[0].kwargs
        self.assertIs(first_call_kwargs["world_info"], sentinel_world_info)
        # Secondary stage (talker) gets world_info=None to avoid port conflict
        second_call_kwargs = mock_create_engine.call_args_list[1].kwargs
        self.assertIsNone(second_call_kwargs["world_info"])

    @patch("rtp_llm.omni.engine.omni_engine.ModelFactory")
    @patch("rtp_llm.omni.engine.omni_engine.create_engine")
    def test_start_rollback_on_failure(self, mock_create_engine, mock_model_factory):
        """If a stage fails to start, previously started stages must be stopped."""
        mock_model_cls = MagicMock()
        mock_model_factory.get_model_cls.return_value = mock_model_cls

        mock_thinker_sub = MagicMock()
        mock_talker_sub = MagicMock()
        mock_talker_sub.start.side_effect = RuntimeError("port already in use")
        mock_create_engine.side_effect = [mock_thinker_sub, mock_talker_sub]

        engine = OmniEngine(pipeline_config=self._make_config())

        model_config = MagicMock()
        model_config.max_seq_len = 8192
        model_config.task_type = MagicMock()
        model_config.attn_config.tokens_per_block = 64
        model_config.attn_config.kernel_tokens_per_block = 64
        engine_config = self._make_engine_config()
        engine.initialize_stages(
            model_config=model_config,
            engine_config=engine_config,
            world_info=None,
        )

        with self.assertRaises(RuntimeError):
            engine.start()

        # thinker was started successfully, so it must be stopped during rollback
        mock_thinker_sub.stop.assert_called_once()
        # talker start failed, so stop should not be called on it
        mock_talker_sub.stop.assert_not_called()
        self.assertFalse(engine.started)


if __name__ == "__main__":
    unittest.main()
