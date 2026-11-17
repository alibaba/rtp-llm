"""Check startup parsing, spawn serialization, and an incompatible configuration."""

import os
import pickle
import unittest
from unittest.mock import patch

from rtp_llm.config.engine_config import EngineConfig, configure_batch_invariant_model
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import CPRotateMethod, DataType, TaskType
from rtp_llm.server.server_args.server_args import setup_args


class BatchInvariantConfigTest(unittest.TestCase):
    def test_switch_defaults_off_and_survives_spawn(self):
        with patch.dict(os.environ):
            os.environ.pop("RTP_BATCH_INVARIANT", None)
            self.assertFalse(setup_args([]).py_hw_kernel_config.batch_invariant)
            os.environ["RTP_BATCH_INVARIANT"] = "1"
            parsed = setup_args([])
            os.environ["RTP_BATCH_INVARIANT"] = "0"
            engine = EngineConfig.create(pickle.loads(pickle.dumps(parsed)))
        self.assertTrue(engine.hw_kernel_config.batch_invariant)
        self.assertTrue(engine.collective_config.batch_invariant)

    def test_invalid_switch(self):
        with patch.dict(os.environ, {"RTP_BATCH_INVARIANT": "true"}):
            with self.assertRaisesRegex(
                ValueError, "RTP_BATCH_INVARIANT must be 0 or 1"
            ):
                setup_args([])

    def test_model_conflicts(self):
        env = {
            "RTP_BATCH_INVARIANT": "1",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
            "TP_SIZE": "1",
            "DP_SIZE": "1",
            "EP_SIZE": "1",
            "PP_SIZE": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            engine = EngineConfig.create(setup_args([]))
        engine.kv_cache_config.reuse_cache = True
        model = ModelConfig()
        model.model_type = "qwen35_moe"
        model.task_type = TaskType.LANGUAGE_MODEL
        model.num_layers = 40
        model.hidden_size = 2048
        model.expert_num = 256
        model.moe_k = 8
        model.moe_inter_size = 512
        model.data_type = "bf16"
        model.attn_config.head_num = 16
        model.attn_config.kv_head_num = 2
        model.attn_config.size_per_head = 256
        model.attn_config.tokens_per_block = 64
        model.attn_config.kernel_tokens_per_block = 64
        model.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP32
        with (
            patch("rtp_llm.config.engine_config.is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 3)),
            patch("rtp_llm.config.engine_config.get_config_from_path") as read_config,
        ):
            read_config.return_value = {"text_config": {"dtype": "bfloat16"}}
            with self.assertRaisesRegex(ValueError, "prefix reuse"):
                configure_batch_invariant_model(engine, model)
            engine.kv_cache_config.reuse_cache = False
            read_config.return_value["text_config"]["dtype"] = "float16"
            with self.assertRaisesRegex(ValueError, "requires a BF16 checkpoint"):
                configure_batch_invariant_model(engine, model)
            read_config.return_value["text_config"]["dtype"] = "bfloat16"
            model.mm_model_config.is_multimodal = True
            configure_batch_invariant_model(engine, model)
            model.num_layers = 60
            model.hidden_size = 4096
            model.expert_num = 512
            model.moe_k = 10
            model.moe_inter_size = 1024
            model.attn_config.head_num = 32
            with self.assertRaisesRegex(
                ValueError, "Qwen3.5-397B-A17B requires .*TP_SIZE=4"
            ):
                configure_batch_invariant_model(engine, model)
            parallel = engine.parallelism_config
            parallel.world_size = parallel.local_world_size = parallel.tp_size = 4
            parallel.ffn_tp_size = 4
            configure_batch_invariant_model(engine, model)
            parallel.ep_size = 4
            with self.assertRaisesRegex(ValueError, "DP_SIZE=EP_SIZE=PP_SIZE=1"):
                configure_batch_invariant_model(engine, model)
            parallel.ep_size = 1
            parallel.prefill_cp_config.method = CPRotateMethod.PREFILL_CP
            with self.assertRaisesRegex(ValueError, "CP disabled"):
                configure_batch_invariant_model(engine, model)
        self.assertFalse(model.mm_model_config.is_multimodal)
        self.assertFalse(model.enable_fp32_lm_head)


if __name__ == "__main__":
    unittest.main()
