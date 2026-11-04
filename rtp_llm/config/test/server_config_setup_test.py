import json
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import (
    set_parallelism_config,
    setup_and_configure_server,
    setup_default_args,
)
from rtp_llm.ops import RoleType
from rtp_llm.server.server_args.server_args import setup_args


class GenerateConfigTest(TestCase):

    # EnvArgumentParser in setup_args() reads these env vars (START_PORT, TP_SIZE, etc.)
    # and binds them to py_env_configs; server_port = start_port + rank_id * worker_info_port_num (rank_id=0 here).
    @patch.dict(
        "os.environ",
        {
            "TP_SIZE": "4",
            "PP_SIZE": "1",
            "WORLD_SIZE": "4",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "CONCURRENCY_LIMIT": "32",
            "START_PORT": "20000",
            "MODEL_TYPE": "fake_model",
            "USE_ALL_GATHER": "0",
        },
        clear=True,
    )
    def test_simple(self):
        from rtp_llm.config.server_config_setup import (
            fetch_model_files_to_local,
            setup_default_args,
        )

        py_env_configs: PyEnvConfigs = setup_args()
        setup_and_configure_server(py_env_configs)
        pc = py_env_configs.parallelism_config
        self.assertEqual(pc.tp_size, 4)
        self.assertEqual(pc.world_size, 4)
        self.assertEqual(pc.local_world_size, 2)
        self.assertEqual(py_env_configs.server_config.server_port, 20000)

        self.assertEqual(py_env_configs.moe_config.use_deepep_moe, True)
        self.assertEqual(py_env_configs.moe_config.use_deepep_low_latency, False)
        self.assertEqual(py_env_configs.moe_config.use_deepep_internode, True)
        self.assertEqual(py_env_configs.moe_config.ll_num_max_token, 32)

    @patch.dict(
        "os.environ",
        {
            "TP_SIZE": "2",
            "PP_SIZE": "1",
            "WORLD_SIZE": "2",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "CONCURRENCY_LIMIT": "32",
            "START_PORT": "20000",
            "MODEL_TYPE": "fake_model",
            "SP_TYPE": "eagle",
            "SP_MODEL_TYPE": "qwen_2-mtp",
            "GEN_NUM_PER_CIRCLE": "4",
            "ROLE_TYPE": "DECODE",
            "USE_ALL_GATHER": "0",
        },
        clear=True,
    )
    def test_sp_deepep_low_latency(self):
        py_env_configs: PyEnvConfigs = setup_args()
        setup_and_configure_server(py_env_configs)

        self.assertEqual(py_env_configs.moe_config.use_deepep_moe, True)
        self.assertEqual(py_env_configs.moe_config.use_deepep_low_latency, True)
        self.assertEqual(py_env_configs.moe_config.use_deepep_internode, False)
        self.assertEqual(py_env_configs.moe_config.ll_num_max_token, 160)

    @patch.dict(
        "os.environ",
        {
            "TP_SIZE": "4",
            "PP_SIZE": "1",
            "WORLD_SIZE": "4",
            "WORLD_RANK": "4",
            "LOCAL_WORLD_SIZE": "2",
            "CONCURRENCY_LIMIT": "32",
            "START_PORT": "20000",
            "MODEL_TYPE": "fake_model",
            "USE_ALL_GATHER": "0",
        },
        clear=True,
    )
    def test_world_rank_consistent_with_env_after_setup_args(self):
        """After setup_args(), set_parallelism_config(parallelism_config) keeps world_rank from env and it is not None."""
        py_env_configs: PyEnvConfigs = setup_args()
        set_parallelism_config(py_env_configs.parallelism_config)
        pc = py_env_configs.parallelism_config
        self.assertIsNotNone(pc.world_rank)
        self.assertEqual(pc.world_rank, 4)

    @patch.dict(
        "os.environ",
        {
            "TP_SIZE": "4",
            "DP_SIZE": "2",
            "PP_SIZE": "1",
            "WORLD_SIZE": "8",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "CONCURRENCY_LIMIT": "32",
            "START_PORT": "20000",
            "MODEL_TYPE": "fake_model",
            "USE_ALL_GATHER": "0",
        },
        clear=True,
    )
    def test_set_parallelism_config_after_setup_and_configure_server_world_rank_not_none(
        self,
    ):
        """After setup_and_configure_server(), set_parallelism_config(..., world_rank=5) assigns world_rank and derived ranks correctly."""
        py_env_configs: PyEnvConfigs = setup_args()
        setup_and_configure_server(py_env_configs)
        set_parallelism_config(py_env_configs.parallelism_config, world_rank=5)
        pc = py_env_configs.parallelism_config
        self.assertEqual(pc.world_rank, 5)
        self.assertEqual(pc.local_rank, 5 % pc.local_world_size)
        self.assertEqual(pc.tp_rank, 5 % pc.tp_size)
        self.assertEqual(pc.dp_rank, 5 // pc.tp_size)
        self.assertEqual(pc.ep_rank, 5 % pc.ep_size)
        self.assertEqual(pc.ffn_tp_rank, pc.tp_rank % pc.ffn_tp_size)
        self.assertEqual(pc.tp_rank, 1)
        self.assertEqual(pc.dp_rank, 1)
        self.assertEqual(pc.local_rank, 1)
        self.assertEqual(pc.ep_size, 8)
        self.assertEqual(pc.ep_rank, 5)
        self.assertEqual(pc.ffn_tp_rank, 1)


class Hy4DecodeCacheConfigTest(TestCase):
    @patch("torch.cuda.is_available", return_value=False)
    def test_cache_sharding_by_model_and_role(self, _cuda_available):
        for model_type in ("hy_v4", "hy_v4_mtp", "deepseek_v4"):
            for role in (RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION):
                for sharded in (False, True):
                    with self.subTest(model=model_type, role=role, sharded=sharded):
                        config = PyEnvConfigs()
                        config.model_args.model_type = model_type
                        config.role_config.role_type = role
                        config.parallelism_config.tp_size = 8
                        config.parallelism_config.world_size = 8
                        config.parallelism_config.local_world_size = 8
                        config.prefill_cp_config.kv_cache_sharded = sharded

                        setup_default_args(config)

                        expected = sharded and not (
                            model_type in ("hy_v4", "hy_v4_mtp")
                            and role == RoleType.DECODE
                        )
                        self.assertEqual(
                            config.prefill_cp_config.kv_cache_sharded, expected
                        )
                        self.assertEqual(
                            config.parallelism_config.prefill_cp_config.kv_cache_sharded,
                            expected,
                        )
                        set_parallelism_config(
                            config.parallelism_config,
                            world_rank=7,
                            py_prefill_cp_config=config.prefill_cp_config,
                        )
                        self.assertEqual(
                            config.parallelism_config.prefill_cp_config.kv_cache_sharded,
                            expected,
                        )

    @patch("torch.cuda.is_available", return_value=False)
    def test_decode_sharding_disabled_for_inferred_hy4(self, _cuda_available):
        with tempfile.TemporaryDirectory() as checkpoint:
            (Path(checkpoint) / "config.json").write_text(
                json.dumps({"architectures": ["HYV4ForCausalLM"]}), encoding="utf-8"
            )
            config = PyEnvConfigs()
            config.model_args.ckpt_path = checkpoint
            config.role_config.role_type = RoleType.DECODE
            config.prefill_cp_config.kv_cache_sharded = True

            setup_default_args(config)

            self.assertEqual(config.model_args.model_type, "hy_v4")
            self.assertFalse(config.prefill_cp_config.kv_cache_sharded)
            self.assertFalse(
                config.parallelism_config.prefill_cp_config.kv_cache_sharded
            )


if __name__ == "__main__":
    unittest.main()
