import pickle
import unittest
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import (
    set_parallelism_config,
    setup_and_configure_server,
)
from rtp_llm.server.server_args.server_args import setup_args


class GenerateConfigTest(TestCase):

    def test_dcp_configuration_reaches_engine_and_spawn(self):
        from rtp_llm.config.engine_config import EngineConfig
        from rtp_llm.config.server_config_setup import setup_default_args
        from rtp_llm.model_factory import ModelFactory
        from rtp_llm.ops import RoleType

        for role, tp, dp, source, enabled, local in (
            ("DECODE", 8, 1, 1, True, 8),
            ("DECODE", 8, 1, 8, True, 8),
            ("DECODE", 16, 1, 16, True, 16),
            ("DECODE", 8, 2, 16, True, 8),
            ("DECODE", 8, 1, 8, False, 1),
            ("PREFILL", 8, 1, 8, False, 8),
        ):
            with self.subTest(role=role, tp=tp, dp=dp, source=source, enabled=enabled):
                env = {
                    "MODEL_TYPE": "kimi_k3",
                    "ROLE_TYPE": role,
                    "TP_SIZE": str(tp),
                    "DP_SIZE": str(dp),
                    "WORLD_SIZE": str(tp * dp),
                    "WORLD_RANK": str(tp * dp - 1),
                    "LOCAL_WORLD_SIZE": str(tp),
                    "PREFILL_CP_SIZE": str(source),
                    "PREFILL_CP_KV_CACHE_SHARDED": "1",
                    "DECODE_CP_KV_CACHE_SHARDED": str(int(enabled)),
                }
                with patch.dict("os.environ", env, clear=True):
                    configs = setup_args()
                    setup_default_args(configs)
                    pc = pickle.loads(pickle.dumps(EngineConfig.create(configs).parallelism_config))
                self.assertEqual(pc.role_type, RoleType.DECODE if role == "DECODE" else RoleType.PREFILL)
                self.assertEqual(pc.decode_cp_kv_cache_sharded, enabled)
                self.assertEqual(pc.local_kv_page_rr_shard_count(), local)
                self.assertEqual(pc.upstream_kv_page_rr_shard_count(), source)
                self.assertEqual((pc.tp_rank, pc.dp_rank), (tp - 1, dp - 1))
                if role == "DECODE":
                    draft = ModelFactory._propose_parallelism_config(pc, "kimi_k3_mtp")
                    self.assertEqual(draft.local_kv_page_rr_shard_count(), local)
                    self.assertEqual(draft.upstream_kv_page_rr_shard_count(), source)

        for overrides, error, message in (
            ({"TP_SIZE": "1", "DP_SIZE": "8", "KTP_SIZE": "8"}, AssertionError, "mutually exclusive"),
            ({"MODEL_TYPE": "fake_model"}, ValueError, "MODEL_TYPE=kimi_k3"),
        ):
            env = {"MODEL_TYPE": "kimi_k3", "ROLE_TYPE": "DECODE", "TP_SIZE": "8", "WORLD_SIZE": "8"}
            env.update(overrides)
            with self.subTest(overrides=overrides), patch.dict("os.environ", env, clear=True):
                with patch("sys.argv", ["test", "--decode_cp_kv_cache_sharded", "1"]):
                    configs = setup_args()
                with self.assertRaisesRegex(error, message):
                    setup_default_args(configs)

    def test_projection_ktp_decode_topology(self):
        from rtp_llm.ops import ParallelismConfig

        pc = ParallelismConfig()
        pc.tp_size = 1
        pc.dp_size = 8
        pc.ep_size = 8
        pc.ktp_size = 8
        pc.world_size = 8
        pc.local_world_size = 8
        set_parallelism_config(pc, world_rank=7)
        self.assertEqual(pc.ktp_rank, 7)
        self.assertEqual(pc.dp_rank, 7)

    def test_projection_ktp_rejects_attention_tp(self):
        from rtp_llm.ops import ParallelismConfig

        pc = ParallelismConfig()
        pc.tp_size = 8
        pc.dp_size = 1
        pc.ep_size = 8
        pc.ktp_size = 8
        pc.world_size = 8
        with self.assertRaisesRegex(AssertionError, "attention tp_size=1"):
            set_parallelism_config(pc)

    def test_projection_ktp_rejects_mismatched_world(self):
        from rtp_llm.ops import ParallelismConfig

        pc = ParallelismConfig()
        pc.tp_size = 1
        pc.dp_size = 8
        pc.ep_size = 8
        pc.ktp_size = 8
        pc.world_size = 16
        with self.assertRaisesRegex(AssertionError, "DP=EP=KTP=world_size"):
            set_parallelism_config(pc)

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


if __name__ == "__main__":
    unittest.main()
