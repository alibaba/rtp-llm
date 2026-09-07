import multiprocessing
import pickle
import unittest
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import (
    set_parallelism_config,
    setup_and_configure_server,
)
from rtp_llm.server.server_args.server_args import setup_args


def _spawned_engine_parallelism(configs, result):
    with result:
        values = []
        for config in configs:
            set_parallelism_config(config.parallelism_config, world_rank=3)
            pc = EngineConfig.create(config).parallelism_config
            values.append(
                (
                    pc.prefill_cp_config.kv_cache_sharded,
                    pc.decode_cp_kv_cache_sharded,
                    pc.tp_size,
                    pc.tp_rank,
                    pc.get_attn_tp_size(),
                    pc.role_type.name,
                )
            )
        result.send(values)


class GenerateConfigTest(TestCase):

    def test_parallelism_pickle_compatibility_preserves_cp_and_ktp(self):
        from rtp_llm.ops import ParallelismConfig, RoleType

        config = ParallelismConfig()
        config.tp_size = 8
        config.tp_rank = 3
        config.ktp_size = 8
        config.ktp_rank = 5
        config.role_type = RoleType.DECODE
        config.prefill_cp_config.kv_cache_sharded = True
        config.decode_cp_kv_cache_sharded = True
        current = config.__getstate__()
        self.assertEqual(len(current), 21)
        upstream = current[:20]
        legacy = tuple(value for i, value in enumerate(upstream) if i not in (1, 10))
        for state, ktp, rank, decode_cp in (
            (legacy, 1, 0, False),
            (legacy + (True,), 1, 0, True),
            (upstream, 8, 5, False),
            (current, 8, 5, True),
        ):
            with self.subTest(fields=len(state)):
                restored = ParallelismConfig.__new__(ParallelismConfig)
                restored.__setstate__(state)
                restored = pickle.loads(pickle.dumps(restored))
                self.assertEqual(restored.tp_size, 8)
                self.assertEqual(restored.tp_rank, 3)
                self.assertEqual(restored.ktp_size, ktp)
                self.assertEqual(restored.ktp_rank, rank)
                self.assertEqual(restored.role_type, RoleType.DECODE)
                self.assertTrue(restored.prefill_cp_config.kv_cache_sharded)
                self.assertEqual(restored.decode_cp_kv_cache_sharded, decode_cp)


    @patch("rtp_llm.config.server_config_setup.torch.cuda.is_available", return_value=False)
    def test_k3_cache_sharding_reaches_spawned_engine(self, _):
        configs, expected = [], []
        for model, role, prefill, decode in (
            ("kimi_k3", "PREFILL", True, False),
            ("kimi_k3", "DECODE", False, True),
            ("kimi_k3", "PDFUSION", True, True),
            ("kimi_k3", "PDFUSION", False, False),
            ("fake_model", "PREFILL", True, False),
            ("fake_model", "DECODE", False, True),
            ("fake_model", "PDFUSION", True, True),
            ("fake_model", "PDFUSION", False, False),
        ):
            for rotate in (None, "ALL_GATHER", "PREFILL_CP"):
                env = {
                    "MODEL_TYPE": model,
                    "ROLE_TYPE": role,
                    "TP_SIZE": "8",
                    "WORLD_SIZE": "8",
                    "LOCAL_WORLD_SIZE": "8",
                    "PREFILL_CP_KV_CACHE_SHARDED": str(int(prefill)),
                    "DECODE_CP_KV_CACHE_SHARDED": str(int(decode)),
                }
                if rotate:
                    env["CP_ROTATE_METHOD"] = rotate
                with patch.dict("os.environ", env, clear=True), patch("sys.argv", ["test"]):
                    config = setup_args()
                    setup_and_configure_server(config)
                    configs.append(config)
                    expected.append(
                        (prefill, decode, 8, 3, 1 if rotate == "ALL_GATHER" else 8, role)
                    )

        # The backend receives parsed configs through spawn, then derives rank and role.
        ctx = multiprocessing.get_context("spawn")
        receive, send = ctx.Pipe(duplex=False)
        worker = ctx.Process(target=_spawned_engine_parallelism, args=(configs, send))
        try:
            worker.start()
            send.close()
            self.assertTrue(receive.poll(60), "spawned config consumer did not finish")
            self.assertEqual(receive.recv(), expected)
            worker.join(10)
            self.assertEqual(worker.exitcode, 0)
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join()
            receive.close()
            send.close()

    @patch("rtp_llm.config.server_config_setup.torch.cuda.is_available", return_value=False)
    def test_k3_cache_sharding_rejects_invalid_role_layout(self, _):
        for model, prefill, decode, error in (
            ("kimi_k3", "1", "0", "must match"),
            ("kimi_k3", "0", "1", "must match"),
            ("kimi_k3_mla_swa_eagle3", "1", "0", "must match"),
            ("qwen", "0", "1", "must match"),
            ("qwen", "1", "0", "must match"),
        ):
            with self.subTest(model=model, prefill=prefill, decode=decode), patch.dict(
                "os.environ",
                {
                    "MODEL_TYPE": model,
                    "ROLE_TYPE": "PDFUSION",
                    "PREFILL_CP_KV_CACHE_SHARDED": prefill,
                    "DECODE_CP_KV_CACHE_SHARDED": decode,
                },
                clear=True,
            ), patch("sys.argv", ["test"]):
                with self.assertRaisesRegex(ValueError, error):
                    setup_and_configure_server(setup_args())

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
