import pickle
import unittest
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import (
    _configure_model_prefill_cp,
    set_parallelism_config,
    setup_and_configure_server,
    setup_default_args,
)
from rtp_llm.model_factory_register import ModelDict
from rtp_llm.ops import CPRotateMethod, ParallelismConfig, PrefillCPConfig, RoleType
from rtp_llm.server.server_args.server_args import setup_args


class GenerateConfigTest(TestCase):
    def test_incompatible_model_cp_cache_block_size_is_rejected(self):
        import rtp_llm.models.qwen3_next.qwen3_next  # noqa: F401

        py_env_configs = self._qwen35_cp_configs()
        py_env_configs.kv_cache_config.seq_size_per_block = 96

        with self.assertRaisesRegex(ValueError, "KV cache block size 96"):
            _configure_model_prefill_cp(py_env_configs)

    def test_cp_with_layer_micro_batch_is_rejected(self):
        py_env_configs = PyEnvConfigs()
        py_env_configs.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        py_env_configs.device_resource_config.enable_layer_micro_batch = 1

        with self.assertRaisesRegex(ValueError, "layer micro-batching"):
            _configure_model_prefill_cp(py_env_configs)

    def test_non_positive_cp_alignment_is_rejected(self):
        py_env_configs = PyEnvConfigs()
        py_env_configs.prefill_cp_config.segment_size_alignment = 0

        with self.assertRaisesRegex(ValueError, "must be greater than 0"):
            set_parallelism_config(
                py_env_configs.parallelism_config,
                py_prefill_cp_config=py_env_configs.prefill_cp_config,
            )

    def test_unknown_cp_model_is_rejected(self):
        py_env_configs = PyEnvConfigs()
        py_env_configs.model_args.model_type = "unregistered_model"
        py_env_configs.prefill_cp_config.method = CPRotateMethod.ALL_GATHER

        with patch.object(
            ModelDict, "get_model_cls", return_value=None
        ), self.assertRaisesRegex(ValueError, "unknown model_type"):
            _configure_model_prefill_cp(py_env_configs)

    def test_non_model_roles_ignore_shared_cp_environment(self):
        for role_type in (RoleType.FRONTEND, RoleType.VIT):
            with self.subTest(role_type=role_type):
                py_env_configs = PyEnvConfigs()
                py_env_configs.model_args.model_type = "unregistered_model"
                py_env_configs.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
                py_env_configs.device_resource_config.enable_layer_micro_batch = 1
                py_env_configs.role_config.role_type = role_type

                _configure_model_prefill_cp(py_env_configs)

    def test_prefill_cp_config_pickle_roundtrip_preserves_alignment(self):
        cp_config = PrefillCPConfig()
        cp_config.method = CPRotateMethod.ALL_GATHER
        cp_config.segment_size_alignment = 64

        restored_cp = pickle.loads(pickle.dumps(cp_config))
        self.assertEqual(restored_cp.method, CPRotateMethod.ALL_GATHER)
        self.assertEqual(restored_cp.segment_size_alignment, 64)

        parallelism_config = ParallelismConfig()
        parallelism_config.prefill_cp_config = cp_config
        restored_parallelism = pickle.loads(pickle.dumps(parallelism_config))
        self.assertEqual(
            restored_parallelism.prefill_cp_config.segment_size_alignment, 64
        )

    def test_qwen35_cp_supported_topology_is_accepted(self):
        import rtp_llm.models.qwen3_next.qwen3_next  # noqa: F401

        py_env_configs = self._qwen35_cp_configs()
        with patch(
            "rtp_llm.config.server_config_setup.torch.cuda.is_available",
            return_value=False,
        ):
            setup_default_args(py_env_configs)

        self.assertEqual(py_env_configs.prefill_cp_config.segment_size_alignment, 64)
        self.assertEqual(
            py_env_configs.parallelism_config.prefill_cp_config.segment_size_alignment,
            64,
        )

    def test_qwen35_cp_unsupported_topologies_are_rejected(self):
        import rtp_llm.models.qwen3_next.qwen3_next  # noqa: F401

        invalid_configs = (
            ("tp_size", 1),
            ("world_size", 8),
            ("dp_size", 2),
            ("ep_size", 4),
            ("pp_size", 2),
            ("ffn_sp_size", 2),
        )
        for field, value in invalid_configs:
            with self.subTest(field=field, value=value):
                py_env_configs = self._qwen35_cp_configs()
                setattr(py_env_configs.parallelism_config, field, value)
                with self.assertRaisesRegex(ValueError, "Qwen3-Next/Qwen3.5"):
                    _configure_model_prefill_cp(py_env_configs)

        py_env_configs = self._qwen35_cp_configs()
        py_env_configs.role_config.role_type = RoleType.PREFILL
        with self.assertRaisesRegex(ValueError, "Qwen3-Next/Qwen3.5"):
            _configure_model_prefill_cp(py_env_configs)

        py_env_configs = self._qwen35_cp_configs()
        py_env_configs.ffn_disaggregate_config.enable_ffn_disaggregate = True
        with self.assertRaisesRegex(ValueError, "Qwen3-Next/Qwen3.5"):
            _configure_model_prefill_cp(py_env_configs)

    def test_qwen35_cp_default_ep_is_rejected_after_normalization(self):
        import rtp_llm.models.qwen3_next.qwen3_next  # noqa: F401

        py_env_configs = self._qwen35_cp_configs()
        py_env_configs.parallelism_config.ep_size = 0
        with patch(
            "rtp_llm.config.server_config_setup.torch.cuda.is_available",
            return_value=False,
        ), self.assertRaisesRegex(ValueError, "set EP_SIZE=1 explicitly"):
            setup_default_args(py_env_configs)
        self.assertEqual(py_env_configs.parallelism_config.ep_size, 4)

    @staticmethod
    def _qwen35_cp_configs() -> PyEnvConfigs:
        py_env_configs = PyEnvConfigs()
        py_env_configs.model_args.model_type = "qwen35_dense"
        py_env_configs.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        py_env_configs.kv_cache_config.seq_size_per_block = 256
        py_env_configs.parallelism_config.tp_size = 4
        py_env_configs.parallelism_config.world_size = 4
        py_env_configs.parallelism_config.dp_size = 1
        py_env_configs.parallelism_config.ep_size = 1
        py_env_configs.parallelism_config.pp_size = 1
        py_env_configs.parallelism_config.ffn_sp_size = 1
        py_env_configs.role_config.role_type = RoleType.PDFUSION
        return py_env_configs

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
