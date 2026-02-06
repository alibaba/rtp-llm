import unittest
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.server.server_args.server_args import setup_args


class GenerateConfigTest(TestCase):

    @patch.dict(
        "os.environ",
        {
            "TP_SIZE": "4",
            "PP_SIZE": "1",
            "EP_SIZE": "1",
            "WORLD_SIZE": "4",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "CONCURRENCY_LIMIT": "32",
            "START_PORT": "20000",
            "MODEL_TYPE": "fake_model",
        },
        clear=True,
    )
    def test_simple(self):
        from rtp_llm.config.server_config_setup import setup_default_args, fetch_model_files_to_local
        py_env_configs: PyEnvConfigs = setup_args()
        worker_info = WorkerInfo.from_env(
            py_env_configs.server_config.start_port,
            py_env_configs.distribute_config.remote_server_port,
            py_env_configs.server_config.worker_info_port_num,
        )
        setup_and_configure_server(py_env_configs)
        pc = py_env_configs.parallelism_config
        self.assertEqual(pc.tp_size, 4)
        self.assertEqual(pc.world_size, 4)
        self.assertEqual(pc.local_world_size, 2)
        self.assertEqual(worker_info.server_port, 20000)

        self.assertEqual(py_env_configs.moe_config.use_deepep_moe, True)
        self.assertEqual(py_env_configs.moe_config.use_deepep_low_latency, False)
        self.assertEqual(py_env_configs.moe_config.use_deepep_internode, True)
        self.assertEqual(py_env_configs.moe_config.ll_num_max_token, 32)

    @patch.dict(
        "os.environ",
        {
            "TP_SIZE": "2",
            "EP_SIZE": "1",
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


if __name__ == "__main__":
    unittest.main()
