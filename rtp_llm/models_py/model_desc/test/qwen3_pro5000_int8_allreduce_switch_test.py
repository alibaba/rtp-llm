import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc.qwen3 import _should_enable_pro5000_int8_allreduce


class Qwen3Pro5000Int8AllReduceSwitchTest(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.device = torch.device("cuda")

    @patch("torch.cuda.get_device_name", return_value="NVIDIA RTX PRO 5000")
    def test_enabled_on_pro5000_by_default(self, _get_device_name):
        self.assertTrue(_should_enable_pro5000_int8_allreduce(self.config, self.device))

    @patch("torch.cuda.get_device_name", return_value="NVIDIA RTX PRO 5000")
    def test_explicit_disable_uses_exact_allreduce(self, _get_device_name):
        self.config.enable_qwen3_pro5000_int8_allreduce = False
        self.assertFalse(
            _should_enable_pro5000_int8_allreduce(self.config, self.device)
        )

    @patch("torch.cuda.get_device_name", return_value="NVIDIA H20")
    def test_other_devices_do_not_enable_the_optimization(self, _get_device_name):
        self.assertFalse(
            _should_enable_pro5000_int8_allreduce(self.config, self.device)
        )

    def test_cpu_does_not_query_cuda_device_name(self):
        with patch("torch.cuda.get_device_name") as get_device_name:
            self.assertFalse(
                _should_enable_pro5000_int8_allreduce(self.config, torch.device("cpu"))
            )
            get_device_name.assert_not_called()


if __name__ == "__main__":
    unittest.main()
