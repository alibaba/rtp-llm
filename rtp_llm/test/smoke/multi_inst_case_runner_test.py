import unittest
from unittest.mock import patch

from smoke.multi_inst_case_runner import _set_visible_devices

from rtp_llm.test.utils.maga_server_manager import _apply_env_args


class MultiInstCaseRunnerTest(unittest.TestCase):
    @patch.dict("os.environ", {"HIP_VISIBLE_DEVICES": "0,1,2,3"}, clear=True)
    def test_pd_tp2x2_visible_devices_are_split_for_rocm(self):
        gpu_ids = ["0", "1", "2", "3"]
        decode_envs = {"WORLD_SIZE": "2"}
        prefill_envs = {"WORLD_SIZE": "2"}

        _set_visible_devices(decode_envs, gpu_ids[:2])
        _set_visible_devices(prefill_envs, gpu_ids[2:])

        self.assertEqual("0,1", decode_envs["HIP_VISIBLE_DEVICES"])
        self.assertEqual("0,1", decode_envs["CUDA_VISIBLE_DEVICES"])
        self.assertIsNone(decode_envs["ROCR_VISIBLE_DEVICES"])

        self.assertEqual("2,3", prefill_envs["HIP_VISIBLE_DEVICES"])
        self.assertEqual("0,1", prefill_envs["CUDA_VISIBLE_DEVICES"])
        self.assertIsNone(prefill_envs["ROCR_VISIBLE_DEVICES"])

    @patch.dict("os.environ", {}, clear=True)
    def test_visible_devices_use_cuda_for_non_rocm(self):
        envs = {"WORLD_SIZE": "2"}

        _set_visible_devices(envs, ["2", "3"])

        self.assertEqual("2,3", envs["CUDA_VISIBLE_DEVICES"])
        self.assertNotIn("HIP_VISIBLE_DEVICES", envs)
        self.assertNotIn("ROCR_VISIBLE_DEVICES", envs)

    def test_none_env_override_removes_inherited_var(self):
        current_env = {"ROCR_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "0,1"}

        _apply_env_args(current_env, {"ROCR_VISIBLE_DEVICES": None})

        self.assertNotIn("ROCR_VISIBLE_DEVICES", current_env)
        self.assertEqual("0,1", current_env["HIP_VISIBLE_DEVICES"])


if __name__ == "__main__":
    unittest.main()
