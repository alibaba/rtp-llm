from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.utils import arch


class ArchTest(TestCase):

    def setUp(self):
        arch._get_sm_for_device.cache_clear()

    def tearDown(self):
        arch._get_sm_for_device.cache_clear()

    def test_get_sm_defaults_to_current_cuda_device(self):
        with (
            patch("rtp_llm.models_py.utils.arch.is_cuda", return_value=True),
            patch("torch.cuda.current_device", return_value=1),
            patch(
                "torch.cuda.get_device_capability",
                side_effect=lambda device_id: {0: (10, 0), 1: (12, 0)}[device_id],
            ) as get_device_capability,
        ):
            self.assertEqual(arch.get_sm(), (12, 0))
            self.assertFalse(arch.is_sm10x())
            self.assertTrue(arch.is_sm12x())
            self.assertTrue(arch.is_blackwell())
            get_device_capability.assert_called_once_with(1)

    def test_arch_helpers_cache_by_resolved_device_id(self):
        queried_devices = []

        def get_device_capability(device_id):
            queried_devices.append(device_id)
            return {0: (10, 0), 1: (12, 0), 2: (9, 0)}[device_id]

        with (
            patch("rtp_llm.models_py.utils.arch.is_cuda", return_value=True),
            patch(
                "torch.cuda.get_device_capability", side_effect=get_device_capability
            ),
        ):
            with patch("torch.cuda.current_device", return_value=1):
                self.assertTrue(arch.is_sm12x())
                self.assertFalse(arch.is_sm10x())
            with patch("torch.cuda.current_device", return_value=0):
                self.assertTrue(arch.is_sm10x())
                self.assertFalse(arch.is_sm12x())

            self.assertTrue(arch.is_blackwell(torch.device("cuda:1")))
            self.assertFalse(arch.is_blackwell(2))
            self.assertEqual(queried_devices, [1, 0, 2])


if __name__ == "__main__":
    main()
