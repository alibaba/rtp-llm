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
            self.assertFalse(arch.is_sm90())
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
            self.assertTrue(arch.is_sm90(2))
            self.assertFalse(arch.is_blackwell(2))
            self.assertEqual(queried_devices, [1, 0, 2])

    def test_cuda_ordinal_strings_match_integer_devices(self):
        with (
            patch.object(arch, "is_cuda", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch(
                "torch.cuda.get_device_capability",
                side_effect=lambda device: {0: (10, 0), 1: (12, 0)}[device],
            ) as capability,
        ):
            for device in ("1", "cuda:1", torch.device("cuda:1"), 1):
                self.assertTrue(arch.is_sm120(device))
                self.assertFalse(arch.is_sm10x(device))
            self.assertTrue(arch.is_sm10x("0"))
            self.assertEqual(capability.call_count, 2)

    def test_sm120_rejects_non_cuda_without_querying_capability(self):
        with patch.object(arch, "get_sm") as get_sm:
            with patch.object(arch, "is_cuda", return_value=False):
                self.assertFalse(arch.is_sm120())
            with patch.object(arch, "is_cuda", side_effect=AssertionError("must not probe CUDA")) as is_cuda_mock:
                for fn in (
                    arch.is_sm90,
                    arch.is_sm10x,
                    arch.is_sm12x,
                    arch.is_sm120,
                    arch.is_blackwell,
                ):
                    for device in ("cpu", torch.device("cpu"), torch.device("cpu", 0)):
                        self.assertFalse(fn(device))
                is_cuda_mock.assert_not_called()
            get_sm.assert_not_called()

    def test_invalid_and_unindexed_cuda_specs(self):
        for bad in ("", "-1", "gpu:0", "cdua:0", "CUDA:0", "cpu:abc", torch.device("cpu")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    arch.get_sm(bad)
        for bad_pred in ("", "-1", "gpu:0", "CUDA:0"):
            with self.subTest(bad_pred=bad_pred):
                with self.assertRaises(ValueError):
                    arch.is_sm120(bad_pred)
        with (
            patch.object(arch, "is_cuda", return_value=True),
            patch("torch.cuda.current_device", return_value=1),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)) as cap_mock,
        ):
            self.assertTrue(arch.is_sm120(" 1 "))
            self.assertTrue(arch.is_sm120("cuda"))
            self.assertEqual(arch.get_sm(" 1 "), (12, 0))
            cap_mock.assert_called_once_with(1)

    def test_sm120_requires_exact_cuda_capability(self):
        for capability in ((9, 0), (10, 0), (10, 3), (12, 0), (12, 1)):
            with (
                self.subTest(capability=capability),
                patch.object(arch, "is_cuda", return_value=True),
                patch.object(arch, "get_sm", return_value=capability),
            ):
                self.assertEqual(arch.is_sm120(0), capability == (12, 0))

    def test_sm10x_excludes_sm120(self):
        with patch.object(arch, "is_cuda", return_value=True):
            with patch.object(arch, "get_sm", return_value=(12, 0)):
                self.assertFalse(arch.is_sm10x(0))
                self.assertTrue(arch.is_sm120(0))
                self.assertTrue(arch.is_sm12x(0))
                self.assertTrue(arch.is_blackwell(0))
            for sm10x_cap in ((10, 0), (10, 3)):
                with self.subTest(capability=sm10x_cap):
                    with patch.object(arch, "get_sm", return_value=sm10x_cap):
                        self.assertTrue(arch.is_sm10x(0))
                        self.assertFalse(arch.is_sm120(0))


if __name__ == "__main__":
    main()
