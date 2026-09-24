import unittest

import torch


class DecodeDeviceTest(unittest.TestCase):
    def test_required_device_is_usable(self):
        self.assertTrue(torch.cuda.is_available(), "PPU decode tests require a device")
        self.assertEqual(
            torch.cuda.get_device_name(),
            "ZW-M890P",
            "PPU decode suite must not pass with its device tests skipped",
        )
        self.assertEqual(torch.ones(1, device="cuda").sum().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
