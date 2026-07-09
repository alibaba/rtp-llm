import unittest
from unittest import mock

from rtp_llm.utils.flash_attn_utils import can_use_flash_attn


class FlashAttnUtilsTest(unittest.TestCase):
    @mock.patch("importlib.util.find_spec", return_value=None)
    @mock.patch("torch.cuda.get_device_name", return_value="AMD Instinct MI308X")
    @mock.patch("torch.cuda.get_device_capability", return_value=(9, 0))
    def test_returns_false_when_flash_attn_package_is_missing(
        self, mock_capability, mock_name, mock_find_spec
    ):
        self.assertFalse(can_use_flash_attn())


if __name__ == "__main__":
    unittest.main()
