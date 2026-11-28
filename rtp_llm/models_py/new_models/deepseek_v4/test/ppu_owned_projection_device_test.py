"""Non-skipping M890P numerical regression: original PPU quantizer and GEMM."""

import os
import tempfile
import unittest

import torch

from rtp_llm.device.device_type import is_ppu
from rtp_llm.models_py.modules.dsv4.ppu.ppu_module_provider import PpuModuleProvider
from rtp_llm.models_py.modules.factory.linear.ppu_impl.fp8_linear import PpuFp8Linear
from rtp_llm.models_py.new_models.deepseek_v4.ppu_linear import PpuOwnedFP8Linear
from rtp_llm.models_py.weight_mapper import get_all_weights


class OwnedProjectionDeviceTest(unittest.TestCase):
    def test_prefill_checkpoint_projection(self):
        self.assertTrue(is_ppu() and torch.cuda.is_available(), "requires a PPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9), "requires M890P")
        with tempfile.TemporaryDirectory() as tmp:
            file = os.path.join(tmp, "model.pt")
            torch.save(
                {
                    "weight": torch.ones((256, 256), dtype=torch.float8_e4m3fn),
                    "scale": torch.tensor(
                        [[127, 128], [129, 127]], dtype=torch.uint8
                    ).view(torch.float8_e8m0fnu),
                },
                file,
            )
            owner = PpuOwnedFP8Linear(256, 256, platform_provider=PpuModuleProvider({}))
            owner.load_weights(
                get_all_weights([file], read_spec_resolver=owner.checkpoint_read_spec)
            )
            owner.cuda()
            executor = owner.bind_runtime()
            self.assertIs(type(executor), PpuFp8Linear)
            self.assertEqual(executor.weight.data_ptr(), owner.weight.data_ptr())
            expected_scales = torch.tensor([[1.0, 2.0], [4.0, 1.0]], device="cuda")
            torch.testing.assert_close(
                executor.weight_scale, expected_scales, atol=0, rtol=0
            )
            inputs = torch.ones((64, 256), device="cuda", dtype=torch.bfloat16)
            expected = (
                inputs.float()
                @ expected_scales.repeat_interleave(128, 0).repeat_interleave(128, 1).T
            )
            torch.testing.assert_close(
                owner(inputs), expected.bfloat16(), atol=4, rtol=0.015625
            )


if __name__ == "__main__":
    unittest.main()
