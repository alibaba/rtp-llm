"""Disk checkpoint to NewLoader's concrete PPU operators, including the fused path."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.modules.dsv4.ppu.manifest import DECODE_EXECUTION_OPTIONS
from rtp_llm.models_py.modules.dsv4.ppu.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.models_py.modules.factory.linear.ppu_impl.fp8_linear import PpuFp8Linear
from rtp_llm.models_py.new_models.deepseek_v4.ppu_mtp_weights import PpuMtpWeights


class OwnedMtpFusionTest(unittest.TestCase):
    def test_real_loaded_projection_and_decode_fusion(self):
        dim, rows, hc = 4096, 2, 4
        provider = PpuDecodeProvider(
            {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_FUSIONS": "1"}
        )
        owner = PpuMtpWeights(dim, 1e-6, torch.bfloat16, provider).to("cuda")
        tensors = {}
        for name in ("enorm", "hnorm"):
            tensors[name + ".weight"] = torch.linspace(
                0.8, 1.2, dim, dtype=torch.bfloat16
            )
        for name, multiplier in (("e_proj", 1), ("h_proj", 2)):
            tensors[name + ".weight"] = (torch.eye(dim) * multiplier).to(
                torch.float8_e4m3fn
            )
            tensors[name + ".scale"] = torch.full(
                (dim // 128, dim // 128), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mtp.safetensors"
            save_file(tensors, path)
            with safe_open(path, framework="pt", device="cpu") as archive:
                owner.load_weights(
                    {name: archive.get_tensor(name) for name in archive.keys()}
                )
        model = SimpleNamespace()
        owner.bind(model, provider)
        self.assertIs(type(model.e_proj), PpuFp8Linear)
        self.assertIs(type(model.h_proj), PpuFp8Linear)
        self.assertEqual(model.enorm.weight.data_ptr(), owner.enorm.weight.data_ptr())
        self.assertEqual(model.hnorm.weight.data_ptr(), owner.hnorm.weight.data_ptr())
        torch.manual_seed(17)
        e = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
        h = torch.randn((rows, hc, dim), device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(rows, device="cuda", dtype=torch.int64)
        result = model._mtp_input_fusion(e, h, positions)
        self.assertIsNotNone(result, "PPU decode fusion declined a supported MTP shape")
        masked = torch.where(positions[:, None] == 0, 0, e)
        ref = model.h_proj(model.hnorm(h.view(-1, dim))).view_as(h)
        ref += model.e_proj(model.enorm(masked)).unsqueeze(1)
        torch.testing.assert_close(result, ref, rtol=0.02, atol=0.02)


if __name__ == "__main__":
    unittest.main()
