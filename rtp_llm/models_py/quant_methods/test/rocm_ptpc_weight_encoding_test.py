import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from rtp_llm.models_py.quant_methods.rocm_ptpc_weight import encode_rocm_ptpc_weight


class RocmPtpcWeightEncodingTest(unittest.TestCase):
    def test_checkpoint_weight_uses_448_grid_and_source_precision_scales(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as d:
                path = str(Path(d) / "weights.safetensors")
                w = torch.tensor([[1., -.5, .25, -1.], [0., 0., 0., 0.],
                                  [.913, -.019, .32, -.44]], dtype=dtype)
                save_file({"projection.weight": w}, path)
                encoded, scales = encode_rocm_ptpc_weight(load_file(path)["projection.weight"])
                self.assertEqual(encoded.dtype, torch.float8_e4m3fn)
                self.assertEqual(scales.dtype, torch.float32)
                expected_scale = (w.abs().max(1).values.clamp(min=1e-4) / 448).float()
                torch.testing.assert_close(scales[:, 0], expected_scale, rtol=0, atol=0)
                self.assertEqual(encoded[0].view(torch.uint8).tolist(), [126, 246, 110, 254])
                self.assertEqual(encoded[1].view(torch.uint8).tolist(), [0]*4)
                # Old map is KxN and its weights/scales have to be transposed
                # into per-output-channel runtime layout.
                legacy = w.t().contiguous()
                legacy_scale = (legacy.abs().amax(0, keepdim=True).clamp(min=1e-4) / 448).float()
                legacy = (legacy / legacy_scale).to(torch.float8_e4m3fn)
                self.assertTrue(torch.equal(encoded.view(torch.uint8), legacy.t().contiguous().view(torch.uint8)))

    def test_noncontiguous_input_and_chunk_boundary(self):
        w = torch.linspace(-1.0, 1.0, 1025 * 1024).reshape(1024, 1025).t()
        encoded, scales = encode_rocm_ptpc_weight(w)
        scale = (w.abs().amax(1, keepdim=True).clamp(min=1e-4) / 448).float()
        expected = (w / scale).to(torch.float8_e4m3fn)
        self.assertTrue(torch.equal(encoded.view(torch.uint8), expected.view(torch.uint8)))
        self.assertTrue(torch.equal(scales, scale))
        self.assertTrue(encoded.is_contiguous())


if __name__ == "__main__":
    unittest.main()
