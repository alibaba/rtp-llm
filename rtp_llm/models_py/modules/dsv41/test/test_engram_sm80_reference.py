"""SM80 Engram selected-row fallback contract.

Triton rejects ``float8e4nv`` loads on SM80; the host table keeps native
FP8/E8M0 bytes and this path stages only selected rows, decodes them to BF16.
"""

import importlib.util
from pathlib import Path
import unittest

import torch

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "engram_triton_tested", SOURCE / "_engram_triton.py"
)
engram = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engram)


def _reference(weight, scales, ids):
    n, dim = weight.shape
    flat = ids.flatten()
    valid = (flat >= 0) & (flat < n)
    safe = flat.clamp(0, n - 1)
    rows = weight.view(torch.float8_e4m3fn).index_select(0, safe).float()
    exponent = scales.index_select(0, safe).to(torch.int32)
    scale = (exponent << 23).view(torch.float32).repeat_interleave(32, 1)
    out = (rows * scale).bfloat16()
    out[~valid] = 0
    return out.reshape(ids.shape[0], ids.shape[1], dim)


class EngramSm80ReferenceTest(unittest.TestCase):
    def test_selected_rows_match_dequant_reference(self):
        torch.manual_seed(20261111)
        for n, dim, tokens, heads in ((23, 256, 9, 8), (5, 64, 3, 2)):
            weight = torch.randint(256, (n, dim), dtype=torch.uint8)
            scales = torch.randint(256, (n, dim // 32), dtype=torch.uint8)
            ids = torch.randint(-2, n + 2, (tokens, heads), dtype=torch.int64)
            expected = _reference(weight, scales, ids)
            actual = engram._lookup_host_rows_reference(weight, scales, ids)
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))

    def test_empty_lookup_returns_empty_bf16(self):
        weight = torch.empty((0, 256), dtype=torch.uint8)
        scales = torch.empty((0, 8), dtype=torch.uint8)
        ids = torch.empty((0, 4), dtype=torch.int64)
        actual = engram._lookup_host_rows_reference(weight, scales, ids)
        self.assertEqual(tuple(actual.shape), (0, 4, 256))
        self.assertEqual(actual.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
