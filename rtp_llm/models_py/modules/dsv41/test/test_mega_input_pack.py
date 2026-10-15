# SPDX-License-Identifier: Apache-2.0
"""Host-side launch contracts for the standalone MegaMoE pack kernel."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

_PATH = Path(__file__).resolve().parent.parent / "moe" / "_mega_input_pack_triton.py"
_SPEC = importlib.util.spec_from_file_location("_mega_input_pack_under_test", _PATH)
_PACK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PACK)


class MegaInputPackLaunchTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("DSV4_MEGA_MOE_PACK_BLOCK_M", None)
        kernel = patch.object(
            _PACK, "_pack_mega_moe_inputs_optimized_kernel", new_callable=MagicMock
        )
        self.kernel = kernel.start()
        self.addCleanup(kernel.stop)

    def _pack(self, tokens, topk=6):
        x = torch.empty((tokens, 5120), device="meta", dtype=torch.bfloat16)
        weights = torch.empty((tokens, topk), device="meta")
        indices = torch.empty((tokens, topk), device="meta", dtype=torch.int64)
        _PACK.fused_pack_mega_moe_inputs_optimized(
            x,
            weights,
            indices,
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty((tokens, 40), device="meta", dtype=torch.int32),
            torch.empty_like(indices),
            torch.empty_like(weights),
        )

    def test_default_tile_and_warps_at_token_boundaries(self):
        for tokens, block_m, warps in (
            (1, 4, 1),
            (63, 4, 1),
            (64, 16, 2),
            (65, 16, 2),
            (2047, 16, 2),
            (2048, 32, 2),
            (2049, 32, 2),
        ):
            for topk in (3, 6):
                with self.subTest(tokens=tokens, topk=topk):
                    self.kernel.reset_mock()
                    self._pack(tokens, topk)
                    self.kernel.__getitem__.assert_called_once_with(
                        ((tokens + block_m - 1) // block_m, 40)
                    )
                    kwargs = self.kernel.__getitem__.return_value.call_args.kwargs
                    self.assertEqual(kwargs["BLOCK_M"], block_m)
                    self.assertEqual(kwargs["num_warps"], warps)
                    self.assertEqual(kwargs["BLOCK_K"], 4 if topk == 3 else 8)

    def test_explicit_override_keeps_four_warps(self):
        with patch.dict(os.environ, {"DSV4_MEGA_MOE_PACK_BLOCK_M": "8"}):
            for tokens in (1, 64, 2048):
                with self.subTest(tokens=tokens):
                    self._pack(tokens)
                    kwargs = self.kernel.__getitem__.return_value.call_args.kwargs
                    self.assertEqual(kwargs["BLOCK_M"], 8)
                    self.assertEqual(kwargs["num_warps"], 4)

    def test_empty_input_does_not_launch(self):
        self._pack(0)
        self.kernel.__getitem__.assert_not_called()


if __name__ == "__main__":
    unittest.main()
