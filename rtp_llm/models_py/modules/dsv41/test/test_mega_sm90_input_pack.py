# SPDX-License-Identifier: Apache-2.0
"""Host-side launch contracts for the fused SM90 MegaMoE input packer."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

_PATH = Path(__file__).resolve().parent.parent / "moe" / "_mega_sm90_input_pack_triton.py"
_SPEC = importlib.util.spec_from_file_location("_mega_sm90_pack_under_test", _PATH)
_PACK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PACK)


class MegaSm90InputPackLaunchTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("DSV4_MEGA_MOE_SM90_PACK_BLOCK_M", None)
        if _PACK.triton is None:
            fake_triton = patch.object(
                _PACK,
                "triton",
                SimpleNamespace(
                    cdiv=lambda x, y: (x + y - 1) // y,
                    next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
                ),
            )
            fake_triton.start()
            self.addCleanup(fake_triton.stop)
        kernel = patch.object(
            _PACK, "_pack_mega_moe_sm90_inputs_kernel", new_callable=MagicMock
        )
        self.kernel = kernel.start()
        self.addCleanup(kernel.stop)

    def _pack(self, tokens, hidden=5120, topk=6):
        x = torch.empty((tokens, hidden), device="meta", dtype=torch.bfloat16)
        weights = torch.empty((tokens, topk), device="meta")
        indices = torch.empty((tokens, topk), device="meta", dtype=torch.int32)
        _PACK.fused_pack_mega_moe_sm90_inputs(
            x,
            weights,
            indices,
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty((tokens, hidden // 128), device="meta"),
            torch.empty((tokens, topk), device="meta", dtype=torch.int64),
            torch.empty_like(weights),
        )

    def test_default_tile_and_grid_at_token_boundaries(self):
        for tokens, block_m in (
            (1, 4),
            (63, 4),
            (64, 16),
            (2047, 16),
            (2048, 32),
        ):
            with self.subTest(tokens=tokens):
                self.kernel.reset_mock()
                self._pack(tokens)
                self.kernel.__getitem__.assert_called_once_with(
                    ((tokens + block_m - 1) // block_m, 40)
                )
                kwargs = self.kernel.__getitem__.return_value.call_args.kwargs
                self.assertEqual(kwargs["BLOCK_M"], block_m)
                self.assertEqual(kwargs["BLOCK_K"], 8)
                self.assertEqual(kwargs["num_warps"], 4)

    def test_explicit_block_m_override(self):
        with patch.dict(os.environ, {"DSV4_MEGA_MOE_SM90_PACK_BLOCK_M": "8"}):
            self._pack(12)
        kwargs = self.kernel.__getitem__.return_value.call_args.kwargs
        self.assertEqual(kwargs["BLOCK_M"], 8)
        self.kernel.__getitem__.assert_called_once_with((2, 40))

    def test_invalid_block_m_override_is_rejected(self):
        for value in ("0", "3", "-4", "64"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"DSV4_MEGA_MOE_SM90_PACK_BLOCK_M": value}
            ):
                with self.assertRaisesRegex(ValueError, "power of two in"):
                    self._pack(12)
        self.kernel.__getitem__.assert_not_called()

    def test_empty_input_does_not_launch(self):
        self._pack(0)
        self.kernel.__getitem__.assert_not_called()

    def test_noncontiguous_inputs_are_normalized_before_launch(self):
        tokens, hidden, topk = 4, 5120, 6
        x = torch.empty((hidden, tokens), device="meta", dtype=torch.bfloat16).t()
        weights = torch.empty((topk, tokens), device="meta").t()
        indices = torch.empty(
            (topk, tokens), device="meta", dtype=torch.int32
        ).t()
        _PACK.fused_pack_mega_moe_sm90_inputs(
            x,
            weights,
            indices,
            torch.empty((tokens, hidden), device="meta", dtype=torch.float8_e4m3fn),
            torch.empty((tokens, hidden // 128), device="meta"),
            torch.empty((tokens, topk), device="meta", dtype=torch.int64),
            torch.empty((tokens, topk), device="meta"),
        )
        args = self.kernel.__getitem__.return_value.call_args.args
        self.assertTrue(args[0].is_contiguous())
        self.assertTrue(args[1].is_contiguous())
        self.assertTrue(args[2].is_contiguous())
        self.assertEqual(args[1].dtype, torch.float32)
        self.assertEqual(args[2].dtype, torch.int64)

    def test_hidden_width_must_match_k128_protocol(self):
        with self.assertRaisesRegex(ValueError, "divisible by 128"):
            self._pack(4, hidden=5119)

    def test_router_and_output_shapes_are_validated(self):
        tokens, hidden, topk = 4, 5120, 6
        x = torch.empty((tokens, hidden), device="meta", dtype=torch.bfloat16)
        weights = torch.empty((tokens, topk), device="meta")
        indices = torch.empty((tokens - 1, topk), device="meta", dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, r"matching \[tokens, topk\]"):
            _PACK.fused_pack_mega_moe_sm90_inputs(
                x,
                weights,
                indices,
                torch.empty_like(x, dtype=torch.float8_e4m3fn),
                torch.empty((tokens, hidden // 128), device="meta"),
                torch.empty((tokens, topk), device="meta", dtype=torch.int64),
                torch.empty((tokens, topk), device="meta"),
            )
        self.kernel.__getitem__.assert_not_called()


if __name__ == "__main__":
    unittest.main()
