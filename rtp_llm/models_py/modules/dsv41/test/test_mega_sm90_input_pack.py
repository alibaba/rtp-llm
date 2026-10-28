# SPDX-License-Identifier: Apache-2.0
"""Host-side launch contracts for the fused SM90 MegaMoE input packer."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.modules.dsv41.moe import mega_sm90_input_packer as packer

_PATH = Path(__file__).resolve().parent.parent / "moe" / "_mega_sm90_input_pack_triton.py"
_SPEC = importlib.util.spec_from_file_location("_mega_sm90_pack_under_test", _PATH)
_PACK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PACK)


class MegaSm90InputPackLaunchTest(unittest.TestCase):
    def setUp(self):
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

    def test_factory_selects_sm90_backend_and_falls_back_safely(self):
        from rtp_llm.device import device_type
        from rtp_llm.models_py.modules.dsv41.moe import _mega_sm90_input_pack_triton

        with patch.object(device_type, "is_sm90", return_value=True), patch.object(
            _mega_sm90_input_pack_triton, "triton", object()
        ):
            self.assertIsInstance(
                packer.get_mega_sm90_input_packer(),
                packer.FusedMegaMoeSM90InputPacker,
            )
        with patch.object(device_type, "is_sm90", return_value=False):
            self.assertIsInstance(
                packer.get_mega_sm90_input_packer(),
                packer.MegaMoeSM90InputPacker,
            )
        with patch.object(device_type, "is_sm90", return_value=True), patch.object(
            _mega_sm90_input_pack_triton, "triton", None
        ):
            self.assertIsInstance(
                packer.get_mega_sm90_input_packer(),
                packer.MegaMoeSM90InputPacker,
            )

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
