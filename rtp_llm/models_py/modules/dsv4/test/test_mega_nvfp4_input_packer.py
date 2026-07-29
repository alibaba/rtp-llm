"""Correctness tests for the fused NVFP4 MegaMoE input packer."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.moe._mega_nvfp4_input_pack_triton import (
    fused_pack_mega_nvfp4_inputs,
)
from rtp_llm.models_py.modules.dsv4.moe.mega_nvfp4_input_packer import (
    TorchMegaNVFP4InputPacker,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MegaNVFP4InputPackerTest(unittest.TestCase):
    def test_matches_deep_gemm_reference_bitwise(self):
        from deep_gemm.utils import per_token_cast_to_nvfp4

        for tokens in (0, 1, 3, 17, 129):
            torch.manual_seed(100 + tokens)
            x = (
                torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16) * 3
            ).contiguous()
            weights = torch.softmax(
                torch.randn(tokens, 6, device="cuda", dtype=torch.float32), -1
            ).contiguous()
            indices = torch.randint(
                0, 256, (tokens, 6), device="cuda", dtype=torch.int64
            ).contiguous()
            out_x = torch.empty((tokens, 2048), device="cuda", dtype=torch.int8)
            out_sf = torch.empty((tokens, 64), device="cuda", dtype=torch.int32)
            out_gsf = torch.empty((tokens,), device="cuda", dtype=torch.float32)
            out_indices = torch.empty_like(indices)
            out_weights = torch.empty_like(weights)

            fused_pack_mega_nvfp4_inputs(
                x,
                weights,
                indices,
                out_x,
                out_sf,
                out_gsf,
                out_indices,
                out_weights,
            )
            ref_x, ref_sf, ref_gsf = per_token_cast_to_nvfp4(
                x,
                gran_k=16,
                use_packed_ue4m3=True,
            )
            self.assertTrue(torch.equal(out_x, ref_x))
            self.assertTrue(torch.equal(out_sf, ref_sf))
            self.assertEqual(out_gsf.dtype, torch.float32)
            self.assertTrue(torch.equal(out_gsf, ref_gsf))
            self.assertTrue(torch.equal(out_indices, indices))
            self.assertTrue(torch.equal(out_weights, weights))

    def test_nonfinite_activations_are_zeroed(self):
        from deep_gemm.utils import per_token_cast_to_nvfp4

        tokens, hidden, topk = 3, 256, 6
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        x[0, 0] = float("nan")
        x[1, 1] = float("inf")
        x[2, 2] = -float("inf")
        weights = torch.softmax(
            torch.randn(tokens, topk, device="cuda", dtype=torch.float32), -1
        ).contiguous()
        indices = torch.randint(
            0, 256, (tokens, topk), device="cuda", dtype=torch.int64
        ).contiguous()

        safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        ref_x, ref_sf, ref_gsf = per_token_cast_to_nvfp4(
            safe_x,
            gran_k=16,
            use_packed_ue4m3=True,
        )

        fused_buf = SimpleNamespace(
            x=torch.empty_like(ref_x),
            x_sf=torch.empty_like(ref_sf),
            x_gsf=torch.empty_like(ref_gsf),
            topk_idx=torch.empty_like(indices),
            topk_weights=torch.empty_like(weights),
        )
        fused_pack_mega_nvfp4_inputs(
            x,
            weights,
            indices,
            fused_buf.x,
            fused_buf.x_sf,
            fused_buf.x_gsf,
            fused_buf.topk_idx,
            fused_buf.topk_weights,
        )

        torch_buf = SimpleNamespace(
            x=torch.empty_like(ref_x),
            x_sf=torch.empty_like(ref_sf),
            x_gsf=torch.empty_like(ref_gsf),
            topk_idx=torch.empty_like(indices),
            topk_weights=torch.empty_like(weights),
        )
        with mock.patch.dict(os.environ, {"DSV4_MOE_STRICT_FUSED": "0"}):
            TorchMegaNVFP4InputPacker().pack(x, weights, indices, torch_buf, tokens)
        torch.cuda.synchronize()

        for buf in (fused_buf, torch_buf):
            self.assertTrue(torch.equal(buf.x, ref_x))
            self.assertTrue(torch.equal(buf.x_sf, ref_sf))
            self.assertTrue(torch.equal(buf.x_gsf, ref_gsf))
            self.assertTrue(torch.isfinite(buf.x_gsf).all().item())
            self.assertTrue(torch.equal(buf.topk_idx, indices))
            self.assertTrue(torch.equal(buf.topk_weights, weights))
