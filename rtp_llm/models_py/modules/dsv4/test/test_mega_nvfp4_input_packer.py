"""Correctness tests for the fused NVFP4 MegaMoE input packer."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.moe._mega_nvfp4_input_pack_triton import (
    fused_pack_mega_nvfp4_inputs,
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
                return_gsf=True,
            )
            self.assertTrue(torch.equal(out_x, ref_x))
            self.assertTrue(torch.equal(out_sf, ref_sf))
            self.assertEqual(out_gsf.dtype, torch.float32)
            self.assertTrue(torch.equal(out_gsf, ref_gsf))
            self.assertTrue(torch.equal(out_indices, indices))
            self.assertTrue(torch.equal(out_weights, weights))
