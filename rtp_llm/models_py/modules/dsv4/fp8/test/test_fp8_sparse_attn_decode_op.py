"""Unit tests for the FP8 sparse decode FlashMLA wrapper."""

from __future__ import annotations

import os
import sys
import types
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)
from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import canonical_topk


class TestSparseAttnV4DecodeFp8Op(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_sm120_flashinfer_matches_precise_reference_for_mtp132(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("requires SM120")

        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )

        torch.manual_seed(20260820)
        device = torch.device("cuda")
        token_count = 256
        valid_count = 132
        head_count = 64
        head_dim = 512
        kv_cache = torch.zeros(
            1, token_count, 584, dtype=torch.uint8, device=device
        )
        kv = (torch.randn(token_count, head_dim, device=device) * 0.125).to(
            torch.bfloat16
        )
        quantize_and_insert_k_cache(
            kv,
            kv_cache,
            torch.arange(token_count, dtype=torch.int64, device=device),
        )
        q = (torch.randn(1, 1, head_count, head_dim, device=device) * 0.125).to(
            torch.bfloat16
        )
        sinks = torch.randn(head_count, dtype=torch.float32, device=device)
        indices = torch.full(
            (1, 1, token_count), -1, dtype=torch.int32, device=device
        )
        indices[..., :valid_count] = torch.randperm(
            token_count, dtype=torch.int64, device=device
        )[:valid_count].to(torch.int32)
        lengths = torch.tensor([valid_count], dtype=torch.int32, device=device)
        op = SparseAttnV4DecodeFp8Op(
            n_heads=head_count,
            head_dim=head_dim,
            softmax_scale=head_dim**-0.5,
        )

        old_precise = os.environ.pop("DSV4_SM120_PRECISE_SPARSE_MLA", None)
        try:
            actual = op._forward_sm120_flashinfer(
                q, kv_cache, sinks, indices, lengths
            )
            os.environ["DSV4_SM120_PRECISE_SPARSE_MLA"] = "1"
            expected = op._forward_sm120_flashinfer(
                q, kv_cache, sinks, indices, lengths
            )
        finally:
            if old_precise is None:
                os.environ.pop("DSV4_SM120_PRECISE_SPARSE_MLA", None)
            else:
                os.environ["DSV4_SM120_PRECISE_SPARSE_MLA"] = old_precise

        diff = (actual.float() - expected.float()).abs()
        relative_l2 = diff.norm() / expected.float().norm()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        print(
            "SM120 sparse MLA parity: "
            f"max_abs={diff.max().item():.8f} "
            f"mean_abs={diff.mean().item():.8f} "
            f"relative_l2={relative_l2.item():.8f} "
            f"cosine={cosine.item():.8f}"
        )
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

        extra_cache = torch.zeros(
            4, 64, 584, dtype=torch.uint8, device=device
        )
        extra_kv = (torch.randn(token_count, head_dim, device=device) * 0.125).to(
            torch.bfloat16
        )
        quantize_and_insert_k_cache(
            extra_kv,
            extra_cache,
            torch.arange(token_count, dtype=torch.int64, device=device),
        )
        extra_valid_count = 97
        extra_indices = torch.full(
            (1, 1, 128), -1, dtype=torch.int32, device=device
        )
        extra_indices[..., :extra_valid_count] = torch.randperm(
            token_count, dtype=torch.int64, device=device
        )[:extra_valid_count].to(torch.int32)
        extra_lengths = torch.tensor(
            [extra_valid_count], dtype=torch.int32, device=device
        )
        old_precise = os.environ.pop("DSV4_SM120_PRECISE_SPARSE_MLA", None)
        try:
            actual_dual = op._forward_sm120_flashinfer(
                q,
                kv_cache,
                sinks,
                indices,
                lengths,
                extra_cache,
                extra_indices,
                extra_lengths,
            )
            os.environ["DSV4_SM120_PRECISE_SPARSE_MLA"] = "1"
            expected_dual = op._forward_sm120_flashinfer(
                q,
                kv_cache,
                sinks,
                indices,
                lengths,
                extra_cache,
                extra_indices,
                extra_lengths,
            )
        finally:
            if old_precise is None:
                os.environ.pop("DSV4_SM120_PRECISE_SPARSE_MLA", None)
            else:
                os.environ["DSV4_SM120_PRECISE_SPARSE_MLA"] = old_precise
        dual_diff = (actual_dual.float() - expected_dual.float()).abs()
        dual_relative_l2 = dual_diff.norm() / expected_dual.float().norm()
        dual_cosine = torch.nn.functional.cosine_similarity(
            actual_dual.float().flatten(), expected_dual.float().flatten(), dim=0
        )
        print(
            "SM120 dual-pool sparse MLA parity: "
            f"max_abs={dual_diff.max().item():.8f} "
            f"mean_abs={dual_diff.mean().item():.8f} "
            f"relative_l2={dual_relative_l2.item():.8f} "
            f"cosine={dual_cosine.item():.8f}"
        )
        torch.testing.assert_close(
            actual_dual, expected_dual, rtol=0.03, atol=0.02
        )

    def test_dspark_mtp_topk_keeps_132_valid_entries(self):
        indices = torch.arange(256, dtype=torch.int32).view(1, 256)
        lengths = torch.tensor([132], dtype=torch.int32)

        canonical, canonical_lens = canonical_topk(
            indices,
            lengths,
            (128, 512, 1024),
            trim_dspark_padding=False,
            pad_to_supported=True,
        )

        self.assertEqual(tuple(canonical.shape), (1, 512))
        self.assertTrue(torch.equal(canonical[:, :256], indices))
        self.assertTrue(bool((canonical[:, 256:] == -1).all()))
        self.assertEqual(canonical_lens.tolist(), [132])

    def test_sparse_indices_drop_dense_cache_metadata(self):
        calls = []
        fake_flash_mla = types.ModuleType("flash_mla")

        def fake_flash_mla_with_kvcache(**kwargs):
            calls.append(kwargs)
            q = kwargs["q"]
            head_dim_v = kwargs["head_dim_v"]
            out = torch.zeros(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                head_dim_v,
                dtype=q.dtype,
                device=q.device,
            )
            lse = torch.zeros(
                q.shape[0],
                q.shape[2],
                q.shape[1],
                dtype=torch.float32,
                device=q.device,
            )
            return out, lse

        fake_flash_mla.flash_mla_with_kvcache = fake_flash_mla_with_kvcache
        old_flash_mla = sys.modules.get("flash_mla")
        sys.modules["flash_mla"] = fake_flash_mla
        try:
            op = SparseAttnV4DecodeFp8Op(
                n_heads=4,
                head_dim=512,
                softmax_scale=1.0,
            )
            q = torch.zeros(2, 3, 4, 512, dtype=torch.bfloat16)
            kv_cache = torch.zeros(8, 256, 584, dtype=torch.uint8)
            attn_sink = torch.zeros(4, dtype=torch.float32)
            topk = torch.arange(128, dtype=torch.int32).view(1, 1, 128).expand(
                2, 3, 128
            )
            block_table = torch.full((2, 257), -1, dtype=torch.int32)
            cache_seqlens = torch.tensor([65537, 65537], dtype=torch.int32)

            out = op._forward_flash_mla(
                q=q,
                kv_cache=kv_cache,
                attn_sink=attn_sink,
                topk_idxs=topk,
                sched_meta=object(),
                cache_seqlens=cache_seqlens,
                block_table=block_table,
            )

            self.assertEqual(tuple(out.shape), (2, 3, 4, 512))
            self.assertEqual(len(calls), 1)
            self.assertIsNone(calls[0]["block_table"])
            self.assertIsNone(calls[0]["cache_seqlens"])
            self.assertTrue(torch.equal(calls[0]["indices"], topk))
        finally:
            if old_flash_mla is None:
                sys.modules.pop("flash_mla", None)
            else:
                sys.modules["flash_mla"] = old_flash_mla


if __name__ == "__main__":
    unittest.main()
