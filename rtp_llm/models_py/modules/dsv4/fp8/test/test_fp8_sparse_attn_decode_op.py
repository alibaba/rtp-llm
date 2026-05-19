"""Unit tests for the FP8 sparse decode FlashMLA wrapper."""

from __future__ import annotations

import sys
import types
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class TestSparseAttnV4DecodeFp8Op(unittest.TestCase):
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
            topk_length = torch.tensor([128, 127], dtype=torch.int32)

            out = op._forward_flash_mla(
                q=q,
                kv_cache=kv_cache,
                attn_sink=attn_sink,
                topk_idxs=topk,
                sched_meta=object(),
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                topk_length=topk_length,
            )

            self.assertEqual(tuple(out.shape), (2, 3, 4, 512))
            self.assertEqual(len(calls), 1)
            self.assertIsNone(calls[0]["block_table"])
            self.assertIsNone(calls[0]["cache_seqlens"])
            self.assertTrue(torch.equal(calls[0]["indices"], topk))
            self.assertTrue(torch.equal(calls[0]["topk_length"], topk_length))
        finally:
            if old_flash_mla is None:
                sys.modules.pop("flash_mla", None)
            else:
                sys.modules["flash_mla"] = old_flash_mla

    def test_multi_query_lengths_flatten_like_vllm_decode(self):
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
            extra_k_cache = torch.zeros(8, 64, 584, dtype=torch.uint8)
            attn_sink = torch.zeros(4, dtype=torch.float32)
            topk = torch.arange(128, dtype=torch.int32).view(1, 1, 128).expand(
                2, 3, 128
            )
            extra_topk = torch.arange(512, dtype=torch.int32).view(1, 1, 512).expand(
                2, 3, 512
            )
            topk_length = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)
            extra_topk_length = torch.tensor([6, 5, 4, 3, 2, 1], dtype=torch.int32)

            out = op._forward_flash_mla(
                q=q,
                kv_cache=kv_cache,
                attn_sink=attn_sink,
                topk_idxs=topk,
                sched_meta=object(),
                cache_seqlens=None,
                block_table=None,
                topk_length=topk_length,
                extra_k_cache=extra_k_cache,
                extra_topk_idxs=extra_topk,
                extra_topk_length=extra_topk_length,
            )

            self.assertEqual(tuple(out.shape), (2, 3, 4, 512))
            self.assertEqual(tuple(calls[0]["q"].shape), (6, 1, 4, 512))
            self.assertEqual(tuple(calls[0]["indices"].shape), (6, 1, 128))
            self.assertEqual(
                tuple(calls[0]["extra_indices_in_kvcache"].shape), (6, 1, 512)
            )
            self.assertTrue(torch.equal(calls[0]["topk_length"], topk_length))
            self.assertTrue(
                torch.equal(calls[0]["extra_topk_length"], extra_topk_length)
            )
        finally:
            if old_flash_mla is None:
                sys.modules.pop("flash_mla", None)
            else:
                sys.modules["flash_mla"] = old_flash_mla


if __name__ == "__main__":
    unittest.main()
