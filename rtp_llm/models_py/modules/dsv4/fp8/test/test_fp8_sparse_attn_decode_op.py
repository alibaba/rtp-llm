"""Unit tests for the FP8 sparse decode FlashMLA wrapper."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode import fp8_sparse_attn_decode_op
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class TestSparseAttnV4DecodeFp8Op(unittest.TestCase):
    def test_flash_mla_optional_wheel_failures_are_lazy_and_nonfatal(self):
        old_available = fp8_sparse_attn_decode_op._FLASH_MLA_AVAILABLE
        old_attempted = fp8_sparse_attn_decode_op._flash_mla_load_attempted
        try:
            for error in (OSError("ABI mismatch"), RuntimeError("load failure")):
                with self.subTest(error=type(error).__name__):
                    fp8_sparse_attn_decode_op._FLASH_MLA_AVAILABLE = False
                    fp8_sparse_attn_decode_op._flash_mla_load_attempted = False
                    with patch.object(torch.version, "cuda", "12.9"), patch.object(
                        fp8_sparse_attn_decode_op.importlib,
                        "import_module",
                        side_effect=error,
                    ) as import_module:
                        self.assertFalse(fp8_sparse_attn_decode_op._load_flash_mla())
                        self.assertFalse(fp8_sparse_attn_decode_op._load_flash_mla())
                    import_module.assert_called_once_with("flash_mla")
        finally:
            fp8_sparse_attn_decode_op._FLASH_MLA_AVAILABLE = old_available
            fp8_sparse_attn_decode_op._flash_mla_load_attempted = old_attempted

    def test_sm120_passes_original_paged_caches_without_static_width_copy(self):
        # This unit verifies only the cache ABI.  Keep construction independent
        # of whether the host running the Python test has the optional CUDA 13
        # FlashInfer wheel installed; the dedicated hardware target exercises
        # the real entry point.
        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8.decode."
            "fp8_sparse_attn_decode_op._load_sm120_sparse_mla",
            return_value=object(),
        ):
            op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        query = torch.zeros(1, 1, 8, 512, dtype=torch.bfloat16)
        sink = torch.zeros(8, dtype=torch.float32)
        swa_cache = torch.zeros(2, 64, 584, dtype=torch.uint8)
        extra_cache = torch.zeros(3, 2, 584, dtype=torch.uint8)
        swa_indices = torch.tensor([[[1, -1, 3]]], dtype=torch.int32)
        extra_indices = torch.tensor([[[2, -1]]], dtype=torch.int32)

        with patch("rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla.warmup"), patch(
            "rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla.run"
        ) as mock_run:
            out = op._forward_sm120_flashinfer(
                query,
                swa_cache,
                sink,
                swa_indices,
                torch.tensor([3], dtype=torch.int32),
                extra_cache,
                extra_indices,
                torch.tensor([2], dtype=torch.int32),
            )

        self.assertEqual(tuple(out.shape), tuple(query.shape))
        self.assertIs(mock_run.call_args.kwargs["swa_cache"], swa_cache)
        self.assertIs(mock_run.call_args.kwargs["extra_cache"], extra_cache)

    def test_attn_sink_cache_tracks_source_identity_and_inplace_updates(self):
        op = SparseAttnV4DecodeFp8Op(
            n_heads=2,
            head_dim=128,
            softmax_scale=1.0,
        )
        source = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

        first = op._cached_attn_sink(source)
        self.assertIs(first, op._cached_attn_sink(source))

        source.add_(1)
        second = op._cached_attn_sink(source)
        self.assertIsNot(first, second)
        torch.testing.assert_close(second, torch.tensor([2.0, 3.0]))

        replacement = source.clone()
        third = op._cached_attn_sink(replacement)
        self.assertIsNot(second, third)

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
            topk = (
                torch.arange(128, dtype=torch.int32).view(1, 1, 128).expand(2, 3, 128)
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
