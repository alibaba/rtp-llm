"""SM120 cases split out of test_fp8_sparse_attn_decode_op.py.
These cases live in this sibling file, leaving the original test file
untouched."""

from __future__ import annotations

from unittest.mock import patch

from rtp_llm.models_py.modules.dsv4.fp8 import (
    attention_backend,
    backend_selection,
    sm120_sparse_mla,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode import fp8_sparse_attn_decode_op

"""Unit tests for the FP8 sparse decode FlashMLA wrapper."""
import contextlib
import sys
import types
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class TestSparseAttnV4DecodeFp8OpSm120(unittest.TestCase):
    def setUp(self):
        for cached in (
            backend_selection._is_cuda_runtime,
            backend_selection._is_sm120_index,
            attention_backend._backend_for_cuda,
        ):
            cached.cache_clear()
            self.addCleanup(cached.cache_clear)
        runtime = patch.object(backend_selection, "is_cuda", return_value=True)
        runtime.start()
        self.addCleanup(runtime.stop)

    """Sibling of ``TestSparseAttnV4DecodeFp8Op`` carrying the SM120 cases."""

    def test_decode_sink_conversion_tracks_inplace_updates(self):
        fallback_width = sm120_sparse_mla.SM120_SWA_TOPK_WIDTHS[-1] + 128
        query = torch.zeros(1, 1, 8, 512, dtype=torch.bfloat16)
        cache = torch.zeros(2, 64, 584, dtype=torch.uint8)
        indices = torch.zeros(1, 1, fallback_width, dtype=torch.int32)
        lengths = torch.tensor([1], dtype=torch.int32)

        def passed_sink(sink):
            with patch.object(sm120_sparse_mla, "run_chunked_reference") as runner:
                sm120_sparse_mla.forward_decode(
                    query, cache, sink, indices, lengths, softmax_scale=1.0
                )
            return runner.call_args.kwargs["sinks"]

        for mode in (contextlib.nullcontext, torch.inference_mode):
            with self.subTest(mode=mode.__name__), mode():
                source = torch.arange(8, dtype=torch.bfloat16)
                first = passed_sink(source)
                source.add_(1)
                second = passed_sink(source)
                self.assertEqual(first.dtype, torch.float32)
                torch.testing.assert_close(first, torch.arange(8.0))
                torch.testing.assert_close(second, torch.arange(8.0) + 1)
                fp32_sink = torch.arange(8, dtype=torch.float32)
                self.assertIs(passed_sink(fp32_sink), fp32_sink)

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
                    import_module.assert_called_once_with(
                        "rtp_llm.models_py.utils.flash_mla_legacy"
                    )
        finally:
            fp8_sparse_attn_decode_op._FLASH_MLA_AVAILABLE = old_available
            fp8_sparse_attn_decode_op._flash_mla_load_attempted = old_attempted

    def test_sm120_packs_decode_slots_to_paged_caches(self):
        # The kernel derives page geometry from the cache tensor and requires
        # page-64 (extra pool also page-2) layout, so the op gathers the
        # selected pool slots into packed caches and passes those to ``run``.
        # The pack helper is mocked: this unit verifies only the cache ABI.
        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla._load_sm120_sparse_mla",
            return_value=object(),
        ):
            op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        query = torch.zeros(1, 1, 8, 512, dtype=torch.bfloat16)
        sink = torch.zeros(8, dtype=torch.float32)
        swa_cache = torch.zeros(2, 64, 584, dtype=torch.uint8)
        extra_cache = torch.zeros(3, 2, 584, dtype=torch.uint8)
        swa_indices = torch.tensor([[[1, -1, 3]]], dtype=torch.int32)
        extra_indices = torch.tensor([[[2, -1]]], dtype=torch.int32)
        packed_swa = torch.zeros(1, 64, 584, dtype=torch.uint8)
        packed_extra = torch.zeros(1, 2, 584, dtype=torch.uint8)
        remap_swa = torch.zeros(1, 3, dtype=torch.int32)
        remap_extra = torch.zeros(1, 2, dtype=torch.int32)
        packs = []

        def fake_pack(cache, indices, page_size, row_lens=None):
            packs.append((cache, indices.clone(), page_size, row_lens))
            if cache is swa_cache:
                return packed_swa, remap_swa
            return packed_extra, remap_extra

        with patch("rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla.warmup"), patch(
            "rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla.run"
        ) as mock_run, patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton": (
                    types.SimpleNamespace(pack_slots_to_paged=fake_pack)
                )
            },
        ):
            out = sm120_sparse_mla.forward_decode(
                query,
                swa_cache,
                sink,
                swa_indices,
                torch.tensor([3], dtype=torch.int32),
                extra_cache,
                extra_indices,
                torch.tensor([2], dtype=torch.int32),
                softmax_scale=op.softmax_scale,
            )

        self.assertEqual(tuple(out.shape), tuple(query.shape))
        self.assertEqual(len(packs), 2)
        self.assertIs(packs[0][0], swa_cache)
        self.assertEqual(packs[0][2], 64)
        self.assertIs(packs[1][0], extra_cache)
        self.assertEqual(packs[1][2], 2)
        self.assertIs(mock_run.call_args.kwargs["swa_cache"], packed_swa)
        self.assertIs(mock_run.call_args.kwargs["swa_indices"], remap_swa)
        self.assertIs(mock_run.call_args.kwargs["extra_cache"], packed_extra)
        self.assertIs(mock_run.call_args.kwargs["extra_indices"], remap_extra)

    def test_sm120_rejects_unsupported_heads_before_loading_provider(self):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            backend_selection, "is_sm120", return_value=True
        ), patch.object(sm120_sparse_mla, "_load_sm120_sparse_mla") as load:
            for heads in (1, 2, 4, 12, 256):
                with self.subTest(heads=heads), self.assertRaisesRegex(
                    ValueError, "local query heads"
                ):
                    SparseAttnV4DecodeFp8Op(heads, 512, 1.0).bind_device(
                        torch.device("cuda:0")
                    )
            load.assert_not_called()

    def test_sm120_remaps_topk_padding_to_first_entry(self):
        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla._load_sm120_sparse_mla",
            return_value=object(),
        ):
            op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        query = torch.zeros(1, 1, 8, 512, dtype=torch.bfloat16)
        sink = torch.zeros(8, dtype=torch.float32)
        swa_cache = torch.zeros(2, 64, 584, dtype=torch.uint8)
        swa_indices = torch.full((1, 1, 128), -1, dtype=torch.int32)
        swa_indices[0, 0, :2] = torch.tensor([5, 7], dtype=torch.int32)

        packed_inputs = []

        def fake_pack(cache, indices, page_size, row_lens=None):
            packed_inputs.append(indices.clone())
            return cache, indices

        with patch("rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla.warmup"), patch(
            "rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla.run"
        ) as mock_run, patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton": (
                    types.SimpleNamespace(pack_slots_to_paged=fake_pack)
                )
            },
        ):
            sm120_sparse_mla.forward_decode(
                query,
                swa_cache,
                sink,
                swa_indices,
                torch.tensor([2], dtype=torch.int32),
                softmax_scale=op.softmax_scale,
            )

        # Padding is remapped to the row's first live slot before packing.
        self.assertEqual(len(packed_inputs), 1)
        self.assertTrue(
            torch.equal(
                packed_inputs[0][:, :2], torch.tensor([[5, 7]], dtype=torch.int32)
            )
        )
        self.assertTrue((packed_inputs[0][:, 2:] == 5).all())

        passed = mock_run.call_args.kwargs["swa_indices"]
        self.assertTrue(
            torch.equal(passed[:, :2], torch.tensor([[5, 7]], dtype=torch.int32))
        )
        self.assertTrue((passed[:, 2:] == 5).all())
        self.assertTrue(
            torch.equal(
                mock_run.call_args.kwargs["swa_lens"],
                torch.tensor([2], dtype=torch.int32),
            )
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
