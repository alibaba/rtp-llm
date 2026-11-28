"""Unit tests for the FP8 sparse decode FlashMLA wrapper."""

from __future__ import annotations

import contextlib
import importlib
import sys
import threading
import types
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import (
    attention_backend,
    backend_selection,
    sm120_sparse_mla,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode import fp8_sparse_attn_decode_op
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class _TrackingLock:
    def __init__(self):
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._lock.acquire()
        return self

    def __exit__(self, *args):
        self._lock.release()
        return False


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
        old_flash_mla_legacy = sys.modules.get("flash_mla_legacy")
        # The production shim prefers the isolated legacy wheel; alias the
        # fake under both names so the mock still intercepts it.
        sys.modules["flash_mla"] = fake_flash_mla
        sys.modules["flash_mla_legacy"] = fake_flash_mla
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
        if old_flash_mla_legacy is None:
            sys.modules.pop("flash_mla_legacy", None)
        else:
            sys.modules["flash_mla_legacy"] = old_flash_mla_legacy

    def test_concurrent_first_probe_waits_for_flash_mla_load(self):
        entered = threading.Event()
        release = threading.Event()
        tracking_lock = _TrackingLock()

        fake_flash_mla = types.ModuleType("rtp_llm.models_py.utils.flash_mla_legacy")
        fake_flash_mla.flash_mla_with_kvcache = object()
        fake_flash_mla.get_mla_metadata = object()

        real_import_module = importlib.import_module

        def slow_import(name, package=None):
            if name == "rtp_llm.models_py.utils.flash_mla_legacy":
                entered.set()
                assert release.wait(timeout=5)
                return fake_flash_mla
            return real_import_module(name, package)

        saved_available = fp8_sparse_attn_decode_op._FLASH_MLA_AVAILABLE
        saved_attempted = fp8_sparse_attn_decode_op._flash_mla_load_attempted
        try:
            fp8_sparse_attn_decode_op._FLASH_MLA_AVAILABLE = False
            fp8_sparse_attn_decode_op._flash_mla_load_attempted = False
            results = [None, None]
            errors = []

            def worker(idx):
                try:
                    results[idx] = fp8_sparse_attn_decode_op._load_flash_mla()
                except BaseException as exc:  # pragma: no cover
                    errors.append(exc)

            with (
                patch.object(
                    fp8_sparse_attn_decode_op, "_FLASH_MLA_LOCK", tracking_lock
                ),
                patch.object(torch.version, "cuda", "12.9"),
                patch.object(importlib, "import_module", side_effect=slow_import),
            ):
                t0 = threading.Thread(target=worker, args=(0,))
                t0.start()
                self.assertTrue(entered.wait(timeout=5))
                t1 = threading.Thread(target=worker, args=(1,))
                t1.start()
                self.assertTrue(tracking_lock.contended.wait(timeout=5))
                release.set()
                t0.join(timeout=5)
                t1.join(timeout=5)
            self.assertEqual(errors, [])
            self.assertEqual(results, [True, True])
        finally:
            fp8_sparse_attn_decode_op._FLASH_MLA_AVAILABLE = saved_available
            fp8_sparse_attn_decode_op._flash_mla_load_attempted = saved_attempted


    def test_concurrent_first_bind_serializes_device_binding(self):
        op = fp8_sparse_attn_decode_op.SparseAttnV4DecodeFp8Op(
            n_heads=16, head_dim=512, softmax_scale=0.1
        )
        entered = threading.Event()
        release = threading.Event()
        call_count = 0

        def slow_prepare(device):
            nonlocal call_count
            call_count += 1
            entered.set()
            assert release.wait(timeout=5)
            return (lambda *args, **kwargs: torch.empty(0), False)

        errors = []
        with patch.object(op, "_prepare_backend", side_effect=slow_prepare):
            t0 = threading.Thread(
                target=lambda: op.bind_device(torch.device("cpu"))
            )

            def second_bind():
                try:
                    op.bind_device(torch.device("meta"))
                except BaseException as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=second_bind)
            t0.start()
            self.assertTrue(entered.wait(timeout=5))
            t1.start()
            release.set()
            t0.join(timeout=5)
            t1.join(timeout=5)

        self.assertEqual(call_count, 1)
        self.assertEqual(op._device, torch.device("cpu"))
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)

    def test_get_sched_meta_enters_query_cuda_device_context(self):
        op = fp8_sparse_attn_decode_op.SparseAttnV4DecodeFp8Op(
            n_heads=16, head_dim=512, softmax_scale=0.1
        )
        op._device = torch.device("cuda", 1)
        op._uses_flash_mla = True
        entered = []
        fake_flash_mla = types.SimpleNamespace(
            get_mla_metadata=lambda **kwargs: (object(), None)
        )
        metadata = types.SimpleNamespace(sched_meta_cache={}, pool_block_tables={})
        from contextlib import nullcontext

        with (
            patch.dict(
                sys.modules,
                {"rtp_llm.models_py.utils.flash_mla_legacy": fake_flash_mla},
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.cuda.device",
                side_effect=lambda dev: (entered.append(dev), nullcontext())[1],
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
        ):
            op.get_sched_meta(
                torch.device("cuda", 1),
                metadata,
                batch_size=2,
                q_len=1,
                topk=128,
                extra_attn_type="compressed",
            )
        self.assertTrue(entered)
        self.assertTrue(all(d == torch.device("cuda", 1) for d in entered))
        self.assertTrue(metadata._sched_meta_capturing)


class TestSparseAttnV4DecodeFp8OpSm120(unittest.TestCase):
    """SM120 decode backend & slot-packing contract cases."""

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

        owner = sm120_sparse_mla.Sm120SparseMlaWorkspace()
        with patch(
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
                workspace_owner=owner,
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
        self.assertIs(mock_run.call_args.kwargs["workspace_owner"], owner)

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

        with patch(
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
    unittest.main()
