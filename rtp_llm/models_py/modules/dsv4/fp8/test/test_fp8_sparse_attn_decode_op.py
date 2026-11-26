"""Unit tests for the FP8 sparse decode FlashMLA wrapper."""

from __future__ import annotations

import importlib
import sys
import threading
import types
import unittest
from unittest.mock import patch

import torch

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


if __name__ == "__main__":
    unittest.main()
