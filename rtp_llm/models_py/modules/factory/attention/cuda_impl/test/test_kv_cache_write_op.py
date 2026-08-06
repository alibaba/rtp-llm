"""Unit tests for PyFlashinfer's paged KV cache write operation."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.ops.compute_ops import LayerKVCache


class KVCacheWriteOpDtypeTest(unittest.TestCase):
    def _make_inputs(self, magnitude: float = 1.0):
        key = torch.randn((1, 2, 4), dtype=torch.bfloat16) * magnitude
        value = torch.randn((1, 2, 4), dtype=torch.bfloat16) * magnitude
        return key, value

    def _make_op(self, kv_cache_dtype=None):
        op = KVCacheWriteOp(
            num_kv_heads=2,
            head_size=4,
            token_per_block=8,
            kv_cache_dtype=kv_cache_dtype,
        )
        op.set_params(
            SimpleNamespace(
                batch_indice_d=torch.tensor([0], dtype=torch.int32),
                positions_d=torch.tensor([0], dtype=torch.int32),
                page_indice_d=torch.tensor([0], dtype=torch.int32),
                decode_page_indptr_d=torch.tensor([0, 1], dtype=torch.int32),
                paged_kv_last_page_len_d=torch.tensor([1], dtype=torch.int32),
            )
        )
        return op

    def _run_write(self, cache_dtype: torch.dtype, magnitude: float = 1.0):
        op = self._make_op()
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = torch.empty((1, 2, 2, 8, 4), dtype=cache_dtype)
        if cache_dtype == torch.float8_e4m3fn:
            kv_cache.kv_scale_base = torch.ones((1, 32), dtype=torch.float32)
        key, value = self._make_inputs(magnitude)

        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op.page.append_paged_kv_cache"
        ) as append:
            op.forward(key, value, kv_cache)

        written_key, written_value = append.call_args.args[:2]
        return key, value, written_key, written_value

    def test_fp8_cache_casts_bf16_inputs_before_append(self):
        _, _, written_key, written_value = self._run_write(torch.float8_e4m3fn)
        self.assertEqual(written_key.dtype, torch.float8_e4m3fn)
        self.assertEqual(written_value.dtype, torch.float8_e4m3fn)

    def test_bf16_cache_preserves_input_tensors(self):
        key, value, written_key, written_value = self._run_write(torch.bfloat16)
        self.assertIs(written_key, key)
        self.assertIs(written_value, value)

    def test_fp8_cache_clamps_outliers_to_finite_values(self):
        _, _, written_key, written_value = self._run_write(
            torch.float8_e4m3fn, magnitude=1e4
        )
        limit = torch.finfo(torch.float8_e4m3fn).max
        for written in (written_key, written_value):
            self.assertFalse(written.float().isnan().any())
            self.assertLessEqual(written.float().abs().max().item(), limit)

    def test_non_fp8_dtype_mismatch_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "only converts activations for FP8"):
            self._run_write(torch.float16)

    def test_fp8_warmup_uses_real_cache_dtype(self):
        op = self._make_op(torch.float8_e4m3fn)
        key, value = self._make_inputs()
        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op.page.append_paged_kv_cache"
        ) as append:
            op.forward(key, value, None)

        written_key, written_value, _, _, caches = append.call_args.args[:5]
        self.assertEqual(written_key.dtype, torch.float8_e4m3fn)
        self.assertEqual(written_value.dtype, torch.float8_e4m3fn)
        self.assertEqual(caches[0].dtype, torch.float8_e4m3fn)
        self.assertEqual(caches[1].dtype, torch.float8_e4m3fn)

    def test_fp8_cache_requires_scale_buffer(self):
        op = self._make_op()
        key, value = self._make_inputs()
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = torch.empty((1, 2, 2, 8, 4), dtype=torch.float8_e4m3fn)

        with self.assertRaisesRegex(RuntimeError, "requires an initialized"):
            op.forward(key, value, kv_cache)

    def test_fp8_cache_rejects_empty_scale_buffer(self):
        op = self._make_op()
        key, value = self._make_inputs()
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = torch.empty((1, 2, 2, 8, 4), dtype=torch.float8_e4m3fn)
        kv_cache.kv_scale_base = torch.empty(0, dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "requires an initialized"):
            op.forward(key, value, kv_cache)

    def test_fp8_cache_does_not_synchronize_scale_view_in_hot_path(self):
        op = self._make_op()
        key, value = self._make_inputs()
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = torch.empty((1, 2, 2, 8, 4), dtype=torch.float8_e4m3fn)
        scale_storage = torch.ones((32,), dtype=torch.float32)
        kv_cache.kv_scale_base = scale_storage.view(1, 32)

        with (
            mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("sync")),
            mock.patch(
                "rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op.page.append_paged_kv_cache"
            ),
        ):
            op.forward(key, value, kv_cache)
            kv_cache.kv_scale_base = scale_storage.view(1, 32)
            op.forward(key, value, kv_cache)

    def test_fp8_cache_rejects_configured_dtype_mismatch(self):
        op = self._make_op(torch.bfloat16)
        key, value = self._make_inputs()
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = torch.empty((1, 2, 2, 8, 4), dtype=torch.float8_e4m3fn)
        kv_cache.kv_scale_base = torch.ones((1, 32), dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "dtype mismatch"):
            op.forward(key, value, kv_cache)

    def test_non_fp8_cache_rejects_configured_fp8_dtype(self):
        op = self._make_op(torch.float8_e4m3fn)
        key, value = self._make_inputs()
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = torch.empty((1, 2, 2, 8, 4), dtype=torch.bfloat16)

        with self.assertRaisesRegex(RuntimeError, "dtype mismatch"):
            op.forward(key, value, kv_cache)

    def test_fp8_scale_warning_is_emitted_once(self):
        with (
            mock.patch.object(KVCacheWriteOp, "_fp8_scale_warning_emitted", False),
            self.assertLogs(
                "rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op",
                level="WARNING",
            ) as logs,
        ):
            self._make_op(torch.float8_e4m3fn)
            self._make_op(torch.float8_e4m3fn)

        self.assertEqual(len(logs.output), 1)
        self.assertIn("implicit scale=1", logs.output[0])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_real_fp8_append_writes_single_and_cross_page_tokens(self):
        num_tokens = 9
        head_size = 64
        op = KVCacheWriteOp(num_kv_heads=2, head_size=head_size, token_per_block=8)
        op.set_params(
            SimpleNamespace(
                batch_indice_d=torch.zeros(
                    num_tokens, dtype=torch.int32, device="cuda"
                ),
                positions_d=torch.arange(num_tokens, dtype=torch.int32, device="cuda"),
                page_indice_d=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
                decode_page_indptr_d=torch.tensor(
                    [0, 2], dtype=torch.int32, device="cuda"
                ),
                paged_kv_last_page_len_d=torch.tensor(
                    [1], dtype=torch.int32, device="cuda"
                ),
            )
        )
        key = torch.linspace(
            -600,
            600,
            num_tokens * 2 * head_size,
            dtype=torch.bfloat16,
            device="cuda",
        ).reshape(num_tokens, 2, head_size)
        value = -key
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = torch.zeros(
            (2, 2, 2, 8, head_size), dtype=torch.float8_e4m3fn, device="cuda"
        )
        kv_cache.kv_scale_base = torch.ones((2, 32), dtype=torch.float32, device="cuda")

        op.forward(key, value, kv_cache)

        written_key = torch.cat(
            (
                kv_cache.kv_cache_base[0, 0].permute(1, 0, 2),
                kv_cache.kv_cache_base[1, 0, :, :1, :].permute(1, 0, 2),
            )
        )
        written_value = torch.cat(
            (
                kv_cache.kv_cache_base[0, 1].permute(1, 0, 2),
                kv_cache.kv_cache_base[1, 1, :, :1, :].permute(1, 0, 2),
            )
        )
        # Keep the oracle independent from the production helper under test.
        expected_key = key.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        expected_value = value.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        torch.testing.assert_close(written_key.float(), expected_key.float())
        torch.testing.assert_close(written_value.float(), expected_value.float())
        torch.testing.assert_close(
            kv_cache.kv_scale_base, torch.ones_like(kv_cache.kv_scale_base)
        )


if __name__ == "__main__":
    unittest.main()
