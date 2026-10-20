"""Regression coverage for checkpoint-derived tensors discarded by level 2."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.model_loader.weight_memory_saver import (
    current_model_scope,
    model_build_scope,
)
from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.modules.dsv4 import rope
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8, _v4_fp8_linear
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8
from rtp_llm.models_py.modules.dsv4.utils import LinearFactory
from rtp_llm.models_py.modules.dsv4.utils import _v4_fp8_linear as model_fp8_linear
from rtp_llm.models_py.modules.dsv4.utils import iter_fp8_linears


class SleepComputedWeightsTest(unittest.TestCase):

    def test_attention_and_compressor_registration_is_scoped_and_weak(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import attention, compressor

        for module, register_name, registry_name in (
            (attention, "_register_attention", "_ATTENTION_REGISTRY"),
            (compressor, "_register_compressor", "_COMPRESSOR_REGISTRY"),
        ):
            with self.subTest(module=module.__name__):
                obj = torch.nn.Module()
                with mock.patch.object(module, registry_name, weakref.WeakSet()):
                    with model_build_scope("draft"):
                        getattr(module, register_name)(obj)
                    self.assertEqual(obj._sleep_model_scope, "draft")
                    self.assertIn(obj, getattr(module, registry_name))
                    ref = weakref.ref(obj)
                    del obj
                    gc.collect()
                    self.assertIsNone(ref())
                    self.assertEqual(list(getattr(module, registry_name)), [])

    def test_attention_and_compressor_registration_failures_propagate(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import attention, compressor

        for module, register_name, registry_name in (
            (attention, "_register_attention", "_ATTENTION_REGISTRY"),
            (compressor, "_register_compressor", "_COMPRESSOR_REGISTRY"),
        ):
            register = getattr(module, register_name)
            with self.subTest(module=module.__name__, stage="scope"):
                with mock.patch(
                    "rtp_llm.model_loader.weight_memory_saver.current_model_scope",
                    side_effect=RuntimeError("scope failed"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "scope failed"):
                        register(torch.nn.Module())
            with self.subTest(module=module.__name__, stage="weak registration"):
                with mock.patch.object(module, registry_name) as registry:
                    registry.add.side_effect = TypeError("weak registration failed")
                    with self.assertRaisesRegex(TypeError, "weak registration failed"):
                        register(torch.nn.Module())

    def test_deferred_extra_weights_reopen_the_owning_model_scope(self):
        seen = []
        weights = object()

        def load(value):
            self.assertIs(value, weights)
            seen.append(current_model_scope())

        model = SimpleNamespace(_build_scope_token="draft", _load_extra_weights=load)
        with model_build_scope("worker_previous_scope"):
            DeepSeekV4Model._load_scoped_extra_weights(model, weights)
            self.assertEqual(current_model_scope(), "worker_previous_scope")
        self.assertEqual(seen, ["draft"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for scale packing")
    def test_mtp_packed_fp8_scale_rebuild_preserves_storage(self):
        def create(weights, weight_key, scale_key, **kwargs):
            linear = torch.nn.Module()
            linear.weight = weights[weight_key]
            linear.weight_scales = weights[scale_key]
            return linear

        raw_w = torch.zeros((128, 512), device="cuda", dtype=torch.float8_e4m3fn)
        raw_s = torch.full((1, 4), 2.0, device="cuda").to(torch.float8_e8m0fnu)
        with mock.patch.object(LinearFactory, "create_linear_from_weights", create):
            with model_build_scope("draft"):
                linear = model_fp8_linear(raw_w, raw_s)
        self.assertEqual(linear.weight_scales.dtype, torch.int32)
        expected = linear.weight_scales.clone()
        ptr = linear.weight_scales.data_ptr()
        for _ in range(2):
            linear.weight_scales.zero_()
            AttentionFP8._reload_linear_scale(linear)
            torch.testing.assert_close(linear.weight_scales, expected, rtol=0, atol=0)
            self.assertEqual(ptr, linear.weight_scales.data_ptr())

    def test_model_level_mtp_linears_are_scoped_and_restore_in_place(self):
        # e_proj/h_proj are constructed by the same factory, but do not belong
        # to an AttentionFP8. They must still participate in level-2 restore.
        def create(weights, weight_key, scale_key, **kwargs):
            linear = torch.nn.Module()
            linear.weight = weights[weight_key].clone()
            linear.weight_scales = weights[scale_key].clone()
            return linear

        raw_w = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        raw_s = torch.arange(4, dtype=torch.int32).reshape(4, 1)
        with mock.patch.object(LinearFactory, "create_linear_from_weights", create):
            with model_build_scope("target"):
                target = model_fp8_linear(raw_w, raw_s)
            with model_build_scope("draft"):
                draft = model_fp8_linear(raw_w, raw_s)
        self.assertEqual(target._sleep_model_scope, "target")
        self.assertEqual(draft._sleep_model_scope, "draft")
        self.assertIn(draft, iter_fp8_linears())
        ptrs = draft.weight.data_ptr(), draft.weight_scales.data_ptr()
        for _ in range(2):
            draft.weight.zero_()
            draft.weight_scales.zero_()
            AttentionFP8._reload_linear_scale(draft)
            torch.testing.assert_close(draft.weight, raw_w, rtol=0, atol=0)
            torch.testing.assert_close(draft.weight_scales, raw_s, rtol=0, atol=0)
            self.assertEqual(
                ptrs, (draft.weight.data_ptr(), draft.weight_scales.data_ptr())
            )
        ref = weakref.ref(draft)
        del draft
        gc.collect()
        self.assertIsNone(ref())

    def test_attention_linear_retains_checkpoint_sources_for_owner_reload(self):
        def create(weights, weight_key, scale_key, **kwargs):
            linear = torch.nn.Module()
            linear.weight = weights[weight_key].clone()
            linear.weight_scales = weights[scale_key].clone()
            return linear

        weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        scale = torch.arange(4, dtype=torch.int32).reshape(4, 1)
        with mock.patch.object(LinearFactory, "create_linear_from_weights", create):
            linear = _v4_fp8_linear(weight, scale)
        self.assertIs(linear._sleep_raw_weight_source, weight)
        self.assertIs(linear._sleep_raw_scale_source, scale)
        pointers = linear.weight.data_ptr(), linear.weight_scales.data_ptr()
        linear.weight.zero_()
        linear.weight_scales.zero_()
        AttentionFP8._reload_linear_scale(linear)
        torch.testing.assert_close(linear.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(linear.weight_scales, scale, rtol=0, atol=0)
        self.assertEqual(
            pointers, (linear.weight.data_ptr(), linear.weight_scales.data_ptr())
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for device RoPE")
    def test_rope_reload_ignores_meta_cache_and_preserves_shared_storage(self):
        params = (8, 16, 0, 10000.0, 1.0, 32, 1)
        with mock.patch.object(rope, "_FREQS_CIS_CACHE", {}):
            with torch.device("meta"):
                self.assertEqual(rope.precompute_freqs_cis(*params).device.type, "meta")
            expected = rope.precompute_freqs_cis(
                *params, device=torch.device("cpu")
            ).cuda()
            obj = SimpleNamespace(
                freqs_cis=expected.clone(),
                _rope_dim=8,
                _rope_max_seq_len=16,
                _rope_o_seq_len=0,
                _rope_base=10000.0,
                _rope_factor=1.0,
                _rope_beta_fast=32,
                _rope_beta_slow=1,
            )
            ptr = obj.freqs_cis.data_ptr()
            obj.freqs_cis.zero_()
            seen = set()
            AttentionFP8.reload_sleep_computed_weights(obj, seen)
            torch.testing.assert_close(obj.freqs_cis, expected, rtol=0, atol=0)
            self.assertEqual(ptr, obj.freqs_cis.data_ptr())
            self.assertIn(id(obj.freqs_cis), seen)
            comp = SimpleNamespace(
                freqs_cis=obj.freqs_cis,
                _cos_sin_cache=torch.zeros((16, 8), device="cuda"),
            )
            cache_ptr = comp._cos_sin_cache.data_ptr()
            cache_seen = set()
            CompressorFP8.reload_rope_cache(comp, cache_seen)
            torch.testing.assert_close(
                comp._cos_sin_cache,
                torch.cat([expected.real, expected.imag], dim=-1),
                rtol=0,
                atol=0,
            )
            self.assertEqual(cache_ptr, comp._cos_sin_cache.data_ptr())
            self.assertIn(id(comp._cos_sin_cache), cache_seen)

    def test_prepacked_tp_linear_restores_weight_and_scale_in_place(self):
        raw_w = torch.arange(512 * 1024, dtype=torch.float32).reshape(512, 1024)
        raw_s = torch.arange(512 * 2, dtype=torch.int32).reshape(512, 2)
        rows, cols = slice(128, 256), slice(512, 1024)
        expected_w = raw_w[rows, cols].contiguous()
        expected_s = raw_s[rows, 1:2].contiguous()
        linear = SimpleNamespace(
            weight=torch.empty_like(expected_w),
            weight_scales=torch.empty_like(expected_s),
            _sleep_raw_weight_source=raw_w,
            _sleep_raw_scale_source=raw_s,
            _sleep_row_slice=rows,
            _sleep_col_slice=cols,
        )
        ptrs = linear.weight.data_ptr(), linear.weight_scales.data_ptr()
        for _ in range(2):
            linear.weight.zero_()
            linear.weight_scales.zero_()
            AttentionFP8._reload_linear_scale(linear)
            torch.testing.assert_close(linear.weight, expected_w, rtol=0, atol=0)
            torch.testing.assert_close(linear.weight_scales, expected_s, rtol=0, atol=0)
            self.assertEqual(
                ptrs, (linear.weight.data_ptr(), linear.weight_scales.data_ptr())
            )

    def test_indexer_refolds_projection_without_replacing_storage(self):
        source = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
        obj = SimpleNamespace(
            _sleep_weights_proj_src=source,
            _sleep_weights_proj_scale=0.125,
            weights_proj=torch.zeros_like(source),
            compressor=None,
        )
        ptr = obj.weights_proj.data_ptr()
        IndexerFP8.reload_sleep_computed_weights(obj)
        torch.testing.assert_close(obj.weights_proj, source * 0.125, rtol=0, atol=0)
        self.assertEqual(ptr, obj.weights_proj.data_ptr())


if __name__ == "__main__":
    unittest.main()
