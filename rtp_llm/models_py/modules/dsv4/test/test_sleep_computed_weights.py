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
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    _v4_fp8_linear,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8
from rtp_llm.models_py.modules.dsv4.moe.mega_se_buf import (
    iter_mega_se_strategies,
    register_mega_se_strategy,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies.mega_se import MegaMoEStrategySE
from rtp_llm.models_py.modules.dsv4.utils import (
    LinearFactory,
)
from rtp_llm.models_py.modules.dsv4.utils import _v4_fp8_linear as model_fp8_linear
from rtp_llm.models_py.modules.dsv4.utils import (
    iter_fp8_linears,
)
from rtp_llm.utils.model_weight import W


class SleepComputedWeightsTest(unittest.TestCase):
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
                target = _v4_fp8_linear(raw_w, raw_s)
            with model_build_scope("draft"):
                draft = model_fp8_linear(raw_w, raw_s)
        self.assertIs(_v4_fp8_linear, model_fp8_linear)
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

    def test_se_registry_is_weak_and_scoped_separately_for_target_and_draft(self):
        target, draft = torch.nn.Module(), torch.nn.Module()
        with model_build_scope("target"):
            register_mega_se_strategy(target)
        with model_build_scope("draft"):
            register_mega_se_strategy(draft)
        self.assertEqual(target._sleep_model_scope, "target")
        self.assertEqual(draft._sleep_model_scope, "draft")
        self.assertIn(target, iter_mega_se_strategies())
        ref = weakref.ref(target)
        del target
        gc.collect()
        self.assertIsNone(ref())

    def test_se_reload_requires_all_four_popped_shared_tensors(self):
        self.assertEqual(
            MegaMoEStrategySE.sleep_reload_extra_weight_names(),
            {
                W.v4_shared_w13_w,
                W.v4_shared_w13_s,
                W.v4_shared_w2_w,
                W.v4_shared_w2_s,
            },
        )

    def test_se_shared_transform_restores_all_buffers_at_stable_addresses(self):
        # Exercise the real ownership/reload method; only the CUDA-only kernel
        # is replaced by a deterministic transform for this CPU unit test.
        def transform(l1, l2):
            return ((l1[0].clone(), l1[1].clone()), (l2[0], l2[1]))

        dg = SimpleNamespace(transform_weights_for_mega_moe=transform)
        obj = MegaMoEStrategySE.__new__(MegaMoEStrategySE)
        torch.nn.Module.__init__(obj)
        weights = {
            W.v4_shared_w13_w: torch.arange(64).reshape(8, 8).to(torch.float8_e4m3fn),
            W.v4_shared_w13_s: torch.arange(8, dtype=torch.int32).reshape(8, 1),
            W.v4_shared_w2_w: torch.arange(32).reshape(8, 4).to(torch.float8_e4m3fn),
            W.v4_shared_w2_s: torch.arange(8, dtype=torch.int32).reshape(8, 1),
        }
        with mock.patch.object(torch.cuda, "empty_cache"):
            obj._setup_shared_expert_weights(dict(weights), dg, W, 8, 4)
            names = ("_se_l1_w", "_se_l1_sf", "_se_l2_w", "_se_l2_sf")
            expected = {name: getattr(obj, name).clone() for name in names}
            ptrs = {name: getattr(obj, name).data_ptr() for name in names}
            for _ in range(2):
                for name in names:
                    getattr(obj, name).zero_()
                obj._setup_shared_expert_weights(
                    dict(weights), dg, W, 8, 4, isolate_scratch=True
                )
                for name in names:
                    self.assertTrue(
                        torch.equal(
                            getattr(obj, name).view(torch.uint8),
                            expected[name].view(torch.uint8),
                        ),
                        name,
                    )
                    self.assertEqual(ptrs[name], getattr(obj, name).data_ptr())


if __name__ == "__main__":
    unittest.main()
