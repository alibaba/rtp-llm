"""Unit tests for inline FP8 quantization during fastsafetensors loading.

Covers:
  - TensorCollector: FP8 pre-quantization storage (store/load/clear)
  - _build_stacked_key_config: no overwrite when kernel and scale share checkpoint key
  - per_channel_cast_to_fp8_expert: matches 3D per_channel_cast_to_fp8 semantics
  - _load_moe_inline_quant: transpose_stack_moe_w1 gate/up swap
  - LoadQuantPerChannelFp8Weight.get_tensor_names: excludes scale keys
"""

import functools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.quant_config import Fp8PerChannelCompressedQuantConfig
from rtp_llm.model_loader.ffn_weight import (
    MoeAtomicWeight,
    MoeConfig,
    iter_stacked_moe_weights,
)
from rtp_llm.model_loader.per_channel_fp8_quant_weight import (
    LoadQuantPerChannelFp8Weight,
    per_channel_cast_to_fp8,
    per_channel_cast_to_fp8_expert,
)
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource, TensorCollector
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    stack_,
    stack_moe_w1,
    stack_moe_w1_pad,
    stack_pad,
)


class FakeDatabase:
    def load_tensor(self, name, data_type=torch.float16):
        return []

    def has_tensor(self, name):
        return False


class CompositeWeightGroup(CompositeWeight):
    @classmethod
    def support(cls, quant_config, src_weight_info):
        return True


class TestTensorCollectorFp8(unittest.TestCase):
    """TensorCollector pre-quantized FP8 storage."""

    def _make_collector(self, keys):
        return TensorCollector(keys, FakeDatabase())

    def test_store_fp8_quantized_and_load_skips_dtype_conversion(self):
        c = self._make_collector({"k1", "k2"})
        fp8 = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
        scale = torch.ones(4, 1, dtype=torch.float32)
        c.store_fp8_quantized("k1", fp8, scale)

        loaded = c.load_tensor("k1", torch.bfloat16)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].dtype, torch.float8_e4m3fn)

    def test_load_tensor_normal_converts_dtype(self):
        c = self._make_collector({"k1"})
        t = torch.zeros(4, 8, dtype=torch.bfloat16)
        c.store_tensor("k1", t)

        loaded = c.load_tensor("k1", torch.float32)
        self.assertEqual(loaded[0].dtype, torch.float32)

    def test_has_prequantized_scale(self):
        c = self._make_collector({"k1", "k2"})
        fp8 = torch.zeros(2, 4, dtype=torch.float8_e4m3fn)
        scale = torch.ones(2, 1, dtype=torch.float32)
        c.store_fp8_quantized("k1", fp8, scale)

        self.assertTrue(c.has_prequantized_scale("k1"))
        self.assertFalse(c.has_prequantized_scale("k2"))

    def test_get_scale(self):
        c = self._make_collector({"k1"})
        scale = torch.tensor([[1.5], [2.5]])
        c.store_fp8_quantized("k1", torch.zeros(2, 4, dtype=torch.float8_e4m3fn), scale)
        torch.testing.assert_close(c.get_scale("k1"), scale)

    def test_clear_resets_scales(self):
        c = self._make_collector({"k1"})
        c.store_fp8_quantized(
            "k1", torch.zeros(2, 4, dtype=torch.float8_e4m3fn), torch.ones(2, 1)
        )
        c.clear()
        self.assertFalse(c.has_prequantized_scale("k1"))

    def test_completion_with_fp8_storage(self):
        c = self._make_collector({"k1", "k2"})
        fp8 = torch.zeros(2, 4, dtype=torch.float8_e4m3fn)
        scale = torch.ones(2, 1)

        done = c.store_fp8_quantized("k1", fp8, scale)
        self.assertFalse(done)
        done = c.store_fp8_quantized("k2", fp8, scale)
        self.assertTrue(done)


class TestBuildStackedKeyConfigNoOverwrite(unittest.TestCase):
    """_build_stacked_key_config must not overwrite kernel template with scale template."""

    def test_same_checkpoint_key_preserves_first_template(self):
        """When kernel and scale share the same stacked checkpoint key,
        _build_stacked_key_config must keep the kernel's template (first seen)."""
        from unittest.mock import patch

        from rtp_llm.model_loader.loader import ModelLoader

        config = MoeConfig(expert_num=4)
        shared_ckpt_key = "model.layers.{i}.mlp.experts.gate_up_proj"

        kernel = MoeAtomicWeight(
            name=W.moe_w1,
            weights=[CkptWeightInfo(shared_ckpt_key)],
            config=config,
            stacked_ckpt_keys=True,
        )
        scale = MoeAtomicWeight(
            name=W.moe_s1,
            weights=[CkptWeightInfo(shared_ckpt_key)],
            config=config,
            stacked_ckpt_keys=True,
        )

        fake_weight = MagicMock()
        wi = MagicMock()
        wi.weight = fake_weight
        wi.layer_id = 0

        with patch(
            "rtp_llm.model_loader.loader.iter_stacked_moe_weights",
            return_value=iter([kernel, scale]),
        ):
            result = ModelLoader._build_stacked_key_config([wi])

        resolved_key = shared_ckpt_key.format(i="0")
        self.assertIn(resolved_key, result)

        template = result[resolved_key]
        self.assertIn(W.moe_w1, template)
        self.assertNotIn(W.moe_s1, template)


class TestPerChannelCastToFp8Expert(unittest.TestCase):
    """per_channel_cast_to_fp8_expert matches the 3D path semantics."""

    def test_expert_quant_matches_3d_stacked_path(self):
        num_experts = 4
        dim0, dim1 = 16, 32
        experts = [torch.randn(dim0, dim1) for _ in range(num_experts)]
        stacked_3d = torch.stack(experts, dim=0)

        fp8_3d, scale_3d = per_channel_cast_to_fp8(stacked_3d)

        fp8_parts = []
        scale_parts = []
        for e in experts:
            fp8_e, scale_e = per_channel_cast_to_fp8_expert(e)
            fp8_parts.append(fp8_e)
            scale_parts.append(scale_e)

        fp8_stacked = torch.stack(fp8_parts, dim=0)
        scale_stacked = torch.stack(scale_parts, dim=0)

        self.assertEqual(fp8_stacked.shape, fp8_3d.shape)
        self.assertEqual(scale_stacked.shape, scale_3d.shape)
        self.assertTrue(torch.equal(fp8_stacked, fp8_3d))
        torch.testing.assert_close(scale_stacked, scale_3d)

    def test_output_shapes(self):
        x = torch.randn(64, 128)
        fp8, scale = per_channel_cast_to_fp8_expert(x)
        self.assertEqual(fp8.shape, (64, 128))
        self.assertEqual(fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (64, 1))
        self.assertEqual(scale.dtype, torch.float32)

    def test_rejects_non_2d(self):
        with self.assertRaises(AssertionError):
            per_channel_cast_to_fp8_expert(torch.randn(4, 8, 16))


class TestTransposeStackMoeW1Swap(unittest.TestCase):
    """_load_moe_inline_quant applies gate/up swap for transpose_stack_moe_w1."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gate_up_swap_applied(self):
        from rtp_llm.model_loader.per_channel_fp8_quant_weight import (
            W8A8Fp8PerChannelMoeAtomicWeight,
        )

        num_experts = 2
        gate_dim = 4
        hidden_dim = 8

        def transpose_stack_moe_w1(ts):
            stacked = torch.stack(ts, dim=0)
            half = stacked.shape[1] // 2
            return torch.cat([stacked[:, half:, :], stacked[:, :half, :]], dim=1)

        config = MoeConfig(expert_num=num_experts)
        kernel = W8A8Fp8PerChannelMoeAtomicWeight(
            name=W.moe_w1,
            weights=[CkptWeightInfo("layers.{i}.gate_up")],
            config=config,
            stacked_ckpt_keys=True,
            process_fun=transpose_stack_moe_w1,
        )
        scale = W8A8Fp8PerChannelMoeAtomicWeight(
            name=W.moe_s1,
            weights=[CkptWeightInfo("layers.{i}.gate_up")],
            config=config,
            stacked_ckpt_keys=True,
            process_fun=transpose_stack_moe_w1,
        )

        from rtp_llm.config.quant_config import Fp8PerChannelCompressedQuantConfig
        from rtp_llm.model_loader.weight_module import CompositeWeight

        lqw = LoadQuantPerChannelFp8Weight.__new__(LoadQuantPerChannelFp8Weight)
        CompositeWeight.__init__(
            lqw,
            {kernel.name: kernel, scale.name: scale},
            quant_config=Fp8PerChannelCompressedQuantConfig(),
            name=W.moe_w1,
        )
        lqw.kernel = kernel
        lqw.scale = scale

        gate_data = torch.randn(num_experts, gate_dim, hidden_dim)
        up_data = torch.randn(num_experts, gate_dim, hidden_dim)
        fused_experts = torch.cat([gate_data, up_data], dim=1)

        expert_keys = {
            f"layers.0.moe.{W.moe_w1}.{eid}.0"
            for eid in range(num_experts)
        }
        collector = TensorCollector(expert_keys, FakeDatabase())
        for eid in range(num_experts):
            key = f"layers.0.moe.{W.moe_w1}.{eid}.0"
            expert_tensor = fused_experts[eid]
            fp8_e, scale_e = per_channel_cast_to_fp8_expert(expert_tensor.cuda())
            collector.store_fp8_quantized(key, fp8_e, scale_e)

        lc = MagicMock()
        lc.get_selected_experts.return_value = list(range(num_experts))
        lc.compute_dtype = torch.bfloat16

        result = lqw._load_moe_inline_quant(collector, 0, "cuda:0", lc)
        self.assertIsNotNone(result)

        fp8_out = result[W.moe_w1]
        self.assertEqual(fp8_out.shape, (num_experts, gate_dim * 2, hidden_dim))

        ref_stacked = transpose_stack_moe_w1(
            [fused_experts[i] for i in range(num_experts)]
        )
        ref_fp8, _ = per_channel_cast_to_fp8(ref_stacked.cuda())
        self.assertTrue(torch.equal(fp8_out, ref_fp8))


class TestInlineFp8LayoutCompatibility(unittest.TestCase):
    """Padded layouts must retain original weights until merge and quantization."""

    def _case(self, is_w1, padded):
        parts = 2 if is_w1 else 1
        name, scale_name = (W.moe_w1, W.moe_s1) if is_w1 else (W.moe_w2, W.moe_s2)
        process = stack_moe_w1 if is_w1 else stack_
        if padded:
            process = functools.partial(
                stack_moe_w1_pad if is_w1 else stack_pad,
                moe_align_size=4,
                dim=1 if is_w1 else 2,
            )
        ckpts = [
            CkptWeightInfo("layer.{i}.expert.{expert_id}.part" + str(p), identity)
            for p in range(parts)
        ]
        kernel = MoeAtomicWeight(
            name, ckpts, process, data_type=torch.bfloat16, config=MoeConfig(expert_num=2)
        )
        weight = object.__new__(LoadQuantPerChannelFp8Weight)
        weight.kernel = kernel
        weight.scale = SimpleNamespace(name=scale_name)
        config = SimpleNamespace(
            compute_dtype=torch.bfloat16,
            moe_pure_tp_preshard=False,
            get_selected_experts=lambda layer, n: list(range(n)),
        )
        raw = {
            ckpt.name.format(i=0, expert_id=e): (
                torch.arange(1, 36).reshape(5, 7).float() * (0.001 * (p + e + 1))
            ).to(torch.bfloat16)
            for p, ckpt in enumerate(ckpts)
            for e in range(2)
        }
        return weight, config, raw

    def test_fastsafetensors_matches_scratch_for_padded_and_supported_layouts(self):
        from rtp_llm.model_loader.loader import ModelLoader

        for is_w1 in (True, False):
            for padded in (True, False):
                with self.subTest(is_w1=is_w1, padded=padded):
                    weight, config, raw = self._case(is_w1, padded)
                    database = SimpleNamespace(
                        load_tensor=lambda name, dtype: [raw[name].to(dtype)]
                    )
                    scratch = weight._load_raw_tensor(
                        DatabaseTensorSource(database), 0, "cpu", config
                    )
                    collector = TensorCollector(set(raw), database)
                    inline = ModelLoader._should_inline_fp8_quantize(
                        None, SimpleNamespace(weight=weight)
                    )
                    self.assertEqual(inline, not padded)
                    for key, tensor in raw.items():
                        if inline:
                            fp8, scale = per_channel_cast_to_fp8_expert(tensor)
                            collector.store_fp8_quantized(key, fp8, scale)
                        else:
                            collector.store_tensor(key, tensor)
                    fast = weight._load_raw_tensor(collector, 0, "cpu", config)
                    for key in (weight.kernel.name, weight.scale.name):
                        torch.testing.assert_close(
                            fast[key].float(), scratch[key].float(), rtol=0, atol=0
                        )
                    self.assertLess(float(fast[weight.scale.name].max()), 0.01)

    def test_unsupported_prequantized_layout_fails_instead_of_losing_scale(self):
        for is_w1 in (True, False):
            with self.subTest(is_w1=is_w1):
                weight, config, raw = self._case(is_w1, padded=True)
                collector = TensorCollector(set(raw), FakeDatabase())
                for key, tensor in raw.items():
                    fp8, scale = per_channel_cast_to_fp8_expert(tensor)
                    collector.store_fp8_quantized(key, fp8, scale)
                with self.assertRaisesRegex(ValueError, "require their original scales"):
                    weight._load_raw_tensor(collector, 0, "cpu", config)

    def _auto_loader(self, layers, free_gib):
        from rtp_llm.model_loader.load_config import LoadMethod
        from rtp_llm.model_loader.loader import ModelLoader

        gib = 1024**3
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.AUTO
        loader._weights_info = SimpleNamespace(
            _quant_config=Fp8PerChannelCompressedQuantConfig(),
            model_config=SimpleNamespace(
                eval_model_weight_size=lambda: 80 * gib,
                moe_weight_param_count=lambda: 90,
                layer_weight_param_count=lambda: 100,
            ),
        )
        loader._model_weights_info = SimpleNamespace(layer_weights=layers)
        loader._load_config = SimpleNamespace(
            tp_size=1,
            ep_size=1,
            exported_device=SimpleNamespace(
                get_mem_info=lambda: SimpleNamespace(free=free_gib * gib)
            ),
            database=SimpleNamespace(
                is_safetensor=True,
                get_max_file_size=lambda: 4 * gib,
                get_pretrain_tensor_names=lambda: ["expert.weight"],
            ),
        )
        # Only isolate environment and loading side effects; exercise the actual
        # AUTO selection, model traversal and inline eligibility implementations.
        loader._choose_weight_convert_device = MagicMock(return_value="cuda:0")
        loader._load_from_fastsafetensor = MagicMock(return_value="fast")
        loader._load_from_scratch = MagicMock(return_value="scratch")
        return loader

    def _assert_auto_choice(self, layers, free_gib, expected):
        loader = self._auto_loader(layers, free_gib)
        with patch("rtp_llm.model_loader.loader.has_module", return_value=True):
            self.assertEqual(loader._load_weight("cuda:0"), expected)
        selected = (
            loader._load_from_fastsafetensor
            if expected == "fast"
            else loader._load_from_scratch
        )
        selected.assert_called_once_with("cuda:0")

    def test_auto_keeps_inline_memory_discount_for_supported_moe(self):
        weights = [self._case(is_w1, padded=False)[0] for is_w1 in (True, False)]
        # 80 GiB * (90% MoE + 2 * 10% dense) + 3 * 4 GiB shards = 100 GiB.
        self._assert_auto_choice([weights], 101, "fast")
        self._assert_auto_choice([weights], 99, "scratch")

    def test_auto_uses_noninline_budget_for_padded_moe(self):
        for is_w1 in (True, False):
            with self.subTest(is_w1=is_w1):
                weight = self._case(is_w1, padded=True)[0]
                self._assert_auto_choice([[weight]], 120, "scratch")
                # Conservative non-inline budget: 2 * 80 + 3 * 4 = 172 GiB.
                self._assert_auto_choice([[weight]], 173, "fast")
                self._assert_auto_choice([[weight]], 172, "scratch")

    def test_auto_checks_all_layers_and_nested_components(self):
        supported = self._case(True, padded=False)[0]
        padded = self._case(False, padded=True)[0]
        for last in (padded, supported.kernel):
            with self.subTest(weight_type=type(last).__name__):
                nested = CompositeWeightGroup(
                    {"experts": CompositeWeightGroup({"w2": last}, name="experts")},
                    name="layer",
                )
                self._assert_auto_choice([[supported], nested], 120, "scratch")
        self._assert_auto_choice(
            [CompositeWeightGroup({"w1": supported}, name="layer")], 120, "fast"
        )

    def test_auto_without_moe_weights_does_not_assume_inline(self):
        self._assert_auto_choice([], 120, "scratch")


class TestGetTensorNamesExcludesScale(unittest.TestCase):
    """LoadQuantPerChannelFp8Weight.get_tensor_names only returns kernel keys."""

    def test_excludes_scale_expert_keys(self):
        from rtp_llm.config.quant_config import Fp8PerChannelCompressedQuantConfig
        from rtp_llm.model_loader.per_channel_fp8_quant_weight import (
            W8A8Fp8PerChannelMoeAtomicWeight,
        )
        from rtp_llm.model_loader.weight_module import CompositeWeight

        num_experts = 4
        config = MoeConfig(expert_num=num_experts)
        shared_ckpt = "model.layers.{i}.mlp.experts.gate_up_proj"

        kernel = W8A8Fp8PerChannelMoeAtomicWeight(
            name=W.moe_w1,
            weights=[CkptWeightInfo(shared_ckpt)],
            config=config,
            stacked_ckpt_keys=True,
        )
        scale = W8A8Fp8PerChannelMoeAtomicWeight(
            name=W.moe_s1,
            weights=[CkptWeightInfo(shared_ckpt)],
            config=config,
            stacked_ckpt_keys=True,
        )

        lqw = LoadQuantPerChannelFp8Weight.__new__(LoadQuantPerChannelFp8Weight)
        CompositeWeight.__init__(
            lqw,
            {kernel.name: kernel, scale.name: scale},
            quant_config=Fp8PerChannelCompressedQuantConfig(),
            name=W.moe_w1,
        )
        lqw.kernel = kernel
        lqw.scale = scale

        lc = MagicMock()
        lc.get_selected_experts.return_value = list(range(num_experts))

        names = lqw.get_tensor_names(layer_id=0, load_config=lc)

        for name in names:
            self.assertIn(W.moe_w1, name)
            self.assertNotIn(W.moe_s1, name)

        self.assertEqual(len(names), num_experts)


if __name__ == "__main__":
    unittest.main()
