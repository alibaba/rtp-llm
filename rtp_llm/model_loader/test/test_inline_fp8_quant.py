"""Unit tests for inline FP8 quantization during fastsafetensors loading.

Covers:
  - TensorCollector: FP8 pre-quantization storage (store/load/clear)
  - _build_stacked_key_config: no overwrite when kernel and scale share checkpoint key
  - per_channel_cast_to_fp8_expert: matches 3D per_channel_cast_to_fp8 semantics
  - _load_moe_inline_quant: transpose_stack_moe_w1 gate/up swap
  - LoadQuantPerChannelFp8Weight.get_tensor_names: excludes scale keys
"""

import unittest
from unittest.mock import MagicMock

import torch

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
from rtp_llm.model_loader.tensor_source import TensorCollector
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


class FakeDatabase:
    def load_tensor(self, name, data_type=torch.float16):
        return []

    def has_tensor(self, name):
        return False


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
