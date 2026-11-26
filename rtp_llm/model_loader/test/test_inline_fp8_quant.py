"""Unit tests for inline FP8 quantization during fastsafetensors loading.

Covers:
  - TensorCollector: FP8 pre-quantization storage (store/load/clear)
  - _build_stacked_key_config: no overwrite when kernel and scale share checkpoint key
  - per_channel_cast_to_fp8_expert: matches 3D per_channel_cast_to_fp8 semantics
  - _load_moe_inline_quant: transpose_stack_moe_w1 gate/up swap
  - LoadQuantPerChannelFp8Weight.get_tensor_names: excludes scale keys
"""

import functools
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.device.device_type import DeviceType, is_hip
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import (
    MoeAtomicWeight,
    MoeConfig,
    iter_stacked_moe_weights,
)
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    LoadQuantPerBlockFp8Weight,
    per_block_cast_to_fp8,
)
from rtp_llm.model_loader.per_channel_fp8_quant_weight import (
    LoadQuantPerChannelFp8Weight,
    per_channel_cast_to_fp8,
    per_channel_cast_to_fp8_expert,
)
from rtp_llm.model_loader.tensor_source import TensorCollector
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.models.llama_weight import merge_qkv_hf as llama_merge_qkv_hf
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    transpose,
    transpose_pad,
)


from rtp_llm.utils.database import BaseDatabase


class FakeDatabase(BaseDatabase):
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

        result = ModelLoader._build_stacked_key_config(
            ModelWeightInfo(weights=[], layer_weights=[[kernel, scale]])
        )

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


@unittest.skipIf(is_hip(), "SM120 per-block CUDA layout tests are skipped on ROCm")
class TestPerBlockOnlineSm120Layout(unittest.TestCase):
    def setUp(self):
        self._e8m0_patcher = patch(
            "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
            return_value=True,
        )
        self._e8m0_patcher.start()
        self.addCleanup(self._stop_e8m0_patcher)

    def _stop_e8m0_patcher(self):
        if self._e8m0_patcher is not None:
            self._e8m0_patcher.stop()
            self._e8m0_patcher = None

    @staticmethod
    def _fake_quant_weight_ue8m0_packed(weight: torch.Tensor):
        q, _ = per_block_cast_to_fp8(weight, 128)
        n, k = weight.shape
        packed_scale = torch.ones((n, (k + 511) // 512), dtype=torch.int32)
        return q, packed_scale

    def _mock_loader(self, raw):
        loader = LoadQuantPerBlockFp8Weight.__new__(LoadQuantPerBlockFp8Weight)
        loader.group_size = 128
        loader.quant_config = Fp8BlockWiseQuantConfig()
        loader.kernel = MagicMock()
        loader.kernel.name = W.attn_o_w
        loader.kernel.need_transpose = False
        loader.scale = MagicMock()
        loader.scale.name = W.attn_o_s
        loader.scale.need_transpose = False
        loader.kernel._load_raw_tensor.return_value = {W.attn_o_w: raw}
        return loader

    def _load_descriptor(self, descriptor, tensors):
        loader = LoadQuantPerBlockFp8Weight(
            descriptor,
            Fp8BlockWiseQuantConfig(is_quanted=False),
            name=descriptor.name,
        )
        source = MagicMock()
        source.load_tensor.side_effect = lambda name, dtype: [tensors[name]]
        load_config = MagicMock(compute_dtype=torch.float32)
        with patch(
            "rtp_llm.models_py.kernels.cuda.fp8_kernel.quant_weight_ue8m0_packed",
            side_effect=self._fake_quant_weight_ue8m0_packed,
        ):
            loaded = loader._load_raw_tensor(
                source, layer_id=0, device="cpu", load_config=load_config
            )
        return loader, source, loaded

    def test_raw_and_postprocess_select_scale_format_for_inference_device(self):
        for current, target in ((0, 1), (1, 0)):
            with self.subTest(current=current, target=target):
                raw = torch.ones(128, 128)
                loader = self._mock_loader(raw)
                config = MagicMock()
                config.exported_device.get_device_type.return_value = DeviceType.Cuda
                config.exported_device.get_device_id.return_value = str(target)
                config.exported_device.maybe_rewrite_weight_by_key.side_effect = (
                    lambda key, tensor, **kwargs: tensor
                )
                packed = (
                    raw.to(torch.float8_e4m3fn),
                    torch.ones(128, 1, dtype=torch.int32),
                )
                with (
                    patch("torch.cuda.current_device", return_value=current),
                    patch(
                        "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
                        side_effect=lambda device=None: (
                            torch.cuda.current_device() if device is None else device
                        )
                        == 0,
                    ) as scale_policy,
                    patch(
                        "rtp_llm.models_py.kernels.cuda.fp8_kernel.quant_weight_ue8m0_packed",
                        return_value=packed,
                    ) as quantize_packed,
                ):
                    loaded = loader._load_raw_tensor(MagicMock(), 0, "cpu", config)
                    with patch.object(
                        CompositeWeight, "_postprocess", return_value=loaded.copy()
                    ):
                        processed = loader._postprocess(loaded, "cpu", config)
                self.assertEqual(
                    [call.args for call in scale_policy.call_args_list],
                    [(target,), (target,)],
                )
                self.assertEqual(quantize_packed.call_count, int(target == 0))
                self.assertEqual(
                    processed[W.attn_o_s].dtype == torch.int32, target == 0
                )

    def test_moe_weights_keep_packed_deepgemm_scales(self):
        for kernel_name, scale_name in (
            (W.moe_w1, W.moe_s1),
            (W.moe_w2, W.moe_s2),
        ):
            with self.subTest(kernel=kernel_name):
                raw = torch.linspace(-1.0, 1.0, 128 * 256).reshape(128, 256)
                loader = self._mock_loader(raw)
                loader.kernel.name = kernel_name
                loader.scale.name = scale_name
                loader.kernel._load_raw_tensor.return_value = {kernel_name: raw}
                config = MagicMock()
                config.exported_device.maybe_rewrite_weight_by_key.side_effect = (
                    lambda key, tensor, **kwargs: tensor
                )
                packed_scale = torch.ones(2, 1, dtype=torch.int32)
                with (
                    patch(
                        "rtp_llm.models_py.kernels.cuda.fp8_kernel.quant_weight_ue8m0_packed",
                    ) as quantize_packed,
                    patch(
                        "rtp_llm.models_py.kernels.cuda.fp8_kernel.requant_weight_ue8m0",
                        side_effect=lambda weight, scale: (weight, packed_scale),
                    ) as requant,
                ):
                    loaded = loader._load_raw_tensor(MagicMock(), 0, "cpu", config)
                    with patch.object(
                        CompositeWeight, "_postprocess", return_value=loaded.copy()
                    ):
                        processed = loader._postprocess(loaded, "cpu", config)
                quantize_packed.assert_not_called()
                requant.assert_called_once()
                self.assertNotEqual(loaded[scale_name].dtype, torch.int32)
                self.assertEqual(processed[scale_name].dtype, torch.int32)

    def test_rectangular_input_major_weight_becomes_output_major(self):
        raw_weight = torch.linspace(-1.0, 1.0, 256 * 384).reshape(384, 256)
        loader = self._mock_loader(raw_weight)
        expected_weight, _ = per_block_cast_to_fp8(raw_weight, loader.group_size)

        with patch(
            "rtp_llm.models_py.kernels.cuda.fp8_kernel.quant_weight_ue8m0_packed",
            side_effect=self._fake_quant_weight_ue8m0_packed,
        ):
            loaded = loader._load_raw_tensor(
                MagicMock(), layer_id=0, device="cpu", load_config=MagicMock()
            )

        self.assertEqual(loaded[W.attn_o_w].shape, (256, 384))
        self.assertEqual(loaded[W.attn_o_s].shape, (256, 1))
        self.assertEqual(loaded[W.attn_o_s].dtype, torch.int32)
        self.assertTrue(torch.equal(loaded[W.attn_o_w], expected_weight.T))

    def test_real_qkv_merge_descriptor_restores_output_major_layout(self):
        tensors = {
            "q": torch.linspace(-1.0, 1.0, 256 * 384).reshape(256, 384),
            "k": torch.linspace(-0.7, 0.9, 128 * 384).reshape(128, 384),
            "v": torch.linspace(-0.4, 0.3, 128 * 384).reshape(128, 384),
        }
        descriptor = AttnAtomicWeight(
            W.attn_qkv_w,
            [CkptWeightInfo(key) for key in tensors],
            functools.partial(
                llama_merge_qkv_hf, hidden_size=384, head_num_kv=1, head_num=2
            ),
        )
        loader, tensor_source, loaded = self._load_descriptor(descriptor, tensors)
        expected_weight, _ = per_block_cast_to_fp8(
            torch.cat(list(tensors.values())), loader.group_size
        )
        self.assertEqual(loaded[W.attn_qkv_w].shape, (512, 384))
        self.assertEqual(loaded[W.attn_qkv_s].shape, (512, 1))
        self.assertEqual(loaded[W.attn_qkv_s].dtype, torch.int32)
        torch.testing.assert_close(
            loaded[W.attn_qkv_w].float(), expected_weight.float(), rtol=0, atol=0
        )

    def test_real_transpose_descriptor_restores_output_major_layout(self):
        raw_weight = torch.linspace(-1.0, 1.0, 256 * 384).reshape(256, 384)
        descriptor = AttnAtomicWeight(
            W.attn_o_w,
            [CkptWeightInfo("layers.{i}.self_attn.o_proj.weight")],
            transpose,
        )
        loader, tensor_source, loaded = self._load_descriptor(
            descriptor, {"layers.0.self_attn.o_proj.weight": raw_weight}
        )
        expected_weight, _ = per_block_cast_to_fp8(raw_weight, loader.group_size)

        tensor_source.load_tensor.assert_called_once_with(
            "layers.0.self_attn.o_proj.weight", torch.float32
        )
        self.assertEqual(loaded[W.attn_o_w].shape, (256, 384))
        self.assertEqual(loaded[W.attn_o_s].shape, (256, 1))
        self.assertEqual(loaded[W.attn_o_s].dtype, torch.int32)
        self.assertTrue(torch.equal(loaded[W.attn_o_w], expected_weight))

    def test_real_transpose_pad_descriptor_quantizes_non_aligned_tensor(self):
        raw_weight = torch.linspace(-1.0, 1.0, 130 * 257).reshape(130, 257)
        descriptor = AttnAtomicWeight(
            W.attn_o_w,
            [CkptWeightInfo("layers.{i}.self_attn.o_proj.weight")],
            functools.partial(transpose_pad, align_size=128, dim=0),
        )
        loader, tensor_source, loaded = self._load_descriptor(
            descriptor, {"layers.0.self_attn.o_proj.weight": raw_weight}
        )

        processed = transpose_pad([raw_weight], align_size=128, dim=0)
        expected_weight, _ = per_block_cast_to_fp8(processed, loader.group_size)
        expected_weight = expected_weight.T.contiguous()

        self.assertTrue(loader.kernel.need_transpose)
        self.assertEqual(loaded[W.attn_o_w].shape, (256, 257))
        self.assertEqual(loaded[W.attn_o_s].shape, (256, 1))
        self.assertEqual(loaded[W.attn_o_s].dtype, torch.int32)
        self.assertTrue(torch.equal(loaded[W.attn_o_w], expected_weight))

    def test_real_load_config_cpu_staging_preserves_target_device_and_resident_override(
        self,
    ):
        from contextlib import nullcontext

        from rtp_llm.device.device_impl import GpuImpl
        from rtp_llm.model_loader.load_config import LoadConfig
        from rtp_llm.model_loader.loader import ModelLoader
        from rtp_llm.model_loader.per_block_fp8_quant_weight import (
            _resolve_target_cuda_device,
        )
        from rtp_llm.model_loader.weight_module import WeightModule
        from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import fp8_kernel
        from rtp_llm.models_py.utils import arch

        raw = torch.ones((128, 128), dtype=torch.bfloat16)
        with patch.object(sys, "argv", [sys.argv[0]]):
            exported_device = GpuImpl()
        load_config = LoadConfig(
            database=FakeDatabase(),
            num_layers=1,
            hidden_size=128,
            head_num=1,
            head_num_kv=1,
            size_per_head=128,
            moe_pure_tp_mode=False,
            align_size=128,
            moe_align_size=128,
            moe_layer_index=[],
            moe_n_group=1,
            expert_num=0,
            enable_eplb=False,
            phy_exp_num=0,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            dp_size=1,
            dp_rank=0,
            lm_head_tp_size=1,
            lm_head_tp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            num_nodes=1,
            layer_range=range(1),
            compute_dtype=torch.bfloat16,
            exported_device=exported_device,
            force_cpu_load_weights=True,
        )

        # Real GpuImpl.get_device_id() returns current_device() == 0, yet
        # CPU-staged prepare_weights(..., target_device="cuda:1") must still
        # resolve target_device=1 for both layer and global descriptors without
        # mutating base load_config.target_device.
        seen_layer_targets = []
        seen_global_targets = []

        class _ProbeWeight(WeightModule):
            @classmethod
            def support(cls, quant_config, src_weight_info):
                return True

            def _load_raw_tensor(self, tensor_source, layer_id, device, lc):
                resolved = _resolve_target_cuda_device(raw, lc)
                if layer_id is not None:
                    seen_layer_targets.append((device, lc.target_device, resolved))
                else:
                    seen_global_targets.append((device, lc.target_device, resolved))
                return {self.name: raw}

            def get_tensor_names(self, layer_id, lc):
                return set()

            def _split(self, tensor, load_config):
                return tensor

            def _postprocess(self, tensor, device, load_config):
                return tensor

            def _merge_lora(self, tensor, database, layer_id, load_config):
                return tensor

            def _load_raw_lora(self, database, layer_id, device, load_config, lora_name, data_type):
                return None

            def _split_lora(self, tensor, load_config):
                return tensor

        layer_w = _ProbeWeight("layer_probe")
        cpu_resident_w = _ProbeWeight("cpu_resident_probe")
        cpu_resident_w.resident_device = "cpu"
        global_w = _ProbeWeight("global_probe")

        loader = ModelLoader.__new__(ModelLoader)
        loader._is_attn_model = False
        loader._task_type = None
        loader._global_weight_aliases = {}
        loader._load_config = load_config
        loader._model_weights_info = MagicMock(
            layer_weights=[[layer_w, cpu_resident_w]],
            weights=[global_w],
        )
        loader._misc_weights_info = []
        loader._maybe_skip_weight = lambda w: False

        with patch("torch.cuda.current_device", return_value=0):
            items = list(loader.prepare_weights("cpu", target_device="cuda:1"))

        self.assertEqual(len(items), 3)
        self.assertIsNone(load_config.target_device)
        self.assertEqual(seen_layer_targets, [("cpu", "cuda:1", 1), ("cpu", "cpu", "cpu")])
        self.assertEqual(seen_global_targets, [("cpu", "cuda:1", 1)])

        # Also verify _transform_scale_ue8m0 stages onto cuda:1 (never cuda:0) and
        # returns on CPU when given a CPU scale tensor + cuda_device=1.
        sf_cpu = torch.ones((1, 2), dtype=torch.float32)
        to_calls = []
        entered_devices = []
        orig_to = torch.Tensor.to

        def spy_to(self_tensor, *args, **kwargs):
            to_calls.append(args[0] if args else kwargs.get("device"))
            return self_tensor

        def spy_get_mn_major(x):
            return torch.ones((128, 1), dtype=torch.int32)

        fake_layout = MagicMock(get_mn_major_tma_aligned_packed_ue8m0_tensor=spy_get_mn_major)
        fake_deep_gemm = MagicMock(utils=MagicMock(layout=fake_layout))
        with (
            patch("torch.cuda.current_device", return_value=0),
            patch(
                "torch.cuda.device",
                side_effect=lambda dev: (entered_devices.append(str(dev)), nullcontext())[1],
            ),
            patch("torch.Tensor.to", new=spy_to),
            patch("torch.Tensor.index_select", new=lambda self_t, dim, idx: self_t),
            patch("torch.arange", return_value=torch.zeros(128, dtype=torch.long)),
            patch.dict("sys.modules", {"deep_gemm": fake_deep_gemm, "deep_gemm.utils": fake_deep_gemm.utils, "deep_gemm.utils.layout": fake_layout}),
        ):
            out = fp8_kernel._transform_scale_ue8m0(sf_cpu, mn=128, cuda_device=1)

        self.assertEqual(entered_devices, ["cuda:1"])
        self.assertEqual(out.device.type, "cpu")
        self.assertNotIn(torch.device("cuda", 0), to_calls)
        self.assertIn(torch.device("cuda", 1), to_calls)

        # Verify is_deep_gemm_e8m0_used distinguishes SM90 vs SM120 per target ordinal.
        self._stop_e8m0_patcher()
        deepgemm_wrapper.is_deep_gemm_e8m0_used.cache_clear()
        arch._get_sm_for_device.cache_clear()
        try:
            with (
                patch.object(arch, "is_cuda", return_value=True),
                patch.object(deepgemm_wrapper, "is_cuda", return_value=True),
                patch("torch.cuda.current_device", return_value=0),
                patch(
                    "torch.cuda.get_device_capability",
                    side_effect=lambda d: {0: (9, 0), 1: (12, 0)}[int(d)],
                ),
            ):
                self.assertFalse(deepgemm_wrapper.is_deep_gemm_e8m0_used(0))
                self.assertTrue(deepgemm_wrapper.is_deep_gemm_e8m0_used(1))
        finally:
            deepgemm_wrapper.is_deep_gemm_e8m0_used.cache_clear()
            arch._get_sm_for_device.cache_clear()


if __name__ == "__main__":
    unittest.main()
