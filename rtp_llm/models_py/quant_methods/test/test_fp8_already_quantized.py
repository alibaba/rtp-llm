"""Tests for FP8 already-quantized direct-read paths in the new loader.

Covers:
- Fp8LinearMethod (key="fp8"): per-tensor compressed ckpt -> ColumnParallel.
- Fp8PerChannelLinearMethod (key="fp8_per_channel"): per-channel compressed
  / Quark ckpt -> ColumnParallel / RowParallel.
- MergedColumnParallelLinear: per-channel weight_scale cat across shards,
  per-tensor weight_scale max-merge across shards.
- QKVParallelLinear: per-channel weight_scale cat across q/k/v.

The forward (`apply`) path uses `torch._scaled_mm` and requires CUDA; those
tests are gated by `torch.cuda.is_available()`. Layer-level load_weights
routing tests run on CPU.
"""
import unittest
from unittest import mock

import torch

import rtp_llm.models_py.quant_methods.fp8_moe as fp8_moe
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
from rtp_llm.models_py.quant_methods.fp8 import _runtime_fp8_dtype
from rtp_llm.models_py.utils.arch import is_cuda


def _make_qc(quant_type: str) -> QuantizationConfig:
    return QuantizationConfig(quant_type=quant_type)


class TestRuntimeFp8Dtype(unittest.TestCase):
    def test_cuda_runtime_dtype_is_e4m3fn_for_linear_and_moe(self):
        with mock.patch.object(torch.version, "hip", None):
            self.assertEqual(_runtime_fp8_dtype(), torch.float8_e4m3fn)
            self.assertEqual(fp8_moe._runtime_fp8_dtype(), torch.float8_e4m3fn)

    def test_cuda_fp8_apply_uses_matching_activation_and_weight_dtype(self):
        with mock.patch.object(torch.version, "hip", None):
            layer = ColumnParallelLinear(
                input_size=4,
                output_size=4,
                quant_config=_make_qc("fp8"),
                prefix="fp8",
                params_dtype=torch.bfloat16,
            )
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        layer.weight_scale.data.fill_(1.0)
        qinput = torch.zeros(2, 4, dtype=torch.float8_e4m3fn)
        x_scale = torch.ones(1, dtype=torch.float32)

        def fake_scaled_mm(input, weight, **kwargs):
            self.assertEqual(input.dtype, torch.float8_e4m3fn)
            self.assertEqual(weight.dtype, torch.float8_e4m3fn)
            return torch.zeros(2, 4, dtype=torch.bfloat16)

        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._resolve_per_tensor_quant",
            return_value=lambda tensor: (qinput, x_scale),
        ), mock.patch("torch._scaled_mm", side_effect=fake_scaled_mm):
            output = layer(torch.zeros(2, 4, dtype=torch.bfloat16))
        self.assertEqual(output.dtype, torch.bfloat16)

    @unittest.skipUnless(
        hasattr(torch, "float8_e4m3fnuz"), "requires torch.float8_e4m3fnuz"
    )
    def test_rocm_runtime_dtype_uses_rocm_helper_for_linear_and_moe(self):
        helper = (
            "rtp_llm.models_py.modules.factory.fused_moe.impl.rocm._utils."
            "get_rocm_fp8_dtype"
        )
        with mock.patch.object(torch.version, "hip", "6.4"), mock.patch(
            helper, return_value=torch.float8_e4m3fnuz
        ) as get_dtype:
            self.assertEqual(_runtime_fp8_dtype(), torch.float8_e4m3fnuz)
            self.assertEqual(fp8_moe._runtime_fp8_dtype(), torch.float8_e4m3fnuz)
            self.assertEqual(get_dtype.call_count, 2)


class TestFp8PerTensorLoad(unittest.TestCase):
    """Already-quantized FP8 per-tensor compressed -> ColumnParallel."""

    def test_online_quant_rejects_cpu_post_load_device(self):
        layer = ColumnParallelLinear(
            input_size=4,
            output_size=4,
            quant_config=_make_qc("fp8_online"),
            prefix="online",
            params_dtype=torch.bfloat16,
        )
        layer.load_weights({"online.weight": torch.zeros(4, 4, dtype=torch.bfloat16)})
        with self.assertRaisesRegex(RuntimeError, "requires weights to be on CUDA"):
            layer.process_weights_after_loading()

    def test_online_quant_allows_cpu_post_load_when_force_cpu_enabled(self):
        layer = ColumnParallelLinear(
            input_size=4,
            output_size=4,
            quant_config=_make_qc("fp8_online"),
            prefix="online",
            params_dtype=torch.bfloat16,
        )
        layer.load_weights({"online.weight": torch.zeros(4, 4, dtype=torch.bfloat16)})
        layer._new_loader_force_cpu_load_weights = True
        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8.cpu_per_tensor_quant_like_legacy",
            return_value=(
                torch.zeros(4, 4, dtype=fp8_moe._runtime_fp8_dtype()),
                torch.ones(1, dtype=torch.float32),
            ),
        ) as cpu_quant:
            layer.process_weights_after_loading()
        cpu_quant.assert_called_once()
        self.assertEqual(layer.weight.dtype, fp8_moe._runtime_fp8_dtype())
        self.assertEqual(layer.weight.device.type, "cpu")

    def test_load_into_column_parallel(self):
        N, K = 16, 32
        qc = _make_qc("fp8")
        layer = ColumnParallelLinear(
            input_size=K,
            output_size=N,
            tp_size=1,
            tp_rank=0,
            quant_config=qc,
            prefix="test",
            params_dtype=torch.bfloat16,
        )
        # Hand-craft an "already-quantized" ckpt:
        weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16)
        scale = (weight_bf16.abs().max() / 448.0).float()
        fp8_weight = (weight_bf16 / scale).to(_runtime_fp8_dtype())

        layer.load_weights(
            {
                "test.weight": fp8_weight,
                "test.weight_scale": scale.reshape(1),
                "test.input_scale": torch.tensor([1.0], dtype=torch.float32),
            }
        )
        layer.process_weights_after_loading()

        self.assertEqual(layer.weight.dtype, _runtime_fp8_dtype())
        self.assertEqual(layer.weight.shape, (N, K))
        self.assertEqual(layer.weight_scale.shape, (1,))
        self.assertAlmostEqual(layer.weight_scale.item(), scale.item(), places=4)


class TestFp8PerChannelLoad(unittest.TestCase):
    """Already-quantized FP8 per-channel (compressed/Quark) -> {Col,Row}Parallel."""

    def test_load_into_column_parallel_normalize_1d_scale(self):
        N, K = 16, 32
        qc = _make_qc("fp8_per_channel")
        layer = ColumnParallelLinear(
            input_size=K,
            output_size=N,
            tp_size=1,
            tp_rank=0,
            quant_config=qc,
            prefix="test",
            params_dtype=torch.bfloat16,
        )
        # ckpt scale comes as [N] (1D); process_weights_after_loading should
        # normalize it to [1, N] for torch._scaled_mm scale_b.
        scale_1d = torch.rand(N, dtype=torch.float32) + 0.01
        fp8_weight = torch.zeros(N, K, dtype=_runtime_fp8_dtype())

        layer.load_weights(
            {
                "test.weight": fp8_weight,
                "test.weight_scale": scale_1d,
            }
        )
        layer.process_weights_after_loading()

        self.assertEqual(layer.weight_scale.shape, (1, N))
        torch.testing.assert_close(
            layer.weight_scale.flatten(), scale_1d, rtol=0, atol=0
        )

    def test_load_into_row_parallel_no_scale_shard(self):
        # RowParallel splits along input dim; per-channel weight_scale
        # is on output side and must NOT be sharded.
        N, K = 16, 32
        tp = 2
        qc = _make_qc("fp8_per_channel")
        layer = RowParallelLinear(
            input_size=K,
            output_size=N,
            tp_size=tp,
            tp_rank=1,
            quant_config=qc,
            prefix="test",
            params_dtype=torch.bfloat16,
        )
        fp8_weight = torch.zeros(N, K, dtype=_runtime_fp8_dtype())
        # weight is sharded along K (dim 1), so ckpt has full K.
        # weight_scale is per output-channel, full N.
        scale_2d = torch.rand(N, 1, dtype=torch.float32) + 0.01

        layer.load_weights(
            {
                "test.weight": fp8_weight,
                "test.weight_scale": scale_2d,
            }
        )
        layer.process_weights_after_loading()

        self.assertEqual(layer.weight.shape, (N, K // tp))
        self.assertEqual(layer.weight_scale.shape, (1, N))


class TestMergedColumnPerChannelScale(unittest.TestCase):
    """gate_up_proj per-channel weight_scale should cat across shards."""

    def test_merged_column_per_channel_scale_cat(self):
        H = 32  # input
        N = 16  # gate/up shard size, total 2*N output
        qc = _make_qc("fp8_per_channel")
        layer = MergedColumnParallelLinear(
            input_size=H,
            output_size=2 * N,
            tp_size=1,
            tp_rank=0,
            quant_config=qc,
            prefix="gate_up_proj",
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.bfloat16,
        )
        gate_w = torch.zeros(N, H, dtype=_runtime_fp8_dtype())
        up_w = torch.zeros(N, H, dtype=_runtime_fp8_dtype())
        gate_scale = torch.full((N,), 0.1, dtype=torch.float32)
        up_scale = torch.full((N,), 0.2, dtype=torch.float32)

        layer.load_weights(
            {
                "gate_up_proj.gate_proj.weight": gate_w,
                "gate_up_proj.gate_proj.weight_scale": gate_scale,
                "gate_up_proj.up_proj.weight": up_w,
                "gate_up_proj.up_proj.weight_scale": up_scale,
            }
        )
        layer.process_weights_after_loading()

        # process_weights_after_loading reshapes [2N] -> [1, 2N]
        self.assertEqual(layer.weight_scale.shape, (1, 2 * N))
        torch.testing.assert_close(
            layer.weight_scale[0, :N], gate_scale, rtol=0, atol=0
        )
        torch.testing.assert_close(layer.weight_scale[0, N:], up_scale, rtol=0, atol=0)

    def test_merged_column_per_tensor_scale_max_merge(self):
        H = 32
        N = 16
        qc = _make_qc("fp8")
        layer = MergedColumnParallelLinear(
            input_size=H,
            output_size=2 * N,
            tp_size=1,
            tp_rank=0,
            quant_config=qc,
            prefix="gate_up_proj",
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.bfloat16,
        )
        gate_w = torch.zeros(N, H, dtype=torch.float8_e4m3fn)
        up_w = torch.zeros(N, H, dtype=torch.float8_e4m3fn)
        gate_scale = torch.tensor([0.1], dtype=torch.float32)
        up_scale = torch.tensor([0.3], dtype=torch.float32)

        layer.load_weights(
            {
                "gate_up_proj.gate_proj.weight": gate_w,
                "gate_up_proj.gate_proj.weight_scale": gate_scale,
                "gate_up_proj.up_proj.weight": up_w,
                "gate_up_proj.up_proj.weight_scale": up_scale,
            }
        )
        layer.process_weights_after_loading()

        # Max of {0.1, 0.3} = 0.3
        self.assertAlmostEqual(layer.weight_scale.item(), 0.3, places=5)

    def test_merged_column_per_tensor_scale_streaming_rescales_shards(self):
        H = 8
        N = 4
        qc = _make_qc("fp8")
        layer = MergedColumnParallelLinear(
            input_size=H,
            output_size=2 * N,
            tp_size=1,
            tp_rank=0,
            quant_config=qc,
            prefix="gate_up_proj",
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.bfloat16,
        )
        gate_w = torch.full((N, H), 16.0).to(_runtime_fp8_dtype())
        up_w = torch.full((N, H), 16.0).to(_runtime_fp8_dtype())

        # Simulate RtpModule's streaming dispatch: each call carries one tensor,
        # so gate/up scalar scales are only both available in post-load.
        layer.load_weights({"gate_up_proj.gate_proj.weight": gate_w})
        layer.load_weights(
            {"gate_up_proj.gate_proj.weight_scale": torch.tensor([0.1])}
        )
        layer.load_weights({"gate_up_proj.up_proj.weight": up_w})
        layer.load_weights({"gate_up_proj.up_proj.weight_scale": torch.tensor([0.2])})
        layer.process_weights_after_loading()

        self.assertAlmostEqual(layer.weight_scale.item(), 0.2, places=5)
        torch.testing.assert_close(
            layer.weight[:N].float(),
            torch.full((N, H), 8.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layer.weight[N:].float(),
            torch.full((N, H), 16.0),
            rtol=0,
            atol=0,
        )


class TestQkvPerChannelScale(unittest.TestCase):
    """q_proj/k_proj/v_proj per-channel weight_scale should cat in qkv order."""

    def test_qkv_per_channel_scale_cat(self):
        hidden = 32
        head_dim = 8
        num_heads = 4  # q_size = 32
        num_kv_heads = 2  # kv_size = 16

        qc = _make_qc("fp8_per_channel")
        layer = QKVParallelLinear(
            hidden_size=hidden,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tp_size=1,
            tp_rank=0,
            quant_config=qc,
            prefix="qkv_proj",
            params_dtype=torch.bfloat16,
        )

        q_w = torch.zeros(num_heads * head_dim, hidden, dtype=_runtime_fp8_dtype())
        k_w = torch.zeros(num_kv_heads * head_dim, hidden, dtype=_runtime_fp8_dtype())
        v_w = torch.zeros(num_kv_heads * head_dim, hidden, dtype=_runtime_fp8_dtype())

        q_scale = torch.full((num_heads * head_dim,), 0.1, dtype=torch.float32)
        k_scale = torch.full((num_kv_heads * head_dim,), 0.2, dtype=torch.float32)
        v_scale = torch.full((num_kv_heads * head_dim,), 0.3, dtype=torch.float32)

        layer.load_weights(
            {
                "qkv_proj.q_proj.weight": q_w,
                "qkv_proj.q_proj.weight_scale": q_scale,
                "qkv_proj.k_proj.weight": k_w,
                "qkv_proj.k_proj.weight_scale": k_scale,
                "qkv_proj.v_proj.weight": v_w,
                "qkv_proj.v_proj.weight_scale": v_scale,
            }
        )
        layer.process_weights_after_loading()

        total = num_heads * head_dim + 2 * num_kv_heads * head_dim
        self.assertEqual(layer.weight_scale.shape, (1, total))
        # q segment
        torch.testing.assert_close(
            layer.weight_scale[0, : num_heads * head_dim], q_scale, rtol=0, atol=0
        )
        # k segment
        torch.testing.assert_close(
            layer.weight_scale[
                0,
                num_heads * head_dim : num_heads * head_dim + num_kv_heads * head_dim,
            ],
            k_scale,
            rtol=0,
            atol=0,
        )
        # v segment
        torch.testing.assert_close(
            layer.weight_scale[0, num_heads * head_dim + num_kv_heads * head_dim :],
            v_scale,
            rtol=0,
            atol=0,
        )



class TestFp8MoEAlreadyQuantizedLoad(unittest.TestCase):
    def _make_moe(self, quant_type="fp8"):
        return BaseMoEExperts(
            num_experts=1,
            hidden_size=4,
            moe_intermediate_size=4,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            params_dtype=torch.bfloat16,
            model_config=type("Cfg", (), {"data_type": "bf16", "quant_config": None, "exported_device": None})(),
            parallelism_config=type("Parallel", (), {"dp_size": 1})(),
            moe_config=type("MoeCfg", (), {})(),
            quant_config=_make_qc(quant_type),
            layer_idx=0,
        )

    def test_per_tensor_requants_to_runtime_dtype_and_preserves_scale_semantics(self):
        layer = self._make_moe("fp8")
        source_dtype = torch.float8_e4m3fn
        runtime_dtype = fp8_moe._runtime_fp8_dtype()
        gate_scale = torch.tensor([0.25], dtype=torch.float32)
        up_scale = torch.tensor([0.5], dtype=torch.float32)
        down_scale = torch.tensor([0.125], dtype=torch.float32)
        gate = torch.full((4, 4), 4.0, dtype=torch.float32).to(source_dtype)
        up = torch.full((4, 4), 8.0, dtype=torch.float32).to(source_dtype)
        down = torch.full((4, 4), 2.0, dtype=torch.float32).to(source_dtype)

        expected_w13 = torch.cat(
            [up.float() * up_scale.item(), gate.float() * gate_scale.item()], dim=0
        )
        expected_w2 = down.float() * down_scale.item()

        layer.load_weights(
            {
                "0.gate_proj.weight": gate,
                "0.gate_proj.weight_scale": gate_scale,
                "0.up_proj.weight": up,
                "0.up_proj.weight_scale": up_scale,
                "0.down_proj.weight": down,
                "0.down_proj.weight_scale": down_scale,
            }
        )
        with mock.patch.object(BaseMoEExperts, "_maybe_build_fused_moe", return_value=None):
            layer.process_weights_after_loading()

        self.assertEqual(layer.w13.dtype, runtime_dtype)
        self.assertEqual(layer.w2.dtype, runtime_dtype)
        actual_w13 = layer.w13.float() * layer.w13_scale.view(-1, 1, 1)
        actual_w2 = layer.w2.float() * layer.w2_scale.view(-1, 1, 1)
        torch.testing.assert_close(actual_w13[0], expected_w13, rtol=0.1, atol=0.1)
        torch.testing.assert_close(actual_w2[0], expected_w2, rtol=0.1, atol=0.1)

    def test_online_per_tensor_moe_allows_cpu_post_load_when_force_cpu_enabled(self):
        layer = self._make_moe("fp8_online")
        layer._new_loader_force_cpu_load_weights = True
        for name in (
            "0.gate_proj.weight",
            "0.up_proj.weight",
            "0.down_proj.weight",
        ):
            layer.load_weights({name: torch.zeros(4, 4, dtype=torch.bfloat16)})

        def fake_cpu_quant(weight):
            return (
                torch.zeros_like(weight, dtype=fp8_moe._runtime_fp8_dtype()),
                torch.ones(1, dtype=torch.float32),
            )

        with mock.patch.object(BaseMoEExperts, "_maybe_build_fused_moe", return_value=None), \
             mock.patch(
                 "rtp_llm.models_py.quant_methods.fp8.cpu_per_tensor_quant_like_legacy",
                 side_effect=fake_cpu_quant,
             ) as cpu_quant:
            layer.process_weights_after_loading()

        self.assertEqual(cpu_quant.call_count, 2)
        self.assertEqual(layer.w13.dtype, fp8_moe._runtime_fp8_dtype())
        self.assertEqual(layer.w2.dtype, fp8_moe._runtime_fp8_dtype())
        self.assertEqual(layer.w13.device.type, "cpu")
        self.assertEqual(layer.w2.device.type, "cpu")

    def test_online_block_moe_rejects_force_cpu_post_load(self):
        layer = self._make_moe("fp8_block_online")
        layer._new_loader_force_cpu_load_weights = True
        for name in (
            "0.gate_proj.weight",
            "0.up_proj.weight",
            "0.down_proj.weight",
        ):
            layer.load_weights({name: torch.zeros(4, 4, dtype=torch.bfloat16)})
        with mock.patch.object(BaseMoEExperts, "_maybe_build_fused_moe", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "force_cpu_load_weights"):
                layer.process_weights_after_loading()

    def test_online_block_moe_rejects_custom_block_size(self):
        qc = _make_qc("fp8_block_online")
        qc.weight_block_size = [2, 4]
        layer = BaseMoEExperts(
            num_experts=1,
            hidden_size=4,
            moe_intermediate_size=4,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            params_dtype=torch.bfloat16,
            model_config=type(
                "Cfg",
                (),
                {"data_type": "bf16", "quant_config": None, "exported_device": None},
            )(),
            parallelism_config=type("Parallel", (), {"dp_size": 1})(),
            moe_config=type("MoeCfg", (), {})(),
            quant_config=qc,
            layer_idx=0,
        )
        for name in (
            "0.gate_proj.weight",
            "0.up_proj.weight",
            "0.down_proj.weight",
        ):
            layer.load_weights({name: torch.zeros(4, 4, dtype=torch.bfloat16)})
        with mock.patch.object(BaseMoEExperts, "_maybe_build_fused_moe", return_value=None):
            with self.assertRaisesRegex(ValueError, "only supports"):
                layer.process_weights_after_loading()

    def test_custom_block_moe_source_scales_materialized_after_meta_to_empty(self):
        qc = _make_qc("fp8_block")
        qc.weight_block_size = [2, 4]
        with torch.device("meta"):
            layer = BaseMoEExperts(
                num_experts=1,
                hidden_size=4,
                moe_intermediate_size=4,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                params_dtype=torch.bfloat16,
                model_config=type(
                    "Cfg",
                    (),
                    {"data_type": "bf16", "quant_config": None, "exported_device": None},
                )(),
                parallelism_config=type("Parallel", (), {"dp_size": 1})(),
                moe_config=type("MoeCfg", (), {})(),
                quant_config=qc,
                layer_idx=0,
            )
        layer.to_empty(device="cpu")
        self.assertFalse(layer.w13_scale.is_meta)
        self.assertFalse(layer.w2_scale.is_meta)
        self.assertEqual(layer.w13_scale.device.type, "cpu")
        self.assertEqual(layer.w2_scale.device.type, "cpu")

        fp8_weight = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
        layer.load_weights(
            {
                "0.gate_proj.weight": fp8_weight,
                "0.gate_proj.weight_scale_inv": torch.ones(2, 1),
                "0.up_proj.weight": fp8_weight,
                "0.up_proj.weight_scale_inv": torch.ones(2, 1) * 2,
                "0.down_proj.weight": fp8_weight,
                "0.down_proj.weight_scale_inv": torch.ones(2, 1) * 3,
            }
        )
        self.assertFalse(layer.w13_scale.is_meta)
        self.assertFalse(layer.w2_scale.is_meta)

        def fake_dequant(weight, scale, block_size, dtype):
            self.assertFalse(scale.is_meta)
            return torch.zeros_like(weight, dtype=torch.bfloat16)

        def fake_cast(weight, use_ue8m0=False):
            return (
                torch.zeros_like(weight, dtype=fp8_moe._runtime_fp8_dtype()),
                torch.ones(2, 1, dtype=torch.float32),
            )

        with mock.patch.object(BaseMoEExperts, "_maybe_build_fused_moe", return_value=None), \
             mock.patch(
                 "rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel.block_quant_dequant",
                 side_effect=fake_dequant,
             ), \
             mock.patch(
                 "rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel.per_block_cast_to_fp8",
                 side_effect=fake_cast,
             ):
            layer.process_weights_after_loading()

        self.assertFalse(layer.w13_scale.is_meta)
        self.assertFalse(layer.w2_scale.is_meta)
        self.assertEqual(layer.w13_scale.shape, (1, 2, 1))
        self.assertEqual(layer.w2_scale.shape, (1, 2, 1))

class TestFp8BlockLoad(unittest.TestCase):
    """Already-quantized FP8 per-block (128x128) -> {Col,Row,Merged,QKV}Parallel.

    The ckpt provides:
      - weight: float8_e4m3fn [N, K]
      - weight_scale_inv: fp32 [ceil(N/128), ceil(K/128)]
    After load_weights + process_weights_after_loading, the param is renamed
    to `weight_scale` so apply() can share the contract with the online sibling.
    """

    BLOCK = 128

    def test_block_online_still_rejects_cpu_post_load_when_force_cpu_enabled(self):
        layer = ColumnParallelLinear(
            input_size=4,
            output_size=4,
            quant_config=_make_qc("fp8_block_online"),
            prefix="online_block",
            params_dtype=torch.bfloat16,
        )
        layer.load_weights(
            {"online_block.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}
        )
        layer._new_loader_force_cpu_load_weights = True
        with self.assertRaisesRegex(RuntimeError, "requires weights to be on CUDA"):
            layer.process_weights_after_loading()

    def test_load_into_column_parallel(self):
        N, K = 256, 256  # 2 blocks each
        qc = _make_qc("fp8_block")
        layer = ColumnParallelLinear(
            input_size=K,
            output_size=N,
            tp_size=1,
            tp_rank=0,
            quant_config=qc,
            prefix="test",
            params_dtype=torch.bfloat16,
        )
        fp8_weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
        scale_inv = (
            torch.rand(N // self.BLOCK, K // self.BLOCK, dtype=torch.float32) + 0.01
        )

        layer.load_weights(
            {
                "test.weight": fp8_weight,
                "test.weight_scale_inv": scale_inv,
            }
        )
        layer.process_weights_after_loading()

        self.assertEqual(layer.weight.shape, (N, K))
        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        if has_deep_gemm():
            self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(layer.weight_scale.shape, (2, 2))
            torch.testing.assert_close(layer.weight_scale, scale_inv, rtol=0, atol=0)
        else:
            self.assertEqual(layer.weight.dtype, torch.bfloat16)
            self.assertFalse(hasattr(layer, "weight_scale"))

    def test_load_into_column_parallel_tp(self):
        N, K = 512, 256  # N=4 blocks total, 2 blocks per rank
        tp = 2
        qc = _make_qc("fp8_block")
        layer = ColumnParallelLinear(
            input_size=K,
            output_size=N,
            tp_size=tp,
            tp_rank=1,
            quant_config=qc,
            prefix="test",
            params_dtype=torch.bfloat16,
        )
        fp8_weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
        scale_inv = (
            torch.arange(
                (N // self.BLOCK) * (K // self.BLOCK), dtype=torch.float32
            ).reshape(N // self.BLOCK, K // self.BLOCK)
            + 0.01
        )

        layer.load_weights(
            {
                "test.weight": fp8_weight,
                "test.weight_scale_inv": scale_inv,
            }
        )
        layer.process_weights_after_loading()

        # Rank 1 should get the second half of the dim-0 blocks.
        if has_deep_gemm():
            self.assertEqual(layer.weight_scale.shape, (2, 2))
            torch.testing.assert_close(
                layer.weight_scale, scale_inv[2:4, :], rtol=0, atol=0
            )
        else:
            self.assertEqual(layer.weight.dtype, torch.bfloat16)
            self.assertFalse(hasattr(layer, "weight_scale"))

    def test_load_into_row_parallel_tp(self):
        N, K = 256, 512  # K=4 blocks, 2 per rank along K
        tp = 2
        qc = _make_qc("fp8_block")
        layer = RowParallelLinear(
            input_size=K,
            output_size=N,
            tp_size=tp,
            tp_rank=1,
            quant_config=qc,
            prefix="test",
            params_dtype=torch.bfloat16,
        )
        fp8_weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
        scale_inv = (
            torch.arange(
                (N // self.BLOCK) * (K // self.BLOCK), dtype=torch.float32
            ).reshape(N // self.BLOCK, K // self.BLOCK)
            + 0.01
        )

        layer.load_weights(
            {
                "test.weight": fp8_weight,
                "test.weight_scale_inv": scale_inv,
            }
        )
        layer.process_weights_after_loading()

        # Rank 1 should get the second half of the dim-1 blocks.
        if has_deep_gemm():
            self.assertEqual(layer.weight_scale.shape, (2, 2))
            torch.testing.assert_close(
                layer.weight_scale, scale_inv[:, 2:4], rtol=0, atol=0
            )
        else:
            self.assertEqual(layer.weight.dtype, torch.bfloat16)
            self.assertFalse(hasattr(layer, "weight_scale"))

    def test_load_into_merged_column(self):
        H = 256  # input
        N = 256  # gate/up shard size each (= 2 blocks); total 4 blocks output
        qc = _make_qc("fp8_block")
        layer = MergedColumnParallelLinear(
            input_size=H,
            output_size=2 * N,
            tp_size=1,
            tp_rank=0,
            quant_config=qc,
            prefix="gate_up_proj",
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.bfloat16,
        )
        gate_w = torch.zeros(N, H, dtype=torch.float8_e4m3fn)
        up_w = torch.zeros(N, H, dtype=torch.float8_e4m3fn)
        gate_scale = torch.full(
            (N // self.BLOCK, H // self.BLOCK), 0.1, dtype=torch.float32
        )
        up_scale = torch.full(
            (N // self.BLOCK, H // self.BLOCK), 0.2, dtype=torch.float32
        )

        layer.load_weights(
            {
                "gate_up_proj.gate_proj.weight": gate_w,
                "gate_up_proj.gate_proj.weight_scale_inv": gate_scale,
                "gate_up_proj.up_proj.weight": up_w,
                "gate_up_proj.up_proj.weight_scale_inv": up_scale,
            }
        )
        layer.process_weights_after_loading()

        # Total N_blocks = 2 (gate) + 2 (up) = 4; K_blocks = 2.
        if has_deep_gemm():
            self.assertEqual(layer.weight_scale.shape, (4, 2))
            torch.testing.assert_close(layer.weight_scale[:2], gate_scale, rtol=0, atol=0)
            torch.testing.assert_close(layer.weight_scale[2:], up_scale, rtol=0, atol=0)
        else:
            self.assertEqual(layer.weight.dtype, torch.bfloat16)
            self.assertFalse(hasattr(layer, "weight_scale"))

    def test_load_into_qkv_parallel(self):
        hidden = 256  # K = 2 blocks
        head_dim = 128
        num_heads = 4  # q_size = 512 = 4 blocks
        num_kv_heads = 2  # kv_size = 256 = 2 blocks per kv shard

        qc = _make_qc("fp8_block")
        layer = QKVParallelLinear(
            hidden_size=hidden,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tp_size=1,
            tp_rank=0,
            quant_config=qc,
            prefix="qkv_proj",
            params_dtype=torch.bfloat16,
        )

        q_w = torch.zeros(num_heads * head_dim, hidden, dtype=torch.float8_e4m3fn)
        k_w = torch.zeros(num_kv_heads * head_dim, hidden, dtype=torch.float8_e4m3fn)
        v_w = torch.zeros(num_kv_heads * head_dim, hidden, dtype=torch.float8_e4m3fn)

        q_scale = torch.full(
            (num_heads * head_dim // self.BLOCK, hidden // self.BLOCK),
            0.1,
            dtype=torch.float32,
        )
        k_scale = torch.full(
            (num_kv_heads * head_dim // self.BLOCK, hidden // self.BLOCK),
            0.2,
            dtype=torch.float32,
        )
        v_scale = torch.full(
            (num_kv_heads * head_dim // self.BLOCK, hidden // self.BLOCK),
            0.3,
            dtype=torch.float32,
        )

        layer.load_weights(
            {
                "qkv_proj.q_proj.weight": q_w,
                "qkv_proj.q_proj.weight_scale_inv": q_scale,
                "qkv_proj.k_proj.weight": k_w,
                "qkv_proj.k_proj.weight_scale_inv": k_scale,
                "qkv_proj.v_proj.weight": v_w,
                "qkv_proj.v_proj.weight_scale_inv": v_scale,
            }
        )
        layer.process_weights_after_loading()

        q_blocks = num_heads * head_dim // self.BLOCK
        kv_blocks = num_kv_heads * head_dim // self.BLOCK
        total_blocks = q_blocks + 2 * kv_blocks
        k_blocks_in = hidden // self.BLOCK
        if has_deep_gemm():
            self.assertEqual(layer.weight_scale.shape, (total_blocks, k_blocks_in))
            torch.testing.assert_close(
                layer.weight_scale[:q_blocks], q_scale, rtol=0, atol=0
            )
            torch.testing.assert_close(
                layer.weight_scale[q_blocks : q_blocks + kv_blocks],
                k_scale,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                layer.weight_scale[q_blocks + kv_blocks :], v_scale, rtol=0, atol=0
            )
        else:
            self.assertEqual(layer.weight.dtype, torch.bfloat16)
            self.assertFalse(hasattr(layer, "weight_scale"))

    def test_qkv_block_scale_uses_custom_block_size_with_tp(self):
        hidden = 256
        head_dim = 64
        num_heads = 4
        num_kv_heads = 4
        tp = 2
        tp_rank = 1
        block_n = 64
        block_k = 128

        qc = _make_qc("fp8_block")
        qc.weight_block_size = [block_n, block_k]
        layer = QKVParallelLinear(
            hidden_size=hidden,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tp_size=tp,
            tp_rank=tp_rank,
            quant_config=qc,
            prefix="qkv_proj",
            params_dtype=torch.bfloat16,
        )

        q_w = torch.zeros(num_heads * head_dim, hidden, dtype=torch.float8_e4m3fn)
        k_w = torch.zeros(num_kv_heads * head_dim, hidden, dtype=torch.float8_e4m3fn)
        v_w = torch.zeros(num_kv_heads * head_dim, hidden, dtype=torch.float8_e4m3fn)

        in_blocks = hidden // block_k
        rows_per_rank = (num_heads // tp) * head_dim
        blocks_per_rank = rows_per_rank // block_n
        q_blocks = num_heads * head_dim // block_n
        kv_blocks = num_kv_heads * head_dim // block_n
        start = tp_rank * blocks_per_rank

        q_scale = torch.arange(q_blocks * in_blocks, dtype=torch.float32).reshape(
            q_blocks, in_blocks
        )
        k_scale = (
            torch.arange(kv_blocks * in_blocks, dtype=torch.float32).reshape(
                kv_blocks, in_blocks
            )
            + 100
        )
        v_scale = (
            torch.arange(kv_blocks * in_blocks, dtype=torch.float32).reshape(
                kv_blocks, in_blocks
            )
            + 200
        )

        layer.load_weights(
            {
                "qkv_proj.q_proj.weight": q_w,
                "qkv_proj.q_proj.weight_scale_inv": q_scale,
                "qkv_proj.k_proj.weight": k_w,
                "qkv_proj.k_proj.weight_scale_inv": k_scale,
                "qkv_proj.v_proj.weight": v_w,
                "qkv_proj.v_proj.weight_scale_inv": v_scale,
            }
        )

        expected = torch.cat(
            [
                q_scale[start : start + blocks_per_rank],
                k_scale[start : start + blocks_per_rank],
                v_scale[start : start + blocks_per_rank],
            ],
            dim=0,
        )
        torch.testing.assert_close(
            layer.weight_scale_inv.detach(), expected, rtol=0, atol=0
        )

    def test_fp8_block_dequant_uses_custom_block_size(self):
        N, K = 4, 8
        qc = _make_qc("fp8_block_dequant")
        qc.weight_block_size = [2, 4]
        layer = ColumnParallelLinear(
            input_size=K,
            output_size=N,
            tp_size=1,
            tp_rank=0,
            quant_config=qc,
            prefix="dequant",
            params_dtype=torch.bfloat16,
        )
        weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
        scale = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        expected = torch.full((N, K), 3.0, dtype=torch.bfloat16)
        layer.load_weights(
            {
                "dequant.weight": weight,
                "dequant.weight_scale_inv": scale,
            }
        )
        with mock.patch(
            "rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel.block_quant_dequant",
            return_value=expected,
        ) as dequant:
            layer.process_weights_after_loading()
        dequant.assert_called_once()
        self.assertEqual(dequant.call_args.args[2], [2, 4])
        torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)
        self.assertFalse(hasattr(layer, "weight_scale_inv"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA + DeepGEMM")
class TestFp8BlockForward(unittest.TestCase):
    """End-to-end: load fp8-block ckpt vs online path, forward should match."""

    def test_forward_matches_online_path(self):
        try:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import per_block_cast_to_fp8
        except ImportError:
            self.skipTest("deepgemm/fp8_kernel not available")
        if not has_deep_gemm():
            self.skipTest("DeepGEMM kernel not available at runtime")

        N, K, M = 256, 256, 32  # 2x2 weight blocks
        device = "cuda"

        weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.05
        # Use the same kernel the online path uses to produce fp8 + scale.
        fp8_weight, scale = per_block_cast_to_fp8(weight_bf16, use_ue8m0=False)

        # Offline path
        offline = ColumnParallelLinear(
            input_size=K,
            output_size=N,
            tp_size=1,
            tp_rank=0,
            quant_config=_make_qc("fp8_block"),
            prefix="off",
            params_dtype=torch.bfloat16,
        ).to(device)
        offline.load_weights(
            {
                "off.weight": fp8_weight,
                "off.weight_scale_inv": scale,
            }
        )
        offline.process_weights_after_loading()

        # Online path
        online = ColumnParallelLinear(
            input_size=K,
            output_size=N,
            tp_size=1,
            tp_rank=0,
            quant_config=_make_qc("fp8_block_online"),
            prefix="on",
            params_dtype=torch.bfloat16,
        ).to(device)
        online.load_weights({"on.weight": weight_bf16})
        online.process_weights_after_loading()

        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
        out_offline = offline(x)
        out_online = online(x)

        torch.testing.assert_close(out_offline, out_online, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available() and is_cuda(), "requires CUDA + torch._scaled_mm")
class TestFp8AlreadyQuantizedForward(unittest.TestCase):
    """End-to-end: load already-quantized ckpt -> forward via _scaled_mm."""

    def test_fp8_per_tensor_forward_matches_dequant_reference(self):
        N, K, M = 32, 64, 8
        device = "cuda"
        weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        scale = (weight_bf16.abs().max() / 448.0).float()
        fp8_weight = (weight_bf16 / scale).to(_runtime_fp8_dtype())

        qc = _make_qc("fp8")
        layer = ColumnParallelLinear(
            input_size=K,
            output_size=N,
            tp_size=1,
            tp_rank=0,
            quant_config=qc,
            prefix="test",
            params_dtype=torch.bfloat16,
        ).to(device)
        layer.load_weights(
            {
                "test.weight": fp8_weight.to(device),
                "test.weight_scale": scale.reshape(1).to(device),
                "test.input_scale": torch.tensor(
                    [1.0], dtype=torch.float32, device=device
                ),
            }
        )
        layer.process_weights_after_loading()

        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        out = layer(x)

        # Reference: dequant fp8 weight back to bf16 and compute regular linear.
        weight_dequant = fp8_weight.to(torch.bfloat16).to(device) * scale.to(device)
        ref = torch.nn.functional.linear(x, weight_dequant)

        # _scaled_mm with dynamic per-tensor activation quant introduces a
        # small relative error vs full-precision reference.
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.05)


if __name__ == "__main__":
    unittest.main()
