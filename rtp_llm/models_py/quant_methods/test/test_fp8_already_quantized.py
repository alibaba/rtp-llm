# type: ignore
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
import subprocess
import sys
import unittest
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import torch

from rtp_llm.config.quant_config import Fp8PerTensorCompressedQuantConfig
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.fp8 import (
    Fp8BlockLinearMethod,
    Fp8LinearMethod,
    Fp8OnlineLinearMethod,
    Fp8PerChannelLinearMethod,
    Fp8PerChannelOnlineLinearMethod,
    _convert_e4m3fn_to_fnuz,
    _is_hip_runtime,
    _resolve_per_tensor_quant,
    _runtime_fp8_dtype,
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod


def _make_qc(
    quant_type: str, activation_dynamic: Optional[bool] = None
) -> QuantizationConfig:
    source_config = None
    if activation_dynamic is not None:
        source_config = Fp8PerTensorCompressedQuantConfig(
            is_quanted=True,
            dynamic=activation_dynamic,
        )
    return QuantizationConfig(quant_type=quant_type, source_config=source_config)


def _runtime_scale(scale: torch.Tensor) -> torch.Tensor:
    if _runtime_fp8_dtype() == torch.float8_e4m3fnuz:
        return scale.float() * 2.0
    return scale.float()


def _channel_scale_shape(channels: int):
    return (channels, 1) if _is_hip_runtime() else (1, channels)


class TestQuantMethodDispatch(unittest.TestCase):
    def test_unquantized_dispatch_does_not_import_fp8_provider(self):
        script = """
import sys
from types import SimpleNamespace

from rtp_llm.models_py.quant_methods.base import QuantizationConfig

method = QuantizationConfig("none").get_quant_method(
    SimpleNamespace(shard_names=[]), "proj"
)
assert type(method).__name__ == "UnquantizedLinearMethod"
assert "rtp_llm.models_py.quant_methods.fp8" not in sys.modules
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_canonical_fp8_aliases_dispatch_to_expected_methods(self):
        layer = SimpleNamespace(shard_names=[])
        expected = {
            "FP8_PER_TENSOR_COMPRESSED": Fp8LinearMethod,
            "FP8_DYNAMIC_PER_TENSOR": Fp8OnlineLinearMethod,
            "FP8_PER_BLOCK": Fp8BlockLinearMethod,
            "FP8_PER_CHANNEL_COMPRESSED": Fp8PerChannelLinearMethod,
            "FP8_PER_CHANNEL_QUARK": Fp8PerChannelLinearMethod,
        }
        for quant_type, method_type in expected.items():
            with self.subTest(quant_type=quant_type):
                method = QuantizationConfig(quant_type).get_quant_method(
                    layer, "layers.0.self_attn.q_proj"
                )
                self.assertIsInstance(method, method_type)

    def test_fused_layer_requires_consistent_exclusions(self):
        layer = SimpleNamespace(shard_names=["gate_proj", "up_proj"])
        prefix = "layers.0.mlp.gate_up_proj"

        fully_ignored = QuantizationConfig(
            "fp8",
            ignored_layers=[
                "layers.0.mlp.gate_proj",
                "layers.0.mlp.up_proj",
            ],
        )
        self.assertIsInstance(
            fully_ignored.get_quant_method(layer, prefix), UnquantizedLinearMethod
        )

        partially_ignored = QuantizationConfig(
            "fp8", ignored_layers=["layers.0.mlp.gate_proj"]
        )
        with self.assertRaisesRegex(ValueError, "partially match fused layer"):
            partially_ignored.get_quant_method(layer, prefix)

        with self.assertRaisesRegex(ValueError, "stable module prefix"):
            partially_ignored.get_quant_method(layer)


class TestFp8PerTensorLoad(unittest.TestCase):
    """Already-quantized FP8 per-tensor compressed -> ColumnParallel."""

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
        fp8_weight = (weight_bf16 / scale).to(torch.float8_e4m3fn)

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
        self.assertAlmostEqual(
            layer.weight_scale.item(), _runtime_scale(scale).item(), places=4
        )

    def test_static_input_scale_is_required_but_dynamic_scale_is_not(self):
        weight = torch.ones(4, 4, dtype=torch.float8_e4m3fn)
        weight_scale = torch.tensor([0.25], dtype=torch.float32)

        for activation_dynamic in (True, False):
            with self.subTest(activation_dynamic=activation_dynamic):
                layer = ColumnParallelLinear(
                    input_size=4,
                    output_size=4,
                    quant_config=_make_qc("fp8", activation_dynamic),
                    prefix="test",
                    params_dtype=torch.bfloat16,
                )
                layer.load_weights(
                    {
                        "test.weight": weight,
                        "test.weight_scale": weight_scale,
                    }
                )
                if activation_dynamic:
                    layer.validate_weights_loaded()
                else:
                    with self.assertRaisesRegex(RuntimeError, "input_scale"):
                        layer.validate_weights_loaded()
                    layer.load_weights(
                        {"test.input_scale": torch.tensor([0.5], dtype=torch.float32)}
                    )
                    layer.validate_weights_loaded()

    def test_static_input_scale_must_be_finite_and_positive(self):
        for input_scale in (0.0, -0.5, float("nan"), float("inf")):
            with self.subTest(input_scale=input_scale):
                layer = ColumnParallelLinear(
                    input_size=4,
                    output_size=4,
                    quant_config=_make_qc("fp8", activation_dynamic=False),
                    prefix="test",
                    params_dtype=torch.bfloat16,
                )
                layer.load_weights(
                    {
                        "test.weight": torch.ones(4, 4, dtype=torch.float8_e4m3fn),
                        "test.weight_scale": torch.tensor([0.25], dtype=torch.float32),
                        "test.input_scale": torch.tensor(
                            [input_scale], dtype=torch.float32
                        ),
                    }
                )
                layer.validate_weights_loaded()
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    layer.process_weights_after_loading()


class TestFp8PerChannelLoad(unittest.TestCase):
    """Already-quantized FP8 per-channel (compressed/Quark) -> {Col,Row}Parallel."""

    def test_e4m3fn_to_fnuz_matches_legacy_bit_conversion(self):
        source_bits = torch.tensor([-128, -1, 0, 1, 127], dtype=torch.int8)
        weight = source_bits.view(torch.float8_e4m3fn)
        scale = torch.tensor([[0.5]], dtype=torch.float32)

        converted, converted_scale = _convert_e4m3fn_to_fnuz(weight, scale)

        self.assertEqual(converted.dtype, torch.float8_e4m3fnuz)
        torch.testing.assert_close(
            converted.view(torch.int8),
            torch.tensor([0, -1, 0, 1, 127], dtype=torch.int8),
        )
        torch.testing.assert_close(converted_scale, torch.tensor([[1.0]]))

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
        # normalize it to [N, 1].
        scale_1d = torch.rand(N, dtype=torch.float32) + 0.01
        fp8_weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)

        layer.load_weights(
            {
                "test.weight": fp8_weight,
                "test.weight_scale": scale_1d,
            }
        )
        layer.process_weights_after_loading()

        self.assertEqual(layer.weight_scale.shape, _channel_scale_shape(N))
        torch.testing.assert_close(
            layer.weight_scale.flatten(), _runtime_scale(scale_1d), rtol=0, atol=0
        )

    def test_load_into_row_parallel_no_scale_shard(self):
        # RowParallel splits along input dim; per-channel weight_scale
        # is on output side and must NOT be sharded.
        N, K = 16, 64
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
        fp8_weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
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
        self.assertEqual(layer.weight_scale.shape, _channel_scale_shape(N))
        torch.testing.assert_close(
            layer.weight_scale.flatten(),
            _runtime_scale(scale_2d).flatten(),
            rtol=0,
            atol=0,
        )

    def test_rocm_process_preshuffles_weight(self):
        N, K = 16, 32
        layer = ColumnParallelLinear(
            input_size=K,
            output_size=N,
            quant_config=_make_qc("fp8_per_channel"),
            prefix="test",
            params_dtype=torch.bfloat16,
        )
        weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
        scale = torch.ones(N, 1, dtype=torch.float32)
        layer.load_weights({"test.weight": weight, "test.weight_scale": scale})

        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._is_hip_runtime",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._runtime_fp8_dtype",
            return_value=torch.float8_e4m3fn,
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._shuffle_rocm_fp8_weight",
            side_effect=lambda tensor: tensor.contiguous(),
        ) as shuffle:
            layer.process_weights_after_loading()

        shuffle.assert_called_once()
        self.assertEqual(layer.weight_scale.shape, (N, 1))

    def test_rocm_apply_uses_legacy_aiter_ptpc_path(self):
        N, K = 16, 32
        method = Fp8PerChannelLinearMethod()
        layer = SimpleNamespace(
            weight=torch.zeros(N, K, dtype=torch.float8_e4m3fn),
            weight_scale=torch.ones(N, 1, dtype=torch.float32),
            prefix="test",
        )
        x = torch.randn(2, K, dtype=torch.bfloat16)
        expected = torch.randn(2, N, dtype=torch.bfloat16)

        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._is_hip_runtime",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._apply_rocm_fp8_per_channel",
            return_value=expected,
        ) as rocm_apply:
            actual = method.apply(layer, x)

        rocm_apply.assert_called_once()
        args = rocm_apply.call_args.args
        torch.testing.assert_close(args[0], x)
        self.assertIs(args[1], layer.weight)
        self.assertIs(args[2], layer.weight_scale)
        self.assertEqual(args[3], torch.bfloat16)
        torch.testing.assert_close(actual, expected)


@unittest.skipUnless(
    torch.cuda.is_available() and _is_hip_runtime(),
    "requires a ROCm GPU and AITer",
)
class TestRocmFp8PerChannelOnline(unittest.TestCase):
    def test_fp16_bf16_tp_weights_use_ptpc_runtime(self):
        import aiter
        from aiter.ops.shuffle import shuffle_weight

        from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8

        # Match the production PTPC shape family used by the tuned AITer path:
        # TP=2 leaves each rank with N=1024 and K=1024.
        output_size, input_size, batch = 2048, 1024, 4
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                weight = torch.randn(
                    output_size,
                    input_size,
                    dtype=dtype,
                    device="cuda",
                )
                layer = ColumnParallelLinear(
                    input_size=input_size,
                    output_size=output_size,
                    tp_size=2,
                    tp_rank=1,
                    quant_config=_make_qc("fp8_per_channel_online"),
                    prefix="test",
                    params_dtype=dtype,
                ).to("cuda")
                layer.load_weights({"test.weight": weight})
                layer.process_weights_after_loading()

                self.assertIsInstance(
                    layer.quant_method, Fp8PerChannelOnlineLinearMethod
                )
                self.assertEqual(layer.weight_scale.shape, (output_size // 2, 1))

                local_weight = weight[output_size // 2 :].contiguous()
                quant_input = (
                    local_weight
                    if dtype == torch.bfloat16
                    else local_weight.to(torch.bfloat16)
                )
                expected_weight, expected_weight_scale = rocm_per_token_quant_fp8(
                    quant_input,
                    eps=1e-10,
                )
                expected_weight = shuffle_weight(expected_weight, layout=(16, 16))
                torch.testing.assert_close(layer.weight, expected_weight)
                torch.testing.assert_close(
                    layer.weight_scale,
                    expected_weight_scale.to(torch.float32),
                )

                x = torch.randn(batch, input_size, dtype=dtype, device="cuda")
                actual = layer(x)
                x_bf16 = x if dtype == torch.bfloat16 else x.to(torch.bfloat16)
                qinput, x_scale = rocm_per_token_quant_fp8(x_bf16, eps=1e-10)
                expected = aiter.gemm_a8w8_bpreshuffle(
                    qinput,
                    expected_weight,
                    x_scale.to(torch.float32),
                    expected_weight_scale.to(torch.float32),
                    None,
                    torch.bfloat16,
                )
                if dtype == torch.float16:
                    expected = expected.to(dtype)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, expected)


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
        gate_w = torch.zeros(N, H, dtype=torch.float8_e4m3fn)
        up_w = torch.zeros(N, H, dtype=torch.float8_e4m3fn)
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

        # process_weights_after_loading reshapes [2N] -> [2N, 1]
        self.assertEqual(layer.weight_scale.shape, _channel_scale_shape(2 * N))
        loaded_scale = layer.weight_scale.flatten()
        torch.testing.assert_close(
            loaded_scale[:N], _runtime_scale(gate_scale), rtol=0, atol=0
        )
        torch.testing.assert_close(
            loaded_scale[N:], _runtime_scale(up_scale), rtol=0, atol=0
        )

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

        expected = _runtime_scale(torch.tensor([0.3])).item()
        self.assertAlmostEqual(layer.weight_scale.item(), expected, places=5)


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

        q_w = torch.zeros(num_heads * head_dim, hidden, dtype=torch.float8_e4m3fn)
        k_w = torch.zeros(num_kv_heads * head_dim, hidden, dtype=torch.float8_e4m3fn)
        v_w = torch.zeros(num_kv_heads * head_dim, hidden, dtype=torch.float8_e4m3fn)

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
        self.assertEqual(layer.weight_scale.shape, _channel_scale_shape(total))
        loaded_scale = layer.weight_scale.flatten()
        # q segment
        torch.testing.assert_close(
            loaded_scale[: num_heads * head_dim],
            _runtime_scale(q_scale),
            rtol=0,
            atol=0,
        )
        # k segment
        torch.testing.assert_close(
            loaded_scale[
                num_heads * head_dim : num_heads * head_dim + num_kv_heads * head_dim
            ],
            _runtime_scale(k_scale),
            rtol=0,
            atol=0,
        )
        # v segment
        torch.testing.assert_close(
            loaded_scale[num_heads * head_dim + num_kv_heads * head_dim :],
            _runtime_scale(v_scale),
            rtol=0,
            atol=0,
        )


class TestFp8BlockLoad(unittest.TestCase):
    """Already-quantized FP8 per-block (128x128) -> {Col,Row,Merged,QKV}Parallel.

    The ckpt provides:
      - weight: float8_e4m3fn [N, K]
      - weight_scale_inv: fp32 [ceil(N/128), ceil(K/128)]
    With DeepGEMM, post-processing renames the scale to `weight_scale`.
    Without it, the method dequantizes once to the BF16 fallback layout.
    """

    BLOCK = 128

    def _assert_runtime_state(self, layer, expected_scale):
        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        if has_deep_gemm():
            self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
            if is_deep_gemm_e8m0_used():
                self.assertEqual(layer.weight_scale.dtype, torch.int32)
            else:
                torch.testing.assert_close(
                    layer.weight_scale, expected_scale, rtol=0, atol=0
                )
        else:
            self.assertEqual(layer.weight.dtype, torch.bfloat16)
            self.assertFalse(hasattr(layer, "weight_scale"))

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
        torch.testing.assert_close(layer.weight_scale_inv, scale_inv, rtol=0, atol=0)
        layer.process_weights_after_loading()

        self.assertEqual(layer.weight.shape, (N, K))
        self._assert_runtime_state(layer, scale_inv)

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
        expected_scale = scale_inv[2:4, :]
        torch.testing.assert_close(
            layer.weight_scale_inv, expected_scale, rtol=0, atol=0
        )
        layer.process_weights_after_loading()

        self._assert_runtime_state(layer, expected_scale)

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
        expected_scale = scale_inv[:, 2:4]
        torch.testing.assert_close(
            layer.weight_scale_inv, expected_scale, rtol=0, atol=0
        )
        layer.process_weights_after_loading()

        self._assert_runtime_state(layer, expected_scale)

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
        expected_scale = torch.cat([gate_scale, up_scale])
        torch.testing.assert_close(
            layer.weight_scale_inv, expected_scale, rtol=0, atol=0
        )
        layer.process_weights_after_loading()

        self._assert_runtime_state(layer, expected_scale)

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
        expected_scale = torch.cat([q_scale, k_scale, v_scale])
        torch.testing.assert_close(
            layer.weight_scale_inv, expected_scale, rtol=0, atol=0
        )
        layer.process_weights_after_loading()

        self._assert_runtime_state(layer, expected_scale)

    def test_qkv_tp2_square_q_weight_keeps_output_row_layout(self):
        hidden = 256
        head_dim = 128
        num_heads = 2
        num_kv_heads = 2

        def row_pattern(rows, offset):
            values = (torch.arange(rows) % 16 + offset).view(rows, 1)
            return values.expand(rows, hidden).to(torch.float8_e4m3fn)

        q_w = row_pattern(num_heads * head_dim, 1)
        k_w = row_pattern(num_kv_heads * head_dim, 33)
        v_w = row_pattern(num_kv_heads * head_dim, 65)
        q_scale = torch.arange(4, dtype=torch.float32).view(2, 2) + 1
        k_scale = q_scale + 10
        v_scale = q_scale + 20

        for rank in range(2):
            layer = QKVParallelLinear(
                hidden_size=hidden,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tp_size=2,
                tp_rank=rank,
                quant_config=_make_qc("fp8_block"),
                prefix="qkv_proj",
                params_dtype=torch.bfloat16,
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

            start = rank * head_dim
            expected_weight = torch.cat(
                [
                    q_w[start : start + head_dim],
                    k_w[start : start + head_dim],
                    v_w[start : start + head_dim],
                ]
            )
            expected_scale = torch.cat(
                [
                    q_scale[rank : rank + 1],
                    k_scale[rank : rank + 1],
                    v_scale[rank : rank + 1],
                ]
            )
            torch.testing.assert_close(layer.weight, expected_weight, rtol=0, atol=0)
            torch.testing.assert_close(
                layer.weight_scale_inv, expected_scale, rtol=0, atol=0
            )


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

        if is_deep_gemm_e8m0_used():
            self.assertEqual(offline.weight_scale.dtype, torch.int32)
            self.assertEqual(online.weight_scale.dtype, torch.int32)
        else:
            self.assertEqual(offline.weight_scale.dtype, torch.float32)
            self.assertEqual(online.weight_scale.dtype, torch.float32)

        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
        out_offline = offline(x)
        out_online = online(x)

        # The offline checkpoint and online quantizer use equivalent FP8 block
        # semantics but may round one BF16 ULP differently before GEMM.
        torch.testing.assert_close(out_offline, out_online, rtol=0.02, atol=0.005)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA + torch._scaled_mm")
class TestFp8AlreadyQuantizedForward(unittest.TestCase):
    """End-to-end: load already-quantized ckpt -> forward via _scaled_mm."""

    def test_fp8_per_tensor_static_and_dynamic_forward_match_reference(self):
        N, K, M = 32, 64, 8
        device = "cuda"
        weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        scale = (weight_bf16.abs().max() / 448.0).float()
        fp8_weight = (weight_bf16 / scale).to(torch.float8_e4m3fn)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        dynamic_scale = (x.abs().max() / 448.0).float().reshape(1)
        static_scale = dynamic_scale * 2.0

        for activation_dynamic in (True, False):
            with self.subTest(activation_dynamic=activation_dynamic):
                layer = ColumnParallelLinear(
                    input_size=K,
                    output_size=N,
                    tp_size=1,
                    tp_rank=0,
                    quant_config=_make_qc("fp8", activation_dynamic),
                    prefix="test",
                    params_dtype=torch.bfloat16,
                ).to(device)
                checkpoint = {
                    "test.weight": fp8_weight.to(device),
                    "test.weight_scale": scale.reshape(1).to(device),
                }
                if not activation_dynamic:
                    checkpoint["test.input_scale"] = static_scale
                layer.load_weights(checkpoint)
                layer.validate_weights_loaded()
                layer.process_weights_after_loading()

                out = layer(x)
                expected_input_scale = None if activation_dynamic else static_scale
                qinput, x_scale = _resolve_per_tensor_quant()(x, expected_input_scale)
                if not activation_dynamic:
                    torch.testing.assert_close(x_scale, static_scale)
                input_dequant = qinput.float() * x_scale.float()
                weight_dequant = layer.weight.float() * layer.weight_scale.float()
                ref = torch.nn.functional.linear(input_dequant, weight_dequant).to(
                    out.dtype
                )

                # _scaled_mm and the BF16 reference differ only in accumulation.
                torch.testing.assert_close(out, ref, rtol=0.05, atol=0.05)


if __name__ == "__main__":
    unittest.main()
