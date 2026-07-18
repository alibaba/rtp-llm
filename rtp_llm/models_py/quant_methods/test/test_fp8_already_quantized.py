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
    _select_fp8_runtime_backend,
    is_deep_gemm_e8m0_used,
    per_block_quant_like_legacy,
)
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod


def _make_qc(
    quant_type: str,
    activation_dynamic: Optional[bool] = None,
    hw_kernel_config=None,
) -> QuantizationConfig:
    source_config = None
    if activation_dynamic is not None:
        source_config = Fp8PerTensorCompressedQuantConfig(
            is_quanted=True,
            dynamic=activation_dynamic,
        )
    return QuantizationConfig(
        quant_type=quant_type,
        source_config=source_config,
        hw_kernel_config=hw_kernel_config,
    )


def _runtime_scale(scale: torch.Tensor) -> torch.Tensor:
    return scale.float()


class _LoadBackendMixin:
    runtime_backend = "cuda_scaled_mm"

    def setUp(self):
        self._platform_patchers = [
            mock.patch(
                "rtp_llm.models_py.quant_methods.fp8._select_fp8_runtime_backend",
                return_value=self.runtime_backend,
            ),
            mock.patch(
                "rtp_llm.models_py.quant_methods.fp8._is_hip_runtime",
                return_value=False,
            ),
            mock.patch(
                "rtp_llm.models_py.quant_methods.fp8._runtime_fp8_dtype",
                return_value=torch.float8_e4m3fn,
            ),
            mock.patch(
                "rtp_llm.models_py.quant_methods.fp8.is_deep_gemm_e8m0_used",
                return_value=False,
            ),
            mock.patch(
                "rtp_llm.models_py.quant_methods.fp8._shuffle_rocm_fp8_weight",
                side_effect=AssertionError(
                    "CPU routing tests must not invoke ROCm weight shuffle"
                ),
            ),
            mock.patch(
                "rtp_llm.models_py.quant_methods.fp8._resolve_requant_weight_ue8m0",
                side_effect=AssertionError(
                    "CPU routing tests must not invoke SM100 UE8M0 conversion"
                ),
            ),
        ]
        for patcher in self._platform_patchers:
            patcher.start()
        super().setUp()

    def tearDown(self):
        for patcher in reversed(self._platform_patchers):
            patcher.stop()
        super().tearDown()


def _channel_scale_shape(channels: int):
    return (1, channels)


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

    def test_rocm_ptpc_selector_honors_swizzle_configuration(self):
        properties = SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-")
        with mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 4)
        ), mock.patch.object(
            torch.cuda, "get_device_properties", return_value=properties
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._is_hip_runtime",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._aiter_has_symbol",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._rocm_cktile_ptpc_available",
            return_value=True,
        ):
            self.assertEqual(
                _select_fp8_runtime_backend(
                    torch.device("cuda:0"),
                    "per_channel",
                    SimpleNamespace(use_swizzleA=False),
                ),
                "rocm_aiter_ptpc",
            )
            self.assertEqual(
                _select_fp8_runtime_backend(
                    torch.device("cuda:0"),
                    "per_channel",
                    SimpleNamespace(use_swizzleA=True),
                ),
                "rocm_hipblaslt_ptpc",
            )


class TestFp8PerTensorLoad(_LoadBackendMixin, unittest.TestCase):
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

        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
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


class TestFp8PerChannelLoad(_LoadBackendMixin, unittest.TestCase):
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

        executor = mock.Mock()
        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._select_fp8_runtime_backend",
            return_value="rocm_aiter_ptpc",
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._is_hip_runtime",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._runtime_fp8_dtype",
            return_value=torch.float8_e4m3fn,
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._prepare_rocm_fp8_ptpc_executor",
            return_value=executor,
        ) as prepare:
            layer.process_weights_after_loading()

        prepare.assert_called_once()
        self.assertIs(layer.quant_method._rocm_executor, executor)
        self.assertEqual(layer.weight_scale.shape, (N, 1))

    def test_rocm_apply_uses_legacy_aiter_ptpc_path(self):
        N, K = 16, 32
        expected = torch.randn(2, N, dtype=torch.bfloat16)
        method = Fp8PerChannelLinearMethod()
        method._runtime_backend = "rocm_aiter_ptpc"
        method._rocm_executor = mock.Mock(return_value=expected)
        layer = SimpleNamespace(
            weight=torch.zeros(N, K, dtype=torch.float8_e4m3fn),
            weight_scale=torch.ones(N, 1, dtype=torch.float32),
            prefix="test",
            output_size=N,
        )
        x = torch.randn(2, K, dtype=torch.bfloat16)
        actual = method.apply(layer, x)

        method._rocm_executor.assert_called_once()
        torch.testing.assert_close(method._rocm_executor.call_args.args[0], x)
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
        for use_swizzle_a in (False, True):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype, use_swizzleA=use_swizzle_a):
                    self._check_online_ptpc_runtime(
                        aiter,
                        shuffle_weight,
                        rocm_per_token_quant_fp8,
                        output_size,
                        input_size,
                        batch,
                        dtype,
                        use_swizzle_a,
                    )

    def _check_online_ptpc_runtime(
        self,
        aiter,
        shuffle_weight,
        rocm_per_token_quant_fp8,
        output_size,
        input_size,
        batch,
        dtype,
        use_swizzle_a,
    ):
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
            quant_config=_make_qc(
                "fp8_per_channel_online",
                hw_kernel_config=SimpleNamespace(use_swizzleA=use_swizzle_a),
            ),
            prefix="test",
            params_dtype=dtype,
        ).to("cuda")
        layer.load_weights({"test.weight": weight})
        layer.process_weights_after_loading()

        self.assertIsInstance(layer.quant_method, Fp8PerChannelOnlineLinearMethod)
        expected_executor = (
            "RocmFp8PTPCLinearWithSwizzle"
            if use_swizzle_a
            else "run_rocm_fp8_ptpc_no_swizzle"
        )
        actual_executor = (
            type(layer.quant_method._rocm_executor).__name__
            if use_swizzle_a
            else layer.quant_method._rocm_executor.__name__
        )
        self.assertEqual(
            actual_executor,
            expected_executor,
        )

        local_weight = weight[output_size // 2 :].contiguous()
        quant_input = (
            local_weight if dtype == torch.bfloat16 else local_weight.to(torch.bfloat16)
        )
        expected_weight, expected_weight_scale = rocm_per_token_quant_fp8(
            quant_input,
            eps=1e-10,
        )
        expected_weight = shuffle_weight(expected_weight, layout=(16, 16))
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
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


class TestMergedColumnPerChannelScale(_LoadBackendMixin, unittest.TestCase):
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


class TestQkvPerChannelScale(_LoadBackendMixin, unittest.TestCase):
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


class TestFp8BlockLoad(_LoadBackendMixin, unittest.TestCase):
    """Already-quantized FP8 per-block (128x128) -> {Col,Row,Merged,QKV}Parallel.

    The ckpt provides:
      - weight: float8_e4m3fn [N, K]
      - weight_scale_inv: fp32 [ceil(N/128), ceil(K/128)]
    Load-routing tests use an explicit fake backend; real CUDA and ROCm block
    kernels are exercised by TestFp8BlockForward.
    """

    BLOCK = 128
    runtime_backend = "cuda_deep_gemm"

    def test_noncanonical_block_size_fails_before_weight_allocation(self):
        quant_config = QuantizationConfig(
            "fp8_block",
            source_config=SimpleNamespace(weight_block_size=[64, 128]),
        )
        with self.assertRaisesRegex(ValueError, "supports only.*128, 128"):
            ColumnParallelLinear(
                input_size=256,
                output_size=256,
                quant_config=quant_config,
                prefix="test",
                params_dtype=torch.bfloat16,
            )

    def _assert_runtime_state(self, layer, expected_scale):
        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        self.assertTrue(layer.quant_method._use_deep_gemm)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(layer.weight_scale, expected_scale, rtol=0, atol=0)

    def test_unsupported_runtime_fails_before_expanding_block_weights(self):
        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._select_fp8_runtime_backend",
            side_effect=RuntimeError("unsupported FP8 block runtime"),
        ):
            online = ColumnParallelLinear(
                input_size=self.BLOCK,
                output_size=self.BLOCK,
                quant_config=_make_qc("fp8_block_online"),
                prefix="online",
                params_dtype=torch.bfloat16,
            )
            online.load_weights(
                {
                    "online.weight": torch.ones(
                        self.BLOCK, self.BLOCK, dtype=torch.bfloat16
                    )
                }
            )
            with self.assertRaisesRegex(RuntimeError, "unsupported FP8 block"):
                online.process_weights_after_loading()

            offline = ColumnParallelLinear(
                input_size=self.BLOCK,
                output_size=self.BLOCK,
                quant_config=_make_qc("fp8_block"),
                prefix="offline",
                params_dtype=torch.bfloat16,
            )
            offline.load_weights(
                {
                    "offline.weight": torch.zeros(
                        self.BLOCK, self.BLOCK, dtype=torch.float8_e4m3fn
                    ),
                    "offline.weight_scale_inv": torch.ones(1, 1, dtype=torch.float32),
                }
            )
            with self.assertRaisesRegex(RuntimeError, "unsupported FP8 block"):
                offline.process_weights_after_loading()

        self.assertEqual(online.weight.dtype, torch.bfloat16)
        self.assertTrue(hasattr(online, "weight_scale"))
        self.assertEqual(offline.weight.dtype, torch.float8_e4m3fn)
        self.assertTrue(hasattr(offline, "weight_scale_inv"))

    def test_scaled_mm_rejects_unsupported_cuda_architecture(self):
        properties = SimpleNamespace(gcnArchName="")
        with mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 0)
        ), mock.patch.object(
            torch.cuda, "get_device_properties", return_value=properties
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._is_hip_runtime",
            return_value=False,
        ), mock.patch.object(
            torch, "_scaled_mm", create=True
        ):
            with self.assertRaisesRegex(RuntimeError, "SM89 or newer"):
                _select_fp8_runtime_backend(torch.device("cuda:0"), "per_tensor")

    def test_rocm_block_requires_aiter_kernel_symbol(self):
        properties = SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-")
        with mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 4)
        ), mock.patch.object(
            torch.cuda, "get_device_properties", return_value=properties
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._is_hip_runtime",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._aiter_has_symbol",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "No executable ROCm"):
                _select_fp8_runtime_backend(torch.device("cuda:0"), "block")

    @unittest.skipUnless(
        torch.cuda.is_available() and not _is_hip_runtime(),
        "requires a CUDA runtime",
    )
    def test_runtime_predicate_checks_symbol_and_cuda_arch(self):
        from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper

        with mock.patch.object(
            deepgemm_wrapper, "has_deep_gemm", return_value=True
        ), mock.patch.object(deepgemm_wrapper, "_fp8_gemm_nt_impl", None):
            self.assertFalse(
                deepgemm_wrapper.is_deep_gemm_runtime_available(
                    torch.device("cuda", torch.cuda.current_device())
                )
            )

        with mock.patch.object(
            deepgemm_wrapper, "has_deep_gemm", return_value=True
        ), mock.patch.object(
            deepgemm_wrapper, "_fp8_gemm_nt_impl", object()
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 9)
        ):
            self.assertFalse(
                deepgemm_wrapper.is_deep_gemm_runtime_available(
                    torch.device("cuda", torch.cuda.current_device())
                )
            )

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


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA/ROCm accelerator")
class TestFp8BlockForward(unittest.TestCase):
    """End-to-end: load fp8-block ckpt vs online path, forward should match."""

    def test_forward_matches_online_path(self):
        if not _is_hip_runtime():
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
                is_deep_gemm_runtime_available,
            )

            if not is_deep_gemm_runtime_available(torch.device("cuda")):
                self.skipTest("DeepGEMM kernel not available at runtime")

        N, K, M = 256, 256, 32  # 2x2 weight blocks
        device = "cuda"

        weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.05
        fp8_weight, scale = per_block_quant_like_legacy(weight_bf16, 128)

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

        self.assertEqual(offline.weight.dtype, _runtime_fp8_dtype())
        self.assertEqual(online.weight.dtype, _runtime_fp8_dtype())
        self.assertTrue(hasattr(offline, "weight_scale"))
        self.assertTrue(hasattr(online, "weight_scale"))

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

    @unittest.skipUnless(_is_hip_runtime(), "requires ROCm")
    def test_static_rocm_forward_does_not_revalidate_scale(self):
        N, K = 32, 64
        weight_scale = torch.tensor([0.25], dtype=torch.float32, device="cuda")
        layer = ColumnParallelLinear(
            input_size=K,
            output_size=N,
            quant_config=_make_qc("fp8", activation_dynamic=False),
            prefix="static",
            params_dtype=torch.bfloat16,
        ).to("cuda")
        layer.load_weights(
            {
                "static.weight": torch.ones(
                    N, K, dtype=torch.float8_e4m3fn, device="cuda"
                ),
                "static.weight_scale": weight_scale,
                "static.input_scale": torch.tensor(
                    [0.5], dtype=torch.float32, device="cuda"
                ),
            }
        )
        layer.validate_weights_loaded()
        layer.process_weights_after_loading()

        with mock.patch.object(
            torch,
            "isfinite",
            side_effect=AssertionError("forward must not validate static scale"),
        ):
            output = layer(torch.ones(2, K, dtype=torch.bfloat16, device="cuda"))
        self.assertEqual(tuple(output.shape), (2, N))


if __name__ == "__main__":
    unittest.main()
