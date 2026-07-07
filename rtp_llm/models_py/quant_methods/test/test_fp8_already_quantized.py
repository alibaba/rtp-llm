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
import unittest

import torch

from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


def _make_qc(quant_type: str) -> QuantizationConfig:
    return QuantizationConfig(quant_type=quant_type)


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

        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
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

        self.assertEqual(layer.weight_scale.shape, (N, 1))
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
        self.assertEqual(layer.weight_scale.shape, (N, 1))


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
        self.assertEqual(layer.weight_scale.shape, (2 * N, 1))
        torch.testing.assert_close(
            layer.weight_scale[:N, 0], gate_scale, rtol=0, atol=0
        )
        torch.testing.assert_close(layer.weight_scale[N:, 0], up_scale, rtol=0, atol=0)

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
        self.assertEqual(layer.weight_scale.shape, (total, 1))
        # q segment
        torch.testing.assert_close(
            layer.weight_scale[: num_heads * head_dim, 0], q_scale, rtol=0, atol=0
        )
        # k segment
        torch.testing.assert_close(
            layer.weight_scale[
                num_heads * head_dim : num_heads * head_dim + num_kv_heads * head_dim,
                0,
            ],
            k_scale,
            rtol=0,
            atol=0,
        )
        # v segment
        torch.testing.assert_close(
            layer.weight_scale[num_heads * head_dim + num_kv_heads * head_dim :, 0],
            v_scale,
            rtol=0,
            atol=0,
        )


class TestFp8BlockLoad(unittest.TestCase):
    """Already-quantized FP8 per-block (128x128) -> {Col,Row,Merged,QKV}Parallel.

    The ckpt provides:
      - weight: float8_e4m3fn [N, K]
      - weight_scale_inv: fp32 [ceil(N/128), ceil(K/128)]
    After load_weights + process_weights_after_loading, the param is renamed
    to `weight_scale` so apply() can share the contract with the online sibling.
    """

    BLOCK = 128

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

        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.weight.shape, (N, K))
        self.assertEqual(layer.weight_scale.shape, (2, 2))
        # Renamed: weight_scale_inv must be gone.
        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        torch.testing.assert_close(layer.weight_scale, scale_inv, rtol=0, atol=0)

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
        self.assertEqual(layer.weight_scale.shape, (2, 2))
        torch.testing.assert_close(
            layer.weight_scale, scale_inv[2:4, :], rtol=0, atol=0
        )

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
        self.assertEqual(layer.weight_scale.shape, (2, 2))
        torch.testing.assert_close(
            layer.weight_scale, scale_inv[:, 2:4], rtol=0, atol=0
        )

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
        self.assertEqual(layer.weight_scale.shape, (4, 2))
        torch.testing.assert_close(layer.weight_scale[:2], gate_scale, rtol=0, atol=0)
        torch.testing.assert_close(layer.weight_scale[2:], up_scale, rtol=0, atol=0)

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


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA + torch._scaled_mm")
class TestFp8AlreadyQuantizedForward(unittest.TestCase):
    """End-to-end: load already-quantized ckpt -> forward via _scaled_mm."""

    def test_fp8_per_tensor_forward_matches_dequant_reference(self):
        N, K, M = 32, 64, 8
        device = "cuda"
        weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        scale = (weight_bf16.abs().max() / 448.0).float()
        fp8_weight = (weight_bf16 / scale).to(torch.float8_e4m3fn)

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
