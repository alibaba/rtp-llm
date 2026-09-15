"""Tests for Qwen3Next attention QKV + output-gate projection fusion."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextAttention
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.test.utils.numeric_util import calc_diff, per_block_cast_to_fp8
from rtp_llm.utils.model_weight import W

HEAD_NUM = 4
KV_HEAD_NUM = 2
HEAD_DIM = 32
Q_SIZE = HEAD_NUM * HEAD_DIM
QKV_SIZE = Q_SIZE + 2 * KV_HEAD_NUM * HEAD_DIM
GATE_SIZE = Q_SIZE


class _FakeLinear(nn.Module):
    """Records the tensors LinearFactory would have received and runs a plain matmul."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.weight_scales = weight_scales

    def maybe_cache_quant_scale(self, max_len: int) -> None:
        del max_len

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight


class _RecordingQKNorm(nn.Module):
    """Records the shape and contiguity of the tensor handed to the QK norm."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Size, bool]] = []

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        self.calls.append((qkv.shape, qkv.is_contiguous()))
        return qkv * 2


class _StubQuantConfig:
    def __init__(self, method: str) -> None:
        self._method = method

    def get_method(self) -> str:
        return self._method


class _FirstQFmha:
    """Minimal FMHA stand-in that returns the Q slice of a packed qkv tensor."""

    def __init__(self, q_size: int) -> None:
        self.q_size = q_size

    def forward(
        self, qkv: torch.Tensor, kv_cache: object, layer_idx: int
    ) -> torch.Tensor:
        del kv_cache, layer_idx
        return qkv[..., : self.q_size]


def _fp8_tensor(shape: tuple[int, int], offset: int = 0) -> torch.Tensor:
    values = torch.arange(offset, offset + shape[0] * shape[1])
    values = (values.remainder(31).to(torch.float32) - 15) / 8
    return values.reshape(shape).to(torch.float8_e4m3fn)


def _mn_major_int32(mn: int, packed_k: int, offset: int = 0) -> torch.Tensor:
    """Build a packed-UE8M0 scale of shape [mn, packed_k] whose stride(-2) is 1."""
    values = torch.arange(offset, offset + mn * packed_k, dtype=torch.int32)
    return values.reshape(packed_k, mn).t()


def _block_dequant(weight_nk: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """Dequantize an FP8 per-block weight the way the DeepGEMM linear consumes it."""
    n, k = weight_nk.shape
    n_index = torch.arange(n) // 128
    k_index = torch.arange(k) // 128
    return weight_nk.to(torch.float32) * block_scales[n_index][:, k_index]


class TestQkvGateFusionBlocker(unittest.TestCase):
    """The pure decision function: which weight layouts may be fused."""

    def _blocker(
        self,
        weights: dict[str, torch.Tensor],
        quant_config: object | None = None,
        qkv_size: int = QKV_SIZE,
        gate_size: int = GATE_SIZE,
    ) -> str | None:
        attention = Qwen3NextAttention.__new__(Qwen3NextAttention)
        attention._qkv_size = qkv_size
        attention._gate_size = gate_size
        return attention._qkv_gate_fusion_blocker(weights, quant_config)

    def test_accepts_bf16(self) -> None:
        weights = {
            W.attn_qkv_w: torch.randn(384, QKV_SIZE, dtype=torch.bfloat16),
            W.attn_gate_w: torch.randn(384, GATE_SIZE, dtype=torch.bfloat16),
        }
        self.assertIsNone(self._blocker(weights))

    def test_accepts_fp8_per_block_float32_scales(self) -> None:
        hidden_size = 512
        weights = {
            W.attn_qkv_w: _fp8_tensor((hidden_size, QKV_SIZE)),
            W.attn_gate_w: _fp8_tensor((hidden_size, GATE_SIZE)),
            W.attn_qkv_s: torch.ones(hidden_size // 128, QKV_SIZE // 128),
            W.attn_gate_s: torch.ones(hidden_size // 128, GATE_SIZE // 128),
        }
        self.assertIsNone(self._blocker(weights, _StubQuantConfig("FP8_PER_BLOCK")))

    def test_accepts_fp8_per_block_e8m0_scales(self) -> None:
        hidden_size = 1024
        packed_k = (hidden_size + 511) // 512
        weights = {
            W.attn_qkv_w: _fp8_tensor((QKV_SIZE, hidden_size)),
            W.attn_gate_w: _fp8_tensor((GATE_SIZE, hidden_size)),
            W.attn_qkv_s: _mn_major_int32(QKV_SIZE, packed_k),
            W.attn_gate_s: _mn_major_int32(GATE_SIZE, packed_k),
        }
        self.assertIsNone(self._blocker(weights, _StubQuantConfig("FP8_PER_BLOCK")))

    def test_declines_when_qkv_has_bias_or_fp4_markers(self) -> None:
        for key in (W.attn_qkv_b, W.attn_qkv_s2, W.attn_qkv_i_s):
            with self.subTest(key=key):
                weights = {
                    W.attn_qkv_w: torch.randn(384, QKV_SIZE, dtype=torch.bfloat16),
                    W.attn_gate_w: torch.randn(384, GATE_SIZE, dtype=torch.bfloat16),
                    key: torch.ones(1, dtype=torch.float32),
                }
                self.assertIsNotNone(self._blocker(weights))

    def test_declines_non_fp8_per_block_quant_scheme(self) -> None:
        hidden_size = 384
        weights = {
            W.attn_qkv_w: _fp8_tensor((hidden_size, QKV_SIZE)),
            W.attn_gate_w: _fp8_tensor((hidden_size, GATE_SIZE)),
            W.attn_qkv_s: torch.ones(1, QKV_SIZE, dtype=torch.float32),
            W.attn_gate_s: torch.ones(1, GATE_SIZE, dtype=torch.float32),
        }
        self.assertIsNotNone(
            self._blocker(weights, _StubQuantConfig("FP8_PER_CHANNEL_COMPRESSED"))
        )

    def test_declines_when_only_one_projection_is_quantized(self) -> None:
        hidden_size = 512
        weights = {
            W.attn_qkv_w: _fp8_tensor((hidden_size, QKV_SIZE)),
            W.attn_gate_w: _fp8_tensor((hidden_size, GATE_SIZE)),
            W.attn_qkv_s: torch.ones(hidden_size // 128, QKV_SIZE // 128),
        }
        self.assertIsNotNone(self._blocker(weights, _StubQuantConfig("FP8_PER_BLOCK")))

    def test_declines_fp8_float32_when_output_not_128_aligned(self) -> None:
        # A tensor-parallel split can leave the per-rank output size off a 128 block
        # boundary, at which point the fused scale blocks no longer line up.
        hidden_size = 512
        qkv_size, gate_size = 256, 64
        weights = {
            W.attn_qkv_w: _fp8_tensor((hidden_size, qkv_size)),
            W.attn_gate_w: _fp8_tensor((hidden_size, gate_size)),
            W.attn_qkv_s: torch.ones(hidden_size // 128, qkv_size // 128),
            W.attn_gate_s: torch.ones(hidden_size // 128, 1),
        }
        self.assertIsNotNone(
            self._blocker(
                weights,
                _StubQuantConfig("FP8_PER_BLOCK"),
                qkv_size=qkv_size,
                gate_size=gate_size,
            )
        )

    def test_declines_fp8_e8m0_when_output_not_16byte_aligned(self) -> None:
        # The packed UE8M0 scale keeps the output dim as its stride-1 axis and DeepGEMM
        # TMA-aligns it to 16 bytes, so a fused output that is not a multiple of four
        # int32 lanes cannot be represented.
        hidden_size = 512
        qkv_size, gate_size = 4, 2
        weights = {
            W.attn_qkv_w: _fp8_tensor((qkv_size, hidden_size)),
            W.attn_gate_w: _fp8_tensor((gate_size, hidden_size)),
            W.attn_qkv_s: _mn_major_int32(qkv_size, 1),
            W.attn_gate_s: _mn_major_int32(gate_size, 1),
        }
        self.assertIsNotNone(
            self._blocker(
                weights,
                _StubQuantConfig("FP8_PER_BLOCK"),
                qkv_size=qkv_size,
                gate_size=gate_size,
            )
        )


class TestQkvGateFusionWeights(unittest.TestCase):
    """The fused weight/scale buffers built for the single GEMM."""

    def _fuse(
        self,
        weights: dict[str, torch.Tensor],
        quant_config: object,
        qkv_size: int = QKV_SIZE,
        gate_size: int = GATE_SIZE,
    ) -> tuple[Qwen3NextAttention, MagicMock]:
        config = AttentionConfigs()
        config.head_num = HEAD_NUM
        config.kv_head_num = KV_HEAD_NUM
        config.size_per_head = HEAD_DIM
        parallelism = ParallelismConfig()
        parallelism.tp_size = 1
        parallelism.tp_rank = 0

        def fake_from_weights(weights, weight_key, scale_key=None, *a, **k):
            scales = weights.get(scale_key) if scale_key else None
            return _FakeLinear(weights[weight_key], scales)

        def fake_linear(weight, bias, weight_scales, quant_config, **k):
            return _FakeLinear(weight, weight_scales)

        with patch.object(
            LinearFactory, "create_linear_from_weights", side_effect=fake_from_weights
        ), patch.object(
            LinearFactory, "create_linear", side_effect=fake_linear
        ) as create_linear:
            attention = Qwen3NextAttention(
                config,
                parallelism,
                weights,
                layernorm_eps=1e-6,
                quant_config=quant_config,
            )
        return attention, create_linear

    def test_fp8_float32_fusion_matches_separate_projections(self) -> None:
        hidden_size = 512
        qkv_blocks, gate_blocks = QKV_SIZE // 128, GATE_SIZE // 128
        k_blocks = hidden_size // 128
        qkv_real = _fp8_tensor((QKV_SIZE, hidden_size))
        gate_real = _fp8_tensor((GATE_SIZE, hidden_size), offset=qkv_real.numel())
        qkv_scale_real = torch.arange(
            1, qkv_blocks * k_blocks + 1, dtype=torch.float32
        ).reshape(qkv_blocks, k_blocks)
        gate_scale_real = torch.arange(
            100, 100 + gate_blocks * k_blocks, dtype=torch.float32
        ).reshape(gate_blocks, k_blocks)
        weights = {
            # The loader exposes the [N, K] storage as a [K, N] view; mirror that here.
            W.attn_qkv_w: qkv_real.reshape(hidden_size, QKV_SIZE),
            W.attn_gate_w: gate_real.reshape(hidden_size, GATE_SIZE),
            W.attn_qkv_s: qkv_scale_real.reshape(k_blocks, qkv_blocks),
            W.attn_gate_s: gate_scale_real.reshape(k_blocks, gate_blocks),
            W.attn_o_w: torch.randn(Q_SIZE, hidden_size, dtype=torch.bfloat16),
        }

        attention, create_linear = self._fuse(
            weights, _StubQuantConfig("FP8_PER_BLOCK")
        )

        self.assertTrue(attention._qkv_gate_fused)
        self.assertEqual(create_linear.call_count, 1)
        fused_n = QKV_SIZE + GATE_SIZE
        fused_weight = attention.qkv_proj.weight.reshape(fused_n, hidden_size)
        fused_scales = attention.qkv_proj.weight_scales.reshape(
            qkv_blocks + gate_blocks, k_blocks
        )
        fused_dequant = _block_dequant(fused_weight, fused_scales)
        torch.testing.assert_close(
            fused_dequant[:QKV_SIZE],
            _block_dequant(qkv_real, qkv_scale_real),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            fused_dequant[QKV_SIZE:],
            _block_dequant(gate_real, gate_scale_real),
            atol=0,
            rtol=0,
        )
        # The original dict entries are rebound onto the fused buffer so the unfused
        # copies can be released.
        self.assertEqual(
            weights[W.attn_gate_w].untyped_storage().data_ptr(),
            attention.qkv_proj.weight.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            weights[W.attn_gate_s].untyped_storage().data_ptr(),
            attention.qkv_proj.weight_scales.untyped_storage().data_ptr(),
        )

    def test_e8m0_fusion_concatenates_and_keeps_scale_layout(self) -> None:
        hidden_size = 1024
        packed_k = (hidden_size + 511) // 512
        qkv_weight = _fp8_tensor((QKV_SIZE, hidden_size))
        gate_weight = _fp8_tensor((GATE_SIZE, hidden_size), offset=qkv_weight.numel())
        qkv_scales = _mn_major_int32(QKV_SIZE, packed_k)
        gate_scales = _mn_major_int32(GATE_SIZE, packed_k, offset=QKV_SIZE * packed_k)
        self.assertEqual(qkv_scales.stride(-2), 1)
        weights = {
            W.attn_qkv_w: qkv_weight,
            W.attn_gate_w: gate_weight,
            W.attn_qkv_s: qkv_scales,
            W.attn_gate_s: gate_scales,
            W.attn_o_w: torch.randn(Q_SIZE, hidden_size, dtype=torch.bfloat16),
        }

        attention, create_linear = self._fuse(
            weights, _StubQuantConfig("FP8_PER_BLOCK")
        )

        self.assertTrue(attention._qkv_gate_fused)
        self.assertEqual(create_linear.call_count, 1)
        fused_scales = attention.qkv_proj.weight_scales
        torch.testing.assert_close(
            attention.qkv_proj.weight,
            torch.cat([qkv_weight, gate_weight], dim=0),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            fused_scales, torch.cat([qkv_scales, gate_scales], dim=0), atol=0, rtol=0
        )
        # DeepGEMM requires the packed UE8M0 scale to keep the output dim as its
        # stride-1 axis; a plain cat + contiguous would flip that to a stride of 1 on
        # the wrong axis.
        self.assertEqual(fused_scales.stride(-2), 1)
        self.assertEqual(weights[W.attn_qkv_s].stride(-2), 1)
        self.assertEqual(weights[W.attn_gate_s].stride(-2), 1)
        self.assertEqual(
            weights[W.attn_gate_s].untyped_storage().data_ptr(),
            fused_scales.untyped_storage().data_ptr(),
        )


class TestQkvGateFusionForward(unittest.TestCase):
    """The forward path that consumes the fused (or unfused) projections."""

    def _build(
        self, weights: dict[str, torch.Tensor], quant_config: object | None = None
    ) -> Qwen3NextAttention:
        config = AttentionConfigs()
        config.head_num = HEAD_NUM
        config.kv_head_num = KV_HEAD_NUM
        config.size_per_head = HEAD_DIM
        parallelism = ParallelismConfig()
        parallelism.tp_size = 1
        parallelism.tp_rank = 0

        def fake_from_weights(weights, weight_key, scale_key=None, *a, **k):
            scales = weights.get(scale_key) if scale_key else None
            return _FakeLinear(weights[weight_key], scales)

        def fake_linear(weight, bias, weight_scales, quant_config, **k):
            return _FakeLinear(weight, weight_scales)

        with patch.object(
            LinearFactory, "create_linear_from_weights", side_effect=fake_from_weights
        ), patch.object(LinearFactory, "create_linear", side_effect=fake_linear):
            return Qwen3NextAttention(
                config,
                parallelism,
                weights,
                layernorm_eps=1e-6,
                quant_config=quant_config,
            )

    def test_fused_forward_normalizes_qkv_half_and_gates_output(self) -> None:
        torch.manual_seed(2)
        hidden_size = 384
        qkv_weight = torch.randn(hidden_size, QKV_SIZE, dtype=torch.bfloat16)
        gate_weight = torch.randn(hidden_size, GATE_SIZE, dtype=torch.bfloat16)
        qkv_reference, gate_reference = qkv_weight.clone(), gate_weight.clone()
        weights = {
            W.attn_qkv_w: qkv_weight,
            W.attn_gate_w: gate_weight,
            W.attn_o_w: torch.randn(Q_SIZE, hidden_size, dtype=torch.bfloat16),
        }
        output_weight = weights[W.attn_o_w].clone()

        attention = self._build(weights)
        self.assertTrue(attention._qkv_gate_fused)
        qk_norm = _RecordingQKNorm()
        attention.qk_fuse_norm = qk_norm

        hidden_states = torch.randn(5, hidden_size, dtype=torch.bfloat16)
        actual = attention(
            hidden_states, _FirstQFmha(Q_SIZE), kv_cache=None, attention_inputs=None
        )

        # The norm must see only the materialized, contiguous qkv half.
        self.assertEqual(len(qk_norm.calls), 1)
        shape, is_contiguous = qk_norm.calls[0]
        self.assertEqual(tuple(shape), (5, QKV_SIZE))
        self.assertTrue(is_contiguous)

        qkv = (hidden_states @ qkv_reference) * 2
        gate = hidden_states @ gate_reference
        expected = (qkv[..., :Q_SIZE] * torch.sigmoid(gate)) @ output_weight
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_two_projection_fallback_forward_matches_reference(self) -> None:
        torch.manual_seed(3)
        hidden_size = 384
        qkv_weight = torch.randn(hidden_size, QKV_SIZE, dtype=torch.bfloat16)
        gate_weight = torch.randn(hidden_size, GATE_SIZE, dtype=torch.bfloat16)
        weights = {
            W.attn_qkv_w: qkv_weight,
            W.attn_gate_w: gate_weight,
            W.attn_o_w: torch.randn(Q_SIZE, hidden_size, dtype=torch.bfloat16),
            # An fp4 marker keeps both weights untouched while declining fusion.
            W.attn_qkv_s2: torch.ones(1, dtype=torch.float32),
        }
        output_weight = weights[W.attn_o_w].clone()

        attention = self._build(weights)
        self.assertFalse(attention._qkv_gate_fused)
        self.assertIsNotNone(attention.gate)

        hidden_states = torch.randn(6, hidden_size, dtype=torch.bfloat16)
        actual = attention(
            hidden_states, _FirstQFmha(Q_SIZE), kv_cache=None, attention_inputs=None
        )
        qkv = hidden_states @ qkv_weight
        gate = hidden_states @ gate_weight
        expected = (qkv[..., :Q_SIZE] * torch.sigmoid(gate)) @ output_weight
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class TestQkvGateFusionOnDevice(unittest.TestCase):
    """End-to-end equivalence through the real LinearFactory and GEMM kernels."""

    head_num = 4
    kv_head_num = 2
    head_dim = 128
    hidden_size = 1024
    q_size = head_num * head_dim
    qkv_size = q_size + 2 * kv_head_num * head_dim
    gate_size = q_size

    def setUp(self) -> None:
        torch.manual_seed(4)
        torch.cuda.manual_seed(4)
        self.device = torch.device("cuda:0")

    def _attention(self, weights, quant_config=None) -> Qwen3NextAttention:
        config = AttentionConfigs()
        config.head_num = self.head_num
        config.kv_head_num = self.kv_head_num
        config.size_per_head = self.head_dim
        parallelism = ParallelismConfig()
        parallelism.tp_size = 1
        parallelism.tp_rank = 0
        return Qwen3NextAttention(
            config, parallelism, weights, layernorm_eps=1e-6, quant_config=quant_config
        )

    def _reference_output(
        self, hidden_states, qkv_proj, gate_proj, o_proj
    ) -> torch.Tensor:
        qkv = qkv_proj(hidden_states)
        gate = gate_proj(hidden_states)
        attn_output = qkv[..., : self.q_size] * torch.sigmoid(gate)
        return o_proj(attn_output)

    def test_bf16_fusion_matches_two_projections(self) -> None:
        qkv_weight = torch.randn(
            self.hidden_size, self.qkv_size, dtype=torch.bfloat16, device=self.device
        )
        gate_weight = torch.randn(
            self.hidden_size, self.gate_size, dtype=torch.bfloat16, device=self.device
        )
        weights = {
            W.attn_qkv_w: qkv_weight,
            W.attn_gate_w: gate_weight,
            W.attn_o_w: torch.randn(
                self.q_size, self.hidden_size, dtype=torch.bfloat16, device=self.device
            ),
        }
        reference_qkv = LinearFactory.create_linear(
            weight=qkv_weight.clone(), bias=None, weight_scales=None, quant_config=None
        )
        reference_gate = LinearFactory.create_linear(
            weight=gate_weight.clone(), bias=None, weight_scales=None, quant_config=None
        )

        attention = self._attention(weights)
        self.assertTrue(attention._qkv_gate_fused)

        hidden_states = torch.randn(
            7, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )
        actual = attention(
            hidden_states,
            _FirstQFmha(self.q_size),
            kv_cache=None,
            attention_inputs=None,
        )
        expected = self._reference_output(
            hidden_states, reference_qkv, reference_gate, attention.o_proj
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_fp8_per_block_fusion_matches_two_projections(self) -> None:
        if is_deep_gemm_e8m0_used():
            self.skipTest("E8M0 packed scales require the sm_100+ DeepGEMM kernels")
        quant_config = init_quant_config("FP8_PER_BLOCK")

        def make_fp8(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
            reference = (
                torch.randn(n, k, dtype=torch.bfloat16, device=self.device) * 0.1
            )
            weight_nk, scale_nk = per_block_cast_to_fp8(reference, use_ue8m0=False)
            # The loader exposes the [N, K] storage as a [K, N] view.
            return weight_nk.reshape(k, n), scale_nk.reshape(k // 128, n // 128)

        qkv_weight, qkv_scales = make_fp8(self.qkv_size, self.hidden_size)
        gate_weight, gate_scales = make_fp8(self.gate_size, self.hidden_size)
        o_weight, o_scales = make_fp8(self.hidden_size, self.q_size)
        weights = {
            W.attn_qkv_w: qkv_weight,
            W.attn_qkv_s: qkv_scales,
            W.attn_gate_w: gate_weight,
            W.attn_gate_s: gate_scales,
            W.attn_o_w: o_weight,
            W.attn_o_s: o_scales,
        }
        reference_qkv = LinearFactory.create_linear(
            weight=qkv_weight.clone(),
            bias=None,
            weight_scales=qkv_scales.clone(),
            quant_config=quant_config,
        )
        reference_gate = LinearFactory.create_linear(
            weight=gate_weight.clone(),
            bias=None,
            weight_scales=gate_scales.clone(),
            quant_config=quant_config,
        )

        attention = self._attention(weights, quant_config)
        self.assertTrue(attention._qkv_gate_fused)
        # The fused tensors must satisfy the real linear's own shape validation.
        self.assertEqual(attention.qkv_proj.N, self.qkv_size + self.gate_size)
        self.assertEqual(attention.qkv_proj.K, self.hidden_size)

        for tokens in (1, 33, 128):
            with self.subTest(tokens=tokens):
                hidden_states = (
                    torch.randn(
                        tokens,
                        self.hidden_size,
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    * 0.1
                )
                actual = attention(
                    hidden_states,
                    _FirstQFmha(self.q_size),
                    kv_cache=None,
                    attention_inputs=None,
                )
                expected = self._reference_output(
                    hidden_states, reference_qkv, reference_gate, attention.o_proj
                )
                self.assertLess(calc_diff(actual.float(), expected.float()), 1e-4)


if __name__ == "__main__":
    unittest.main()
