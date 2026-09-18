"""HY4 full-Indexer q_c MXFP8 reuse plumbing tests."""

from types import MethodType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention


class _Projection(torch.nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.calls = []

    def forward(self, x: torch.Tensor, input_scales=None) -> torch.Tensor:
        self.calls.append((x, input_scales))
        return torch.zeros(
            x.shape[0], self.output_size, dtype=torch.bfloat16, device=x.device
        )


class _Norm(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.variance_epsilon = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FMHA:
    pinned_mla_groups = {}
    fmha_params = object()
    attn_inputs = object()
    cp_params = None

    def __init__(self):
        self.q_view = None

    def forward(self, q, compressed_kv, k_pe, *unused_args, **unused_kwargs):
        self.q_view = q
        return torch.zeros(
            q.shape[0], q.shape[1], 256, dtype=torch.bfloat16, device=q.device
        )


class TestHy4Mxfp8QcReuse(TestCase):
    def test_full_indexer_uses_fp8_qc_without_bf16_output(self):
        """The full-Indexer path must pass only MXFP8 q_c to both WQ-B uses."""
        attention = MlaAttention.__new__(MlaAttention)
        torch.nn.Module.__init__(attention)
        attention.q_lora_rank = 2048
        attention.kv_lora_rank = 512
        attention.qk_rope_head_dim = 64
        attention.q_head_dim = 256
        attention.num_heads = 64
        attention.v_head_dim = 256
        attention.layer_idx = 0
        attention._reuse_mxfp8_hidden_quant = False
        attention._fuse_q_a_norm_mode = "mxfp8_indexer"
        attention._fuse_kv_a_norm = False
        attention.attn_sink = None
        attention.gate_proj = None
        attention.parallelism_config = SimpleNamespace(get_attn_tp_size=lambda: 1)

        attention.fused_qkv_a_proj = _Projection(2048 + 512 + 64)
        attention.q_a_layernorm = _Norm(2048)
        attention.q_b_proj = _Projection(64 * 256)
        attention.kv_a_layernorm = _Norm(512)
        attention.o_proj = _Projection(6144)
        attention.indexer = object()

        captured = {}

        def _capture_indexer(_self, *args):
            captured["q_c"] = args[1]
            captured["q_c_fp8"] = args[7]
            captured["q_c_scale"] = args[8]
            return torch.zeros((args[0].shape[0], 1), dtype=torch.int32)

        attention._run_sparse_indexer = MethodType(_capture_indexer, attention)
        fmha = _FMHA()
        hidden = torch.zeros(2, 6144, dtype=torch.bfloat16)
        expected_fp8 = torch.zeros(2, 2048, dtype=torch.float8_e4m3fn)
        expected_scale = torch.zeros(2, 16, dtype=torch.int32)

        with patch(
            "rtp_llm.models_py.modules.hybrid.mla_attention."
            "fused_strided_rmsnorm_per_token_fp8_quant",
            return_value=(expected_fp8, expected_scale),
        ) as quant:
            output = attention(hidden, fmha)

        quant.assert_called_once()
        self.assertIsNone(captured["q_c"])
        self.assertIs(captured["q_c_fp8"], expected_fp8)
        self.assertIs(captured["q_c_scale"], expected_scale)
        self.assertEqual(len(attention.q_b_proj.calls), 1)
        self.assertIs(attention.q_b_proj.calls[0][0], expected_fp8)
        self.assertIs(attention.q_b_proj.calls[0][1], expected_scale)
        self.assertEqual(tuple(fmha.q_view.shape), (2, 64, 256))
        self.assertEqual(tuple(output.shape), (2, 6144))
