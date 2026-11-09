"""HY4 full-Indexer q_c MXFP8 reuse plumbing tests."""

from types import MethodType, SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch

from rtp_llm.config.prefill_optimization import PREFILL_OPTIMIZATIONS_ENABLED
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
    def test_indexer_hidden_quant_format_compatibility(self):
        hidden = torch.zeros(2, 128, dtype=torch.bfloat16)
        quantized = torch.zeros(2, 128, dtype=torch.float8_e4m3fn)
        scales = torch.ones(2, 1)
        for mxfp8, compatible in ((False, False), (True, False), (True, True)):
            with self.subTest(mxfp8=mxfp8, compatible=compatible):
                attention = MlaAttention.__new__(MlaAttention)
                torch.nn.Module.__init__(attention)
                attention.reuse_topk_indices = False
                attention.q_lora_rank = 128
                attention.accepts_mxfp8_input = mxfp8
                attention._reuse_mxfp8_hidden_quant = compatible
                attention.indexer = Mock(return_value=torch.zeros(2, 1))
                fmha = SimpleNamespace(
                    is_sparse=lambda: True,
                    fmha_params=None,
                    attn_inputs=None,
                    cp_params=None,
                )
                attention._run_sparse_indexer(
                    hidden, hidden, hidden, None, fmha, quantized, scales
                )
                kwargs = attention.indexer.call_args.kwargs
                self.assertIs(
                    kwargs["x_fp8"], quantized if not mxfp8 or compatible else None
                )
                self.assertIs(
                    kwargs["x_scale"], scales if not mxfp8 or compatible else None
                )

    def test_full_indexer_uses_fp8_qc_without_bf16_output(self):
        self.check_qc_reuse(paired=False)

    def test_paired_norm_keeps_shared_fp8_qc_without_bf16_output(self):
        self.check_qc_reuse(paired=True)

    def test_paired_norm_rejection_preserves_single_output_consumers(self):
        self.check_qc_reuse(paired=True, reject_pair=True)

    def check_qc_reuse(self, paired, reject_pair=False):
        paired = paired and PREFILL_OPTIMIZATIONS_ENABLED
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
        attention.accepts_mxfp8_input = False
        attention._fuse_q_a_norm_mode = "mxfp8_indexer"
        attention._fuse_kv_a_norm = paired
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

        def _capture_indexer(_self, *args, **kwargs):
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
            "rtp_llm.models_py.modules.hybrid.mla_attention._prefill_optimization.ENABLED",
            paired,
        ), patch(
            "rtp_llm.models_py.triton_kernels.common.mla_norm_pair.try_mla_norm_pair",
            return_value=(
                None
                if reject_pair
                else (
                    None,
                    expected_fp8,
                    expected_scale,
                    torch.zeros(2, 512, dtype=torch.bfloat16),
                )
            ),
        ) as pair, patch(
            "rtp_llm.models_py.modules.hybrid.mla_attention."
            "fused_strided_rmsnorm_per_token_fp8_quant",
            return_value=(expected_fp8, expected_scale),
        ) as quant, patch(
            "rtp_llm.models_py.modules.hybrid.mla_attention.fused_strided_rmsnorm",
            return_value=torch.zeros(2, 512, dtype=torch.bfloat16),
        ):
            output = attention(hidden, fmha)

        self.assertEqual(quant.call_count, int(not paired or reject_pair))
        self.assertEqual(pair.call_count, int(paired))
        if paired:
            self.assertTrue(pair.call_args.kwargs["mxfp8"])
            self.assertFalse(pair.call_args.kwargs["bf16_output"])
        self.assertIsNone(captured["q_c"])
        self.assertIs(captured["q_c_fp8"], expected_fp8)
        self.assertIs(captured["q_c_scale"], expected_scale)
        self.assertEqual(len(attention.q_b_proj.calls), 1)
        self.assertIs(attention.q_b_proj.calls[0][0], expected_fp8)
        self.assertIs(attention.q_b_proj.calls[0][1], expected_scale)
        self.assertEqual(tuple(fmha.q_view.shape), (2, 64, 256))
        self.assertEqual(tuple(output.shape), (2, 6144))


if __name__ == "__main__":
    main()
