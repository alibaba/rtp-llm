"""GPU regression checks for K3 scale, latent-norm and MTP integration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpLayer
from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)
from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MLA
from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_scale_layout import (
    repack_ag_scale_wire,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.moe_decode import add_moe_output
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class KimiK3RemainingSmallKernelsTest(unittest.TestCase):
    def test_scale_repack_graph_replay_preserves_all_bits_and_zeroes_global_tail(self):
        for ranks, rows in ((8, 1), (8, 2), (8, 4), (8, 8), (8, 16), (8, 32), (3, 3)):
            groups, local_pad = 14, (rows + 3) // 4 * 4
            wire = torch.zeros(
                ranks * groups, local_pad, device="cuda", dtype=torch.int32
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                repack_ag_scale_wire(wire, rows, ranks)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = repack_ag_scale_wire(wire, rows, ranks)
            for step in range(3):
                wire.random_(-(2**31), 2**31)
                actual.fill_(-123)  # Global padding must be rewritten on replay.
                graph.replay()
                expected = torch.zeros_like(actual)
                for rank in range(ranks):
                    expected[:, rank * rows : (rank + 1) * rows] = wire[
                        rank * groups : (rank + 1) * groups, :rows
                    ]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @staticmethod
    def _mla(role):
        config = SimpleNamespace(
            attn_config=SimpleNamespace(
                head_num=96,
                nope_head_dim=128,
                rope_head_dim=64,
                q_lora_rank=1536,
                kv_lora_rank=512,
                v_head_dim=128,
                kernel_tokens_per_block=64,
                is_sparse=False,
            ),
            k3_attention_quant_config=object(),
            quant_config=None,
            k3_runtime_config=SimpleNamespace(
                mla_use_nope=True, mla_use_output_gate=True
            ),
        )
        parallel = SimpleNamespace(
            get_attn_tp_size=lambda: 8,
            decode_cp_q_replicated=False,
            role_type=role,
        )
        weights = {
            W.mla_q_a_ln_gamma: torch.ones(1536, device="cuda", dtype=torch.bfloat16),
            W.mla_kv_a_ln_gamma: torch.ones(512, device="cuda", dtype=torch.bfloat16),
            W.mla_fusedqkrope_w: torch.empty(0, device="cuda"),
            W.mla_fusedqkrope_s: torch.empty(0, device="cuda"),
            W.attn_gate_s: torch.empty(0, device="cuda"),
        }
        # Projection weights are irrelevant here: keep the real constructor and
        # native/Triton norms, replacing only the unused GEMM factory boundary.
        with patch.object(
            LinearFactory,
            "create_linear_from_weights",
            side_effect=lambda *a, **kw: nn.Identity(),
        ):
            return KimiK3MLA(config, parallel, weights)

    def test_fp8_decode_kv_norm_consumes_strided_latent_without_staging(self):
        optimized = self._mla(RoleType.DECODE)
        native_norm = RMSNorm(optimized._kv_a_norm, 1e-6)
        for rows in (1, 2, 4, 8, 16, 32):
            with self.subTest(rows=rows):
                latent = torch.randn(
                    rows, 1536 + 512 + 64, device="cuda", dtype=torch.bfloat16
                )[:, 1536:2048]
                reference = native_norm(latent.contiguous())
                call = lambda: optimized._normalize_latent(
                    optimized.kv_a_layernorm, latent
                )
                result = call()
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.dtype, torch.bfloat16)
                torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-3)
                if rows == 32:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = call()
                    for _ in range(3):
                        latent.normal_()
                        graph.replay()
                        torch.testing.assert_close(captured, call(), rtol=0, atol=0)
                        reference = native_norm(latent.contiguous())
                        torch.testing.assert_close(
                            captured, reference, rtol=1e-2, atol=1e-3
                        )

    def test_fp8_prefill_keeps_quantized_and_retained_bf16_kv(self):
        module = self._mla(RoleType.PREFILL)
        latent = torch.randn(32, 576, device="cuda", dtype=torch.bfloat16)[:, :512]
        actual = module._normalize_latent(module.kv_a_layernorm, latent)
        expected = module.kv_a_layernorm(latent.contiguous())
        self.assertIsInstance(actual, QuantizedActivation)
        torch.testing.assert_close(
            actual.values.view(torch.uint8),
            expected.values.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual.scale_wire, expected.scale_wire, rtol=0, atol=0
        )
        torch.testing.assert_close(actual.bf16, expected.bf16, rtol=0, atol=0)

    def test_mtp_reuses_moe_residual_fusion_in_both_phases(self):
        # Real MTP forward and real fused add; expensive attention/expert GEMMs
        # are replaced at their boundaries with exact BF16 fixture values.
        class Attention:
            def tp_input_projection_weights(self):
                return []

            def output_projection_weight(self):
                return None

            def __call__(self, x, *args, **kwargs):
                return torch.full_like(x, 0.125)

        def moe(x, *, residual=None, **kwargs):
            return add_moe_output(
                torch.full_like(x, 1),
                torch.full_like(x, 0.25),
                residual,
            )

        for prefill in (False, True):
            with self.subTest(prefill=prefill):
                layer = SimpleNamespace(
                    enorm=lambda x: x,
                    hnorm=lambda x: x,
                    eh_proj=lambda x: x[:, :128].contiguous(),
                    input_norm=lambda x: x,
                    attention=Attention(),
                    attn_tp_size=1,
                    _local_projection=lambda x, w: x,
                    post_norm=lambda x: x,
                    moe=moe,
                )
                x = torch.ones(3, 128, device="cuda", dtype=torch.bfloat16)
                positions = torch.tensor([0, 1, 2], device="cuda")
                layout = SimpleNamespace(
                    tokens=SimpleNamespace(
                        local_valid_tokens=2, local_tokens=3, physical_tokens=3
                    )
                )
                actual = KimiK3MtpLayer.forward(
                    layer,
                    x,
                    x,
                    positions,
                    SimpleNamespace(release_forward_workspace=lambda: None),
                    None,
                    SimpleNamespace(is_prefill=prefill),
                    sp_layout=layout,
                )
                expected = torch.full_like(x, 2.375)
                expected[0].fill_(1.375)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
