import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import rtp_llm.models_py.model_desc.kimi_k3 as kimi_k3
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig
from rtp_llm.models.kimi_k3.kimi_k3_weight import _merge_mla_input_projections
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3MLA
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention
from rtp_llm.models_py.modules.kimi_k3.parallel_mode import KimiK3ParallelMode
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W


class _CountingProjection(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.mm(hidden_states, self.weight)


class KimiK3MLAProjectionFusionUnitTest(unittest.TestCase):
    @staticmethod
    def _projection_module() -> tuple[KimiK3MLA, _CountingProjection]:
        module = KimiK3MLA.__new__(KimiK3MLA)
        nn.Module.__init__(module)
        module.q_lora_rank = 3
        module.kv_lora_rank = 2
        module.suffix_dim = 1
        module.local_heads = 2
        module.value_dim = 4
        module.attn_tp_size = 1
        module.attn_tp_rank = 0
        module.parallel_mode = KimiK3ParallelMode.TP_SP
        module.use_output_gate = True
        module._sp_layout_for_forward = SimpleNamespace(
            tokens=SimpleNamespace(physical_tokens=7)
        )
        projection = _CountingProjection(torch.randn(5, 14))
        module.fused_qkv_a_proj = projection
        module._packed_qkv_gate_w = projection.weight
        module.weights = {W.mla_fusedqkrope_w: projection.weight}
        return module, projection

    def test_loader_packs_replicated_latents_and_local_gate(self) -> None:
        q_a = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        kv_a = torch.arange(20, dtype=torch.float32).reshape(4, 5) + 100
        output_gate = torch.arange(40, dtype=torch.float32).reshape(8, 5) + 200

        for tp_size in (1, 2, 4, 8):
            for tp_rank in range(tp_size):
                with self.subTest(tp_size=tp_size, tp_rank=tp_rank):
                    actual = _merge_mla_input_projections(
                        [q_a, kv_a, output_gate],
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                    )
                    expected = torch.cat(
                        (q_a, kv_a, output_gate.chunk(tp_size, dim=0)[tp_rank]),
                        dim=0,
                    ).T.contiguous()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(actual.is_contiguous())

    def test_k3_splits_precomputed_qkv_and_gate_without_another_gemm(self) -> None:
        module, projection = self._projection_module()
        hidden_states = torch.randn(7, 5)

        expected = torch.mm(hidden_states, projection.weight)
        module._projected_qkv_a_for_forward = [expected]
        from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MLAContext

        qkv_a, output_gate = module._project_qkv_a_input(
            hidden_states, KimiK3MLAContext(module._sp_layout_for_forward, [expected])
        )

        self.assertEqual(projection.calls, 0)
        torch.testing.assert_close(qkv_a, expected[:, :6], rtol=0, atol=0)
        torch.testing.assert_close(output_gate, expected[:, 6:], rtol=0, atol=0)

    def test_model_config_accepts_only_bf16_without_runtime_quantization(
        self,
    ) -> None:
        config = KimiK3ModelConfig()
        with patch.object(ModelConfig, "init_precision_config", return_value=None):
            config.data_type = "bf16"
            config.quant_config = None
            config.init_precision_config(None, None)

            config.data_type = "fp16"
            with self.assertRaisesRegex(ValueError, "only BF16 compute"):
                config.init_precision_config(None, None)

            config.data_type = "bf16"
            config.quant_config = object()
            with self.assertRaisesRegex(ValueError, "runtime weight quantization"):
                config.init_precision_config(None, None)

    def test_base_mla_keeps_gate_hook_optional(self) -> None:
        module = MlaAttention.__new__(MlaAttention)
        nn.Module.__init__(module)
        module.fused_qkv_a_proj = nn.Linear(5, 7, bias=False)
        hidden_states = torch.randn(3, 5)

        projected, output_gate = module._project_qkv_a_input(hidden_states)

        torch.testing.assert_close(
            projected,
            module.fused_qkv_a_proj(hidden_states),
            rtol=0,
            atol=0,
        )
        self.assertIsNone(output_gate)
        self.assertIs(module._apply_output_gate(projected, output_gate), projected)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
    and has_deep_gemm(),
    "requires SM100-family DeepGEMM UE8M0",
)
class KimiK3MLAFp8ProjectionGpuTest(unittest.TestCase):
    @staticmethod
    def _fp8_module(device="cpu", *, local_projection=False, k=512, ue8m0=True):
        # Keep the actual TP8 MLA output widths, including the N=2112 boundary.
        attn = SimpleNamespace(
            head_num=12 if local_projection else 96,
            nope_head_dim=128,
            rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            q_lora_rank=1536,
            kernel_tokens_per_block=64,
            is_sparse=False,
        )
        parallel = SimpleNamespace(
            get_attn_tp_size=lambda: 1 if local_projection else 8,
            ktp_size=8 if local_projection else 1,
            decode_cp_q_replicated=False,
            role_type=RoleType.DECODE if local_projection else RoleType.PREFILL,
        )
        config = SimpleNamespace(
            attn_config=attn,
            quant_config=None,
            k3_attention_quant_config=Fp8BlockWiseQuantConfig(),
            k3_runtime_config=SimpleNamespace(
                mla_use_nope=True, mla_use_output_gate=True
            ),
        )
        weights = {}
        torch.manual_seed(613)
        for wk, sk, n, width, exponent in (
            (W.mla_fusedqkrope_w, W.mla_fusedqkrope_s, 2112, k, 124),
            (W.attn_gate_w, W.attn_gate_s, 1536, k, 126),
            (W.mla_q_b_w, W.mla_q_b_s, 2304, 1536, 124),
            (W.mla_kv_b_w, W.mla_kv_b_s, 3072, 512, 124),
            (W.attn_o_w, W.attn_o_s, k, 1536, 124),
        ):
            weights[wk] = torch.randn(n, width, device=device).to(torch.float8_e4m3fn)
            # Four different K-group scales per packed word; the two projections
            # have deliberately different scales on either side of N=2112.
            row_group = torch.arange(n, device=device)[None, :] // 128
            k_group = torch.arange((width + 511) // 512, device=device)[:, None]
            packed = torch.zeros_like(row_group + k_group, dtype=torch.int64)
            for byte in range(4):
                packed |= (exponent + (row_group + k_group + byte) % 4) << (8 * byte)
            weights[sk] = packed.to(torch.int32).T
            if not ue8m0:
                weights[sk] = torch.ones(
                    ((n + 127) // 128, (width + 127) // 128),
                    device=device,
                    dtype=torch.float32,
                )
                if not is_deep_gemm_e8m0_used():
                    weights[wk] = weights[wk].reshape(width, n)
                    weights[sk] = weights[sk].reshape(
                        (width + 127) // 128, (n + 127) // 128
                    )
        weights[W.mla_q_a_ln_gamma] = torch.ones(
            1536, device=device, dtype=torch.bfloat16
        )
        weights[W.mla_kv_a_ln_gamma] = torch.ones(
            512, device=device, dtype=torch.bfloat16
        )
        original = {key: value.clone() for key, value in weights.items()}
        return KimiK3MLA(config, parallel, weights), original, weights

    @unittest.skipUnless(
        torch.cuda.is_available() and is_deep_gemm_e8m0_used() and has_deep_gemm(),
        "requires UE8M0 FP8 Linear",
    )
    def test_fp8_initialization_packs_final_weights_without_requantizing(self):
        module, original, weights = self._fp8_module()
        projections = module.tp_input_projection_weights()
        self.assertEqual(len(projections), 1)
        projection = projections[0]
        self.assertEqual((projection.N, projection.K), (3648, 512))
        self.assertEqual(projection.weight_scales.stride(), (1, 3648))
        for begin, end, wk, sk in (
            (0, 2112, W.mla_fusedqkrope_w, W.mla_fusedqkrope_s),
            (2112, 3648, W.attn_gate_w, W.attn_gate_s),
        ):
            torch.testing.assert_close(
                projection.weight[begin:end].view(torch.uint8),
                original[wk].view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                projection.weight_scales[begin:end], original[sk], rtol=0, atol=0
            )
            # Keep large weights as views, while each original scale tensor
            # retains the pitch required to remain usable by its own GEMM.
            self.assertEqual(
                weights[wk].untyped_storage().data_ptr(),
                projection.weight.untyped_storage().data_ptr(),
            )
            self.assertEqual(weights[sk].stride(), original[sk].stride())
            for key in (wk, sk):
                actual, expected = weights[key], original[key]
                if actual.dtype == torch.float8_e4m3fn:
                    actual = actual.view(torch.uint8)
                    expected = expected.view(torch.uint8)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(
        torch.cuda.is_available() and is_deep_gemm_e8m0_used() and has_deep_gemm(),
        "requires UE8M0 FP8 Linear",
    )
    def test_fp8_tp_projection_splits_packed_output_as_views(self):
        from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MLAContext

        module, _, _ = self._fp8_module()
        packed = torch.arange(3 * 3648, dtype=torch.float32).reshape(3, 3648)
        layout = SimpleNamespace(tokens=SimpleNamespace(physical_tokens=3))
        qkv, gate = module._project_qkv_a_input(
            torch.empty(0), KimiK3MLAContext(layout, [packed])
        )
        torch.testing.assert_close(qkv, packed[:, :2112], rtol=0, atol=0)
        torch.testing.assert_close(gate, packed[:, 2112:], rtol=0, atol=0)
        for value in (qkv, gate):
            self.assertEqual(
                value.untyped_storage().data_ptr(), packed.untyped_storage().data_ptr()
            )
            self.assertEqual(value.stride(0), 3648)

    @unittest.skipUnless(
        torch.cuda.is_available() and has_deep_gemm(), "requires CUDA FP8 Linear"
    )
    def test_fp32_block_scales_keep_separate_projection_contract(self):
        from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MLAContext

        module, _, _ = self._fp8_module(ue8m0=False)
        self.assertEqual(len(module.tp_input_projection_weights()), 2)
        outputs = [torch.randn(3, n) for n in (2112, 1536)]
        layout = SimpleNamespace(tokens=SimpleNamespace(physical_tokens=3))
        qkv, gate = module._project_qkv_a_input(
            torch.empty(0), KimiK3MLAContext(layout, outputs)
        )
        self.assertIs(qkv, outputs[0])
        self.assertIs(gate, outputs[1])

    def test_fused_projection_matches_original_through_graph_replay(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            CudaFp8DeepGEMMLinear,
        )
        from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
            QuantizedActivation,
        )
        from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MLAContext
        from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_producers import (
            rmsnorm_fp8,
            sigmoid_gate_fp8,
        )

        for local_projection in (False, True):
            module, _, weights = self._fp8_module(
                "cuda", local_projection=local_projection, k=7168
            )
            reference = [
                CudaFp8DeepGEMMLinear(weights[w], weights[s])
                for w, s in (
                    (W.mla_fusedqkrope_w, W.mla_fusedqkrope_s),
                    (W.attn_gate_w, W.attn_gate_s),
                )
            ]
            for m in (1, 7, 128, 129):
                with self.subTest(local_projection=local_projection, m=m):
                    values = torch.randn(m, 7168, device="cuda").to(torch.float8_e4m3fn)
                    wire = torch.full(
                        (14, (m + 3) // 4 * 4),
                        0x7F7E7D7C,
                        device="cuda",
                        dtype=torch.int32,
                    )
                    payload = QuantizedActivation(values, wire)
                    layout = SimpleNamespace(tokens=SimpleNamespace(physical_tokens=m))

                    def run():
                        if local_projection:
                            return module._project_qkv_a_input(
                                payload, KimiK3MLAContext(layout)
                            )
                        outputs = [
                            p.forward_quantized(values, payload.scales)
                            for p in module.tp_input_projection_weights()
                        ]
                        return module._project_qkv_a_input(
                            payload, KimiK3MLAContext(layout, outputs)
                        )

                    expected = [
                        p.forward_quantized(values, payload.scales) for p in reference
                    ]
                    actual = run()
                    for a, e in zip(actual, expected):
                        torch.testing.assert_close(a, e, rtol=0, atol=0)
                    graph = torch.cuda.CUDAGraph()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph):
                        actual = run()
                    for epoch in (1, 2):
                        values.copy_(
                            torch.randn(m, 7168, device="cuda").to(values.dtype)
                        )
                        packed_scale = 0x7F7E7D7C + epoch * 0x01010101
                        wire.fill_(packed_scale - 2**32)
                        expected = [
                            p.forward_quantized(values, payload.scales)
                            for p in reference
                        ]
                        graph.replay()
                        torch.cuda.synchronize()
                        for a, e in zip(actual, expected):
                            torch.testing.assert_close(a, e, rtol=0, atol=0)
                    # Exercise actual downstream consumers of the strided views.
                    qkv, gate = actual
                    q = qkv[:, :1536]
                    norm = torch.ones(1536, device="cuda", dtype=torch.bfloat16)
                    norm_view = rmsnorm_fp8(q, norm, 1e-6)
                    norm_dense = rmsnorm_fp8(q.contiguous(), norm, 1e-6)
                    kv = qkv[:, 1536:2048]
                    kv_norm = torch.ones(512, device="cuda", dtype=torch.bfloat16)
                    kv_view = rmsnorm_fp8(kv, kv_norm, 1e-6, retain_bf16=True)
                    kv_dense = rmsnorm_fp8(
                        kv.contiguous(), kv_norm, 1e-6, retain_bf16=True
                    )
                    torch.testing.assert_close(
                        kv_view.bf16, kv_dense.bf16, rtol=0, atol=0
                    )
                    context = torch.randn_like(gate)
                    gate_view = sigmoid_gate_fp8(context, gate)
                    gate_dense = sigmoid_gate_fp8(context, gate.contiguous())
                    for a, e in (
                        (norm_view, norm_dense),
                        (kv_view, kv_dense),
                        (gate_view, gate_dense),
                    ):
                        torch.testing.assert_close(
                            a.values.view(torch.uint8),
                            e.values.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            a.scale_wire, e.scale_wire, rtol=0, atol=0
                        )

            if local_projection:
                bf16_input = torch.randn(7, 7168, device="cuda", dtype=torch.bfloat16)
                layout = SimpleNamespace(tokens=SimpleNamespace(physical_tokens=7))
                quantized = reference[0].quantize_input(bf16_input)
                expected = [p.forward_quantized(*quantized) for p in reference]
                actual = module._project_qkv_a_input(
                    bf16_input, KimiK3MLAContext(layout)
                )
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
