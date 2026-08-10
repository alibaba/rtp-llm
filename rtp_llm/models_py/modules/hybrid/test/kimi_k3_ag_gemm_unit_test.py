import os
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch

import torch
from torch import nn

import rtp_llm.models_py.model_desc.kimi_k3 as kimi_k3
from rtp_llm.models_py.model_desc.kimi_k3 import (
    KimiK3DecoderLayer,
    KimiK3KDA,
    KimiK3MLA,
    KimiK3Model,
)
from rtp_llm.utils.model_weight import W


class KimiK3AllGatherMatmulUnitTest(unittest.TestCase):
    @staticmethod
    def _packed_kda_stub(tp_size: int) -> KimiK3KDA:
        total_heads = 8
        head_dim = 4
        local_heads = total_heads // tp_size
        projection_size = local_heads * head_dim
        forget_rank = 3
        hidden_size = 16

        module = KimiK3KDA.__new__(KimiK3KDA)
        nn.Module.__init__(module)
        module.attn_tp_size = tp_size
        module.attn_tp_rank = tp_size - 1
        module.total_heads = total_heads
        module.local_heads = local_heads
        module.projection_size = projection_size
        module.forget_latent_size = forget_rank
        module.trace_prefix = "test.kda"
        module.kda_fused_w = torch.randn(
            hidden_size,
            4 * projection_size + forget_rank + total_heads,
            dtype=torch.bfloat16,
        )
        module._full_column_weights = {}
        module.weights = {
            W.linear_attn_f_b_w: torch.randn(
                forget_rank,
                projection_size,
                dtype=torch.bfloat16,
            )
        }
        return module

    def test_packed_kda_projection_contract_tp1_tp2_tp4_tp8(self) -> None:
        with patch.dict(os.environ, {"KIMI_K3_PERF_FUSIONS": "1"}, clear=False):
            for tp_size in (1, 2, 4, 8):
                with self.subTest(tp_size=tp_size):
                    module = self._packed_kda_stub(tp_size)
                    hidden = torch.randn(5, 16, dtype=torch.bfloat16)
                    packed = torch.mm(hidden, module.kda_fused_w)
                    q, k, v, gate, forget, beta = torch.split(
                        packed,
                        (
                            module.projection_size,
                            module.projection_size,
                            module.projection_size,
                            module.projection_size,
                            module.forget_latent_size,
                            module.total_heads,
                        ),
                        dim=1,
                    )
                    beta_begin = module.attn_tp_rank * module.local_heads
                    expected = (
                        q,
                        k,
                        v,
                        torch.mm(forget, module.weights[W.linear_attn_f_b_w]),
                        beta[:, beta_begin : beta_begin + module.local_heads],
                        gate,
                    )
                    actual = module._project_fused_kda_inputs(
                        hidden,
                        prefill_input_is_sharded=False,
                    )
                    for actual_tensor, expected_tensor in zip(actual, expected):
                        torch.testing.assert_close(
                            actual_tensor,
                            expected_tensor,
                            rtol=0,
                            atol=0,
                        )

    def test_sharded_mla_projection_uses_loader_packed_weight(self) -> None:
        module = KimiK3MLA.__new__(KimiK3MLA)
        nn.Module.__init__(module)
        module._sp_prefill_input_is_sharded = True
        module.attn_tp_size = 2
        module.q_lora_rank = 3
        module.kv_lora_rank = 2
        module.suffix_dim = 1
        module.local_heads = 2
        module.value_dim = 4
        module._mla_backend = "kernel"
        module.use_output_gate = True
        module.attn_tp_rank = 0
        module._accuracy_full_weight_cache = {}
        packed_weight = torch.randn(5, 14)
        module._packed_qkv_gate_w = packed_weight
        module.weights = {W.mla_fusedqkrope_w: packed_weight}
        local_input = torch.randn(2, 5)
        projected = torch.randn(4, 14)

        with (
            patch.dict(
                os.environ,
                {
                    "KIMI_K3_PERF_FUSIONS": "1",
                    "KIMI_K3_ACCURACY_CANONICAL_TP": "0",
                    "KIMI_K3_ACCURACY_CANONICAL_MLA": "0",
                    "KIMI_K3_ACCURACY_LOCAL_EAGER_MLA": "0",
                    "KIMI_K3_ACCURACY_TRACE_DIR": "",
                },
                clear=False,
            ),
            patch.object(
                kimi_k3,
                "_prefill_all_gather_matmul",
                return_value=projected,
            ) as project,
        ):
            actual_qkv_a, actual_gate = module._project_qkv_a_input(local_input)

        project.assert_called_once_with(
            local_input,
            packed_weight,
            tp_size=2,
        )
        torch.testing.assert_close(actual_qkv_a, projected[:, :6], rtol=0, atol=0)
        torch.testing.assert_close(actual_gate, projected[:, 6:], rtol=0, atol=0)

    def test_prefill_fusion_is_a_direct_model_policy(self) -> None:
        local_input = torch.empty((8192, 1))
        weight = torch.empty((1, 1))
        output = torch.empty((65536, 1))
        group = SimpleNamespace(group_name="tp-test")
        with (
            patch.dict(
                os.environ,
                {"KIMI_K3_FUSED_AG_GEMM": "force"},
                clear=False,
            ),
            patch.object(kimi_k3, "get_process_group", return_value=group),
            patch.object(
                kimi_k3,
                "fused_all_gather_matmul",
                return_value=(None, [output]),
            ) as fused,
        ):
            actual = kimi_k3._prefill_all_gather_matmul(
                local_input,
                weight,
                tp_size=8,
            )

        fused.assert_called_once_with(
            local_input,
            [weight],
            group,
            return_gathered=False,
        )
        self.assertIs(actual, output)

    def test_small_prefill_uses_nccl_even_when_fusion_is_forced(self) -> None:
        local_input = torch.randn(2, 3)
        gathered_input = torch.randn(4, 3)
        weight = torch.randn(3, 5)
        with (
            patch.dict(
                os.environ,
                {"KIMI_K3_FUSED_AG_GEMM": "force"},
                clear=False,
            ),
            patch.object(
                kimi_k3,
                "all_gather_into",
                return_value=gathered_input,
            ) as gather,
            patch.object(kimi_k3, "fused_all_gather_matmul") as fused,
        ):
            actual = kimi_k3._prefill_all_gather_matmul(
                local_input,
                weight,
                tp_size=2,
            )

        gather.assert_called_once_with(local_input, ANY, kimi_k3.Group.TP)
        fused.assert_not_called()
        torch.testing.assert_close(actual, torch.mm(gathered_input, weight))

    def test_auto_policy_starts_at_32k_global_tokens(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "KIMI_K3_FUSED_AG_GEMM": "auto",
                    "KIMI_K3_PERF_FUSIONS": "1",
                    "KIMI_K3_ACCURACY_CANONICAL_TP": "0",
                    "KIMI_K3_ACCURACY_CANONICAL_MLA": "0",
                    "KIMI_K3_ACCURACY_LOCAL_EAGER_MLA": "0",
                    "KIMI_K3_ACCURACY_TRACE_DIR": "",
                },
                clear=False,
            ),
        ):
            self.assertFalse(kimi_k3._use_fused_prefill_ag_gemm(32767))
            self.assertTrue(kimi_k3._use_fused_prefill_ag_gemm(32768))

    def test_decoder_delegates_cuda_prefill_shard_to_attention(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        class StopAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.parallelism_config = SimpleNamespace(
                    get_attn_tp_size=lambda: 2,
                    get_attn_tp_rank=lambda: 0,
                )
                self.received_prefill_shard = False

            def forward(self, *args, **kwargs):
                self.received_prefill_shard = kwargs["prefill_input_is_sharded"]
                raise RuntimeError("stop after attention dispatch")

        layer = KimiK3DecoderLayer.__new__(KimiK3DecoderLayer)
        nn.Module.__init__(layer)
        layer.layer_idx = 1
        layer.eps = 1e-5
        layer.attn_res_block_size = 2
        layer.is_kda = True
        layer.self_attn = StopAttention()
        layer.mlp = nn.Identity()
        hidden = torch.randn((4, 8), dtype=torch.bfloat16, device="cuda")
        block_residual = torch.empty(
            (4, 0, 8),
            dtype=torch.bfloat16,
            device="cuda",
        )
        layer.weights = {W.pre_ln_gamma: torch.ones_like(hidden[0])}
        cu_seqlens = torch.tensor([0, 4], dtype=torch.int32, device="cuda")

        with (
            patch.object(kimi_k3, "_rms_norm", side_effect=lambda x, *_: x),
            patch.object(kimi_k3, "all_gather") as gather,
            self.assertRaisesRegex(RuntimeError, "stop after attention dispatch"),
        ):
            layer(
                hidden,
                block_residual,
                cu_seqlens,
                mode="prefill",
                sequence_parallel=True,
            )

        gather.assert_not_called()
        self.assertTrue(layer.self_attn.received_prefill_shard)

    def test_model_initialize_reserves_max_prefill_workspace(self) -> None:
        model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(max_seq_len=1024 * 1024, hidden_size=7168)
        model.parallelism_config = SimpleNamespace(
            get_attn_tp_size=lambda: 8,
        )
        model.embedding_weight = torch.empty(1, dtype=torch.bfloat16)
        model._fused_ag_gemm_workspace_ready = False
        init_resource = SimpleNamespace(
            kv_cache=None,
            is_decode_role=False,
            max_context_batch_size=2,
        )
        group = SimpleNamespace(group_name="tp-test")
        expected_bytes = 2 * (1024 * 1024 // 8) * 7168 * 2

        with (
            patch.dict(
                os.environ,
                {"KIMI_K3_FUSED_AG_GEMM": "auto"},
                clear=False,
            ),
            patch.object(
                kimi_k3,
                "_use_fused_prefill_ag_gemm",
                return_value=True,
            ) as use_fused,
            patch.object(kimi_k3, "get_process_group", return_value=group),
            patch.object(
                kimi_k3,
                "reserve_fused_all_gather_matmul_workspace",
            ) as reserve,
        ):
            self.assertTrue(model.initialize(init_resource))

        reserve.assert_called_once_with(group, expected_bytes)
        use_fused.assert_called_once_with(2 * 1024 * 1024)
        self.assertTrue(model._fused_ag_gemm_workspace_ready)

    def test_decode_packed_projection_is_cuda_graph_safe_and_local(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        module = self._packed_kda_stub(8)
        module.kda_fused_w = module.kda_fused_w.cuda()
        module.weights[W.linear_attn_f_b_w] = module.weights[W.linear_attn_f_b_w].cuda()
        hidden = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
        with patch.dict(os.environ, {"KIMI_K3_PERF_FUSIONS": "1"}, clear=False):
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                module._project_fused_kda_inputs(
                    hidden,
                    prefill_input_is_sharded=False,
                )
            torch.cuda.current_stream().wait_stream(warmup_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = module._project_fused_kda_inputs(
                    hidden,
                    prefill_input_is_sharded=False,
                )
            graph.replay()
            expected_packed = torch.mm(hidden, module.kda_fused_w)
            (
                expected_q,
                expected_k,
                expected_v,
                expected_gate,
                expected_forget,
                expected_beta,
            ) = torch.split(
                expected_packed,
                (
                    module.projection_size,
                    module.projection_size,
                    module.projection_size,
                    module.projection_size,
                    module.forget_latent_size,
                    module.total_heads,
                ),
                dim=1,
            )
            beta_begin = module.attn_tp_rank * module.local_heads
            expected = (
                expected_q,
                expected_k,
                expected_v,
                torch.mm(expected_forget, module.weights[W.linear_attn_f_b_w]),
                expected_beta[:, beta_begin : beta_begin + module.local_heads],
                expected_gate,
            )
            for actual_tensor, expected_tensor in zip(captured, expected):
                torch.testing.assert_close(
                    actual_tensor,
                    expected_tensor,
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
