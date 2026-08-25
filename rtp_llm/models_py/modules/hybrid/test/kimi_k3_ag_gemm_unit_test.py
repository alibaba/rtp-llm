import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import torch
from torch import nn

import rtp_llm.models_py.model_desc.kimi_k3 as kimi_k3
import rtp_llm.models_py.modules.factory.linear.parallel as sequence_parallel
import rtp_llm.models_py.modules.kimi_k3.all_gather_gemm as kimi_k3_ag_gemm
import rtp_llm.models_py.modules.kimi_k3.gemm_reduce_scatter as kimi_k3_gemm_reduce_scatter
import rtp_llm.models_py.modules.kimi_k3.kda.module as kimi_k3_kda
import rtp_llm.models_py.modules.kimi_k3.mla as kimi_k3_mla
import rtp_llm.models_py.modules.kimi_k3.moe as kimi_k3_moe
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.model_desc.kimi_k3 import (
    KimiK3DecoderLayer,
    KimiK3KDA,
    KimiK3LatentMoE,
    KimiK3MLA,
    KimiK3Model,
)
from rtp_llm.utils.model_weight import W


class KimiK3CollectiveGemmUnitTest(unittest.TestCase):
    @staticmethod
    def _router_stub() -> KimiK3LatentMoE:
        module = KimiK3LatentMoE.__new__(KimiK3LatentMoE)
        nn.Module.__init__(module)
        module._bf16_fp32_router_enabled = True
        module._group_topk = SimpleNamespace(
            fused_sigmoid_supported=lambda *args: False
        )
        module.num_expert_group = 1
        module.topk_group = 1
        module.top_k = 2
        module.renormalize = True
        module.routed_scaling_factor = 0.75
        return module

    @staticmethod
    def _expected_route(
        module: KimiK3LatentMoE,
        logits: torch.Tensor,
        correction_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = logits.sigmoid()
        expert_ids = (
            (scores + correction_bias.unsqueeze(0))
            .topk(
                module.top_k,
                dim=-1,
                sorted=False,
            )
            .indices
        )
        expert_weights = scores.gather(1, expert_ids)
        if module.renormalize:
            expert_weights = expert_weights / (
                expert_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        return (
            expert_ids,
            expert_weights * module.routed_scaling_factor,
        )

    def test_decode_sp_shard_accepts_replicated_kda_output(self) -> None:
        tensor = torch.arange(9 * 4).reshape(9, 4)
        layout = sequence_parallel.token_shard_layout(9, 8, 4)

        actual = kimi_k3._decode_sp_shard(
            tensor,
            layout,
            name="attention output",
        )

        self.assertEqual(tuple(actual.shape), (2, 4))
        torch.testing.assert_close(actual[0], tensor[8], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(actual[1]).item(), 0)

    def test_decode_sp_shard_preserves_mla_local_output(self) -> None:
        tensor = torch.arange(2 * 4).reshape(2, 4)
        layout = sequence_parallel.token_shard_layout(9, 8, 4)

        actual = kimi_k3._decode_sp_shard(
            tensor,
            layout,
            name="attention output",
        )

        self.assertIs(actual, tensor)

    def test_full_world_mega_topology_accepts_ep16(self) -> None:
        kimi_k3_moe._validate_full_world_mega_topology(
            ep_size=16,
            world_size=16,
            local_expert_count=56,
        )

    def test_full_world_mega_topology_rejects_replicated_ep(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "full distributed world"):
            kimi_k3_moe._validate_full_world_mega_topology(
                ep_size=8,
                world_size=16,
                local_expert_count=112,
            )

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
                    packed[:, : 3 * module.projection_size],
                    q,
                    k,
                    v,
                    torch.mm(forget, module.weights[W.linear_attn_f_b_w]),
                    beta[:, beta_begin : beta_begin + module.local_heads],
                    gate,
                )
                actual = module._project_fused_kda_inputs(
                    hidden,
                    prefill_sp_layout=None,
                )
                for actual_tensor, expected_tensor in zip(actual, expected):
                    torch.testing.assert_close(
                        actual_tensor,
                        expected_tensor,
                        rtol=0,
                        atol=0,
                    )

    def test_router_uses_bf16_fp32_torch_mm(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        torch.manual_seed(20260813)
        module = self._router_stub()
        hidden = (0.1 * torch.randn(7, 13, device="cuda")).to(torch.bfloat16)
        router_weight = (0.1 * torch.randn(13, 5, device="cuda")).to(torch.bfloat16)
        correction_bias = 0.1 * torch.randn(5, dtype=torch.float32, device="cuda")
        module.weights = {
            K3W.MOE_GATE: router_weight,
            K3W.MOE_CORRECTION_BIAS: correction_bias,
        }

        logits = torch.mm(hidden, router_weight, out_dtype=torch.float32)
        reference_logits = torch.mm(hidden.float(), router_weight.float())
        torch.testing.assert_close(logits, reference_logits, rtol=1e-3, atol=1e-3)

        with patch.object(torch, "mm", return_value=logits) as mm:
            actual_ids, actual_weights = module._route(hidden)

        mm.assert_called_once_with(hidden, router_weight, out_dtype=torch.float32)
        expected_ids, expected_weights = self._expected_route(
            module, logits, correction_bias
        )
        reference_ids, _ = self._expected_route(
            module, reference_logits, correction_bias
        )
        torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(expected_ids, reference_ids, rtol=0, atol=0)
        torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)

    def test_router_falls_back_outside_cuda_bf16_path(self) -> None:
        torch.manual_seed(20260814)
        module = self._router_stub()
        hidden = torch.randn(7, 13, dtype=torch.bfloat16)
        router_weight = torch.randn(13, 5, dtype=torch.bfloat16)
        correction_bias = torch.randn(5, dtype=torch.float32)
        module.weights = {
            K3W.MOE_GATE: router_weight,
            K3W.MOE_CORRECTION_BIAS: correction_bias,
        }

        with patch.object(torch, "mm", side_effect=AssertionError("unexpected mm")):
            actual_ids, actual_weights = module._route(hidden)

        logits = torch.matmul(hidden.float(), router_weight.float())
        expected_ids, expected_weights = self._expected_route(
            module, logits, correction_bias
        )
        torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)

    def test_nondivisible_kda_projection_restores_logical_token_domain(
        self,
    ) -> None:
        module = self._packed_kda_stub(8)
        layout = sequence_parallel.token_shard_layout(9, 8, 0)
        local_hidden = torch.randn(2, 16, dtype=torch.bfloat16)
        projected = torch.randn(
            9,
            module.kda_fused_w.shape[1],
            dtype=torch.bfloat16,
        )

        with patch.object(
            kimi_k3_kda,
            "all_gather_gemm",
            return_value=[projected],
        ) as project:
            outputs = module._project_fused_kda_inputs(
                local_hidden,
                prefill_sp_layout=layout,
            )

        project.assert_called_once_with(
            local_hidden,
            [module.kda_fused_w],
            logical_m=9,
        )
        for output in outputs:
            self.assertEqual(output.shape[0], 9)

    def test_sharded_mla_projection_uses_loader_packed_weight(self) -> None:
        module = KimiK3MLA.__new__(KimiK3MLA)
        nn.Module.__init__(module)
        module._sp_prefill_input_is_sharded = True
        module._sp_prefill_layout_for_forward = sequence_parallel.token_shard_layout(
            3, 2, 0
        )
        module.attn_tp_size = 2
        module.q_lora_rank = 3
        module.kv_lora_rank = 2
        module.suffix_dim = 1
        module.local_heads = 2
        module.value_dim = 4
        module._mla_backend = "kernel"
        module.use_output_gate = True
        module.attn_tp_rank = 0
        packed_weight = torch.randn(5, 14)
        module._packed_qkv_gate_w = packed_weight
        module.weights = {W.mla_fusedqkrope_w: packed_weight}
        local_input = torch.randn(2, 5)
        projected = torch.randn(3, 14)

        with patch.object(
            kimi_k3_mla,
            "all_gather_gemm",
            return_value=[projected],
        ) as project:
            actual_qkv_a, actual_gate = module._project_qkv_a_input(local_input)

        project.assert_called_once_with(
            local_input,
            [packed_weight],
            logical_m=3,
        )
        torch.testing.assert_close(actual_qkv_a, projected[:, :6], rtol=0, atol=0)
        torch.testing.assert_close(actual_gate, projected[:, 6:], rtol=0, atol=0)

    def test_kda_prefill_o_proj_uses_gemm_reduce_scatter(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        module = KimiK3KDA.__new__(KimiK3KDA)
        nn.Module.__init__(module)
        module.projection_size = 8
        module.attn_tp_size = 2
        module.attn_tp_rank = 0
        module.eps = 1e-6
        module.weights = {
            W.linear_attn_out_w: torch.empty(
                (8, 16), dtype=torch.bfloat16, device="cuda"
            ),
            W.linear_attn_norm_w: torch.empty(8, dtype=torch.bfloat16, device="cuda"),
        }
        projection_input = torch.randn((8, 8), dtype=torch.bfloat16, device="cuda")
        output_gate = torch.empty((1, 8, 1), dtype=torch.bfloat16, device="cuda")
        hidden_states = torch.empty((4, 16), dtype=torch.bfloat16, device="cuda")
        fused_output = torch.empty((4, 16), dtype=torch.bfloat16, device="cuda")
        group = object()

        with (
            patch.object(
                kimi_k3_kda,
                "kimi_kda_rms_norm_sigmoid_gate",
                return_value=projection_input,
            ),
            patch.object(kimi_k3_kda, "get_process_group", return_value=group),
            patch.object(
                kimi_k3_kda,
                "gemm_reduce_scatter",
                return_value=fused_output,
            ) as gemm_rs,
            patch.object(kimi_k3_kda, "row_parallel_linear") as fallback,
        ):
            actual = module._project_output(
                torch.empty(1, dtype=torch.bfloat16, device="cuda"),
                output_gate,
                is_target_verify=False,
                sequence_parallel=True,
                hidden_states=hidden_states,
                mode="prefill",
            )

        self.assertIs(actual, fused_output)
        gemm_rs.assert_called_once()
        gemm_rs_input, gemm_rs_weight, gemm_rs_group = gemm_rs.call_args.args
        self.assertEqual(gemm_rs_input.data_ptr(), projection_input.data_ptr())
        self.assertIs(gemm_rs_weight, module.weights[W.linear_attn_out_w])
        self.assertIs(gemm_rs_group, group)
        self.assertEqual(gemm_rs.call_args.kwargs, {"pad_rows": False})
        fallback.assert_not_called()

    def test_mla_prefill_o_proj_uses_gemm_reduce_scatter(self) -> None:
        module = KimiK3MLA.__new__(KimiK3MLA)
        nn.Module.__init__(module)
        module.parallelism_config = SimpleNamespace(get_attn_tp_size=lambda: 8)
        module._sp_active_for_forward = True
        module._sp_padded_for_forward = False
        module._sp_prefill_input_is_sharded = True
        module._o_w = torch.empty((8, 16))
        attn_output = torch.empty((32768, 8))
        fused_output = torch.empty((4096, 16))
        group = object()

        with (
            patch.object(kimi_k3_mla, "get_process_group", return_value=group),
            patch.object(
                kimi_k3_mla,
                "gemm_reduce_scatter",
                return_value=fused_output,
            ) as gemm_rs,
            patch.object(kimi_k3_mla, "row_parallel_linear") as fallback,
        ):
            actual = module._project_output(attn_output)

        self.assertIs(actual, fused_output)
        gemm_rs.assert_called_once()
        gemm_rs_input, gemm_rs_weight, gemm_rs_group = gemm_rs.call_args.args
        self.assertIs(gemm_rs_input, attn_output)
        self.assertIs(gemm_rs_weight, module._o_w)
        self.assertIs(gemm_rs_group, group)
        self.assertEqual(gemm_rs.call_args.kwargs, {"pad_rows": False})
        fallback.assert_not_called()

    def test_all_gather_gemm_uses_fused_path_above_threshold(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        device = torch.device("cuda", torch.cuda.current_device())
        local_input = torch.empty((8192, 1), dtype=torch.bfloat16, device=device)
        weight = torch.empty((1, 1), dtype=torch.bfloat16, device=device)
        output = torch.empty((65536, 1), dtype=torch.bfloat16, device=device)
        group = Mock(group_name="tp-test")
        group.size.return_value = 8
        state = kimi_k3_ag_gemm._AllGatherGemmState(
            enabled=True,
            group=group,
            device=device,
            world_size=8,
            max_m=65536,
            k=1,
            dtype=torch.bfloat16,
            workspace_bytes=8192 * 2,
        )
        key = (group, device.index)
        with (
            patch.object(kimi_k3_ag_gemm, "get_process_group", return_value=group),
            patch.object(
                kimi_k3_ag_gemm,
                "fused_all_gather_matmul",
                return_value=(None, [output]),
            ) as fused,
            patch.dict(kimi_k3_ag_gemm._STATES, {key: state}, clear=True),
        ):
            actual = kimi_k3_ag_gemm.all_gather_gemm(
                local_input,
                [weight],
                logical_m=65536,
            )[0]

        fused.assert_called_once_with(
            local_input,
            [weight],
            group,
            return_gathered=False,
        )
        self.assertIs(actual, output)

    def test_small_prefill_uses_all_gather_then_gemm(self) -> None:
        local_input = torch.randn(2, 3)
        gathered_input = torch.randn(4, 3)
        weight = torch.randn(3, 5)
        group = SimpleNamespace(size=lambda: 2)
        with (
            patch.object(
                kimi_k3_ag_gemm,
                "get_process_group",
                return_value=group,
            ),
            patch.object(
                kimi_k3_ag_gemm,
                "all_gather_into",
                return_value=gathered_input,
            ) as gather,
            patch.object(kimi_k3_ag_gemm, "fused_all_gather_matmul") as fused,
            patch.dict(kimi_k3_ag_gemm._STATES, {}, clear=True),
        ):
            actual = kimi_k3_ag_gemm.all_gather_gemm(
                local_input,
                [weight],
                logical_m=4,
            )[0]

        gather.assert_called_once_with(local_input, ANY, kimi_k3_ag_gemm.Group.TP)
        fused.assert_not_called()
        torch.testing.assert_close(actual, torch.mm(gathered_input, weight))

    def test_all_gather_gemm_policy_starts_at_32k_physical_m(self) -> None:
        self.assertFalse(kimi_k3_ag_gemm.should_use_all_gather_gemm(32767))
        self.assertTrue(kimi_k3_ag_gemm.should_use_all_gather_gemm(32768))

    def test_configure_all_gather_gemm_reserves_local_input_bytes(self) -> None:
        group = Mock()
        group.size.return_value = 8
        device = torch.device("cuda", 0)
        with (
            patch.dict(kimi_k3_ag_gemm._STATES, {}, clear=True),
            patch.dict(
                kimi_k3_ag_gemm.os.environ,
                {"LOCAL_WORLD_SIZE": "8"},
                clear=True,
            ),
            patch.object(
                kimi_k3_ag_gemm,
                "reserve_fused_all_gather_matmul_workspace",
            ) as reserve,
            patch.object(
                kimi_k3_ag_gemm,
                "_all_ranks_ready",
                return_value=True,
            ) as all_ranks_ready,
        ):
            enabled = kimi_k3_ag_gemm.configure_all_gather_gemm(
                group,
                device,
                enabled=True,
                max_m=32768,
                k=16,
                dtype=torch.bfloat16,
            )

        self.assertTrue(enabled)
        reserve.assert_called_once_with(group, 4096 * 16 * 2)
        all_ranks_ready.assert_called_once_with(True, group, device)

    def test_configure_all_gather_gemm_tp16_without_nvshmem_falls_back(
        self,
    ) -> None:
        group = Mock()
        group.size.return_value = 16
        device = torch.device("cuda", 0)
        with (
            patch.dict(kimi_k3_ag_gemm._STATES, {}, clear=True),
            patch.dict(
                kimi_k3_ag_gemm.os.environ,
                {"KIMI_K3_SYMM_MEM_BACKEND": "NVSHMEM", "LOCAL_WORLD_SIZE": "8"},
                clear=True,
            ),
            patch.object(
                kimi_k3_ag_gemm,
                "configure_symm_mem_backend_from_env",
                return_value=None,
            ) as configure_symm_mem,
            patch.object(
                kimi_k3_ag_gemm,
                "reserve_fused_all_gather_matmul_workspace",
            ) as reserve,
            patch.object(
                kimi_k3_ag_gemm,
                "_all_ranks_ready",
                return_value=False,
            ) as all_ranks_ready,
        ):
            enabled = kimi_k3_ag_gemm.configure_all_gather_gemm(
                group,
                device,
                enabled=True,
                max_m=32768,
                k=16,
                dtype=torch.bfloat16,
            )

        self.assertFalse(enabled)
        reserve.assert_not_called()
        configure_symm_mem.assert_called_once_with()
        all_ranks_ready.assert_called_once_with(False, group, device)

    def test_configure_all_gather_gemm_workspace_failure_is_fatal(self) -> None:
        group = Mock()
        group.size.return_value = 8
        device = torch.device("cuda", 0)
        with (
            patch.dict(kimi_k3_ag_gemm._STATES, {}, clear=True),
            patch.dict(kimi_k3_ag_gemm.os.environ, {}, clear=True),
            patch.object(
                kimi_k3_ag_gemm,
                "_all_ranks_ready",
                return_value=True,
            ),
            patch.object(
                kimi_k3_ag_gemm,
                "reserve_fused_all_gather_matmul_workspace",
                side_effect=RuntimeError("rendezvous failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "rendezvous failed"),
        ):
            kimi_k3_ag_gemm.configure_all_gather_gemm(
                group,
                device,
                enabled=True,
                max_m=32768,
                k=16,
                dtype=torch.bfloat16,
            )

    def test_gemm_reduce_scatter_policy_starts_at_32k_physical_m(self) -> None:
        self.assertFalse(
            kimi_k3_gemm_reduce_scatter.should_use_gemm_reduce_scatter(32767)
        )
        self.assertTrue(
            kimi_k3_gemm_reduce_scatter.should_use_gemm_reduce_scatter(32768)
        )

        with self.assertRaisesRegex(ValueError, "physical_m"):
            kimi_k3_gemm_reduce_scatter.should_use_gemm_reduce_scatter(-1)

    def test_configure_gemm_reduce_scatter_tp16_falls_back(self) -> None:
        group = Mock()
        group.size.return_value = 16
        device = torch.device("cuda", 0)
        with (
            patch.dict(kimi_k3_gemm_reduce_scatter._STATES, {}, clear=True),
            patch.dict(
                kimi_k3_gemm_reduce_scatter.os.environ,
                {},
                clear=True,
            ),
        ):
            enabled = kimi_k3_gemm_reduce_scatter.configure_gemm_reduce_scatter(
                group,
                device,
                enabled=True,
                max_m=32768,
                n=16,
            )

        self.assertFalse(enabled)

    def test_configure_gemm_reduce_scatter_tp16_pinned_raises(self) -> None:
        group = Mock()
        group.size.return_value = 16
        device = torch.device("cuda", 0)
        with (
            patch.dict(kimi_k3_gemm_reduce_scatter._STATES, {}, clear=True),
            patch.dict(
                kimi_k3_gemm_reduce_scatter.os.environ,
                {"KIMI_K3_GEMM_REDUCE_SCATTER_BACKEND": "deepgemm"},
                clear=True,
            ),
            self.assertRaisesRegex(RuntimeError, "not TP16"),
        ):
            kimi_k3_gemm_reduce_scatter.configure_gemm_reduce_scatter(
                group,
                device,
                enabled=True,
                max_m=32768,
                n=16,
            )

    def test_gemm_reduce_scatter_dispatches_from_current_physical_m(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        device = torch.device("cuda", torch.cuda.current_device())
        group = object()
        launch = Mock()
        workspace = object()
        state = kimi_k3_gemm_reduce_scatter._GemmReduceScatterState(
            enabled=True,
            group=group,
            device=device,
            world_size=8,
            max_m=32768,
            n=16,
            deep_gemm=SimpleNamespace(bf16_gemm_rs_nn=launch),
            workspace=workspace,
        )
        weight = torch.empty((8, 16), dtype=torch.bfloat16, device=device)
        key = (group, device.index)

        with patch.dict(
            kimi_k3_gemm_reduce_scatter._STATES,
            {key: state},
            clear=True,
        ):
            below_threshold = torch.empty(
                (32760, 8), dtype=torch.bfloat16, device=device
            )
            self.assertIsNone(
                kimi_k3_gemm_reduce_scatter.gemm_reduce_scatter(
                    below_threshold,
                    weight,
                    group,
                    pad_rows=False,
                )
            )
            launch.assert_not_called()

            at_threshold = torch.zeros(
                (32768, 16), dtype=torch.bfloat16, device=device
            )[:, ::2]
            self.assertFalse(at_threshold.is_contiguous())
            output = kimi_k3_gemm_reduce_scatter.gemm_reduce_scatter(
                at_threshold,
                weight,
                group,
                pad_rows=False,
            )

        assert output is not None
        self.assertEqual(tuple(output.shape), (4096, 16))
        launch.assert_called_once()
        launch_input, launch_weight, launch_output, launch_workspace = (
            launch.call_args.args
        )
        self.assertTrue(launch_input.is_contiguous())
        torch.testing.assert_close(launch_input, at_threshold, rtol=0, atol=0)
        self.assertIs(launch_weight, weight)
        self.assertIs(launch_output, output)
        self.assertIs(launch_workspace, workspace)
        self.assertEqual(launch.call_args.kwargs, {"compiled_dims": "nk"})

    def test_gemm_reduce_scatter_cpu_input_uses_fallback(self) -> None:
        self.assertIsNone(
            kimi_k3_gemm_reduce_scatter.gemm_reduce_scatter(
                torch.empty((32768, 8), dtype=torch.bfloat16),
                torch.empty((8, 16), dtype=torch.bfloat16),
                object(),
                pad_rows=False,
            )
        )

    def test_padded_shards_cover_logical_tokens_for_tp2_tp4_tp8(self) -> None:
        for tp_size in (2, 4, 8):
            logical_sizes = (*range(1, 2 * tp_size + 2), 32761, 32768, 32769)
            for logical_tokens in logical_sizes:
                source = torch.arange(logical_tokens * 3).reshape(logical_tokens, 3)
                shards = []
                valid_tokens = 0
                for tp_rank in range(tp_size):
                    layout = sequence_parallel.token_shard_layout(
                        logical_tokens,
                        tp_size,
                        tp_rank,
                    )
                    shards.append(sequence_parallel.shard_tokens(source, layout))
                    valid_tokens += layout.local_valid_tokens

                gathered = torch.cat(shards)
                self.assertEqual(valid_tokens, logical_tokens)
                self.assertEqual(gathered.shape[0] % tp_size, 0)
                torch.testing.assert_close(
                    gathered[:logical_tokens],
                    source,
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(
                    torch.count_nonzero(gathered[logical_tokens:]).item(),
                    0,
                )

    def test_nondivisible_fused_projection_uses_padded_physical_m(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        logical_tokens = 32761
        device = torch.device("cuda", torch.cuda.current_device())
        local_input = torch.empty((4096, 1), dtype=torch.bfloat16, device=device)
        weight = torch.empty((1, 1), dtype=torch.bfloat16, device=device)
        physical_output = torch.arange(
            32768, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        group = Mock(group_name="tp-test")
        group.size.return_value = 8
        state = kimi_k3_ag_gemm._AllGatherGemmState(
            enabled=True,
            group=group,
            device=device,
            world_size=8,
            max_m=32768,
            k=1,
            dtype=torch.bfloat16,
            workspace_bytes=4096 * 2,
        )
        key = (group, device.index)
        with (
            patch.object(kimi_k3_ag_gemm, "get_process_group", return_value=group),
            patch.object(
                kimi_k3_ag_gemm,
                "fused_all_gather_matmul",
                return_value=(None, [physical_output]),
            ),
            patch.dict(kimi_k3_ag_gemm._STATES, {key: state}, clear=True),
        ):
            actual = kimi_k3_ag_gemm.all_gather_gemm(
                local_input,
                [weight],
                logical_m=logical_tokens,
            )[0]

        self.assertEqual(actual.shape, (logical_tokens, 1))
        torch.testing.assert_close(
            actual,
            physical_output[:logical_tokens],
            rtol=0,
            atol=0,
        )

    def test_nondivisible_separate_projection_trims_padding(self) -> None:
        local_input = torch.randn(2, 3)
        gathered_input = torch.randn(4, 3)
        weight = torch.randn(3, 5)
        group = SimpleNamespace(size=lambda: 2)
        with (
            patch.object(
                kimi_k3_ag_gemm,
                "get_process_group",
                return_value=group,
            ),
            patch.object(
                kimi_k3_ag_gemm,
                "all_gather_into",
                return_value=gathered_input,
            ),
            patch.dict(kimi_k3_ag_gemm._STATES, {}, clear=True),
        ):
            actual = kimi_k3_ag_gemm.all_gather_gemm(
                local_input,
                [weight],
                logical_m=3,
            )[0]

        torch.testing.assert_close(
            actual,
            torch.mm(gathered_input[:3], weight),
            rtol=0,
            atol=0,
        )

    def test_nondivisible_prefill_row_projection_uses_bf16_padded_partial(
        self,
    ) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        x = torch.randn(9, 16, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
        captured = None

        def fake_reduce_scatter(partial, *, group):
            nonlocal captured
            captured = partial.clone()
            return partial[:2].clone()

        with (
            patch.object(
                sequence_parallel,
                "reduce_scatter",
                side_effect=fake_reduce_scatter,
            ),
            patch.object(sequence_parallel, "reduce_scatter_padded") as legacy_padding,
        ):
            actual = sequence_parallel.row_parallel_linear(
                x,
                weight,
                world_size=8,
                reduce_scatter_tokens=True,
                pad_reduce_scatter_tokens=True,
                use_input_dtype_reduce_scatter=True,
            )

        self.assertEqual(actual.shape, (2, 8))
        self.assertIsNotNone(captured)
        assert captured is not None
        self.assertEqual(captured.shape, (16, 8))
        self.assertEqual(captured.dtype, torch.bfloat16)
        torch.testing.assert_close(captured[:9], torch.mm(x, weight), rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(captured[9:]).item(), 0)
        legacy_padding.assert_not_called()

    def test_divisible_prefill_row_projection_keeps_original_rs_path(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        x = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")

        with (
            patch.object(
                sequence_parallel,
                "reduce_scatter",
                side_effect=lambda partial, *, group: partial[:1].clone(),
            ) as reduce_scatter,
            patch.object(sequence_parallel, "_matmul_with_padded_rows") as padded_mm,
        ):
            actual = sequence_parallel.row_parallel_linear(
                x,
                weight,
                world_size=8,
                reduce_scatter_tokens=True,
                pad_reduce_scatter_tokens=False,
                use_input_dtype_reduce_scatter=True,
            )

        padded_mm.assert_not_called()
        partial = reduce_scatter.call_args.args[0]
        self.assertEqual(partial.shape, (8, 8))
        torch.testing.assert_close(partial, torch.mm(x, weight), rtol=0, atol=0)
        torch.testing.assert_close(actual, partial[:1], rtol=0, atol=0)

    def test_latent_moe_drops_invalid_rows_before_ep_and_zeroes_output(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        module = KimiK3LatentMoE.__new__(KimiK3LatentMoE)
        nn.Module.__init__(module)
        hidden_size = 4
        latent_size = 2
        shared_size = 3
        module.trace_prefix = "test.moe"
        module.attn_tp_size = 8
        module.ep_size = 8
        module.ffn_tp_size = 1
        module.ffn_tp_rank = 0
        module.expert_num = 4
        module.latent_moe_use_norm = False
        module.routed_norm = None
        module.eps = 1e-6
        module.beta = 1.0
        module.linear_beta = None
        module.shared_expert_weight_shard = False
        module._full_column_weights = {}
        module._full_row_weights = {}
        module.weights = {
            K3W.MOE_ROUTED_DOWN: torch.randn(
                hidden_size,
                latent_size,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            K3W.MOE_ROUTED_UP: torch.randn(
                latent_size,
                hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            K3W.MOE_SHARED_GATE_UP: torch.randn(
                2 * shared_size,
                hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            K3W.MOE_SHARED_DOWN: torch.randn(
                shared_size,
                hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            ),
        }
        hidden_states = torch.randn(
            2,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        routed_ids = torch.tensor([[0, 1], [2, 3]], device="cuda")
        routed_weights = torch.ones(
            (2, 2),
            dtype=torch.float32,
            device="cuda",
        )
        captured = {}

        def fake_expert_sum(
            routed_input,
            expert_ids,
            routing_weights,
            *,
            sequence_parallel,
        ):
            captured["expert_ids"] = expert_ids.clone()
            captured["routing_weights"] = routing_weights.clone()
            captured["sequence_parallel"] = sequence_parallel
            return routed_input + 1

        with (
            patch.object(
                module,
                "_route",
                return_value=(routed_ids, routed_weights),
            ),
            patch.object(
                module,
                "_mega_expert_sum",
                side_effect=fake_expert_sum,
            ),
        ):
            output = module(
                hidden_states,
                sequence_parallel=True,
                valid_token_count=1,
            )

        self.assertTrue(captured["sequence_parallel"])
        torch.testing.assert_close(
            captured["expert_ids"][0],
            routed_ids[0],
            rtol=0,
            atol=0,
        )
        self.assertTrue(
            torch.equal(
                captured["expert_ids"][1],
                torch.zeros((2,), dtype=torch.int64, device="cuda"),
            )
        )
        self.assertEqual(
            torch.count_nonzero(captured["routing_weights"][1]).item(),
            0,
        )
        self.assertEqual(torch.count_nonzero(output[1]).item(), 0)

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
                self.received_prefill_layout = None

            def forward(self, *args, **kwargs):
                self.received_prefill_layout = kwargs["prefill_sp_layout"]
                raise RuntimeError("stop after attention dispatch")

        layer = KimiK3DecoderLayer.__new__(KimiK3DecoderLayer)
        nn.Module.__init__(layer)
        layer.layer_idx = 1
        layer.eps = 1e-5
        layer.attn_res_block_size = 2
        layer.layer_type = kimi_k3.HybridAttentionType.LINEAR
        layer.attention_norm = nn.Identity()
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
        layout = sequence_parallel.token_shard_layout(8, 2, 0)

        attn_meta = kimi_k3.KimiK3DecoderMetadata(
            cu_seqlens=cu_seqlens,
            mode="prefill",
            sequence_parallel=True,
            prefill_sp_layout=layout,
        )
        with self.assertRaisesRegex(RuntimeError, "stop after attention dispatch"):
            layer(
                hidden,
                block_residual,
                attn_meta=attn_meta,
            )

        self.assertIs(layer.self_attn.received_prefill_layout, layout)

    def test_prefill_layer_passes_explicit_valid_rows_to_moe(self) -> None:
        class StubAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.parallelism_config = SimpleNamespace(
                    get_attn_tp_size=lambda: 8,
                    get_attn_tp_rank=lambda: 4,
                )

            def forward(self, hidden_states, *args, **kwargs):
                return torch.zeros_like(hidden_states)

        class StubMoe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.valid_token_count = None
                self.sequence_parallel = None

            def forward(
                self,
                hidden_states,
                *,
                sequence_parallel,
                valid_token_count,
            ):
                self.valid_token_count = valid_token_count
                self.sequence_parallel = sequence_parallel
                return torch.zeros_like(hidden_states)

        class IdentityResidual(nn.Module):
            def forward(self, prefix_sum, *args, **kwargs):
                return prefix_sum

        layer = KimiK3DecoderLayer.__new__(KimiK3DecoderLayer)
        nn.Module.__init__(layer)
        layer.layer_idx = 1
        layer.eps = 1e-6
        layer.attn_res_block_size = 2
        layer.layer_type = kimi_k3.HybridAttentionType.LINEAR
        layer.attention_norm = nn.Identity()
        layer.mlp_residual = IdentityResidual()
        layer.mlp_norm = SimpleNamespace(
            weight=torch.ones(4),
            variance_epsilon=1e-6,
        )
        layer.self_attn = StubAttention()
        layer.mlp = StubMoe()
        hidden = torch.randn(2, 4)
        block_residual = torch.empty(2, 0, 4)
        layer.weights = {
            W.pre_ln_gamma: torch.ones(4),
            W.post_ln_gamma: torch.ones(4),
            K3W.MLP_RES_NORM: torch.empty(0),
            K3W.MLP_RES_PROJ: torch.empty(0),
        }
        layout = sequence_parallel.token_shard_layout(9, 8, 4)

        attn_meta = kimi_k3.KimiK3DecoderMetadata(
            cu_seqlens=torch.tensor([0, 9], dtype=torch.int32),
            mode="prefill",
            sequence_parallel=True,
            prefill_sp_layout=layout,
        )
        output = layer(
            hidden,
            block_residual,
            attn_meta=attn_meta,
        )

        self.assertEqual(layout.local_valid_tokens, 1)
        self.assertEqual(layer.mlp.valid_token_count, 1)
        self.assertTrue(layer.mlp.sequence_parallel)
        torch.testing.assert_close(output.hidden_states, hidden, rtol=0, atol=0)

    def test_model_initialize_configures_all_gather_gemm(self) -> None:
        class FakeCudaEmbedding:
            is_cuda = True
            dtype = torch.bfloat16
            device = torch.device("cuda", 0)

        model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(model)
        max_global_tokens = 32761
        model.config = SimpleNamespace(
            max_seq_len=max_global_tokens,
            hidden_size=16,
        )
        model.parallelism_config = SimpleNamespace(
            get_attn_tp_size=lambda: 8,
        )
        model.embedding_weight = FakeCudaEmbedding()
        model._all_gather_gemm_configured = False
        model._gemm_reduce_scatter_configured = True
        init_resource = SimpleNamespace(
            kv_cache=None,
            is_decode_role=False,
            max_context_batch_size=1,
        )
        group = object()

        with (
            patch.object(kimi_k3, "get_process_group", return_value=group),
            patch.object(
                kimi_k3,
                "configure_all_gather_gemm",
                return_value=True,
            ) as configure,
        ):
            self.assertTrue(model.initialize(init_resource))

        configure.assert_called_once_with(
            group,
            torch.device("cuda", 0),
            enabled=True,
            max_m=32768,
            k=16,
            dtype=torch.bfloat16,
        )
        self.assertTrue(model._all_gather_gemm_configured)

    def test_model_initialize_configures_gemm_reduce_scatter(self) -> None:
        class FakeCudaEmbedding:
            is_cuda = True
            dtype = torch.bfloat16
            device = torch.device("cuda", 0)

            @staticmethod
            def element_size() -> int:
                return 2

        model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(max_seq_len=32761, hidden_size=7168)
        model.parallelism_config = SimpleNamespace(
            get_attn_tp_size=lambda: 8,
        )
        model.embedding_weight = FakeCudaEmbedding()
        model._all_gather_gemm_configured = True
        model._gemm_reduce_scatter_configured = False
        init_resource = SimpleNamespace(
            kv_cache=None,
            is_decode_role=False,
            max_context_batch_size=1,
        )
        group = object()

        with (
            patch.object(kimi_k3, "get_process_group", return_value=group),
            patch.object(
                kimi_k3,
                "configure_gemm_reduce_scatter",
                return_value=True,
            ) as configure,
        ):
            self.assertTrue(model.initialize(init_resource))

        configure.assert_called_once_with(
            group,
            torch.device("cuda", 0),
            enabled=True,
            max_m=32768,
            n=7168,
        )
        self.assertTrue(model._gemm_reduce_scatter_configured)

    def test_model_initialize_bounds_collective_workspaces_by_chunk(self) -> None:
        class FakeCudaEmbedding:
            is_cuda = True
            dtype = torch.bfloat16
            device = torch.device("cuda", 0)

        model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(max_seq_len=1 << 20, hidden_size=7168)
        model.parallelism_config = SimpleNamespace(
            get_attn_tp_size=lambda: 8,
        )
        model.embedding_weight = FakeCudaEmbedding()
        model._all_gather_gemm_configured = False
        model._gemm_reduce_scatter_configured = False
        init_resource = SimpleNamespace(
            kv_cache=None,
            is_decode_role=False,
            max_context_batch_size=4,
        )
        group = object()

        with (
            patch.object(kimi_k3, "prefill_chunk_tokens", return_value=1 << 16),
            patch.object(kimi_k3, "get_process_group", return_value=group),
            patch.object(
                kimi_k3,
                "configure_all_gather_gemm",
                return_value=True,
            ) as configure_ag_gemm,
            patch.object(
                kimi_k3,
                "configure_gemm_reduce_scatter",
                return_value=True,
            ) as configure_gemm_rs,
        ):
            self.assertTrue(model.initialize(init_resource))

        configure_ag_gemm.assert_called_once_with(
            group,
            torch.device("cuda", 0),
            enabled=True,
            max_m=1 << 16,
            k=7168,
            dtype=torch.bfloat16,
        )
        configure_gemm_rs.assert_called_once_with(
            group,
            torch.device("cuda", 0),
            enabled=True,
            max_m=1 << 16,
            n=7168,
        )
        self.assertTrue(model._all_gather_gemm_configured)
        self.assertTrue(model._gemm_reduce_scatter_configured)

    def test_decode_eagle3_uses_fixed_hidden_buffer_across_graph_shapes(self) -> None:
        model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            max_seq_len=32768,
            hidden_size=16,
            gen_num_per_cycle=3,
        )
        model.parallelism_config = SimpleNamespace(
            get_attn_tp_size=lambda: 8,
        )
        model.embedding_weight = torch.empty(1, dtype=torch.bfloat16)
        model._max_generate_batch_size = 8
        model._all_gather_gemm_configured = False
        model._mtp_hidden_buffer = None
        model._mtp_hidden_valid_tokens = 0
        init_resource = SimpleNamespace(
            kv_cache=None,
            is_decode_role=True,
            max_context_batch_size=1,
        )

        with patch.dict(kimi_k3.os.environ, {"SP_TYPE": "eagle3"}):
            self.assertTrue(model.initialize(init_resource))

        self.assertEqual(tuple(model._mtp_hidden_buffer.shape), (32, 48))
        original_ptr = model._mtp_hidden_buffer.data_ptr()
        captured_small = torch.arange(4 * 48, dtype=torch.bfloat16).reshape(4, 48)
        captured = torch.arange(12 * 48, dtype=torch.bfloat16).reshape(12, 48)

        model._write_mtp_hidden_buffer(captured_small, is_cuda_graph=True)
        model._write_mtp_hidden_buffer(captured, is_cuda_graph=True)

        self.assertEqual(model._mtp_hidden_buffer.data_ptr(), original_ptr)
        self.assertEqual(model._mtp_hidden_valid_tokens, 0)
        torch.testing.assert_close(
            model.get_mtp_target_hidden_states(12), captured, rtol=0, atol=0
        )

    def test_decode_packed_projection_is_cuda_graph_safe_and_local(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        module = self._packed_kda_stub(8)
        module.kda_fused_w = module.kda_fused_w.cuda()
        module.weights[W.linear_attn_f_b_w] = module.weights[W.linear_attn_f_b_w].cuda()
        hidden = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            module._project_fused_kda_inputs(
                hidden,
                prefill_sp_layout=None,
            )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module._project_fused_kda_inputs(
                hidden,
                prefill_sp_layout=None,
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
            expected_packed[:, : 3 * module.projection_size],
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
