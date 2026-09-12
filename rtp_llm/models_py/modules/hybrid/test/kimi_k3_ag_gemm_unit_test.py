import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

import rtp_llm.models_py.model_desc.kimi_k3 as kimi_k3
import rtp_llm.models_py.distributed.sequence_parallel as sequence_parallel
import rtp_llm.models_py.modules.hybrid.test.collective_gemm_reference as reference
import rtp_llm.models_py.modules.kimi_k3.all_gather_gemm as kimi_k3_ag_gemm
import rtp_llm.models_py.modules.kimi_k3.gemm_reduce_scatter as kimi_k3_gemm_reduce_scatter
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.model_desc.kimi_k3 import (
    KimiK3DecoderLayer,
    KimiK3KDA,
    KimiK3LatentMoE,
    KimiK3MLA,
    KimiK3Model,
)
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.models_py.modules.kimi_k3.parallel_mode import KimiK3ParallelMode
from rtp_llm.models_py.modules.kimi_k3.parallel_mode import (
    resolve_kimi_k3_parallel_mode,
)
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W


def _sp_layout(logical_tokens: int, world_size: int, rank: int):
    physical_tokens = (logical_tokens + world_size - 1) // world_size * world_size
    return sequence_parallel.sequence_parallel_layout(
        mode="prefill",
        logical_requests=1,
        physical_requests=1 + int(physical_tokens > logical_tokens),
        tokens_per_request=0,
        logical_tokens=logical_tokens,
        physical_tokens=physical_tokens,
        world_size=world_size,
        rank=rank,
    )


class KimiK3CollectiveGemmUnitTest(unittest.TestCase):
    @staticmethod
    def _prepare_model_init_stub(model: KimiK3Model) -> None:
        model.parallel_mode = KimiK3ParallelMode.TP_SP
        model._k3_page_tokens = None
        model._kda_checkpoint_tokens = None
        model._ktp_capture_buckets = ()

    def test_parallel_mode_is_fixed_by_topology(self) -> None:
        def config(*, tp_size: int, ktp_size: int, role_type: RoleType):
            return SimpleNamespace(
                get_attn_tp_size=lambda: tp_size,
                ktp_size=ktp_size,
                role_type=role_type,
            )

        self.assertIs(
            resolve_kimi_k3_parallel_mode(
                config(tp_size=8, ktp_size=1, role_type=RoleType.DECODE)
            ),
            KimiK3ParallelMode.TP_SP,
        )
        self.assertIs(
            resolve_kimi_k3_parallel_mode(
                config(tp_size=1, ktp_size=8, role_type=RoleType.DECODE)
            ),
            KimiK3ParallelMode.PROJECTION_KTP,
        )
        with self.assertRaisesRegex(RuntimeError, "Decode"):
            resolve_kimi_k3_parallel_mode(
                config(tp_size=1, ktp_size=8, role_type=RoleType.PREFILL)
            )
        with self.assertRaisesRegex(RuntimeError, "TP=1"):
            resolve_kimi_k3_parallel_mode(
                config(tp_size=8, ktp_size=8, role_type=RoleType.DECODE)
            )

    def test_generic_dense_returns_local_partial_without_collective(self) -> None:
        module = DenseMLP.__new__(DenseMLP)
        nn.Module.__init__(module)
        module.is_gated = True
        module.merge_gate_up = False
        module.act_fn = lambda gate, up: gate * up
        module.gate_proj = nn.Linear(4, 6, bias=False)
        module.up_proj = nn.Linear(4, 6, bias=False)
        module.down_proj = nn.Linear(6, 4, bias=False)
        x = torch.randn(3, 4)

        module.parallelism_config = SimpleNamespace(get_ffn_tp_size=lambda: 8)
        actual = module(x, reduce_output=False)
        expected = module.down_proj(module.gate_proj(x) * module.up_proj(x))

        torch.testing.assert_close(actual, expected)

    def test_decoder_layer_owns_tp_sp_collectives(self) -> None:
        layer = kimi_k3.KimiK3DecoderLayer.__new__(kimi_k3.KimiK3DecoderLayer)
        nn.Module.__init__(layer)
        layer.parallel_mode = KimiK3ParallelMode.TP_SP
        local_input = torch.randn(2, 4)
        weight = torch.randn(4, 6)
        projected = torch.randn(8, 6)
        layout = _sp_layout(8, 4, 0)

        with patch.object(
            kimi_k3, "all_gather_gemm", return_value=[projected]
        ) as ag_gemm:
            actual = layer._project_tp_sp_inputs(local_input, [weight], layout)

        self.assertIs(actual[0], projected)
        ag_gemm.assert_called_once_with(local_input, [weight], logical_m=8)

        partial = torch.randn(8, 6)
        output = torch.randn(2, 4)
        output_weight = torch.randn(6, 4)
        group = Mock()
        with (
            patch.object(kimi_k3, "get_process_group", return_value=group),
            patch.object(
                kimi_k3, "gemm_reduce_scatter", return_value=output
            ) as gemm_rs,
        ):
            actual = layer._project_parallel_output(partial, output_weight)

        self.assertIs(actual, output)
        gemm_rs.assert_called_once_with(
            partial,
            output_weight,
            group,
            pad_rows=False,
        )

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
        module.ktp_size = 1
        module.ktp_rank = 0
        module.parallel_mode = KimiK3ParallelMode.TP_SP
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
        module._fp8_enabled = False
        module._fp8_projections = {}
        module._fp8_strided_forget = False
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
                full_hidden = torch.randn(8, 16, dtype=torch.bfloat16)
                local_hidden = full_hidden.chunk(tp_size, dim=0)[module.attn_tp_rank]
                packed = torch.mm(full_hidden, module.kda_fused_w)
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
                layout = _sp_layout(8, tp_size, module.attn_tp_rank)
                actual = module._project_fused_kda_inputs(
                    local_hidden,
                    sp_layout=layout,
                    projected_fused=packed,
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

    def test_nondivisible_kda_projection_keeps_physical_token_domain(
        self,
    ) -> None:
        module = self._packed_kda_stub(8)
        layout = _sp_layout(9, 8, 0)
        local_hidden = torch.randn(2, 16, dtype=torch.bfloat16)
        projected = torch.randn(
            16,
            module.kda_fused_w.shape[1],
            dtype=torch.bfloat16,
        )

        outputs = module._project_fused_kda_inputs(
            local_hidden,
            sp_layout=layout,
            projected_fused=projected,
        )
        for output in outputs:
            self.assertEqual(output.shape[0], 16)

    def test_sharded_mla_projection_uses_loader_packed_weight(self) -> None:
        module = KimiK3MLA.__new__(KimiK3MLA)
        nn.Module.__init__(module)
        module.parallel_mode = KimiK3ParallelMode.TP_SP
        module._sp_layout_for_forward = _sp_layout(3, 2, 0)
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
        projected = torch.randn(4, 14)
        module._projected_qkv_a_for_forward = [projected]

        actual_qkv_a, actual_gate = module._project_qkv_a_input(local_input)
        torch.testing.assert_close(actual_qkv_a, projected[:, :6], rtol=0, atol=0)
        torch.testing.assert_close(actual_gate, projected[:, 6:], rtol=0, atol=0)

    def test_kda_prepares_output_for_layer_owned_projection(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        module = KimiK3KDA.__new__(KimiK3KDA)
        nn.Module.__init__(module)
        module.parallel_mode = KimiK3ParallelMode.TP_SP
        module.projection_size = 8
        module.attn_tp_size = 2
        module.attn_tp_rank = 0
        module.eps = 1e-6
        module._fp8_enabled = False
        module._fp8_projections = {}
        module.weights = {
            W.linear_attn_out_w: torch.empty(
                (8, 16), dtype=torch.bfloat16, device="cuda"
            ),
            W.linear_attn_norm_w: torch.empty(8, dtype=torch.bfloat16, device="cuda"),
        }
        projection_input = torch.randn((8, 8), dtype=torch.bfloat16, device="cuda")
        output_gate = torch.empty((1, 8, 1), dtype=torch.bfloat16, device="cuda")
        module.output_norm = Mock(return_value=projection_input)

        actual = module._prepare_output_projection(
            torch.empty(1, dtype=torch.bfloat16, device="cuda"),
            output_gate,
            mode="prefill",
        )

        self.assertEqual(actual.data_ptr(), projection_input.data_ptr())
        self.assertIs(
            module.output_projection_weight(), module.weights[W.linear_attn_out_w]
        )

    def test_mla_returns_output_for_layer_owned_projection(self) -> None:
        module = KimiK3MLA.__new__(KimiK3MLA)
        nn.Module.__init__(module)
        module.parallel_mode = KimiK3ParallelMode.TP_SP
        module.parallelism_config = SimpleNamespace(get_attn_tp_size=lambda: 8)
        module._o_w = torch.empty((8, 16))
        attn_output = torch.empty((32768, 8))
        actual = module._project_output(attn_output)

        self.assertIs(actual, attn_output)
        self.assertIs(module.output_projection_weight(), module._o_w)

    def test_all_gather_gemm_fuses_small_and_large_prefill(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda", torch.cuda.current_device())
        group = Mock(group_name="tp-test")
        group.size.return_value = 8
        state = kimi_k3_ag_gemm._AllGatherGemmState(
            False, group, device, 8, 65536, 1, torch.bfloat16, 16384
        )
        weight = torch.empty((1, 1), dtype=torch.bfloat16, device=device)
        for logical in (1, 7, 8, 9, 32767, 32768, 65536):
            with self.subTest(logical=logical):
                rows = (logical + 7) // 8
                x = torch.empty((rows, 1), dtype=torch.bfloat16, device=device)
                out = torch.empty((rows * 8, 1), dtype=torch.bfloat16, device=device)
                with (
                    patch.object(
                        kimi_k3_ag_gemm, "get_process_group", return_value=group
                    ),
                    patch.dict(
                        kimi_k3_ag_gemm._STATES,
                        {(group, device.index, False): state},
                        clear=True,
                    ),
                    patch.object(
                        kimi_k3_ag_gemm,
                        "fused_all_gather_matmul",
                        return_value=(None, [out]),
                    ) as fused,
                ):
                    actual = kimi_k3_ag_gemm.all_gather_gemm(
                        x, [weight], logical_m=logical
                    )[0]
                fused.assert_called_once_with(x, [weight], group, return_gathered=False)
                self.assertEqual(actual.shape, (logical, 1))

    def test_unconfigured_ag_does_not_fall_back_to_separate_ops(self):
        group = Mock()
        group.size.return_value = 2
        with patch.object(kimi_k3_ag_gemm, "get_process_group", return_value=group):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                kimi_k3_ag_gemm.all_gather_gemm(
                    torch.randn(2, 3), [torch.randn(3, 5)], logical_m=4
                )

    def test_ag_has_no_length_policy_or_reference_switch(self):
        self.assertFalse(hasattr(kimi_k3_ag_gemm, "should_use_all_gather_gemm"))
        self.assertFalse(hasattr(kimi_k3_ag_gemm, "all_gather_into"))

    def test_configure_all_gather_gemm_reserves_local_input_bytes(self) -> None:
        group = Mock()
        group.size.return_value = 8
        device = torch.device("cuda", 0)
        with (
            patch.dict(kimi_k3_ag_gemm._STATES, {}, clear=True),
            patch.object(
                kimi_k3_ag_gemm,
                "reserve_fused_all_gather_matmul_workspace",
            ) as reserve,
        ):
            enabled = kimi_k3_ag_gemm.configure_all_gather_gemm(
                group,
                device,
                max_m=8,
                k=16,
                dtype=torch.bfloat16,
            )

        self.assertTrue(enabled)
        reserve.assert_called_once_with(group, 1 * 16 * 2)

    def test_rs_has_no_length_policy_or_backend_switch(self):
        self.assertFalse(
            hasattr(kimi_k3_gemm_reduce_scatter, "should_use_gemm_reduce_scatter")
        )
        self.assertFalse(
            hasattr(kimi_k3_gemm_reduce_scatter, "gemm_reduce_scatter_backend")
        )

    def test_gemm_reduce_scatter_fuses_all_nonempty_prefill_sizes(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda", torch.cuda.current_device())
        group = Mock()
        group.size.return_value = 8
        launch = Mock()
        workspace = object()
        state = kimi_k3_gemm_reduce_scatter._GemmReduceScatterState(
            group,
            device,
            8,
            32768,
            16,
            SimpleNamespace(bf16_gemm_rs_nn=launch),
            workspace,
        )
        weight = torch.empty((8, 16), dtype=torch.bfloat16, device=device)
        for m in (0, 1, 7, 8, 9, 32760, 32768):
            with (
                self.subTest(m=m),
                patch.dict(
                    kimi_k3_gemm_reduce_scatter._STATES,
                    {(group, device.index): state},
                    clear=True,
                ),
            ):
                launch.reset_mock()
                x = torch.ones((m, 16), dtype=torch.bfloat16, device=device)[:, ::2]
                out = kimi_k3_gemm_reduce_scatter.gemm_reduce_scatter(
                    x, weight, group, pad_rows=True
                )
                physical = (m + 7) // 8 * 8
                self.assertEqual(out.shape, (physical // 8, 16))
                if m == 0:
                    launch.assert_not_called()
                    continue
                launch.assert_called_once()
                actual_x = launch.call_args.args[0]
                self.assertTrue(actual_x.is_contiguous())
                torch.testing.assert_close(actual_x[:m], x, atol=0, rtol=0)
                self.assertEqual(torch.count_nonzero(actual_x[m:]).item(), 0)

    def test_gemm_reduce_scatter_rejects_cpu_instead_of_falling_back(self):
        with self.assertRaisesRegex(TypeError, "CUDA"):
            kimi_k3_gemm_reduce_scatter.gemm_reduce_scatter(
                torch.empty((1, 8), dtype=torch.bfloat16),
                torch.empty((8, 16), dtype=torch.bfloat16),
                object(),
                pad_rows=True,
            )

    def test_padded_shards_cover_logical_tokens_for_tp2_tp4_tp8(self) -> None:
        for tp_size in (2, 4, 8):
            logical_sizes = (*range(1, 2 * tp_size + 2), 32761, 32768, 32769)
            for logical_tokens in logical_sizes:
                source = torch.arange(logical_tokens * 3).reshape(logical_tokens, 3)
                physical_tokens = (
                    (logical_tokens + tp_size - 1) // tp_size * tp_size
                )
                padded = source.new_zeros((physical_tokens, 3))
                padded[:logical_tokens].copy_(source)
                shards = []
                valid_tokens = 0
                for tp_rank in range(tp_size):
                    layout = _sp_layout(logical_tokens, tp_size, tp_rank)
                    shards.append(
                        sequence_parallel.local_physical_token_view(padded, layout)
                    )
                    valid_tokens += layout.tokens.local_valid_tokens

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
            fp8=False,
            group=group,
            device=device,
            world_size=8,
            max_m=32768,
            k=1,
            dtype=torch.bfloat16,
            workspace_bytes=4096 * 2,
        )
        key = (group, device.index, False)
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
                reference,
                "get_process_group",
                return_value=group,
            ),
            patch.object(
                reference,
                "all_gather_into",
                return_value=gathered_input,
            ),
        ):
            actual = reference.all_gather_gemm_reference(
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

    def test_latent_moe_routes_invalid_rows_to_zero_weight_expert_zero(self) -> None:
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
        ):
            captured["expert_ids"] = expert_ids.clone()
            captured["routing_weights"] = routing_weights.clone()
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
                valid_token_count=1,
            )

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
        # Padding is kept numerically independent by its reserved block-0 cache
        # mapping and is trimmed before sampling. MegaMoE still receives the
        # physical row, so only its routed contribution is neutralized here;
        # the shared expert output is intentionally not masked in modeling.
        self.assertGreater(torch.count_nonzero(output[1]).item(), 0)

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
                self.received_sp_layout = None

            def tp_input_projection_weights(self):
                return []

            def forward(self, *args, **kwargs):
                self.received_sp_layout = kwargs["sp_layout"]
                raise RuntimeError("stop after attention dispatch")

        layer = KimiK3DecoderLayer.__new__(KimiK3DecoderLayer)
        nn.Module.__init__(layer)
        layer.parallel_mode = KimiK3ParallelMode.TP_SP
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
        layout = _sp_layout(8, 2, 0)

        attn_meta = kimi_k3.KimiK3DecoderMetadata(
            cu_seqlens=cu_seqlens,
            mode="prefill",
            sp_layout=layout,
        )
        with (
            patch.object(
                layer,
                "_project_tp_sp_inputs",
                return_value=[hidden.new_empty((layout.tokens.physical_tokens, 1))],
            ),
            self.assertRaisesRegex(RuntimeError, "stop after attention dispatch"),
        ):
            layer(hidden, block_residual, attn_meta=attn_meta)

        self.assertIs(layer.self_attn.received_sp_layout, layout)

    def test_prefill_layer_passes_explicit_valid_rows_to_moe(self) -> None:
        class StubAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.parallelism_config = SimpleNamespace(
                    get_attn_tp_size=lambda: 8,
                    get_attn_tp_rank=lambda: 4,
                )

            def tp_input_projection_weights(self):
                return []

            def output_projection_weight(self):
                return torch.empty(0)

            def forward(self, hidden_states, *args, **kwargs):
                return torch.zeros_like(hidden_states)

        class StubMoe(KimiK3LatentMoE):
            def __init__(self) -> None:
                nn.Module.__init__(self)
                self.valid_token_count = None

            def forward(
                self,
                hidden_states,
                *,
                valid_token_count,
                valid_token_mask=None,
            ):
                self.valid_token_count = valid_token_count
                return torch.zeros_like(hidden_states)

        class IdentityResidual(nn.Module):
            def forward(self, prefix_sum, *args, **kwargs):
                return prefix_sum

        layer = KimiK3DecoderLayer.__new__(KimiK3DecoderLayer)
        nn.Module.__init__(layer)
        layer.parallel_mode = KimiK3ParallelMode.TP_SP
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
        layout = _sp_layout(9, 8, 4)

        attn_meta = kimi_k3.KimiK3DecoderMetadata(
            cu_seqlens=torch.tensor([0, 9], dtype=torch.int32),
            mode="prefill",
            sp_layout=layout,
        )
        with (
            patch.object(
                layer,
                "_project_tp_sp_inputs",
                return_value=[torch.empty((layout.tokens.physical_tokens, 1))],
            ),
            patch.object(
                layer,
                "_project_parallel_output",
                side_effect=lambda projection_input, _weight: projection_input,
            ),
        ):
            output = layer(hidden, block_residual, attn_meta=attn_meta)

        self.assertEqual(layout.tokens.local_valid_tokens, 1)
        self.assertEqual(layer.mlp.valid_token_count, 1)
        torch.testing.assert_close(output.hidden_states, hidden, rtol=0, atol=0)

    def test_model_initialize_configures_all_gather_gemm(self) -> None:
        class FakeCudaEmbedding:
            is_cuda = True
            dtype = torch.bfloat16
            device = torch.device("cuda", 0)

        model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(model)
        self._prepare_model_init_stub(model)
        max_global_tokens = 3
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
            max_m=8,
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
        self._prepare_model_init_stub(model)
        model.config = SimpleNamespace(max_seq_len=3, hidden_size=7168)
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
            max_m=8,
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
        self._prepare_model_init_stub(model)
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
            max_m=1 << 16,
            k=7168,
            dtype=torch.bfloat16,
        )
        configure_gemm_rs.assert_called_once_with(
            group,
            torch.device("cuda", 0),
            max_m=1 << 16,
            n=7168,
        )
        self.assertTrue(model._all_gather_gemm_configured)
        self.assertTrue(model._gemm_reduce_scatter_configured)

    def test_decode_eagle3_uses_fixed_hidden_buffer_across_graph_shapes(self) -> None:
        model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(model)
        self._prepare_model_init_stub(model)
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
        model._gemm_reduce_scatter_configured = False
        model._mtp_hidden_buffer = None
        model._mtp_hidden_valid_tokens = 0
        init_resource = SimpleNamespace(
            kv_cache=None,
            is_decode_role=True,
            max_context_batch_size=1,
            max_decode_graph_batch_size=8,
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

    def test_cached_bf16_rs_workspace_validates_fp8_abi(self):
        import rtp_llm.models_py.modules.kimi_k3.gemm_reduce_scatter as rs

        for tp in (2, 4, 8):
            group = Mock()
            group.size.return_value = tp
            workspace = SimpleNamespace()
            state = rs._GemmReduceScatterState(
                group,
                torch.device("cuda", 0),
                tp,
                32768,
                7168,
                SimpleNamespace(_C=SimpleNamespace(bf16_gemm_rs_reduce=Mock())),
                workspace,
            )
            with patch.dict(rs._STATES, {(group, 0): state}, clear=True):
                kwargs = dict(max_m=32768, n=7168)
                self.assertTrue(
                    rs.configure_gemm_reduce_scatter(group, "cuda:0", **kwargs)
                )
                with self.assertRaisesRegex(RuntimeError, "FP8 peer-output RS ABI"):
                    rs.configure_gemm_reduce_scatter(
                        group, "cuda:0", fp8=True, **kwargs
                    )
                workspace._data_offset_bytes = 128
                workspace._mapping_handle = object()
                workspace._launch_lock = object()
                workspace._last_stream = None
                workspace._barrier = Mock()
                self.assertTrue(
                    rs.configure_gemm_reduce_scatter(
                        group, "cuda:0", fp8=True, **kwargs
                    )
                )

    def test_missing_fp8_workspace_is_an_error_not_a_reference_path(self):
        import rtp_llm.models_py.modules.kimi_k3.all_gather_gemm as ag
        import rtp_llm.models_py.modules.kimi_k3.gemm_reduce_scatter as rs
        from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
            QuantizedActivation,
        )

        group = Mock()
        group.size.return_value = 8
        payload = Mock(
            spec=QuantizedActivation, shape=(1, 512), device=torch.device("cuda", 0)
        )
        projection = Mock(K=512, N=512, scale_ue8m0=True)
        with (
            patch.object(ag, "get_process_group", return_value=group),
            patch.dict(ag._STATES, {}, clear=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "initialized"):
                ag._all_gather_quantized(payload, [projection], logical_m=1, group=None)
        source = Mock(is_cuda=True, device=torch.device("cuda", 0))
        with patch.dict(rs._STATES, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "initialized"):
                rs.gemm_reduce_scatter(source, projection, group, pad_rows=True)


if __name__ == "__main__":
    unittest.main()
