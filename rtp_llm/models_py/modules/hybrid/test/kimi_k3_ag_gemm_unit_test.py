import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch

import torch
from torch import nn

import rtp_llm.models_py.model_desc.kimi_k3 as kimi_k3
import rtp_llm.models_py.modules.factory.linear.parallel as sequence_parallel
import rtp_llm.models_py.modules.kimi_k3.kda as kimi_k3_kda
import rtp_llm.models_py.modules.kimi_k3.mla as kimi_k3_mla
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.model_desc.kimi_k3 import (
    KimiK3DecoderLayer,
    KimiK3KDA,
    KimiK3LatentMoE,
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
                    prefill_sp_layout=None,
                )
                for actual_tensor, expected_tensor in zip(actual, expected):
                    torch.testing.assert_close(
                        actual_tensor,
                        expected_tensor,
                        rtol=0,
                        atol=0,
                    )

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
            "all_gather_matmul",
            return_value=[projected],
        ) as project:
            outputs = module._project_fused_kda_inputs(
                local_hidden,
                prefill_sp_layout=layout,
            )

        project.assert_called_once_with(
            local_hidden,
            [module.kda_fused_w],
            logical_tokens=9,
            use_fused=False,
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
            "all_gather_matmul",
            return_value=[projected],
        ) as project:
            actual_qkv_a, actual_gate = module._project_qkv_a_input(local_input)

        project.assert_called_once_with(
            local_input,
            [packed_weight],
            logical_tokens=3,
            use_fused=False,
        )
        torch.testing.assert_close(actual_qkv_a, projected[:, :6], rtol=0, atol=0)
        torch.testing.assert_close(actual_gate, projected[:, 6:], rtol=0, atol=0)

    def test_prefill_fusion_is_a_direct_model_policy(self) -> None:
        local_input = torch.empty((8192, 1))
        weight = torch.empty((1, 1))
        output = torch.empty((65536, 1))
        group = SimpleNamespace(group_name="tp-test")
        with (
            patch.object(
                sequence_parallel, "get_process_group", return_value=group
            ),
            patch.object(
                sequence_parallel,
                "fused_all_gather_matmul",
                return_value=(None, [output]),
            ) as fused,
        ):
            actual = sequence_parallel.all_gather_matmul(
                local_input,
                [weight],
                logical_tokens=65536,
                use_fused=True,
            )[0]

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
            patch.object(
                sequence_parallel,
                "get_process_group",
                return_value=SimpleNamespace(),
            ),
            patch("torch.distributed.get_world_size", return_value=2),
            patch.object(
                sequence_parallel,
                "all_gather_into",
                return_value=gathered_input,
            ) as gather,
            patch.object(sequence_parallel, "fused_all_gather_matmul") as fused,
        ):
            actual = sequence_parallel.all_gather_matmul(
                local_input,
                [weight],
                logical_tokens=4,
                use_fused=False,
            )[0]

        gather.assert_called_once_with(local_input, ANY, sequence_parallel.Group.TP)
        fused.assert_not_called()
        torch.testing.assert_close(actual, torch.mm(gathered_input, weight))

    def test_auto_policy_starts_at_32k_global_tokens(self) -> None:
        self.assertFalse(
            sequence_parallel.should_use_fused_all_gather_matmul(32767)
        )
        self.assertTrue(
            sequence_parallel.should_use_fused_all_gather_matmul(32768)
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
        logical_tokens = 32761
        local_input = torch.empty((4096, 1))
        weight = torch.empty((1, 1))
        physical_output = torch.arange(32768, dtype=torch.float32).reshape(-1, 1)
        group = SimpleNamespace(group_name="tp-test")
        with (
            patch.object(
                sequence_parallel, "get_process_group", return_value=group
            ),
            patch.object(
                sequence_parallel,
                "fused_all_gather_matmul",
                return_value=(None, [physical_output]),
            ),
        ):
            actual = sequence_parallel.all_gather_matmul(
                local_input,
                [weight],
                logical_tokens=logical_tokens,
                use_fused=True,
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
        with (
            patch.object(
                sequence_parallel,
                "get_process_group",
                return_value=SimpleNamespace(),
            ),
            patch("torch.distributed.get_world_size", return_value=2),
            patch.object(
                sequence_parallel,
                "all_gather_into",
                return_value=gathered_input,
            ),
        ):
            actual = sequence_parallel.all_gather_matmul(
                local_input,
                [weight],
                logical_tokens=3,
                use_fused=False,
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
            patch.object(
                sequence_parallel, "reduce_scatter_padded"
            ) as legacy_padding,
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
            patch.object(
                sequence_parallel, "_matmul_with_padded_rows"
            ) as padded_mm,
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
        module.eps = 1e-6
        module.beta = 1.0
        module.linear_beta = None
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
            K3W.MOE_SHARED_GATE: torch.randn(
                hidden_size,
                shared_size,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            K3W.MOE_SHARED_UP: torch.randn(
                hidden_size,
                shared_size,
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
                "_distributed_expert_sum",
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
                torch.full((2,), -1, device="cuda"),
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
        layout = sequence_parallel.token_shard_layout(8, 2, 0)

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
                prefill_sp_layout=layout,
            )

        gather.assert_not_called()
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
                return torch.zeros_like(hidden_states), None

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

        layer = KimiK3DecoderLayer.__new__(KimiK3DecoderLayer)
        nn.Module.__init__(layer)
        layer.layer_idx = 1
        layer.eps = 1e-6
        layer.attn_res_block_size = 2
        layer.is_kda = True
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

        with (
            patch.object(kimi_k3, "_rms_norm", side_effect=lambda x, *_: x),
            patch.object(
                kimi_k3,
                "_attention_residual",
                side_effect=lambda prefix, *_: prefix,
            ),
        ):
            output, _ = layer(
                hidden,
                block_residual,
                torch.tensor([0, 9], dtype=torch.int32),
                mode="prefill",
                sequence_parallel=True,
                prefill_sp_layout=layout,
            )

        self.assertEqual(layout.local_valid_tokens, 1)
        self.assertEqual(layer.mlp.valid_token_count, 1)
        self.assertTrue(layer.mlp.sequence_parallel)
        torch.testing.assert_close(output, hidden, rtol=0, atol=0)

    def test_model_initialize_reserves_max_prefill_workspace(self) -> None:
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
        model.embedding_weight = torch.empty(1, dtype=torch.bfloat16)
        model._fused_ag_gemm_workspace_ready = False
        init_resource = SimpleNamespace(
            kv_cache=None,
            is_decode_role=False,
            max_context_batch_size=1,
        )
        group = SimpleNamespace(group_name="tp-test")
        expected_bytes = 4096 * 16 * 2

        with (
            patch.object(kimi_k3, "get_process_group", return_value=group),
            patch.object(
                kimi_k3,
                "reserve_fused_all_gather_matmul_workspace",
            ) as reserve,
        ):
            self.assertTrue(model.initialize(init_resource))

        reserve.assert_called_once_with(group, expected_bytes)
        self.assertTrue(model._fused_ag_gemm_workspace_ready)

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
        model._fused_ag_gemm_workspace_ready = False
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
