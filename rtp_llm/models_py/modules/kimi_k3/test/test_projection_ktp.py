import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import rtp_llm.models_py.modules.kimi_k3.ktp_step as ktp_step
import rtp_llm.models_py.modules.kimi_k3.projection_ktp as projection_ktp

from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnConfig,
    split_kda_dim1_parallel,
    split_kda_qkvg_fa_beta_parallel,
)
from rtp_llm.models_py.modules.kimi_k3.ktp_step import (
    build_ktp_step_plan,
    normalize_capture_buckets,
    pad_ktp_decode_inputs,
)
from rtp_llm.models_py.modules.kimi_k3.moe import validate_mega_moe_topology
from rtp_llm.models_py.modules.kimi_k3.projection_ktp import (
    pack_ktp_projection_payload,
    reassemble_ktp_projection_payload,
    resolve_projection_local_heads,
    validate_projection_ktp_sp_type,
)
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)


class KtpStepPlanTest(unittest.TestCase):
    def test_draft_parallelism_copy_preserves_cache_cp(self):
        from rtp_llm.model_factory import ModelFactory
        from rtp_llm.ops import ParallelismConfig, RoleType

        for role in (RoleType.PREFILL, RoleType.DECODE):
            for prefill_cp, decode_cp in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(role=role, prefill=prefill_cp, decode=decode_cp):
                    target = ParallelismConfig()
                    target.tp_size = 8
                    target.tp_rank = 3
                    target.ktp_size = 8
                    target.ktp_rank = 5
                    target.role_type = role
                    target.prefill_cp_config.kv_cache_sharded = prefill_cp
                    target.decode_cp_kv_cache_sharded = decode_cp
                    draft = ModelFactory._propose_parallelism_config(target)
                    self.assertEqual((draft.ktp_size, draft.ktp_rank), (1, 0))
                    self.assertEqual((target.ktp_size, target.ktp_rank), (8, 5))
                    self.assertEqual((draft.tp_size, draft.tp_rank), (8, 3))
                    self.assertEqual(draft.role_type, role)
                    self.assertEqual(draft.prefill_cp_config.kv_cache_sharded, prefill_cp)
                    self.assertEqual(draft.decode_cp_kv_cache_sharded, decode_cp)

    def test_mega_moe_accepts_projection_ktp_and_ktp1_draft_layouts(self):
        for ktp_size in (8, 1):
            with self.subTest(ktp_size=ktp_size):
                validate_mega_moe_topology(
                    attention_tp_size=1,
                    dp_size=8,
                    ktp_size=ktp_size,
                    ep_size=8,
                    world_size=8,
                    label="test",
                )

    def test_mega_moe_rejects_ambiguous_tp1_ep_layout(self):
        with self.assertRaisesRegex(RuntimeError, "DP-local tokens"):
            validate_mega_moe_topology(
                attention_tp_size=1,
                dp_size=1,
                ktp_size=1,
                ep_size=8,
                world_size=8,
                label="test",
            )

    def test_projection_ktp_allows_supported_score_model_sessions(self):
        self.assertEqual(validate_projection_ktp_sp_type("eagle3"), "eagle3")
        self.assertEqual(validate_projection_ktp_sp_type(" EAGLE3 "), "eagle3")
        self.assertEqual(validate_projection_ktp_sp_type("mtp"), "mtp")
        self.assertEqual(validate_projection_ktp_sp_type(" MTP "), "mtp")
        self.assertEqual(validate_projection_ktp_sp_type(""), "")
        for unsupported in ("eagle", "deterministic"):
            with self.subTest(unsupported=unsupported):
                with self.assertRaisesRegex(RuntimeError, "must use KTP1"):
                    validate_projection_ktp_sp_type(unsupported)

    def test_padding_preserves_undefined_pybind_tensor_members(self):
        attention = PyAttentionInputs()
        attention.input_lengths = torch.tensor([1], dtype=torch.int32)
        attention.sequence_lengths = torch.tensor([0], dtype=torch.int32)
        attention.cu_seqlens_host = torch.tensor([0, 1], dtype=torch.int32)
        inputs = SimpleNamespace(
            input_ids=torch.tensor([3], dtype=torch.int32),
            attention_inputs=attention,
        )
        plan = build_ktp_step_plan([[0, 1, 0, 1], [2, 1, 0, 1]], [2])

        pad_ktp_decode_inputs(inputs, plan, ktp_rank=0)

        self.assertIsNone(attention.kv_cache_block_id_device)
        self.assertIsNone(attention.kv_cache_block_id_host)
        self.assertIsNone(attention.padding_offset)
        self.assertEqual(attention.input_lengths.tolist(), [1, 1])
        self.assertEqual(inputs.ktp_valid_row_mask.tolist(), [0, 0])

    def test_padding_group_major_pybind_block_table_uses_batch_dimension(self):
        attention = PyAttentionInputs()
        attention.input_lengths = torch.tensor([1], dtype=torch.int32)
        attention.sequence_lengths = torch.tensor([0], dtype=torch.int32)
        attention.cu_seqlens_host = torch.tensor([0, 1], dtype=torch.int32)
        attention.kv_cache_block_id_host = torch.tensor(
            [[[9]], [[19]]], dtype=torch.int32
        )
        inputs = SimpleNamespace(
            input_ids=torch.tensor([3], dtype=torch.int32),
            attention_inputs=attention,
        )
        plan = build_ktp_step_plan([[1, 1, 0, 1], [2, 1, 0, 1]], [2])

        pad_ktp_decode_inputs(inputs, plan, ktp_rank=0)

        self.assertEqual(tuple(attention.kv_cache_block_id_host.shape), (2, 2, 1))
        self.assertEqual(
            attention.kv_cache_block_id_host.tolist(),
            [[[9], [0]], [[19], [0]]],
        )

    def test_padding_uses_reserved_block_zero_and_explicit_valid_mask(self):
        attention = SimpleNamespace(
            input_lengths=torch.tensor([17], dtype=torch.int32),
            input_lengths_host=torch.tensor([17], dtype=torch.int32),
            sequence_lengths=torch.tensor([16], dtype=torch.int32),
            sequence_lengths_host=torch.tensor([16], dtype=torch.int32),
            sequence_lengths_plus_1_d=torch.tensor([17], dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.tensor([[9, 10]], dtype=torch.int32),
            kv_cache_kernel_block_id_host=torch.tensor([[9, 10]], dtype=torch.int32),
            kv_cache_block_id_device=torch.tensor([[9]], dtype=torch.int32),
            kv_cache_block_id_host=torch.tensor([[9]], dtype=torch.int32),
            kv_cache_kernel_block_id_device_by_group=[],
            kv_cache_kernel_block_id_host_by_group=[],
            kv_cache_block_id_host_by_group=[],
            cu_seqlens_host=torch.tensor([0, 1], dtype=torch.int32),
            decode_cu_seqlens_host=torch.tensor([0, 1], dtype=torch.int32),
            cu_kv_seqlens=torch.tensor([0, 1], dtype=torch.int32),
            padding_offset=torch.tensor([0], dtype=torch.int32),
        )
        inputs = SimpleNamespace(
            input_ids=torch.tensor([3], dtype=torch.int32),
            attention_inputs=attention,
        )
        plan = build_ktp_step_plan([[1, 1, 0, 1], [3, 1, 0, 1]], [4])
        pad_ktp_decode_inputs(inputs, plan, ktp_rank=0)
        self.assertEqual(inputs.ktp_valid_row_mask.tolist(), [1, 0, 0, 0])
        self.assertEqual(
            attention.kv_cache_kernel_block_id_device.tolist(),
            [[9, 10], [0, 0], [0, 0], [0, 0]],
        )
        self.assertEqual(attention.sequence_lengths_plus_1_d.tolist(), [17, 1, 1, 1])

    def test_arbitrary_capture_buckets_are_sorted_and_deduplicated(self):
        self.assertEqual(normalize_capture_buckets([8, 1, 4, 2, 4]), (1, 2, 4, 8))

    def test_step_coordination_uses_cpu_control_group(self):
        captured = {}

        def fake_all_gather(tensor, *, group):
            captured["device"] = tensor.device.type
            captured["group"] = group
            return torch.tensor(
                [[1, 1, 0, 1], [3, 1, 0, 1]], dtype=torch.int32
            )

        with mock.patch.object(ktp_step, "all_gather", side_effect=fake_all_gather):
            plan = ktp_step.coordinate_ktp_step(
                local_real_batch=1,
                graph_eligible=True,
                forward_mode=ktp_step.KtpForwardMode.DECODE,
                capture_buckets=[1, 2, 4, 8],
                tokens_per_batch=1,
                device=torch.device("cuda"),
                ktp_size=2,
            )

        self.assertEqual(captured["device"], "cpu")
        self.assertEqual(captured["group"], ktp_step.Group.KTP_CONTROL)
        self.assertEqual(plan.valid_batch_sizes, (1, 3))
        self.assertEqual(plan.common_graph_bucket, 4)

    def test_selects_first_common_bucket(self):
        plan = build_ktp_step_plan(
            [[1, 1, 0, 1], [7, 1, 0, 1], [0, 1, 0, 1]], [1, 2, 4, 8, 16]
        )
        self.assertEqual(plan.valid_batch_sizes, (1, 7, 0))
        self.assertEqual(plan.global_max_batch, 7)
        self.assertEqual(plan.common_graph_bucket, 8)
        self.assertEqual(plan.common_physical_batch, 8)
        self.assertTrue(plan.use_cuda_graph)

    def test_one_ineligible_rank_forces_common_eager(self):
        plan = build_ktp_step_plan([[2, 1, 0, 1], [5, 0, 0, 1]], [8])
        self.assertFalse(plan.use_cuda_graph)
        self.assertEqual(plan.common_graph_bucket, 0)
        self.assertEqual(plan.common_physical_batch, 5)

    def test_missing_bucket_forces_common_eager(self):
        plan = build_ktp_step_plan([[9, 1, 0, 1], [3, 1, 0, 1]], [1, 4, 8])
        self.assertFalse(plan.use_cuda_graph)
        self.assertEqual(plan.common_physical_batch, 9)

    def test_all_idle_skips_step(self):
        plan = build_ktp_step_plan([[0, 1, 0, 1], [0, 1, 0, 1]], [1, 2])
        self.assertTrue(plan.all_idle)
        self.assertEqual(plan.common_physical_batch, 0)

    def test_forward_mode_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, "forward mode"):
            build_ktp_step_plan([[1, 1, 0, 1], [1, 1, 1, 1]], [1])

    def test_token_width_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, "token width"):
            build_ktp_step_plan([[1, 1, 2, 4], [1, 1, 2, 3]], [1])

    def test_target_verify_padding_separates_request_and_token_shapes(self):
        attention = SimpleNamespace(
            input_lengths=torch.tensor([4], dtype=torch.int32),
            input_lengths_host=torch.tensor([4], dtype=torch.int32),
            prefix_lengths=torch.tensor([7], dtype=torch.int32),
            prefix_lengths_host=torch.tensor([7], dtype=torch.int32),
            sequence_lengths=torch.tensor([7], dtype=torch.int32),
            sequence_lengths_host=torch.tensor([7], dtype=torch.int32),
            sequence_lengths_plus_1_d=torch.tensor([8], dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.tensor([[9]], dtype=torch.int32),
            kv_cache_kernel_block_id_host=torch.tensor([[9]], dtype=torch.int32),
            kv_cache_block_id_device=None,
            kv_cache_block_id_host=None,
            kv_cache_kernel_block_id_device_by_group=[],
            kv_cache_kernel_block_id_host_by_group=[],
            kv_cache_block_id_host_by_group=[],
            cu_seqlens_host=torch.tensor([0, 4], dtype=torch.int32),
            cu_kv_seqlens=torch.tensor([0, 11], dtype=torch.int32),
            padding_offset=torch.arange(4, dtype=torch.int32),
        )
        inputs = SimpleNamespace(
            input_ids=torch.arange(4, dtype=torch.int32),
            input_hiddens=torch.ones(4, 3),
            combo_position_ids=torch.arange(4, dtype=torch.int32),
            attention_inputs=attention,
        )
        plan = build_ktp_step_plan([[1, 1, 2, 4], [2, 1, 2, 4]], [2])

        pad_ktp_decode_inputs(inputs, plan, ktp_rank=0)

        self.assertEqual(inputs.input_ids.numel(), 8)
        self.assertEqual(tuple(inputs.input_hiddens.shape), (8, 3))
        self.assertEqual(attention.input_lengths.tolist(), [4, 4])
        self.assertEqual(attention.cu_seqlens.tolist(), [0, 4, 8])
        self.assertEqual(attention.cu_kv_seqlens.tolist(), [0, 11, 15])
        self.assertEqual(attention.total_tokens, 8)
        self.assertEqual(
            inputs.ktp_valid_row_mask.tolist(),
            [1, 1, 1, 1, 0, 0, 0, 0],
        )
        self.assertEqual(inputs.ktp_local_real_batch, 4)
        self.assertEqual(inputs.ktp_common_physical_batch, 2)


class KtpProjectionLayoutTest(unittest.TestCase):
    def test_fp8_projection_input_gathers_values_and_repacks_scales(self):
        local_values = torch.arange(256, dtype=torch.uint8).reshape(2, 128).view(
            torch.float8_e4m3fn
        )
        local_scales = torch.tensor([[101, 102, 0, 0]], dtype=torch.int32)
        local = QuantizedActivation(local_values, local_scales)

        rank1_values = (torch.arange(256, dtype=torch.uint8) + 16).reshape(2, 128)
        gathered_values = torch.cat((local_values.view(torch.uint8), rank1_values))
        gathered_scales = torch.tensor(
            [[101, 102, 0, 0], [201, 202, 0, 0]], dtype=torch.int32
        )

        with mock.patch.object(
            projection_ktp,
            "all_gather",
            side_effect=(gathered_values, gathered_scales),
        ) as gather:
            result = projection_ktp._all_gather_projection_input(local, ktp_size=2)

        self.assertIsInstance(result, QuantizedActivation)
        torch.testing.assert_close(result.values.view(torch.uint8), gathered_values)
        self.assertEqual(result.scale_wire.tolist(), [[101, 102, 201, 202]])
        self.assertEqual(gather.call_count, 2)

    def test_fp8_projection_input_removes_each_ranks_local_scale_padding(self):
        local_values = torch.arange(128, dtype=torch.uint8).reshape(1, 128).view(
            torch.float8_e4m3fn
        )
        local = QuantizedActivation(
            local_values, torch.tensor([[11, 0, 0, 0]], dtype=torch.int32)
        )
        gathered_values = torch.cat(
            tuple((local_values.view(torch.uint8) + 8 * rank) for rank in range(8))
        )
        gathered_scales = torch.tensor(
            [[11 * (rank + 1), 0, 0, 0] for rank in range(8)],
            dtype=torch.int32,
        )

        with mock.patch.object(
            projection_ktp,
            "all_gather",
            side_effect=(gathered_values, gathered_scales),
        ):
            result = projection_ktp._all_gather_projection_input(local, ktp_size=8)

        self.assertEqual(
            result.scale_wire.tolist(),
            [[11, 22, 33, 44, 55, 66, 77, 88]],
        )

    def test_projection_payload_accepts_quantized_linear_callables(self):
        torch.manual_seed(5)
        hidden = torch.randn(4, 3)
        fused_weight = torch.randn(3, 7)
        forget_weight = torch.randn(1, 1)

        class Linear:
            def __init__(self, weight):
                self.weight = weight
                self.calls = 0

            def __call__(self, inputs):
                self.calls += 1
                self.assert_contiguous = inputs.is_contiguous()
                return inputs @ self.weight

        fused_linear = Linear(fused_weight)
        forget_linear = Linear(forget_weight)
        expected = pack_ktp_projection_payload(
            hidden,
            fused_weight,
            forget_weight,
            total_heads=2,
            head_dim=1,
            forget_latent_size=1,
            ktp_size=2,
            ktp_rank=1,
        )
        actual = pack_ktp_projection_payload(
            hidden,
            fused_linear,
            forget_linear,
            total_heads=2,
            head_dim=1,
            forget_latent_size=1,
            ktp_size=2,
            ktp_rank=1,
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(fused_linear.calls, 1)
        self.assertEqual(forget_linear.calls, 1)
        self.assertTrue(fused_linear.assert_contiguous)
        self.assertTrue(forget_linear.assert_contiguous)

    def test_projection_local_heads_preserve_attention_tp_when_ktp_is_off(self):
        cases = (
            (8, 1, 12),
            (1, 1, 96),
            (1, 8, 12),
            (1, 16, 6),
        )
        for attention_tp_size, ktp_size, expected_heads in cases:
            with self.subTest(
                attention_tp_size=attention_tp_size, ktp_size=ktp_size
            ):
                self.assertEqual(
                    resolve_projection_local_heads(
                        total_heads=96,
                        attention_tp_size=attention_tp_size,
                        ktp_size=ktp_size,
                    ),
                    expected_heads,
                )

    def test_all_gather_projection_all_to_all_matches_full_head_reference(self):
        torch.manual_seed(7)
        ktp_size = 2
        physical_batch = 3
        hidden_size = 5
        total_heads = 4
        head_dim = 2
        forget_rank = 3
        gathered_hidden = torch.randn(ktp_size * physical_batch, hidden_size)
        full_qkvg = torch.randn(hidden_size, 4 * total_heads * head_dim)
        f_a = torch.randn(hidden_size, forget_rank)
        beta = torch.randn(hidden_size, total_heads)
        full_fused = torch.cat((full_qkvg, f_a, beta), dim=1)
        full_f_b = torch.randn(forget_rank, total_heads * head_dim)

        sends = []
        local_projection = total_heads // ktp_size * head_dim
        for rank in range(ktp_size):
            q, k, v, g = torch.split(
                full_qkvg, [total_heads * head_dim] * 4, dim=1
            )
            begin = rank * local_projection
            local_fused = torch.cat(
                tuple(section.narrow(1, begin, local_projection) for section in (q, k, v, g))
                + (f_a, beta),
                dim=1,
            )
            sends.append(
                pack_ktp_projection_payload(
                    gathered_hidden,
                    local_fused,
                    full_f_b.narrow(1, begin, local_projection),
                    total_heads=total_heads,
                    head_dim=head_dim,
                    forget_latent_size=forget_rank,
                    ktp_size=ktp_size,
                    ktp_rank=rank,
                ).reshape(ktp_size, physical_batch, -1)
            )

        owner = 1
        received = torch.cat([send[owner] for send in sends], dim=0)
        actual = reassemble_ktp_projection_payload(
            received,
            ktp_size=ktp_size,
            physical_batch=physical_batch,
            local_projection_size=local_projection,
            local_heads=total_heads // ktp_size,
        )
        owner_hidden = gathered_hidden.narrow(0, owner * physical_batch, physical_batch)
        full = owner_hidden @ full_fused
        q, k, v, output_gate, forget_latent, raw_beta = torch.split(
            full,
            [
                total_heads * head_dim,
                total_heads * head_dim,
                total_heads * head_dim,
                total_heads * head_dim,
                forget_rank,
                total_heads,
            ],
            dim=1,
        )
        raw_gate = forget_latent @ full_f_b
        torch.testing.assert_close(actual.q, q)
        torch.testing.assert_close(actual.k, k)
        torch.testing.assert_close(actual.v, v)
        torch.testing.assert_close(actual.output_gate, output_gate)
        torch.testing.assert_close(actual.raw_gate, raw_gate)
        torch.testing.assert_close(actual.raw_beta, raw_beta)

    def test_projection_weight_layout_for_ktp8_and_ktp16(self):
        config = LinearAttnConfig.__new__(LinearAttnConfig)
        config.linear_num_key_heads = 96
        config.linear_num_value_heads = 96
        config.linear_key_head_dim = 128
        config.linear_value_head_dim = 128
        hidden = 4
        forget_rank = 3
        global_width = 4 * 96 * 128 + forget_rank + 96
        fused = torch.arange(hidden * global_width).reshape(hidden, global_width)
        f_b = torch.arange(forget_rank * 96 * 128).reshape(forget_rank, 96 * 128)
        for ktp_size, expected_heads in ((8, 12), (16, 6)):
            local = split_kda_qkvg_fa_beta_parallel(
                fused,
                parallel_size=ktp_size,
                parallel_rank=ktp_size - 1,
                linear_config=config,
            )
            self.assertEqual(
                tuple(local.shape),
                (hidden, 4 * expected_heads * 128 + forget_rank + 96),
            )
            local_f_b = split_kda_dim1_parallel(
                f_b, parallel_size=ktp_size, parallel_rank=ktp_size - 1
            )
            self.assertEqual(tuple(local_f_b.shape), (forget_rank, expected_heads * 128))

    def test_source_head_shards_reassemble_for_each_owner(self):
        ktp_size = 2
        batch = 3
        local_projection = 2
        local_heads = 1
        chunks = []
        for source in range(ktp_size):
            for owner in range(batch):
                base = 100 * source + 10 * owner
                chunks.append(
                    torch.tensor(
                        [
                            base + 1,
                            base + 2,
                            base + 3,
                            base + 4,
                            base + 5,
                            base + 6,
                            base + 7,
                            base + 8,
                            base + 9,
                            base + 10,
                            base + 11,
                        ],
                        dtype=torch.float32,
                    )
                )
        result = reassemble_ktp_projection_payload(
            torch.stack(chunks),
            ktp_size=ktp_size,
            physical_batch=batch,
            local_projection_size=local_projection,
            local_heads=local_heads,
        )
        self.assertEqual(tuple(result.q.shape), (batch, 4))
        self.assertEqual(result.q[1].tolist(), [11.0, 12.0, 111.0, 112.0])
        self.assertEqual(result.raw_beta[2].tolist(), [31.0, 131.0])


if __name__ == "__main__":
    unittest.main()
