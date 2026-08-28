import unittest
from types import SimpleNamespace

import torch

from rtp_llm.ops.compute_ops import PyAttentionInputs

from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnConfig,
    split_kda_dim1_parallel,
    split_kda_qkvg_fa_beta_parallel,
)
from rtp_llm.models_py.modules.kimi_k3.projection_ktp import (
    build_ktp_step_plan,
    normalize_capture_buckets,
    pack_ktp_projection_payload,
    pad_ktp_decode_inputs,
    parse_decode_capture_config,
    reassemble_ktp_projection_payload,
    resolve_projection_local_heads,
)


class KtpStepPlanTest(unittest.TestCase):
    def test_padding_preserves_undefined_pybind_tensor_members(self):
        attention = PyAttentionInputs()
        attention.input_lengths = torch.tensor([1], dtype=torch.int32)
        attention.sequence_lengths = torch.tensor([0], dtype=torch.int32)
        attention.cu_seqlens_host = torch.tensor([0, 1], dtype=torch.int32)
        inputs = SimpleNamespace(
            input_ids=torch.tensor([3], dtype=torch.int32),
            attention_inputs=attention,
        )
        plan = build_ktp_step_plan([[0, 1, 0], [2, 1, 0]], [2])

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
        plan = build_ktp_step_plan([[1, 1, 0], [2, 1, 0]], [2])

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
        plan = build_ktp_step_plan([[1, 1, 0], [3, 1, 0]], [4])
        pad_ktp_decode_inputs(inputs, plan, ktp_rank=0)
        self.assertEqual(inputs.ktp_valid_row_mask.tolist(), [1, 0, 0, 0])
        self.assertEqual(
            attention.kv_cache_kernel_block_id_device.tolist(),
            [[9, 10], [0, 0], [0, 0], [0, 0]],
        )
        self.assertEqual(attention.sequence_lengths_plus_1_d.tolist(), [17, 1, 1, 1])

    def test_arbitrary_capture_buckets_are_sorted_and_deduplicated(self):
        self.assertEqual(parse_decode_capture_config("8, 1,4,2,4"), (1, 2, 4, 8))
        self.assertEqual(normalize_capture_buckets([16, 3, 3]), (3, 16))

    def test_selects_first_common_bucket(self):
        plan = build_ktp_step_plan(
            [[1, 1, 0], [7, 1, 0], [0, 1, 0]], [1, 2, 4, 8, 16]
        )
        self.assertEqual(plan.valid_batch_sizes, (1, 7, 0))
        self.assertEqual(plan.global_max_batch, 7)
        self.assertEqual(plan.common_graph_bucket, 8)
        self.assertEqual(plan.common_physical_batch, 8)
        self.assertTrue(plan.use_cuda_graph)

    def test_one_ineligible_rank_forces_common_eager(self):
        plan = build_ktp_step_plan([[2, 1, 0], [5, 0, 0]], [8])
        self.assertFalse(plan.use_cuda_graph)
        self.assertEqual(plan.common_graph_bucket, 0)
        self.assertEqual(plan.common_physical_batch, 5)

    def test_missing_bucket_forces_common_eager(self):
        plan = build_ktp_step_plan([[9, 1, 0], [3, 1, 0]], [1, 4, 8])
        self.assertFalse(plan.use_cuda_graph)
        self.assertEqual(plan.common_physical_batch, 9)

    def test_all_idle_skips_step(self):
        plan = build_ktp_step_plan([[0, 1, 0], [0, 1, 0]], [1, 2])
        self.assertTrue(plan.all_idle)
        self.assertEqual(plan.common_physical_batch, 0)

    def test_forward_mode_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, "forward mode"):
            build_ktp_step_plan([[1, 1, 0], [1, 1, 1]], [1])


class KtpProjectionLayoutTest(unittest.TestCase):
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
