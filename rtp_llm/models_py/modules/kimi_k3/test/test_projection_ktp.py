import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch

import rtp_llm.models_py.modules.kimi_k3.ktp_step as ktp_step
import rtp_llm.models_py.modules.kimi_k3.projection_ktp as projection_ktp
from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnConfig,
    split_kda_dim1_parallel,
    split_kda_qkvg_fa_beta_parallel,
)
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
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
from rtp_llm.ops.compute_ops import LinearReplayInputs, PyAttentionInputs


class KtpProjectionWorkspaceTest(unittest.TestCase):
    def make_workspace(self):
        return projection_ktp.KtpProjectionWorkspace(
            (1, 2), ktp_size=8, total_heads=16, head_dim=4, device="cpu"
        )

    def test_bucket_addresses_are_stable_and_missing_capture_bucket_fails(self):
        workspace = self.make_workspace()
        with mock.patch.object(workspace, "_is_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "not prepared"):
                workspace.get(1)
        first = workspace.get(1)
        self.assertIs(workspace.get(1), first)
        self.assertNotEqual(first[0].data_ptr(), first[1].data_ptr())
        self.assertNotEqual(first[0].data_ptr(), workspace.get(2)[0].data_ptr())
        with mock.patch.object(workspace, "_is_capturing", return_value=False):
            self.assertIsNone(workspace.get(3))
        self.assertEqual(set(workspace.buffers), {1, 2})
        with mock.patch.object(workspace, "_is_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "not prepared"):
                workspace.get(3)

    def test_workspace_transport_matches_temporary_transport(self):
        workspace = self.make_workspace()
        torch.manual_seed(91)
        hidden = torch.randn(2, 8, dtype=torch.bfloat16)
        fused = torch.randn(8, 4 * 8 + 4 + 16, dtype=torch.bfloat16)
        forget = torch.randn(4, 8, dtype=torch.bfloat16)
        kwargs = dict(
            total_heads=16, head_dim=4, forget_latent_size=4, ktp_size=8, ktp_rank=3
        )
        seen = []

        def exchange(send, group, output=None):
            if output is None:
                return send.clone()
            seen.append((send.data_ptr(), output.data_ptr()))
            output.copy_(send)
            return output

        with mock.patch.object(
            projection_ktp,
            "_all_gather_projection_input",
            side_effect=lambda x, **kw: x.repeat(8, 1),
        ), mock.patch.object(projection_ktp, "all_to_all_single", side_effect=exchange):
            expected = projection_ktp.project_kda_inputs_ktp(
                hidden, fused, forget, **kwargs
            )
            for _ in range(2):
                actual = projection_ktp.project_kda_inputs_ktp(
                    hidden, fused, forget, workspace=workspace, **kwargs
                )
                for name, value in vars(expected).items():
                    torch.testing.assert_close(
                        getattr(actual, name), value, rtol=0, atol=0
                    )
        send, receive = workspace.get(2)
        self.assertEqual(seen, [(send.data_ptr(), receive.data_ptr())] * 2)

    def test_reassembled_heads_survive_receive_buffer_reuse(self):
        workspace = self.make_workspace()
        for batch in (1, 2):
            received = workspace.get(batch)[1]
            received.copy_(torch.arange(received.numel()).reshape(received.shape))
            result = reassemble_ktp_projection_payload(
                received,
                ktp_size=8,
                physical_batch=batch,
                local_projection_size=8,
                local_heads=2,
            )
            before = {name: value.clone() for name, value in vars(result).items()}
            received.zero_()
            for name, value in vars(result).items():
                self.assertNotEqual(
                    value.untyped_storage().data_ptr(),
                    received.untyped_storage().data_ptr(),
                )
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)


class KtpStepPlanTest(unittest.TestCase):
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

    def test_mega_moe_accepts_mixed_tp_dp(self):
        for tp, dp in ((4, 2), (4, 4), (8, 1), (16, 1)):
            with self.subTest(tp=tp, dp=dp):
                validate_mega_moe_topology(
                    attention_tp_size=tp,
                    dp_size=dp,
                    ktp_size=1,
                    ep_size=tp * dp,
                    world_size=tp * dp,
                    label="test",
                )
        for tp, dp, ep, world in ((8, 2, 8, 8), (4, 2, 4, 8), (0, 2, 8, 8)):
            with self.assertRaises(RuntimeError):
                validate_mega_moe_topology(
                    attention_tp_size=tp,
                    dp_size=dp,
                    ktp_size=1,
                    ep_size=ep,
                    world_size=world,
                    label="test",
                )

    def test_mega_moe_rejects_ambiguous_tp1_ep_layout(self):
        with self.assertRaisesRegex(RuntimeError, "TP\\*DP=EP=world"):
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
            return torch.tensor([[1, 1, 0, 1], [3, 1, 0, 1]], dtype=torch.int32)

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
        self.assertEqual(attention.logical_request_count, 1)
        self.assertEqual(attention.physical_request_count, 2)
        self.assertEqual(attention.logical_token_count, 4)
        self.assertEqual(attention.physical_token_count, 8)
        self.assertEqual(
            inputs.ktp_valid_row_mask.tolist(),
            [1, 1, 1, 1, 0, 0, 0, 0],
        )
        self.assertEqual(inputs.ktp_local_real_batch, 4)
        self.assertEqual(inputs.ktp_common_physical_batch, 2)

    def test_target_verify_replay_padding_preserves_logical_inputs(self):
        for local_batch, remote_batch, graph, bucket, physical in (
            (1, 2, True, 4, 4),
            (1, 2, False, 4, 2),
            (0, 2, True, 2, 2),
            (1, 1, True, 4, 4),
            (1, 1, True, 1, 1),
        ):
            with self.subTest(local_batch=local_batch, graph=graph, physical=physical):
                source = LinearReplayInputs()
                values = {
                    "slot_ids": ([7 if local_batch else -1], torch.int32),
                    "slot_generations": ([1 << 40], torch.int64),
                    "active_block_ids": ([[11], [21]], torch.int32),
                    "prev_accept_lengths": ([3], torch.int32),
                    "history_valid_lengths": ([4], torch.int32),
                    "history_epochs": ([(1 << 40) + 1], torch.int64),
                    "verify_epochs": ([(1 << 40) + 2], torch.int64),
                    "init_kinds": ([0], torch.int32),
                    "state_read_block_ids": ([[10], [20]], torch.int32),
                    "anchor_processed_lengths": ([127], torch.int32),
                }
                original_tensors = {}
                for name, (data, dtype) in values.items():
                    original_tensors[name] = torch.tensor(data, dtype=dtype)
                    setattr(source, name, original_tensors[name])
                attention = PyAttentionInputs()
                attention.input_lengths = torch.tensor([4], dtype=torch.int32)
                attention.sequence_lengths = torch.tensor([7], dtype=torch.int32)
                attention.cu_seqlens_host = torch.tensor([0, 4], dtype=torch.int32)
                attention.linear_replay = source
                inputs = SimpleNamespace(
                    input_ids=torch.arange(4, dtype=torch.int32),
                    attention_inputs=attention,
                )
                plan = build_ktp_step_plan(
                    [[local_batch, int(graph), 2, 4], [remote_batch, int(graph), 2, 4]],
                    [bucket],
                )

                pad_ktp_decode_inputs(inputs, plan, ktp_rank=0)

                replay = attention.linear_replay
                self.assertEqual(replay.slot_ids.numel(), physical)
                self.assertEqual(inputs.input_ids.numel() // replay.slot_ids.numel(), 4)
                for name, original in original_tensors.items():
                    with self.subTest(field=name):
                        padded = getattr(replay, name)
                        dim = 1 if original.dim() == 2 else 0
                        expected_shape = list(original.shape)
                        expected_shape[dim] = physical
                        self.assertEqual(list(padded.shape), expected_shape)
                        self.assertEqual(padded.dtype, original.dtype)
                        self.assertEqual(padded.device, original.device)
                        torch.testing.assert_close(padded.narrow(dim, 0, 1), original)
                        fill = (
                            -1
                            if name
                            in (
                                "slot_ids",
                                "active_block_ids",
                                "state_read_block_ids",
                            )
                            else 0
                        )
                        tail = padded.narrow(dim, 1, physical - 1)
                        torch.testing.assert_close(tail, torch.full_like(tail, fill))
                        retained = getattr(source, name)
                        self.assertEqual(retained.data_ptr(), original.data_ptr())
                        torch.testing.assert_close(
                            retained,
                            torch.tensor(values[name][0], dtype=values[name][1]),
                        )
                        if physical == 1:
                            self.assertEqual(padded.data_ptr(), original.data_ptr())


class KtpProjectionLayoutTest(unittest.TestCase):
    def test_fp8_projection_input_gathers_values_and_repacks_scales(self):
        local_values = (
            torch.arange(256, dtype=torch.uint8)
            .reshape(2, 128)
            .view(torch.float8_e4m3fn)
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
        local_values = (
            torch.arange(128, dtype=torch.uint8)
            .reshape(1, 128)
            .view(torch.float8_e4m3fn)
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
            with self.subTest(attention_tp_size=attention_tp_size, ktp_size=ktp_size):
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
            q, k, v, g = torch.split(full_qkvg, [total_heads * head_dim] * 4, dim=1)
            begin = rank * local_projection
            local_fused = torch.cat(
                tuple(
                    section.narrow(1, begin, local_projection)
                    for section in (q, k, v, g)
                )
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
            self.assertEqual(
                tuple(local_f_b.shape), (forget_rank, expected_heads * 128)
            )

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


class MixedTpDpInputPreparationTest(unittest.TestCase):
    def test_target_verify_keeps_tp_rows_and_live_mask_in_each_dp(self):
        from rtp_llm.models_py.modules.kimi_k3.input_preparation import (
            KimiK3ExecutionSpec,
            prepare_round,
        )
        from rtp_llm.models_py.modules.kimi_k3.parallel_mode import KimiK3ParallelMode

        for dp in range(2):
            for rank in range(4):
                for logical in (0, 1, 3, 4):
                    with self.subTest(dp=dp, rank=rank, logical=logical):
                        model = SimpleNamespace(
                            kv_cache=object(),
                            layers=[],
                            _layer_group_ids=(),
                            parallel_mode=KimiK3ParallelMode.TP_SP,
                            parallelism_config=SimpleNamespace(
                                tp_size=4,
                                tp_rank=rank,
                                dp_size=2,
                                dp_rank=dp,
                                ktp_size=1,
                                ep_size=8,
                                world_size=8,
                            ),
                            execution_spec=KimiK3ExecutionSpec(
                                4, rank, 0, "mtp", (), frozenset(), ()
                            ),
                            num_attn_res_blocks=1,
                            config=SimpleNamespace(hidden_size=4),
                            embedding_weight=torch.empty(1, 4),
                        )
                        mask = torch.zeros(16, dtype=torch.int32)
                        mask[: logical * 4] = 1
                        inputs = SimpleNamespace(
                            input_ids=torch.arange(16) + dp * 1000,
                            multimodal_inputs=None,
                            ktp_valid_row_mask=mask,
                            attention_inputs=SimpleNamespace(
                                is_prefill=False,
                                is_target_verify=True,
                                is_cuda_graph=True,
                                input_lengths=torch.ones(4, dtype=torch.int32),
                                logical_request_count=logical,
                                physical_request_count=4,
                                logical_token_count=logical * 4,
                                physical_token_count=16,
                                decode_cu_seqlens_d=torch.arange(17, dtype=torch.int32),
                            ),
                        )
                        with mock.patch(
                            "rtp_llm.models_py.modules.kimi_k3.input_preparation.create_write_cache_store_impl",
                            return_value=None,
                        ):
                            prepared = prepare_round(model, inputs)
                        layout = prepared.attn_meta.sp_layout
                        self.assertEqual(layout.tokens.local_tokens, 4)
                        self.assertEqual(layout.tokens.local_start, rank * 4)
                        self.assertEqual(
                            layout.tokens.local_valid_tokens, 4 if rank < logical else 0
                        )
                        torch.testing.assert_close(
                            prepared.embedding_ids, inputs.input_ids
                        )
                        local_mask = prepared.attn_meta.valid_token_mask
                        self.assertEqual(
                            local_mask.data_ptr(), mask[rank * 4 :].data_ptr()
                        )
                        # CUDA Graph metadata must retain the live mask view.
                        mask.fill_(1)
                        torch.testing.assert_close(
                            local_mask, torch.ones(4, dtype=torch.int32)
                        )


class PrefillAttentionWorkspaceLifetimeTest(unittest.TestCase):
    def test_mla_scratch_is_released_before_projection_output_stays_alive(self):
        from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3DecoderLayer
        from rtp_llm.models_py.modules.kimi_k3.parallel_mode import KimiK3ParallelMode

        for prefill, kda in ((True, False), (False, False), (True, True)):
            with self.subTest(prefill=prefill, kda=kda):
                order = []
                fmha = SimpleNamespace(scratch=torch.arange(8.0).reshape(2, 4).clone())
                scratch_ref = weakref.ref(fmha.scratch)
                fmha.kv_staging = torch.ones(32)
                staging_ref = weakref.ref(fmha.kv_staging)

                def release():
                    order.append("release")
                    fmha.scratch = None
                    fmha.kv_staging = None

                fmha.release_forward_workspace = release

                class Attention:
                    def __call__(self, *args, **kwargs):
                        order.append("attention")
                        return fmha.scratch.view(2, 4)

                    def output_projection_weight(self):
                        return None

                def project(value, weight):
                    self.assertIsNotNone(
                        scratch_ref(), "attention output storage freed early"
                    )
                    if prefill and not kda:
                        self.assertIsNone(
                            staging_ref(), "KV scratch still overlaps output projection"
                        )
                    order.append("projection")
                    return value * 2

                def mlp(value, **kwargs):
                    order.append("mlp")
                    if prefill and not kda:
                        self.assertIsNone(
                            scratch_ref(), "MLA workspace still overlaps MLP"
                        )
                    else:
                        self.assertIsNotNone(scratch_ref())
                    return torch.zeros_like(value)

                layer = SimpleNamespace(
                    _previous_blocks=0,
                    _writes_block=False,
                    parallel_mode=KimiK3ParallelMode.PROJECTION_KTP,
                    is_kda=kda,
                    attention_norm=lambda x: x,
                    self_attn=Attention(),
                    _project_parallel_output=project,
                    mlp_residual=lambda x, *args, **kwargs: x,
                    mlp_norm=SimpleNamespace(weight=None, variance_epsilon=1e-5),
                    mlp=mlp,
                )
                meta = SimpleNamespace(
                    cu_seqlens=None,
                    mode=None,
                    sp_layout=SimpleNamespace(
                        tokens=SimpleNamespace(local_valid_tokens=2, local_tokens=2)
                    ),
                    kda_prefill_metadata=None,
                    kda_current_state_registry=None,
                    valid_token_mask=None,
                )
                result = KimiK3DecoderLayer.forward(
                    layer,
                    torch.ones(2, 4),
                    torch.empty(2, 0, 4),
                    attn_meta=meta,
                    attention_inputs=SimpleNamespace(is_prefill=prefill),
                    fmha_impl=fmha,
                )
                expected_order = ["attention"]
                if prefill and not kda:
                    expected_order.append("release")
                expected_order.append("projection")
                self.assertEqual(order, expected_order + ["mlp"])
                torch.testing.assert_close(
                    result.hidden_states, 1 + 2 * torch.arange(8.0).reshape(2, 4)
                )


if __name__ == "__main__":
    unittest.main()
