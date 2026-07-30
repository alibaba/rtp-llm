import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNetDecode,
    Qwen3NextMetadata,
)
from rtp_llm.models_py.model_desc.qwen3_next_mtp import Qwen3NextMTPModel
from rtp_llm.ops import HybridAttentionType


class Qwen3NextGdnDecodeRoutingTest(unittest.TestCase):
    def _make_qwen3_next_decode_fla_case(self):
        batch, key_heads, value_heads, dim = 2, 2, 8, 128
        decode = object.__new__(Qwen3NextGatedDeltaNetDecode)
        nn.Module.__init__(decode)
        decode.local_num_k_heads = key_heads
        decode.local_num_v_heads = value_heads
        decode.head_k_dim = dim
        decode.enable_cuda_graph = True
        decode.alog = torch.randn(value_heads, dtype=torch.float32)
        decode.dt_bias = torch.randn(value_heads, dtype=torch.bfloat16)

        mixed_qkv = torch.randn(
            batch,
            (key_heads * 2 + value_heads) * dim,
            dtype=torch.bfloat16,
        )
        a = torch.randn(batch, value_heads, dtype=torch.bfloat16)
        b = torch.randn_like(a)
        state = torch.randn(batch * 3 + 1, value_heads, dim, dim)
        block_map = torch.arange(1, 7, dtype=torch.int32).reshape(batch, 3)
        lengths = torch.tensor([1002, 1025], dtype=torch.int32)
        attention_inputs = SimpleNamespace(
            is_cuda_graph=True,
            kv_cache_kernel_block_id_device=block_map,
            sequence_lengths=torch.tensor([1001, 1024], dtype=torch.int32),
            sequence_lengths_plus_1_device=lengths,
        )
        output = torch.empty(batch, 1, value_heads, dim, dtype=torch.bfloat16)
        return (
            decode,
            mixed_qkv,
            b,
            a,
            state,
            attention_inputs,
            output,
        )

    def test_qwen3_next_cuda_graph_uses_narrow_block_map_view(self):
        block_map = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        attention_inputs = SimpleNamespace(
            is_cuda_graph=True,
            kv_cache_kernel_block_id_device=block_map,
        )
        decode = object.__new__(Qwen3NextGatedDeltaNetDecode)
        decode.enable_cuda_graph = True

        narrowed = decode._get_fla_block_map(attention_inputs)

        self.assertEqual(narrowed.shape, (3, 1))
        self.assertEqual(narrowed.stride(0), block_map.stride(0))
        self.assertEqual(narrowed[:, 0].tolist(), [0, 4, 8])

    def test_qwen3_next_non_graph_keeps_full_block_map(self):
        block_map = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        attention_inputs = SimpleNamespace(
            is_cuda_graph=False,
            kv_cache_kernel_block_id_device=block_map,
        )
        decode = object.__new__(Qwen3NextGatedDeltaNetDecode)
        decode.enable_cuda_graph = False

        self.assertIs(decode._get_fla_block_map(attention_inputs), block_map)

    def test_qwen3_next_decode_invalid_row_flags_are_checked_out_of_band(self):
        decode = object.__new__(Qwen3NextGatedDeltaNetDecode)
        decode._aiter_flydsl_gdn_decode_invalid_row_flags = torch.tensor([0, 0])
        decode.check_aiter_flydsl_gdn_decode_state_indices()

        decode._aiter_flydsl_gdn_decode_invalid_row_flags = torch.tensor([0, 1])
        with self.assertRaisesRegex(RuntimeError, r"\[1\]"):
            decode.check_aiter_flydsl_gdn_decode_state_indices()

    def test_qwen3_next_decode_routes_aiter_and_triton_with_correct_block_map(self):
        (
            decode,
            mixed_qkv,
            b,
            a,
            state,
            attention_inputs,
            output,
        ) = self._make_qwen3_next_decode_fla_case()
        read_indices = torch.tensor([1, 4], dtype=torch.int32)
        write_indices = torch.tensor([1, 5], dtype=torch.int32)
        invalid_row_flags = torch.zeros(2, dtype=torch.int32)
        split = mixed_qkv.reshape(2, 1, 12, 128)
        expected_q, expected_k, expected_v = torch.split(split, [2, 2, 8], dim=2)

        for name, is_target_verify, gate_supported, expect_aiter in (
            ("target-verify", True, True, False),
            ("unsupported", False, False, False),
            ("supported", False, True, True),
        ):
            with (
                self.subTest(name=name),
                patch.object(
                    Qwen3NextGatedDeltaNetDecode,
                    "_get_bs_from_attenion_input",
                    return_value=(2, 1),
                ),
                patch.object(
                    Qwen3NextGatedDeltaNetDecode,
                    "_get_ssm_states",
                    return_value=state,
                ),
                patch(
                    "rtp_llm.models_py.model_desc.qwen3_next."
                    "is_aiter_flydsl_gdn_decode_supported",
                    autospec=True,
                    return_value=gate_supported,
                ) as support_gate,
                patch(
                    "rtp_llm.models_py.model_desc.qwen3_next."
                    "prepare_aiter_flydsl_gdn_decode_state_indices",
                    autospec=True,
                    return_value=(
                        read_indices,
                        write_indices,
                        invalid_row_flags,
                    ),
                ) as prepare_indices,
                patch(
                    "rtp_llm.models_py.model_desc.qwen3_next."
                    "aiter_flydsl_gdn_decode",
                    autospec=True,
                    return_value=output,
                ) as aiter_decode,
                patch(
                    "rtp_llm.models_py.model_desc.qwen3_next.fused_gdn_gating",
                    return_value=(
                        torch.zeros(2, 8),
                        torch.zeros(2, 8),
                    ),
                ),
                patch(
                    "rtp_llm.models_py.model_desc.qwen3_next."
                    "fused_recurrent_gated_delta_rule",
                    return_value=(output, None),
                ) as triton_decode,
            ):
                metadata = Qwen3NextMetadata(is_target_verify=is_target_verify)
                decode._fla(
                    mixed_qkv,
                    b,
                    a,
                    torch.empty(0),
                    1024,
                    attention_inputs,
                    metadata,
                )

                if expect_aiter:
                    gate_args = support_gate.call_args.args
                    self.assertEqual(gate_args[0].data_ptr(), expected_q.data_ptr())
                    self.assertEqual(gate_args[1].data_ptr(), expected_k.data_ptr())
                    self.assertEqual(gate_args[2].data_ptr(), expected_v.data_ptr())
                    self.assertIs(gate_args[3], a)
                    self.assertIs(gate_args[4], b)
                    self.assertIs(gate_args[5], state)
                    self.assertIs(gate_args[6], decode.alog)
                    self.assertIs(gate_args[7], decode.dt_bias)
                    gate_kwargs = support_gate.call_args.kwargs
                    self.assertIsNone(gate_kwargs["scale"])
                    self.assertIs(
                        gate_kwargs["block_map"],
                        attention_inputs.kv_cache_kernel_block_id_device,
                    )
                    self.assertIs(
                        gate_kwargs["sequence_lengths_plus_1"],
                        attention_inputs.sequence_lengths_plus_1_device,
                    )
                    self.assertEqual(gate_kwargs["seq_size_per_block"], 1024)
                    self.assertIs(
                        gate_kwargs["host_sequence_lengths"],
                        attention_inputs.sequence_lengths,
                    )
                    self.assertEqual(gate_kwargs["state_pool_size"], state.shape[0])
                    state_metadata = prepare_indices.call_args.args[0]
                    prepare_indices.assert_called_once_with(state_metadata)
                    call = aiter_decode.call_args.kwargs
                    for tensor_name, expected in (
                        ("q", expected_q),
                        ("k", expected_k),
                        ("v", expected_v),
                    ):
                        with self.subTest(case=name, tensor=tensor_name):
                            self.assertEqual(
                                call[tensor_name].data_ptr(), expected.data_ptr()
                            )
                            self.assertEqual(
                                call[tensor_name].stride(), expected.stride()
                            )
                    self.assertIs(call["a"], a)
                    self.assertIs(call["b"], b)
                    self.assertIs(call["state"], state)
                    self.assertIs(call["A_log"], decode.alog)
                    self.assertIs(call["dt_bias"], decode.dt_bias)
                    self.assertIs(call["read_indices"], read_indices)
                    self.assertIs(call["write_indices"], write_indices)
                    self.assertIs(
                        decode.get_aiter_flydsl_gdn_decode_invalid_row_flags(),
                        invalid_row_flags,
                    )
                    self.assertTrue(call["use_qk_l2norm_in_kernel"])
                    self.assertTrue(call["already_validated"])
                    self.assertTrue(call["copy_state"])
                    self.assertIsNone(call["scale"])
                    triton_decode.assert_not_called()
                else:
                    prepare_indices.assert_not_called()
                    aiter_decode.assert_not_called()
                    triton_decode.assert_called_once()
                    self.assertIsNone(
                        decode.get_aiter_flydsl_gdn_decode_invalid_row_flags()
                    )
                    triton_block_map = triton_decode.call_args.kwargs["block_map"]
                    self.assertEqual(triton_block_map.shape, (2, 1))
                    self.assertEqual(
                        triton_block_map.stride(0),
                        attention_inputs.kv_cache_kernel_block_id_device.stride(0),
                    )
                    if is_target_verify:
                        support_gate.assert_not_called()

    def test_qwen3_next_nvidia_cuda_gate_preserves_triton_decode_path(self):
        (
            decode,
            mixed_qkv,
            b,
            a,
            state,
            attention_inputs,
            output,
        ) = self._make_qwen3_next_decode_fla_case()

        with (
            patch.object(decode, "_get_bs_from_attenion_input", return_value=(2, 1)),
            patch.object(decode, "_get_ssm_states", return_value=state),
            patch(
                "rtp_llm.models_py.triton_kernels.fla."
                "aiter_flydsl_decode.is_amd_cdna3",
                False,
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "AiterFlydslGdnDecodeStateMetadata",
                autospec=True,
            ) as state_metadata,
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "prepare_aiter_flydsl_gdn_decode_state_indices",
                autospec=True,
            ) as prepare_indices,
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next." "aiter_flydsl_gdn_decode",
                autospec=True,
            ) as aiter_decode,
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next.fused_gdn_gating",
                return_value=(torch.zeros(2, 8), torch.zeros(2, 8)),
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "fused_recurrent_gated_delta_rule",
                return_value=(output, None),
            ) as triton_decode,
        ):
            actual = decode._fla(
                mixed_qkv,
                b,
                a,
                torch.empty(0),
                1024,
                attention_inputs,
                Qwen3NextMetadata(),
            )

        self.assertEqual(actual.data_ptr(), output.data_ptr())
        self.assertEqual(actual.shape, (2, 8, 128))
        state_metadata.assert_not_called()
        prepare_indices.assert_not_called()
        aiter_decode.assert_not_called()
        triton_decode.assert_called_once()
        self.assertIs(
            triton_decode.call_args.kwargs["initial_state"],
            state,
        )
        self.assertTrue(triton_decode.call_args.kwargs["inplace_final_state"])
        self.assertIsNone(decode.get_aiter_flydsl_gdn_decode_invalid_row_flags())

    @unittest.skipUnless(
        torch.version.hip is not None and torch.cuda.is_available(),
        "real model-to-AITER integration requires ROCm",
    )
    def test_qwen3_next_decode_real_aiter_adapter_integration(self):
        (
            decode,
            mixed_qkv,
            b,
            a,
            state,
            attention_inputs,
            _,
        ) = self._make_qwen3_next_decode_fla_case()
        decode.alog = decode.alog.cuda()
        decode.dt_bias = decode.dt_bias.cuda()
        mixed_qkv = mixed_qkv.cuda()
        a = a.cuda()
        b = b.cuda()
        state = state.cuda()
        attention_inputs.kv_cache_kernel_block_id_device = (
            attention_inputs.kv_cache_kernel_block_id_device.cuda()
        )
        attention_inputs.sequence_lengths_plus_1_device = (
            attention_inputs.sequence_lengths_plus_1_device.cuda()
        )

        with (
            patch.object(decode, "_get_bs_from_attenion_input", return_value=(2, 1)),
            patch.object(decode, "_get_ssm_states", return_value=state),
        ):
            output = decode._fla(
                mixed_qkv,
                b,
                a,
                torch.empty(0, device="cuda"),
                1024,
                attention_inputs,
                Qwen3NextMetadata(),
            )
        torch.cuda.synchronize()

        self.assertEqual(output.shape, (2, 8, 128))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_qwen3_next_decode_falls_back_when_block_map_is_none(self):
        (
            decode,
            mixed_qkv,
            b,
            a,
            state,
            attention_inputs,
            output,
        ) = self._make_qwen3_next_decode_fla_case()
        attention_inputs.kv_cache_kernel_block_id_device = None

        with (
            patch.object(decode, "_get_bs_from_attenion_input", return_value=(2, 1)),
            patch.object(decode, "_get_ssm_states", return_value=state),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "is_aiter_flydsl_gdn_decode_supported",
                autospec=True,
            ) as support_gate,
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "prepare_aiter_flydsl_gdn_decode_state_indices",
                autospec=True,
            ) as prepare_indices,
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next.aiter_flydsl_gdn_decode",
                autospec=True,
            ) as aiter_decode,
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next.fused_gdn_gating",
                return_value=(torch.zeros(2, 8), torch.zeros(2, 8)),
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "fused_recurrent_gated_delta_rule",
                return_value=(output, None),
            ) as triton_decode,
        ):
            decode._fla(
                mixed_qkv,
                b,
                a,
                torch.empty(0),
                1024,
                attention_inputs,
                Qwen3NextMetadata(),
            )

        support_gate.assert_not_called()
        prepare_indices.assert_not_called()
        aiter_decode.assert_not_called()
        triton_decode.assert_called_once()
        self.assertIsNone(triton_decode.call_args.kwargs["block_map"])

    def test_qwen3_next_decode_passes_state_copy_policy_to_adapter(self):
        cases = (
            ("eager-non-boundary", False, torch.tensor([1001, 1500]), False),
            ("eager-boundary", False, torch.tensor([1024, 1500]), True),
            ("eager-mixed-boundary", False, torch.tensor([1001, 1024]), True),
            (
                "non-cpu-host-lengths",
                False,
                torch.empty(2, device="meta", dtype=torch.int32),
                True,
            ),
            ("empty-host-lengths", False, torch.empty(0, dtype=torch.int32), True),
            ("tagged-graph-mode", True, torch.tensor([1001, 1500]), False),
        )
        for case_name, graph_enabled, host_lengths, expected_copy_state in cases:
            (
                decode,
                mixed_qkv,
                b,
                a,
                state,
                attention_inputs,
                output,
            ) = self._make_qwen3_next_decode_fla_case()
            decode.enable_cuda_graph = graph_enabled
            attention_inputs.is_cuda_graph = False
            attention_inputs.sequence_lengths = host_lengths
            read_indices = torch.tensor([1, 4], dtype=torch.int32)
            write_indices = torch.tensor([1, 5], dtype=torch.int32)

            with (
                self.subTest(case=case_name),
                patch.object(
                    decode, "_get_bs_from_attenion_input", return_value=(2, 1)
                ),
                patch.object(decode, "_get_ssm_states", return_value=state),
                patch(
                    "rtp_llm.models_py.model_desc.qwen3_next."
                    "is_aiter_flydsl_gdn_decode_supported",
                    autospec=True,
                    return_value=True,
                ),
                patch(
                    "rtp_llm.models_py.model_desc.qwen3_next."
                    "prepare_aiter_flydsl_gdn_decode_state_indices",
                    autospec=True,
                    return_value=(
                        read_indices,
                        write_indices,
                        torch.zeros(2, dtype=torch.int32),
                    ),
                ),
                patch(
                    "rtp_llm.models_py.model_desc.qwen3_next."
                    "aiter_flydsl_gdn_decode",
                    autospec=True,
                    return_value=output,
                ) as aiter_decode,
            ):
                decode._fla(
                    mixed_qkv,
                    b,
                    a,
                    torch.empty(0),
                    1024,
                    attention_inputs,
                    Qwen3NextMetadata(),
                )
                kwargs = aiter_decode.call_args.kwargs
                self.assertEqual(kwargs["copy_state"], expected_copy_state)
                self.assertIsNone(kwargs["scale"])

    def test_qwen3_next_decode_reuses_indices_across_layers_in_one_forward(self):
        (
            decode,
            mixed_qkv,
            b,
            a,
            state,
            attention_inputs,
            output,
        ) = self._make_qwen3_next_decode_fla_case()
        metadata = Qwen3NextMetadata()
        indices = (
            torch.tensor([1, 4], dtype=torch.int32),
            torch.tensor([1, 5], dtype=torch.int32),
        )

        with (
            patch.object(
                Qwen3NextGatedDeltaNetDecode,
                "_get_bs_from_attenion_input",
                return_value=(2, 1),
            ),
            patch.object(
                Qwen3NextGatedDeltaNetDecode,
                "_get_ssm_states",
                return_value=state,
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "is_aiter_flydsl_gdn_decode_supported",
                autospec=True,
                return_value=True,
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "prepare_aiter_flydsl_gdn_decode_state_indices",
                autospec=True,
                return_value=(*indices, torch.zeros(2, dtype=torch.int32)),
            ) as prepare_indices,
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next." "aiter_flydsl_gdn_decode",
                autospec=True,
                return_value=output,
            ) as aiter_decode,
        ):
            for _ in range(2):
                decode._fla(
                    mixed_qkv,
                    b,
                    a,
                    torch.empty(0),
                    1024,
                    attention_inputs,
                    metadata,
                )

        prepare_indices.assert_called_once()
        self.assertEqual(aiter_decode.call_count, 2)

    def test_qwen3_next_decode_cache_includes_state_pool_contract(self):
        (
            decode,
            mixed_qkv,
            b,
            a,
            state,
            attention_inputs,
            output,
        ) = self._make_qwen3_next_decode_fla_case()
        larger_state = torch.empty(
            state.shape[0] + 1,
            *state.shape[1:],
            dtype=state.dtype,
        )
        metadata = Qwen3NextMetadata()

        with (
            patch.object(decode, "_get_bs_from_attenion_input", return_value=(2, 1)),
            patch.object(decode, "_get_ssm_states", side_effect=(state, larger_state)),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "is_aiter_flydsl_gdn_decode_supported",
                autospec=True,
                return_value=True,
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "prepare_aiter_flydsl_gdn_decode_state_indices",
                autospec=True,
                return_value=(
                    torch.tensor([1, 4], dtype=torch.int32),
                    torch.tensor([1, 5], dtype=torch.int32),
                    torch.zeros(2, dtype=torch.int32),
                ),
            ) as prepare_indices,
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next.aiter_flydsl_gdn_decode",
                autospec=True,
                return_value=output,
            ),
        ):
            for _ in range(2):
                decode._fla(
                    mixed_qkv,
                    b,
                    a,
                    torch.empty(0),
                    1024,
                    attention_inputs,
                    metadata,
                )

        self.assertEqual(prepare_indices.call_count, 2)
        self.assertEqual(
            prepare_indices.call_args_list[0].args[0].state_pool_size,
            state.shape[0],
        )
        self.assertEqual(
            prepare_indices.call_args_list[1].args[0].state_pool_size,
            larger_state.shape[0],
        )

    def test_qwen3_next_decode_cache_matches_new_wrappers_for_same_storage(self):
        (
            decode,
            mixed_qkv,
            b,
            a,
            state,
            base_inputs,
            output,
        ) = self._make_qwen3_next_decode_fla_case()

        class WrapperInputs:
            is_cuda_graph = True
            sequence_lengths = base_inputs.sequence_lengths

            @property
            def kv_cache_kernel_block_id_device(self):
                return base_inputs.kv_cache_kernel_block_id_device.view_as(
                    base_inputs.kv_cache_kernel_block_id_device
                )

            @property
            def sequence_lengths_plus_1_device(self):
                return base_inputs.sequence_lengths_plus_1_device.view_as(
                    base_inputs.sequence_lengths_plus_1_device
                )

        metadata = Qwen3NextMetadata()
        with (
            patch.object(decode, "_get_bs_from_attenion_input", return_value=(2, 1)),
            patch.object(decode, "_get_ssm_states", return_value=state),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "is_aiter_flydsl_gdn_decode_supported",
                autospec=True,
                return_value=True,
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "prepare_aiter_flydsl_gdn_decode_state_indices",
                autospec=True,
                return_value=(
                    torch.tensor([1, 4], dtype=torch.int32),
                    torch.tensor([1, 5], dtype=torch.int32),
                    torch.zeros(2, dtype=torch.int32),
                ),
            ) as prepare_indices,
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next.aiter_flydsl_gdn_decode",
                autospec=True,
                return_value=output,
            ),
        ):
            for _ in range(2):
                decode._fla(
                    mixed_qkv,
                    b,
                    a,
                    torch.empty(0),
                    1024,
                    WrapperInputs(),
                    metadata,
                )

        prepare_indices.assert_called_once()
        entry = next(iter(metadata.aiter_flydsl_gdn_decode_indices.values()))
        self.assertEqual(
            entry.state_metadata.block_map.data_ptr(),
            base_inputs.kv_cache_kernel_block_id_device.data_ptr(),
        )
        self.assertEqual(
            entry.state_metadata.sequence_lengths_plus_1.data_ptr(),
            base_inputs.sequence_lengths_plus_1_device.data_ptr(),
        )
        self.assertEqual(entry.read_indices.tolist(), [1, 4])
        self.assertEqual(entry.write_indices.tolist(), [1, 5])
        self.assertEqual(entry.invalid_row_flags.tolist(), [0, 0])

    def test_qwen3_next_decode_cache_is_scoped_to_attention_inputs(self):
        (
            decode,
            mixed_qkv,
            b,
            a,
            state,
            first_inputs,
            output,
        ) = self._make_qwen3_next_decode_fla_case()
        second_inputs = SimpleNamespace(
            is_cuda_graph=True,
            kv_cache_kernel_block_id_device=(
                first_inputs.kv_cache_kernel_block_id_device.clone()
            ),
            sequence_lengths=first_inputs.sequence_lengths.clone(),
            sequence_lengths_plus_1_device=(
                first_inputs.sequence_lengths_plus_1_device.clone()
            ),
        )
        metadata = Qwen3NextMetadata()
        with (
            patch.object(decode, "_get_bs_from_attenion_input", return_value=(2, 1)),
            patch.object(decode, "_get_ssm_states", return_value=state),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "is_aiter_flydsl_gdn_decode_supported",
                autospec=True,
                return_value=True,
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next."
                "prepare_aiter_flydsl_gdn_decode_state_indices",
                autospec=True,
                return_value=(
                    torch.tensor([1, 4], dtype=torch.int32),
                    torch.tensor([1, 5], dtype=torch.int32),
                    torch.zeros(2, dtype=torch.int32),
                ),
            ) as prepare_indices,
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next.aiter_flydsl_gdn_decode",
                autospec=True,
                return_value=output,
            ),
        ):
            for inputs in (first_inputs, second_inputs):
                decode._fla(
                    mixed_qkv,
                    b,
                    a,
                    torch.empty(0),
                    1024,
                    inputs,
                    metadata,
                )

        self.assertEqual(prepare_indices.call_count, 2)
        self.assertIs(
            prepare_indices.call_args_list[0].args[0].block_map,
            first_inputs.kv_cache_kernel_block_id_device,
        )
        self.assertIs(
            prepare_indices.call_args_list[1].args[0].block_map,
            second_inputs.kv_cache_kernel_block_id_device,
        )
        self.assertIs(
            prepare_indices.call_args_list[0].args[0].sequence_lengths_plus_1,
            first_inputs.sequence_lengths_plus_1_device,
        )
        self.assertIs(
            prepare_indices.call_args_list[1].args[0].sequence_lengths_plus_1,
            second_inputs.sequence_lengths_plus_1_device,
        )

    def test_decoder_layer_default_metadata_is_fresh_per_forward(self):
        class IdentityPair(nn.Module):
            def forward(self, hidden_states, residual):
                return hidden_states, residual

        class CaptureAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.metadata = []

            def forward(self, **kwargs):
                self.metadata.append(kwargs["attn_meta"])
                return kwargs["hidden_states"]

        layer = object.__new__(Qwen3NextDecoderLayer)
        nn.Module.__init__(layer)
        layer.input_layernorm = IdentityPair()
        layer.post_attention_layernorm = IdentityPair()
        layer.self_attn = CaptureAttention()
        layer.mlp = nn.Identity()
        hidden = torch.randn(2, 4)
        residual = torch.randn(2, 4)

        for _ in range(2):
            layer.forward(hidden, residual, fmha_impl=None)

        first, second = layer.self_attn.metadata
        self.assertIsNot(first, second)
        self.assertEqual(first.aiter_flydsl_gdn_decode_indices, {})
        self.assertEqual(second.aiter_flydsl_gdn_decode_indices, {})

    def test_qwen3_next_mtp_shares_metadata_across_layers(self):
        class CaptureLayer(nn.Module):
            layer_type = HybridAttentionType.LINEAR

            def __init__(self):
                super().__init__()
                self.metadata = []
                self.target_verify_shapes = []

            def forward(self, hidden_states, residual, _fmha_impl, **kwargs):
                metadata = kwargs["attn_meta"]
                self.metadata.append(metadata)
                decode = object.__new__(Qwen3NextGatedDeltaNetDecode)
                self.target_verify_shapes.append(
                    decode._get_bs_from_attenion_input(
                        torch.empty(6, 4),
                        SimpleNamespace(prefix_lengths=torch.empty(2)),
                        metadata.is_target_verify,
                    )
                )
                return hidden_states, residual

        class IdentityPair(nn.Module):
            def forward(self, hidden_states, residual):
                return hidden_states, residual

        model = object.__new__(Qwen3NextMTPModel)
        nn.Module.__init__(model)
        model.embed_tokens = nn.Identity()
        model.pre_fc_norm_embedding = nn.Identity()
        model.pre_fc_norm_hidden = nn.Identity()
        model.fc = nn.Identity()
        model.layers = nn.ModuleList([CaptureLayer(), CaptureLayer()])
        model.norm = IdentityPair()
        model.kv_cache = None
        inputs = SimpleNamespace(
            input_ids=torch.randn(2, 4),
            input_hiddens=torch.randn(2, 4),
        )

        with (
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next_mtp."
                "get_primary_attention_inputs",
                return_value=SimpleNamespace(is_target_verify=True),
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next_mtp."
                "select_attention_inputs_for_layer",
                return_value=SimpleNamespace(),
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen3_next_mtp.PyModelOutputs",
                side_effect=lambda hidden_states: hidden_states,
            ),
        ):
            model.forward(inputs, fmha_impl={})

        first = model.layers[0].metadata[0]
        second = model.layers[1].metadata[0]
        self.assertIs(first, second)
        self.assertTrue(first.is_target_verify)
        self.assertEqual(model.layers[0].target_verify_shapes, [(2, 3)])
        self.assertEqual(model.layers[1].target_verify_shapes, [(2, 3)])


if __name__ == "__main__":
    unittest.main()
