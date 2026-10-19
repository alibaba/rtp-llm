import logging
import os
import unittest
from typing import List
from unittest.mock import patch

# DeepJIT derives its on-disk kernel cache from $HOME and asserts when it is
# empty ("HOME environment variable must not be empty", cache/disk.hpp). Some
# CI test actions run without HOME, which flaked the decode cases; give the
# process a writable fallback before any rtp-kernel/JIT import runs.
if not os.environ.get("HOME"):
    os.environ["HOME"] = os.environ.get("TEST_TMPDIR") or "/tmp"

import torch
from base_attention_test import BaseAttentionTest

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.ops import RopeStyle
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.ops.fused_rope_kvcache_op import (
    DecodeRopeContractError,
    FusedRopeAttnParams,
    FusedRopeKVCacheDecodeOp,
)
from rtp_llm.test.utils.cuda_graph_util import record_graph_capture

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestFusedRopeKVCacheDecodeSequenceLengths(BaseAttentionTest):
    """Test the shared decode-RoPE sequence-length buffer contract."""

    def _real_kernel_case(self, sequence_lengths: List[int]):
        config = self._create_config()
        config.attn_configs.need_rope_kv_cache = True
        config.attn_configs.max_seq_len = 128
        config.attn_configs.rope_config.style = RopeStyle.Base
        config.attn_configs.rope_config.dim = config.size_per_head
        config.attn_configs.rope_config.base = 10000
        config.attn_configs.rope_config.max_pos = 128

        inputs = self._create_attention_inputs_base(
            len(sequence_lengths), sequence_lengths, config.seq_size_per_block
        )
        inputs.is_cuda_graph = True

        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        params = op.prepare(inputs)
        qkv_width = (config.head_num + 2 * config.head_num_kv) * config.size_per_head
        qkv = torch.randn(
            len(sequence_lengths),
            qkv_width,
            dtype=torch.float16,
            device=self.device,
        )
        kv_cache, _, _ = self._create_kv_cache(
            self._calculate_total_blocks(sequence_lengths, config.seq_size_per_block),
            config.seq_size_per_block,
            config.head_num_kv,
            config.size_per_head,
        )
        return config, inputs, op, params, qkv, kv_cache

    @staticmethod
    def _base_rope_reference(
        qkv: torch.Tensor,
        positions: torch.Tensor,
        head_num: int,
        head_dim: int,
        rope_base: int,
    ) -> torch.Tensor:
        q = qkv[:, : head_num * head_dim].reshape(-1, head_num, head_dim)
        q_float = q.float()
        half_dim = head_dim // 2
        inv_freq = 1.0 / (
            rope_base
            ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim)
        )
        freqs = positions.to(device=q.device, dtype=torch.float32)[:, None] * inv_freq
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)[:, None, :]
        sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)[:, None, :]
        rotated_half = torch.cat(
            (-q_float[..., half_dim:], q_float[..., :half_dim]), dim=-1
        )
        return (q_float * cos + rotated_half * sin).to(q.dtype)

    def _inputs(
        self,
        values: List[int],
        *,
        is_cuda_graph: bool = True,
        pinned: bool = True,
        dtype: torch.dtype = torch.int32,
        with_target: bool = True,
    ) -> PyAttentionInputs:
        inputs = PyAttentionInputs()
        inputs.is_cuda_graph = is_cuda_graph
        source = torch.tensor(values, dtype=dtype)
        inputs.sequence_lengths = source.pin_memory() if pinned else source
        if with_target:
            # Deliberately unrelated values: the op must mirror the authoritative
            # host tensor. CudaGraphRunner, not this op, owns padding-tail zeroing.
            inputs.sequence_lengths_plus_1_device = torch.full(
                (len(values),), 901, dtype=torch.int32, device=self.device
            )
        return inputs

    def _params(
        self,
        sequence_lengths: torch.Tensor,
        kv_cache_offset: torch.Tensor | None,
    ) -> FusedRopeAttnParams:
        empty = torch.empty(0, dtype=torch.int32, device=self.device)
        return FusedRopeAttnParams(
            kv_cache_offset=kv_cache_offset,
            kv_cache_offset_h=None,
            padding_offset=None,
            position_ids=None,
            cu_seqlens=empty,
            cu_kv_seqlens=empty,
            input_lengths=empty,
            prefix_lengths=empty,
            sequence_lengths=sequence_lengths,
            max_seq_len=0,
            max_prefix_length=0,
            context_total_kv_length=0,
            decode_plan=True,
            attn_type=torch.float16,
        )

    def test_replay_reuses_buffer_and_mirrors_source_values(self):
        config = self._create_config()
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)

        capture_inputs = self._inputs([10, 20, 30, 40])
        capture_buffer = op.refresh_sequence_lengths(capture_inputs)
        capture_ptr = capture_buffer.data_ptr()

        replay_inputs = self._inputs([100, 200, 0, 0])
        replay_buffer = op.refresh_sequence_lengths(replay_inputs)
        torch.cuda.synchronize()

        self.assertIs(replay_buffer, capture_buffer)
        self.assertEqual(replay_buffer.data_ptr(), capture_ptr)
        self.assertTrue(replay_buffer.is_cuda)
        self.assertEqual(replay_buffer.dtype, torch.int32)
        torch.testing.assert_close(
            replay_buffer.cpu(), torch.tensor([100, 200, 0, 0], dtype=torch.int32)
        )

    def test_cuda_source_is_copied_into_owned_buffer(self):
        config = self._create_config()
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        inputs = PyAttentionInputs()
        inputs.is_cuda_graph = False
        inputs.sequence_lengths = torch.tensor(
            [7], dtype=torch.int32, device=self.device
        )

        sequence_lengths = op.refresh_sequence_lengths(inputs)

        self.assertTrue(sequence_lengths.is_cuda)
        self.assertNotEqual(
            sequence_lengths.data_ptr(), inputs.sequence_lengths.data_ptr()
        )
        torch.testing.assert_close(sequence_lengths, inputs.sequence_lengths)

    def test_pageable_cpu_source_uses_sync_copy_outside_graph(self):
        config = self._create_config()
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        inputs = self._inputs([1], is_cuda_graph=False, pinned=False)

        sequence_lengths = op.refresh_sequence_lengths(inputs, self.device)

        torch.testing.assert_close(sequence_lengths.cpu(), inputs.sequence_lengths)

    def test_contract_errors(self):
        config = self._create_config()

        missing = PyAttentionInputs()
        missing.is_cuda_graph = False

        cases = [
            ("missing", missing, None, "non-empty"),
            (
                "empty",
                self._inputs([], is_cuda_graph=False),
                None,
                "non-empty",
            ),
            (
                "dtype",
                self._inputs([1], is_cuda_graph=False, dtype=torch.int64),
                None,
                "torch.int32",
            ),
            (
                "pageable_graph_source",
                self._inputs([1], pinned=False),
                self.device,
                "CUDA or pinned",
            ),
            (
                "missing_target",
                self._inputs([1], is_cuda_graph=False, with_target=False),
                None,
                "CUDA target device",
            ),
        ]
        for name, inputs, device, message in cases:
            with self.subTest(name=name):
                op = FusedRopeKVCacheDecodeOp(config.attn_configs)
                with self.assertRaisesRegex(DecodeRopeContractError, message):
                    op.refresh_sequence_lengths(inputs, device)

        pageable_replay = self._inputs([1], is_cuda_graph=False, pinned=False)
        with self.assertRaisesRegex(DecodeRopeContractError, "CUDA or pinned"):
            FusedRopeKVCacheDecodeOp(config.attn_configs).refresh_sequence_lengths(
                pageable_replay,
                self.device,
                forbid_reallocation=True,
            )

    def test_prepare_reuses_owned_sequence_lengths_buffer(self):
        config = self._create_config()
        config.attn_configs.need_rope_kv_cache = True
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        inputs = self._create_attention_inputs_base(1, [2], 64)
        inputs.is_cuda_graph = True

        first = op.prepare(inputs)
        first_ptr = first.sequence_lengths.data_ptr()
        inputs.sequence_lengths.fill_(7)
        second = op.prepare(inputs)
        torch.cuda.synchronize()

        self.assertIs(first.sequence_lengths, second.sequence_lengths)
        self.assertEqual(second.sequence_lengths.data_ptr(), first_ptr)
        torch.testing.assert_close(
            second.sequence_lengths.cpu(), torch.tensor([7], dtype=torch.int32)
        )

    def test_real_kernel_uses_owned_device_sequence_lengths(self):
        cases = ([5], [5, 9, 1, 1])
        for sequence_lengths in cases:
            with self.subTest(sequence_lengths=sequence_lengths):
                config, inputs, op, params, qkv, kv_cache = self._real_kernel_case(
                    list(sequence_lengths)
                )

                actual = op.forward(qkv, kv_cache, params)
                expected = self._base_rope_reference(
                    qkv,
                    inputs.sequence_lengths,
                    config.head_num,
                    config.size_per_head,
                    config.attn_configs.rope_config.base,
                )

                self.assertTrue(params.sequence_lengths.is_cuda)
                self.assertIs(params.sequence_lengths, op._sequence_lengths_device)
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_real_kernel_graph_replay_observes_refreshed_lengths(self):
        capture_lengths = [3, 6, 1, 1]
        config, inputs, op, params, qkv, kv_cache = self._real_kernel_case(
            capture_lengths
        )
        sequence_lengths_ptr = params.sequence_lengths.data_ptr()

        # Materialize lazy kernel/rope-cache state before graph capture.
        op.forward(qkv, kv_cache, params)
        torch.cuda.synchronize()
        with record_graph_capture() as graph:
            graph_output = op.forward(qkv, kv_cache, params)
        graph.replay()
        torch.cuda.synchronize()
        capture_output = graph_output.clone()
        capture_expected = self._base_rope_reference(
            qkv,
            inputs.sequence_lengths,
            config.head_num,
            config.size_per_head,
            config.attn_configs.rope_config.base,
        )
        torch.testing.assert_close(
            capture_output, capture_expected, rtol=1e-2, atol=1e-2
        )

        # Keep the graph-batch tail at zero, matching CudaGraphRunner padding.
        replay_positions = torch.tensor([11, 19, 0, 0], dtype=torch.int32)
        inputs.sequence_lengths.copy_(replay_positions)
        replay_buffer = op.refresh_sequence_lengths(inputs, forbid_reallocation=True)
        self.assertIs(replay_buffer, params.sequence_lengths)
        self.assertEqual(replay_buffer.data_ptr(), sequence_lengths_ptr)

        graph.replay()
        torch.cuda.synchronize()
        replay_output = graph_output.clone()
        replay_expected = self._base_rope_reference(
            qkv,
            replay_positions,
            config.head_num,
            config.size_per_head,
            config.attn_configs.rope_config.base,
        )
        torch.testing.assert_close(replay_output, replay_expected, rtol=1e-2, atol=1e-2)
        self.assertFalse(torch.allclose(capture_output[:2], replay_output[:2]))
        torch.testing.assert_close(capture_output[2:], replay_output[2:])

    def test_prepare_skips_unused_rope_buffer_for_pageable_input(self):
        config = self._create_config()
        config.attn_configs.need_rope_kv_cache = False
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        inputs = self._create_attention_inputs_base(1, [2], 64)
        inputs.sequence_lengths = torch.tensor([1], dtype=torch.int32)
        inputs.is_cuda_graph = True

        params = op.prepare(inputs)

        self.assertIs(params.sequence_lengths, inputs.sequence_lengths)
        self.assertIsNone(op._sequence_lengths_device)

    def test_graph_shape_change_is_rejected(self):
        config = self._create_config()
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        op.refresh_sequence_lengths(self._inputs([1, 2], is_cuda_graph=False))

        with self.assertRaisesRegex(DecodeRopeContractError, "cannot resize or move"):
            op.refresh_sequence_lengths(
                self._inputs([3], is_cuda_graph=False), forbid_reallocation=True
            )

    def test_indexless_device_reuses_buffer_and_eager_resize_rebinds(self):
        config = self._create_config()
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        first_inputs = self._inputs([1, 2], is_cuda_graph=False)
        first = op.refresh_sequence_lengths(first_inputs, torch.device("cuda"))
        first_ptr = first.data_ptr()

        second = op.refresh_sequence_lengths(first_inputs, self.device)
        self.assertIs(second, first)
        self.assertEqual(second.data_ptr(), first_ptr)

        resized_inputs = self._inputs([3], is_cuda_graph=False)
        resized = op.refresh_sequence_lengths(resized_inputs, self.device)
        self.assertIsNot(resized, first)
        self.assertEqual(tuple(resized.shape), (1,))
        torch.testing.assert_close(resized.cpu(), torch.tensor([3], dtype=torch.int32))

    def test_forward_rejects_missing_offset_and_non_owned_lengths(self):
        config = self._create_config()
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        qkv = torch.empty(0, device=self.device)
        host_lengths = torch.tensor([1], dtype=torch.int32).pin_memory()

        with self.assertRaisesRegex(DecodeRopeContractError, "kv_cache_offset"):
            op.forward(qkv, None, self._params(host_lengths, None))

        offset = torch.empty(0, dtype=torch.int32, device=self.device)
        with self.assertRaisesRegex(DecodeRopeContractError, "must be on CUDA"):
            op.forward(qkv, None, self._params(host_lengths, offset))

        foreign_device_lengths = torch.tensor(
            [1], dtype=torch.int32, device=self.device
        )
        with self.assertRaisesRegex(DecodeRopeContractError, "op-owned"):
            op.forward(qkv, None, self._params(foreign_device_lengths, offset))

    def test_factory_reraises_decode_rope_contract_errors(self):
        class ContractFailImpl:
            accepts_fmha_config = False

            @classmethod
            def support(cls, attn_configs, attn_inputs):
                return True

            @classmethod
            def support_parallelism_config(cls, parallelism_config):
                return True

            def __init__(self, *args, **kwargs):
                raise DecodeRopeContractError("invalid decode RoPE input")

        class FallbackImpl:
            accepts_fmha_config = False
            construction_calls = 0

            @classmethod
            def support(cls, attn_configs, attn_inputs):
                return True

            @classmethod
            def support_parallelism_config(cls, parallelism_config):
                return True

            def __init__(self, *args, **kwargs):
                type(self).construction_calls += 1

        inputs = PyAttentionInputs()
        inputs.is_prefill = False
        config = self._create_config().attn_configs

        with patch.object(
            attn_factory,
            "DECODE_MHA_IMPS",
            [ContractFailImpl, FallbackImpl],
        ):
            with self.assertRaisesRegex(
                DecodeRopeContractError, "invalid decode RoPE input"
            ):
                attn_factory.get_fmha_impl(config, None, inputs)

        self.assertEqual(FallbackImpl.construction_calls, 0)


if __name__ == "__main__":
    unittest.main()
