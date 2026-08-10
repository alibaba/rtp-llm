"""Correctness tests for the fused all-gather CP prefill operator."""

import contextlib
import math
from typing import List
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.fused_allgather_cp_impl import (
    PCPFusedPagedAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    CPAttnTestBase,
    build_cp_attn_inputs,
    compute_rank_positions,
    extract_kv_from_paged_cache,
    make_configs,
    make_kv_cache,
    reference_causal_attention,
)

_FUSED_MODULE = (
    "rtp_llm.models_py.modules.factory.attention.cuda_cp_impl."
    "prefill_mha.fused_allgather_cp_impl"
)


class TestPCPFusedPagedAttnOp(CPAttnTestBase):
    OP_CLASS = PCPFusedPagedAttnOp
    AG_MODULE = _FUSED_MODULE

    def _patch_all_gather(
        self,
        stack: contextlib.ExitStack,
        all_local_k: List[torch.Tensor],
        all_local_v: List[torch.Tensor],
        kv_head_num: int,
        head_dim: int,
    ):
        payload = torch.cat(
            [
                torch.stack(
                    (
                        local_k.view(-1, kv_head_num, head_dim),
                        local_v.view(-1, kv_head_num, head_dim),
                    ),
                    dim=1,
                )
                for local_k, local_v in zip(all_local_k, all_local_v)
            ],
            dim=0,
        )

        def mock_all_gather_into_tensor(output, input_, group=None, async_op=False):
            output.copy_(payload.view_as(output))
            return None

        stack.enter_context(patch(f"{_FUSED_MODULE}._get_group", return_value=None))
        stack.enter_context(
            patch(
                f"{_FUSED_MODULE}.torch.distributed.all_gather_into_tensor",
                side_effect=mock_all_gather_into_tensor,
            )
        )

    def test_no_prefix(self):
        cases = [
            ([32], 2, 0, 8, 2, 64),
            ([32], 2, 1, 8, 2, 64),
            ([64], 4, 2, 8, 2, 64),
            ([32, 64], 2, 1, 8, 2, 64),
            ([128], 4, 3, 32, 4, 128),
            ([128], 8, 5, 64, 8, 128),
        ]
        for lengths, cp_size, rank, q_heads, kv_heads, head_dim in cases:
            with self.subTest(lengths=lengths, cp_size=cp_size, rank=rank):
                self.run_no_prefix(
                    batch_size=len(lengths),
                    sequence_lengths=lengths,
                    cp_size=cp_size,
                    cp_rank=rank,
                    head_num=q_heads,
                    kv_head_num=kv_heads,
                    head_dim=head_dim,
                    tokens_per_block=16,
                )

    def test_prefix_cache(self):
        cases = [
            ([32], [64], 2, 0),
            ([32], [64], 2, 1),
            ([32, 64], [64, 128], 2, 0),
            ([64], [64], 4, 2),
        ]
        for new_lengths, prefix_lengths, cp_size, rank in cases:
            with self.subTest(
                new_lengths=new_lengths,
                prefix_lengths=prefix_lengths,
                cp_size=cp_size,
                rank=rank,
            ):
                self.run_with_prefix(
                    batch_size=len(new_lengths),
                    new_lengths=new_lengths,
                    prefix_lengths=prefix_lengths,
                    cp_size=cp_size,
                    cp_rank=rank,
                    tokens_per_block=16,
                )

    def test_non_aligned_sequence_uses_padded_geometry(self):
        actual_length = 65
        cp_size = 4
        cp_rank = 0
        padded_length = math.ceil(actual_length / (2 * cp_size)) * 2 * cp_size
        chunk_length = padded_length // cp_size
        head_num, kv_head_num, head_dim = 8, 2, 64
        tokens_per_block = 16

        attn_cfg, par_cfg = make_configs(
            head_num=head_num,
            kv_head_num=kv_head_num,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        q_padded = torch.randn(
            padded_length,
            head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        k_padded = torch.randn(
            padded_length,
            kv_head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        v_padded = torch.randn_like(k_padded)
        reference = reference_causal_attention(
            q_padded[:actual_length],
            k_padded[:actual_length],
            v_padded[:actual_length],
            [0, actual_length],
        )

        rank_positions = compute_rank_positions([padded_length], cp_size)
        all_local_k = [
            k_padded[torch.tensor(positions, device=self.device)].reshape(
                -1, kv_head_num * head_dim
            )
            for positions in rank_positions
        ]
        all_local_v = [
            v_padded[torch.tensor(positions, device=self.device)].reshape(
                -1, kv_head_num * head_dim
            )
            for positions in rank_positions
        ]
        local_positions = rank_positions[cp_rank]
        local_index = torch.tensor(local_positions, device=self.device)
        qkv = torch.cat(
            (
                q_padded[local_index].reshape(-1, head_num * head_dim),
                all_local_k[cp_rank],
                all_local_v[cp_rank],
            ),
            dim=-1,
        )

        attn_inputs = build_cp_attn_inputs(
            [actual_length],
            [chunk_length],
            cp_size,
            tokens_per_block,
            device=self.device,
        )
        attn_inputs.context_parallel_info.prefill_qkv_padding_mask[actual_length:] = 0
        total_blocks = math.ceil(actual_length / tokens_per_block)
        kv_cache = make_kv_cache(
            total_blocks,
            kv_head_num,
            tokens_per_block,
            head_dim,
            device=self.device,
        )

        with contextlib.ExitStack() as stack:
            self._patch_all_gather(
                stack, all_local_k, all_local_v, kv_head_num, head_dim
            )
            op = self.OP_CLASS(attn_cfg, attn_inputs, par_cfg)
            params = op.prepare(attn_inputs)
            output = op.forward(qkv, kv_cache, params)

        valid_local = [
            local
            for local, position in enumerate(local_positions)
            if position < actual_length
        ]
        valid_positions = [
            position for position in local_positions if position < actual_length
        ]
        self._assert_close(
            output[torch.tensor(valid_local, device=self.device)],
            reference[torch.tensor(valid_positions, device=self.device)],
        )

        cache_k, cache_v = extract_kv_from_paged_cache(
            kv_cache, [actual_length], tokens_per_block
        )
        self.assertTrue(torch.equal(cache_k, k_padded[:actual_length]))
        self.assertTrue(torch.equal(cache_v, v_padded[:actual_length]))
