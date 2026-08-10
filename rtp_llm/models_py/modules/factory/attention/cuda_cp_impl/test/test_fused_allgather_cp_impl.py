"""Correctness tests for the fused all-gather CP prefill operator."""

import contextlib
import logging
import math
import unittest
from typing import List
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer import (
    select_prefill_cp_impl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.allgather_cp_impl import (
    PCPAllGatherAttnOp,
)
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
from rtp_llm.ops import CPAllGatherImpl, CPRotateMethod, KvCacheDataType

_FUSED_MODULE = (
    "rtp_llm.models_py.modules.factory.attention.cuda_cp_impl."
    "prefill_mha.fused_allgather_cp_impl"
)


class TestPCPFusedPagedAttnOp(CPAttnTestBase):
    OP_CLASS = PCPFusedPagedAttnOp

    def _patch_all_gather(
        self,
        stack: contextlib.ExitStack,
        all_local_k: List[torch.Tensor],
        all_local_v: List[torch.Tensor],
        cp_rank: int,
    ):
        def mock_all_gather_into(output, input_, group):
            self.assertEqual(group.name, "TP")
            tokens_local, _, kv_head_num, head_dim = input_.shape
            packed = [
                torch.stack(
                    (
                        local_k.view(tokens_local, kv_head_num, head_dim),
                        local_v.view(tokens_local, kv_head_num, head_dim),
                    ),
                    dim=1,
                )
                for local_k, local_v in zip(all_local_k, all_local_v)
            ]
            self.assertTrue(torch.equal(input_, packed[cp_rank]))
            payload = torch.cat(packed, dim=0)
            output.copy_(payload.view_as(output))
            return output

        stack.enter_context(
            patch(f"{_FUSED_MODULE}.all_gather_into", side_effect=mock_all_gather_into)
        )

    def test_no_prefix(self):
        cases = [
            ([20], 2, 0, 8, 2, 64, 16),
            ([20, 36], 2, 1, 8, 2, 64, 16),
            ([32], 2, 0, 8, 2, 64, 16),
            ([32], 2, 1, 8, 2, 64, 16),
            ([64], 4, 2, 8, 2, 64, 32),
            ([32, 64], 2, 1, 8, 2, 64, 16),
            ([128], 4, 3, 32, 4, 128, 64),
            ([128], 8, 5, 64, 8, 128, 64),
        ]
        for (
            lengths,
            cp_size,
            rank,
            q_heads,
            kv_heads,
            head_dim,
            tokens_per_block,
        ) in cases:
            with self.subTest(lengths=lengths, cp_size=cp_size, rank=rank):
                self.run_no_prefix(
                    batch_size=len(lengths),
                    sequence_lengths=lengths,
                    cp_size=cp_size,
                    cp_rank=rank,
                    head_num=q_heads,
                    kv_head_num=kv_heads,
                    head_dim=head_dim,
                    tokens_per_block=tokens_per_block,
                )

    def test_prefix_cache(self):
        # Same prefix matrix as the legacy all-gather suite: both the single-request
        # and the batched non-block-aligned cases, over 16/32/64-token pages.
        cases = [
            ([20], [48], 2, 0, 8, 2, 64, 16),
            ([20, 36], [48, 32], 2, 0, 8, 2, 64, 16),
            ([32], [64], 2, 0, 8, 2, 64, 16),
            ([32], [64], 2, 1, 8, 2, 64, 16),
            ([32, 64], [64, 128], 2, 0, 8, 2, 64, 32),
            ([64], [128], 2, 0, 32, 8, 128, 64),
            ([64], [64], 4, 2, 8, 2, 64, 16),
        ]
        for (
            new_lengths,
            prefix_lengths,
            cp_size,
            rank,
            q_heads,
            kv_heads,
            head_dim,
            tokens_per_block,
        ) in cases:
            with self.subTest(
                new_lengths=new_lengths,
                prefix_lengths=prefix_lengths,
                cp_size=cp_size,
                rank=rank,
                tokens_per_block=tokens_per_block,
            ):
                self.run_with_prefix(
                    batch_size=len(new_lengths),
                    new_lengths=new_lengths,
                    prefix_lengths=prefix_lengths,
                    cp_size=cp_size,
                    cp_rank=rank,
                    head_num=q_heads,
                    kv_head_num=kv_heads,
                    head_dim=head_dim,
                    tokens_per_block=tokens_per_block,
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
        q_width = head_num * head_dim
        self.assertFalse(qkv[:, :q_width].is_contiguous())

        # A second prefix-only request owns the page immediately after request 0.
        # Padding for request 0 must stay in its allocated last page.
        attn_inputs = build_cp_attn_inputs(
            [actual_length, tokens_per_block],
            [chunk_length, 0],
            cp_size,
            tokens_per_block,
            prefix_lengths=[0, tokens_per_block],
            device=self.device,
        )
        total_blocks = math.ceil(actual_length / tokens_per_block)
        kv_cache = make_kv_cache(
            total_blocks + 1,
            kv_head_num,
            tokens_per_block,
            head_dim,
            device=self.device,
        )
        kv_cache.kv_cache_base[total_blocks].fill_(7)
        adjacent_request_block = kv_cache.kv_cache_base[total_blocks].clone()

        with contextlib.ExitStack() as stack:
            self._patch_all_gather(stack, all_local_k, all_local_v, cp_rank)
            op = self.OP_CLASS(attn_cfg, attn_inputs, par_cfg)
            params = op.prepare(attn_inputs)
            output = op.forward(qkv, kv_cache, params)
            output_storage = output.data_ptr()
            repeated_output = op.forward(qkv, kv_cache, params)
            self.assertEqual(repeated_output.data_ptr(), output_storage)

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

        tail_start = actual_length % tokens_per_block
        tail_end = padded_length % tokens_per_block
        padded_tail_k = kv_cache.kv_cache_base[
            total_blocks - 1, 0, :, tail_start:tail_end
        ]
        padded_tail_v = kv_cache.kv_cache_base[
            total_blocks - 1, 1, :, tail_start:tail_end
        ]
        self.assertTrue(
            torch.equal(
                padded_tail_k.permute(1, 0, 2),
                k_padded[actual_length:padded_length],
            )
        )
        self.assertTrue(
            torch.equal(
                padded_tail_v.permute(1, 0, 2),
                v_padded[actual_length:padded_length],
            )
        )
        self.assertTrue(
            torch.equal(kv_cache.kv_cache_base[total_blocks], adjacent_request_block)
        )

    def test_dispatch_defaults_to_legacy(self):
        attn_cfg, par_cfg, attn_inputs = self._make_dispatch_case()

        self.assertEqual(
            par_cfg.prefill_cp_config.all_gather_impl,
            CPAllGatherImpl.LEGACY,
        )
        self.assertIs(
            select_prefill_cp_impl(attn_cfg, attn_inputs, par_cfg),
            PCPAllGatherAttnOp,
        )

    def test_dispatch_selects_fused_when_supported(self):
        attn_cfg, par_cfg, attn_inputs = self._make_dispatch_case()
        par_cfg.prefill_cp_config.all_gather_impl = CPAllGatherImpl.FUSED

        self.assertIs(
            select_prefill_cp_impl(attn_cfg, attn_inputs, par_cfg),
            PCPFusedPagedAttnOp,
        )

    def test_dispatch_falls_back_for_unsupported_capabilities(self):
        # Every entry also pins *which* guard rejects the geometry. Asserting only
        # the returned class would stay green even if a case tripped an unintended
        # check, leaving the guard it was written for unexercised.
        cases = []

        attn_cfg, par_cfg, attn_inputs = self._make_dispatch_case()
        attn_cfg.kv_cache_dtype = KvCacheDataType.FP8
        cases.append(("FP8 KV cache", "KV cache dtype", attn_cfg, par_cfg, attn_inputs))

        attn_cfg, par_cfg, attn_inputs = self._make_dispatch_case()
        attn_cfg.kernel_tokens_per_block = 32
        cases.append(
            (
                "mismatched page sizes",
                "kernel_tokens_per_block",
                attn_cfg,
                par_cfg,
                attn_inputs,
            )
        )

        attn_cfg, par_cfg, attn_inputs = self._make_dispatch_case()
        attn_inputs.kv_cache_kernel_block_id[0, 1] = -1
        cases.append(
            (
                "unallocated page id",
                "unallocated KV cache pages",
                attn_cfg,
                par_cfg,
                attn_inputs,
            )
        )

        attn_cfg, par_cfg, attn_inputs = self._make_dispatch_case()
        attn_inputs.kv_cache_kernel_block_id = (
            attn_inputs.kv_cache_kernel_block_id.unsqueeze(0)
        )
        cases.append(
            (
                "3-D block table",
                "2-D paged KV block table",
                attn_cfg,
                par_cfg,
                attn_inputs,
            )
        )

        attn_cfg, par_cfg, attn_inputs = self._make_dispatch_case()
        attn_inputs.context_parallel_info.prefill_cp_chunk_lengths = torch.tensor(
            [0], dtype=torch.int32
        )
        attn_inputs.context_parallel_info.prefill_actual_input_lengths_cpu = (
            torch.tensor([0], dtype=torch.int32)
        )
        attn_inputs.context_parallel_info.prefill_qkv_restore_indice = torch.empty(
            0, dtype=torch.int32, device=self.device
        )
        cases.append(
            (
                "empty query batch",
                "at least one query token",
                attn_cfg,
                par_cfg,
                attn_inputs,
            )
        )

        # page_size 12 with cp_size 4: 12 % 8 != 0, so rounding a request up to a
        # multiple of 2 * cp_size could spill past the last page it owns -- and
        # request 1 here owns the very next block, which is what such a spill would
        # corrupt. This divisibility guard is what makes that spill unreachable:
        # once page_size % (2 * cp_size) == 0 and prefixes are page-aligned,
        # roundup(actual, 2 * cp_size) always lands inside the pages already
        # allocated for `actual`, so the later "padding needs N pages but only M are
        # allocated" check can never fire (verified by exhaustive search over
        # cp_size, page_size, prefix and length). It is kept as defence in depth.
        cp_size = 4
        attn_cfg, par_cfg = make_configs(
            tokens_per_block=12, cp_size=cp_size, cp_rank=0
        )
        actual_length = 12
        padded_length = math.ceil(actual_length / (2 * cp_size)) * 2 * cp_size
        attn_inputs = build_cp_attn_inputs(
            [actual_length, actual_length],
            [padded_length // cp_size, 0],
            cp_size,
            attn_cfg.tokens_per_block,
            prefix_lengths=[0, actual_length],
            device=self.device,
        )
        par_cfg.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        par_cfg.prefill_cp_config.all_gather_impl = CPAllGatherImpl.FUSED
        cases.append(
            (
                "page size not divisible by 2 * CP size",
                "is not divisible by 2 * CP size",
                attn_cfg,
                par_cfg,
                attn_inputs,
            )
        )

        for name, expected_reason, case_cfg, case_par_cfg, case_inputs in cases:
            with self.subTest(name=name):
                supported, reason = PCPFusedPagedAttnOp.can_run(
                    case_cfg, case_inputs, case_par_cfg
                )
                self.assertFalse(supported)
                self.assertIn(expected_reason, reason)
                self.assertIs(
                    select_prefill_cp_impl(case_cfg, case_inputs, case_par_cfg),
                    PCPAllGatherAttnOp,
                )

    def _make_dispatch_case(self):
        cp_size = 2
        attn_cfg, par_cfg = make_configs(
            tokens_per_block=16, cp_size=cp_size, cp_rank=0
        )
        attn_cfg.kv_cache_dtype = KvCacheDataType.BASE
        par_cfg.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        attn_inputs = build_cp_attn_inputs(
            [20],
            [10],
            cp_size,
            attn_cfg.tokens_per_block,
            device=self.device,
        )
        return attn_cfg, par_cfg, attn_inputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
