import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import sparse_paged_gqa_attn

_DEVICE = "cuda"


class SparsePagedIndexValidationTest(unittest.TestCase):
    def test_decode_plan_reads_one_verdict_and_fmha_reuses_exact_operands(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.modules.qwen4_exp.qsa.fmha_inputs import (
            _prepare_paged_inputs,
        )

        table = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        cache = torch.randn((3, 2, 1, 4, 8), device="cuda", dtype=torch.bfloat16)
        inputs = SimpleNamespace(
            sequence_lengths=torch.tensor([4], device="cpu", dtype=torch.int32),
            is_cuda_graph=False,
            is_s_padded=False,
            kv_cache_kernel_block_id_device=table,
        )
        owner = SimpleNamespace(
            is_target_verify=False,
            is_prefill=False,
            attn_inputs=inputs,
            q_width=8,
            kv_width=8,
            kv_head_num=1,
            head_dim=8,
            attn_configs=SimpleNamespace(kernel_tokens_per_block=4),
        )
        qkv = torch.zeros((1, 24), device="cuda", dtype=torch.bfloat16)
        selected = torch.tensor([[0, 4, -1]], device="cuda", dtype=torch.int32)
        original_item = torch.Tensor.item
        reads = []

        def count_gpu_item(tensor, *args, **kwargs):
            if tensor.is_cuda:
                reads.append(1)
            return original_item(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "item", count_gpu_item):
            plan = _prepare_paged_inputs(
                owner, qkv, SimpleNamespace(kv_cache_base=cache), selected
            )
        self.assertEqual(len(reads), 1)
        q = torch.zeros((1, 1, 1, 8), device="cuda", dtype=torch.bfloat16)
        expected = sparse_paged_gqa_attn(
            q, cache, table, plan.kv_lens, plan.selected_indices, page_size=4
        )
        with patch(
            "rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha._validate_sparse_paged_indices",
            side_effect=AssertionError("decode repeated validation"),
        ):
            actual = sparse_paged_gqa_attn(
                q,
                cache,
                table,
                plan.kv_lens,
                plan.selected_indices,
                page_size=4,
                _validated_plan=plan,
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "does not own"):
            sparse_paged_gqa_attn(
                q,
                cache,
                table,
                plan.kv_lens,
                plan.selected_indices.clone(),
                page_size=4,
                _validated_plan=plan,
            )
        table[0, 1] = 0
        with self.assertRaisesRegex(RuntimeError, "unallocated"):
            _prepare_paged_inputs(
                owner, qkv, SimpleNamespace(kv_cache_base=cache), selected
            )


if __name__ == "__main__":
    unittest.main()
