import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.fla.block import store_ssm_state_to_block_map
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.chunk_cache_store import (
    chunk_gated_delta_rule_with_cache_store,
)
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ChunkCacheStoreTest(unittest.TestCase):
    def _check_mixed_batch(self, input_lengths, max_tokens, expected_calls):
        torch.manual_seed(417)
        total = sum(input_lengths)
        batch_size = len(input_lengths)
        key_heads, value_heads, dim = 2, 4, 128
        q = torch.randn(1, total, key_heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, total, value_heads, dim, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(1, total, value_heads, device="cuda") * 0.1
        beta = torch.rand_like(g).to(torch.bfloat16)
        initial = (
            torch.randn(
                batch_size, value_heads, dim, dim, device="cuda", dtype=torch.bfloat16
            )
            * 0.01
        )
        offsets = [0]
        for length in input_lengths:
            offsets.append(offsets[-1] + length)
        lengths_cpu = torch.tensor(offsets, dtype=torch.int32)
        lengths = lengths_cpu.cuda()
        prefix = torch.full((batch_size,), 128, dtype=torch.int32, device="cuda")
        nonempty = [batch for batch, length in enumerate(input_lengths) if length]
        selected = torch.tensor(nonempty, dtype=torch.long, device="cuda")
        compact = [0]
        for batch in nonempty:
            compact.append(compact[-1] + input_lengths[batch])
        compact_lengths = torch.tensor(compact, dtype=torch.int32, device="cuda")
        # Normalize with the old per-sequence/per-tile dispatch, then use one
        # independent full-varlen recurrence as the output/cache reference.
        normalized = []
        for tensor in (q, k):
            normalized.append(
                torch.cat(
                    [
                        l2norm_fwd(tensor[:, start : min(start + max_tokens, end)])
                        for begin, end in zip(offsets, offsets[1:])
                        for start in range(begin, end, max_tokens)
                    ],
                    dim=1,
                )
            )
        for cache_block_size in (64, 128):
            with self.subTest(cache_block_size=cache_block_size):
                columns = (max(input_lengths) + 128) // cache_block_size + 2
                table_cpu = torch.zeros(batch_size, columns + 1, dtype=torch.int32)
                next_block = 1
                for batch, length in enumerate(input_lengths):
                    count = (length + 128 + cache_block_size - 1) // cache_block_size
                    table_cpu[batch, :count] = torch.arange(
                        next_block, next_block + count
                    )
                    next_block += count
                table = table_cpu.cuda()[:, :columns]
                blocks = next_block - 1
                table[:, 2] = 0
                state_elements = value_heads * dim * dim
                backing = torch.randn(
                    blocks + 1,
                    state_elements + 256,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                reference_backing = backing.clone()
                states = backing[:, :state_elements].view(
                    blocks + 1, value_heads, dim, dim
                )
                reference_states = reference_backing[:, :state_elements].view_as(states)
                expected, h, final = chunk_gated_delta_rule(
                    *normalized,
                    v,
                    g,
                    beta,
                    initial_state=initial.index_select(0, selected),
                    output_final_state=True,
                    cu_seqlens=compact_lengths,
                    use_qk_l2norm_in_kernel=False,
                )
                store_ssm_state_to_block_map(
                    h,
                    final,
                    prefix.index_select(0, selected),
                    compact_lengths,
                    table.index_select(0, selected),
                    reference_states,
                    cache_block_size,
                    chunk_size=64,
                )
                tensor_tolist = torch.Tensor.tolist

                def cpu_tolist(tensor):
                    self.assertEqual(
                        tensor.device.type, "cpu", "unexpected GPU length readback"
                    )
                    return tensor_tolist(tensor)

                with patch.object(torch.Tensor, "tolist", cpu_tolist), patch(
                    "rtp_llm.models_py.triton_kernels.fla.chunk_cache_store.chunk_gated_delta_rule",
                    wraps=chunk_gated_delta_rule,
                ) as recurrent:
                    actual = chunk_gated_delta_rule_with_cache_store(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        initial,
                        lengths,
                        prefix,
                        table,
                        states,
                        cache_block_size,
                        max_tokens=max_tokens,
                        cu_seqlens_cpu=lengths_cpu,
                    )
                self.assertEqual(
                    recurrent.call_count,
                    expected_calls,
                    "short requests must stay batched within the chunk budget",
                )
                for call in recurrent.call_args_list:
                    cpu_lengths = call.kwargs.get("cu_seqlens_cpu")
                    if cpu_lengths is None:
                        chunks = (call.args[0].shape[1] + 63) // 64
                    else:
                        chunks = sum(
                            (length + 63) // 64
                            for length in (cpu_lengths[1:] - cpu_lengths[:-1]).tolist()
                        )
                    self.assertLessEqual(chunks * 64, max_tokens)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(
                    torch.equal(
                        backing.view(torch.uint8), reference_backing.view(torch.uint8)
                    )
                )

    def test_mixed_lengths_batch_short_requests_across_a_long_sequence(self):
        # Short requests on both sides of the long one must be compacted and
        # restored to their original token/state/cache rows, including empties.
        self._check_mixed_batch([0, 17, 1025, 64, 63, 0, 129, 1], 1024, 4)

    def test_batched_short_requests_preserve_small_normalization_dispatch(self):
        # Two key heads put each 64-token request on the small L2 kernel, but
        # their combined batch would normally select different BF16 arithmetic.
        self._check_mixed_batch([65537] + [64] * 32, 65536, 3)

    def test_short_batch_budget_counts_partial_chunks(self):
        # Five one-token requests need three bounded batches, not one batch
        # whose hidden-state allocation exceeds the two-chunk tile budget.
        self._check_mixed_batch([129, 1, 1, 1, 1, 1], 128, 5)

    def test_varlen_tiles_preserve_outputs_and_cache_without_device_readback(self):
        torch.manual_seed(417)
        # Include an empty sequence, partial chunks, nonzero prefixes, sparse
        # block ids, a strided block table and conv bytes beside each SSM block.
        for cache_block_size in (64, 128):
            with self.subTest(cache_block_size=cache_block_size):
                device = "cuda"
                total, key_heads, value_heads, dim = 457, 8, 24, 128
                q = torch.randn(
                    1, total, key_heads, dim, device=device, dtype=torch.bfloat16
                )
                k = torch.randn_like(q)
                v = torch.randn(
                    1, total, value_heads, dim, device=device, dtype=torch.bfloat16
                )
                g = -torch.rand(1, total, value_heads, device=device) * 0.1
                beta = torch.rand_like(g).to(torch.bfloat16)
                initial = (
                    torch.randn(
                        3, value_heads, dim, dim, device=device, dtype=torch.bfloat16
                    )
                    * 0.01
                )
                lengths = torch.tensor(
                    [0, 0, 137, total], device=device, dtype=torch.int32
                )
                prefix = torch.tensor([0, 128, 256], device=device, dtype=torch.int32)
                table = torch.arange(1, 49, device=device, dtype=torch.int32).reshape(
                    3, 16
                )[:, :10]
                table[:, 2] = 0
                state_elements = value_heads * dim * dim
                backing = torch.randn(
                    49, state_elements + 256, device=device, dtype=torch.bfloat16
                )
                reference_backing = backing.clone()
                states = backing[:, :state_elements].view(49, value_heads, dim, dim)
                reference_states = reference_backing[:, :state_elements].view_as(states)
                # The existing varlen index helper assumes nonempty sequences.
                # Compare against its valid compact batch while also checking
                # that the new wrapper safely skips an empty sequence.
                compact_lengths = lengths[1:]
                expected, h, final = chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=initial[1:],
                    output_final_state=True,
                    cu_seqlens=compact_lengths,
                    use_qk_l2norm_in_kernel=True,
                )
                lengths_cpu = lengths.cpu()
                tensor_tolist = torch.Tensor.tolist

                def cpu_tolist(tensor):
                    self.assertEqual(
                        tensor.device.type,
                        "cpu",
                        "tile metadata must not read lengths back from the GPU",
                    )
                    return tensor_tolist(tensor)

                # Keep the real recurrent and cache-store kernels: a fresh GPU
                # length tensor passed to either varlen path must fail this guard.
                with patch.object(torch.Tensor, "tolist", cpu_tolist):
                    # The default varlen path must reuse the recurrent kernel's
                    # cached indices even when the CPU mirror argument is omitted.
                    store_ssm_state_to_block_map(
                        h,
                        final,
                        prefix[1:],
                        compact_lengths,
                        table[1:],
                        reference_states,
                        cache_block_size,
                        chunk_size=64,
                    )
                    actual = chunk_gated_delta_rule_with_cache_store(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        initial,
                        lengths,
                        prefix,
                        table,
                        states,
                        cache_block_size,
                        max_tokens=128,
                        cu_seqlens_cpu=lengths_cpu,
                    )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(
                    torch.equal(
                        backing.view(torch.uint8), reference_backing.view(torch.uint8)
                    )
                )


if __name__ == "__main__":
    unittest.main()
