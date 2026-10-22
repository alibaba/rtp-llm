import unittest

import torch

from rtp_llm.models_py.triton_kernels.fla.block import store_ssm_state_to_block_map
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.chunk_cache_store import (
    chunk_gated_delta_rule_with_cache_store,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ChunkCacheStoreTest(unittest.TestCase):
    def test_varlen_tiles_preserve_outputs_and_all_cache_bytes(self):
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
                    cu_seqlens_cpu=lengths.cpu(),
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(
                    torch.equal(
                        backing.view(torch.uint8), reference_backing.view(torch.uint8)
                    )
                )


if __name__ == "__main__":
    unittest.main()
