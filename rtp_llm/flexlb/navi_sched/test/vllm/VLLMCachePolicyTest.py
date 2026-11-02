import unittest
from types import SimpleNamespace

from navi_sched.adapters.vllm.VLLMCachePolicy import (
    VLLMCacheDelta,
    VLLMCachePolicy,
    VLLMCacheState,
    get_vllm_hash_block_size,
)


class FullAttentionSpec:
    def __init__(self, block_size):
        self.block_size = block_size


class MambaSpec:
    def __init__(self, block_size, mode="align"):
        self.block_size = block_size
        self.mamba_cache_mode = mode


class SlidingWindowSpec:
    def __init__(self, block_size):
        self.block_size = block_size


def make_vllm_config(block_size=16, dcp_size=1, pcp_size=1):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=pcp_size,
        ),
    )


def make_kv_cache_config(*specs, num_blocks=1024):
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs],
    )


def block_hash(value):
    return bytes([value]) * 32


class VLLMCachePolicyTest(unittest.TestCase):
    def test_single_full_attention_group_uses_flat_cache(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(),
            make_kv_cache_config(FullAttentionSpec(16), num_blocks=1000),
        )

        self.assertEqual(16, policy.block_size)
        self.assertEqual(1000, policy.total_blocks)
        self.assertEqual(1, policy.linear_step)
        self.assertFalse(policy.hybrid_tail_required)

        state = VLLMCacheState(policy)
        value = block_hash(1)
        self.assertEqual(
            VLLMCacheDelta(added=(value,)),
            state.apply("stored", 0, (value,)),
        )
        self.assertEqual(
            VLLMCacheDelta(removed=(value,)),
            state.apply("removed", 0, (value,)),
        )

    def test_all_full_attention_groups_must_contain_a_hash(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(),
            make_kv_cache_config(
                FullAttentionSpec(16),
                FullAttentionSpec(16),
                num_blocks=2000,
            ),
        )
        state = VLLMCacheState(policy)
        value = block_hash(2)

        self.assertEqual(VLLMCacheDelta(), state.apply("stored", 0, (value,)))
        self.assertEqual(
            VLLMCacheDelta(added=(value,)),
            state.apply("stored", 1, (value,)),
        )
        self.assertEqual(1000, policy.total_blocks)

    def test_context_parallelism_defines_hash_block_size(self):
        config = make_vllm_config(block_size=16, dcp_size=2, pcp_size=2)
        policy = VLLMCachePolicy.from_engine_config(
            config,
            make_kv_cache_config(FullAttentionSpec(16)),
        )

        self.assertEqual(64, get_vllm_hash_block_size(config))
        self.assertEqual(64, policy.block_size)
        self.assertEqual((1,), policy.group_block_scales)

    def test_larger_group_blocks_are_split_into_hash_blocks(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(),
            make_kv_cache_config(FullAttentionSpec(32)),
        )
        state = VLLMCacheState(policy)
        first = block_hash(3)
        second = block_hash(4)

        self.assertEqual(
            VLLMCacheDelta(added=(first, second)),
            state.apply("stored", 0, (first + second,)),
        )

    def test_full_attention_and_mamba_map_to_full_and_tail_readiness(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(),
            make_kv_cache_config(
                FullAttentionSpec(16),
                MambaSpec(64),
                num_blocks=2000,
            ),
        )
        state = VLLMCacheState(policy)
        hashes = tuple(block_hash(value) for value in range(1, 5))

        self.assertEqual(1000, policy.total_blocks)
        self.assertEqual(4, policy.linear_step)
        self.assertTrue(policy.hybrid_tail_required)
        self.assertEqual(
            VLLMCacheDelta(added_full_ready=(hashes[3],)),
            state.apply("stored", 0, (hashes[3],)),
        )
        self.assertEqual(
            VLLMCacheDelta(added_tail_ready=(hashes[3],)),
            state.apply("stored", 1, (b"".join(hashes),)),
        )
        self.assertEqual(
            VLLMCacheDelta(
                removed_full_ready=(hashes[3],),
                removed_tail_ready=(hashes[3],),
            ),
            state.apply("cleared", None, ()),
        )

    def test_duplicate_hashes_are_reference_counted(self):
        policy = VLLMCachePolicy.from_engine_config(
            make_vllm_config(),
            make_kv_cache_config(FullAttentionSpec(16)),
        )
        state = VLLMCacheState(policy)
        value = block_hash(5)

        self.assertEqual(
            VLLMCacheDelta(added=(value,)),
            state.apply("stored", 0, (value, value)),
        )
        self.assertEqual(VLLMCacheDelta(), state.apply("removed", 0, (value,)))
        self.assertEqual(
            VLLMCacheDelta(removed=(value,)),
            state.apply("removed", 0, (value,)),
        )

    def test_unsupported_hybrid_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot exactly represent"):
            VLLMCachePolicy.from_engine_config(
                make_vllm_config(),
                make_kv_cache_config(SlidingWindowSpec(16)),
            )

    def test_mamba_only_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Mamba-only"):
            VLLMCachePolicy.from_engine_config(
                make_vllm_config(),
                make_kv_cache_config(MambaSpec(16)),
            )


if __name__ == "__main__":
    unittest.main()
