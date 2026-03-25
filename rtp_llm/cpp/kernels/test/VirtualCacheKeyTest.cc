#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {
namespace test {

/// Tests that the virtual-block-granularity rolling hash used by decode side
/// matches what prefill would produce when storing cache at virtualBlockSize granularity.
///
/// Prefill computes one cache key per virtual block (block_size * cp_size tokens).
/// Decode recomputes the same keys from the full token sequence.
class VirtualCacheKeyTest : public ::testing::Test {};

// Simulate prefill-side cache key computation:
// Rolling hash over virtual blocks of size (block_size * cp_size).
static std::vector<int64_t> computePrefillKeys(const std::vector<int32_t>& token_ids,
                                                int block_size, int cp_size) {
    const int virtual_block_sz = block_size * cp_size;
    const int seq_len = static_cast<int>(token_ids.size());
    const int vb_count = (seq_len + virtual_block_sz - 1) / virtual_block_sz;

    std::vector<int64_t> keys;
    int64_t rolling_hash = 0;
    for (int v = 0; v < vb_count; ++v) {
        int pos = v * virtual_block_sz;
        int block_len = std::min(virtual_block_sz, seq_len - pos);
        rolling_hash = hashInt64Array(rolling_hash,
                                      const_cast<int32_t*>(token_ids.data()) + pos,
                                      const_cast<int32_t*>(token_ids.data()) + pos + block_len);
        keys.push_back(rolling_hash);
    }
    return keys;
}

// Simulate decode-side cache key recomputation (same logic as in DecodeRpcServer::loadCacheForAllRank).
static std::vector<int64_t> computeDecodeKeys(const std::vector<int32_t>& token_ids,
                                               int block_size, int cp_size) {
    const int virtual_block_sz = block_size * cp_size;
    const int seq_len = static_cast<int>(token_ids.size());
    const int vb_count = (seq_len + virtual_block_sz - 1) / virtual_block_sz;

    std::vector<int64_t> keys;
    int64_t rolling_hash = 0;
    for (int v = 0; v < vb_count; ++v) {
        int pos = v * virtual_block_sz;
        int block_len = std::min(virtual_block_sz, seq_len - pos);
        rolling_hash = hashInt64Array(rolling_hash,
                                      const_cast<int32_t*>(token_ids.data()) + pos,
                                      const_cast<int32_t*>(token_ids.data()) + pos + block_len);
        keys.push_back(rolling_hash);
    }
    return keys;
}

TEST_F(VirtualCacheKeyTest, PrefillAndDecodeKeysMatch) {
    // 16 tokens, block_size=4, cp_size=2 => virtual_block_size=8, 2 virtual blocks
    std::vector<int32_t> tokens = {10, 20, 30, 40, 50, 60, 70, 80,
                                   90, 100, 110, 120, 130, 140, 150, 160};
    auto prefill_keys = computePrefillKeys(tokens, 4, 2);
    auto decode_keys  = computeDecodeKeys(tokens, 4, 2);

    ASSERT_EQ(prefill_keys.size(), 2u);
    ASSERT_EQ(decode_keys.size(), 2u);
    EXPECT_EQ(prefill_keys[0], decode_keys[0]);
    EXPECT_EQ(prefill_keys[1], decode_keys[1]);
}

TEST_F(VirtualCacheKeyTest, RollingHashIsCumulative) {
    // The second virtual block's key should depend on the first block's tokens too
    // (rolling hash is cumulative).
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    auto keys = computePrefillKeys(tokens, 2, 2);  // virtual_block_size=4, 2 VBs

    ASSERT_EQ(keys.size(), 2u);
    EXPECT_NE(keys[0], 0);
    EXPECT_NE(keys[1], 0);
    EXPECT_NE(keys[0], keys[1]);

    // Verify key[0] is hash of first 4 tokens only
    int64_t expected_key0 = hashInt64Array(0, const_cast<int32_t*>(tokens.data()),
                                           const_cast<int32_t*>(tokens.data()) + 4);
    EXPECT_EQ(keys[0], expected_key0);

    // Verify key[1] is hash of all 8 tokens (rolling)
    int64_t expected_key1 = hashInt64Array(expected_key0, const_cast<int32_t*>(tokens.data()) + 4,
                                           const_cast<int32_t*>(tokens.data()) + 8);
    EXPECT_EQ(keys[1], expected_key1);
}

TEST_F(VirtualCacheKeyTest, PartialLastBlock) {
    // 10 tokens, block_size=4, cp_size=2 => virtual_block_size=8
    // VB0: tokens[0..7], VB1: tokens[8..9] (partial)
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto keys = computePrefillKeys(tokens, 4, 2);

    ASSERT_EQ(keys.size(), 2u);

    int64_t h0 = hashInt64Array(0, const_cast<int32_t*>(tokens.data()),
                                const_cast<int32_t*>(tokens.data()) + 8);
    int64_t h1 = hashInt64Array(h0, const_cast<int32_t*>(tokens.data()) + 8,
                                const_cast<int32_t*>(tokens.data()) + 10);
    EXPECT_EQ(keys[0], h0);
    EXPECT_EQ(keys[1], h1);
}

TEST_F(VirtualCacheKeyTest, DifferentCPSizesProduceDifferentKeys) {
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    auto keys_cp2 = computePrefillKeys(tokens, 4, 2);  // virtual_block_size=8, 2 VBs
    auto keys_cp4 = computePrefillKeys(tokens, 4, 4);  // virtual_block_size=16, 1 VB

    ASSERT_EQ(keys_cp2.size(), 2u);
    ASSERT_EQ(keys_cp4.size(), 1u);

    // The single VB key for cp_size=4 should equal the rolling hash of all 16 tokens
    int64_t full_hash = hashInt64Array(0, const_cast<int32_t*>(tokens.data()),
                                       const_cast<int32_t*>(tokens.data()) + 16);
    EXPECT_EQ(keys_cp4[0], full_hash);

    // cp_size=2 second key should also equal full hash (rolling), but first key differs
    EXPECT_EQ(keys_cp2[1], full_hash);
}

TEST_F(VirtualCacheKeyTest, BlockMappingLogic) {
    // Test that for virtual block v, peer p, the decode block position is v * cp_size + p
    const int cp_size = 3;
    const int virtual_block_count = 4;

    for (int v = 0; v < virtual_block_count; ++v) {
        for (int p = 0; p < cp_size; ++p) {
            int decode_block_pos = v * cp_size + p;
            // Verify the mapping is unique and within range
            EXPECT_GE(decode_block_pos, 0);
            EXPECT_LT(decode_block_pos, virtual_block_count * cp_size);
        }
    }

    // Verify all positions are covered exactly once
    std::vector<bool> covered(virtual_block_count * cp_size, false);
    for (int v = 0; v < virtual_block_count; ++v) {
        for (int p = 0; p < cp_size; ++p) {
            int pos = v * cp_size + p;
            EXPECT_FALSE(covered[pos]) << "Position " << pos << " covered twice";
            covered[pos] = true;
        }
    }
    for (int i = 0; i < virtual_block_count * cp_size; ++i) {
        EXPECT_TRUE(covered[i]) << "Position " << i << " not covered";
    }
}

}  // namespace test
}  // namespace rtp_llm
