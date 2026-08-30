#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"

namespace rtp_llm {
namespace test {

TEST(LinearKVCacheGroupPartitionTest, PartitionEightSlicesEveryKdaHeadSegment) {
    auto spec                 = std::make_shared<LinearKVCacheSpec>();
    spec->type               = KVCacheSpecType::LinearAttention;
    spec->dtype              = DataType::TYPE_FP16;
    spec->layer_num          = 1;
    spec->local_num_k_heads  = 8;
    spec->local_num_v_heads  = 8;
    spec->local_head_num_kv  = 8;
    spec->head_k_dim         = 2;
    spec->head_v_dim         = 2;
    spec->conv_kernel_dim    = 3;
    spec->ssm_state_dtype    = DataType::TYPE_FP32;
    spec->conv_state_dtype   = DataType::TYPE_FP16;
    spec->seq_size_per_block = 4;

    const size_t block_stride = spec->block_size_bytes();
    auto pool_config = BlockPoolConfigHelper::createConfig(
        /*layer_num=*/1, /*block_num=*/2, block_stride, DataType::TYPE_FP16);
    auto block_pool = std::make_shared<BlockPool>(pool_config, AllocationType::HOST);
    ASSERT_TRUE(block_pool->init());

    LinearKVCacheGroup group(/*layer_ids=*/{0}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());
    auto allocated = block_pool->malloc(1);
    ASSERT_EQ(allocated.size(), 1u);

    auto whole = block_pool->convertIndexToBuffer(/*layer_id=*/0, allocated[0]);
    ASSERT_EQ(whole.size(), 1u);
    auto* base = static_cast<char*>(whole[0].addr);

    constexpr int kPartitions = 8;
    const size_t ssm_bytes = spec->k_block_size_bytes();
    const size_t q_bytes = static_cast<size_t>(spec->local_num_k_heads)
                           * spec->head_k_dim * getTypeSize(spec->conv_state_dtype);
    const size_t k_bytes = q_bytes;
    const size_t v_bytes = static_cast<size_t>(spec->local_num_v_heads)
                           * spec->head_v_dim * getTypeSize(spec->conv_state_dtype);
    const size_t history_stride = q_bytes + k_bytes + v_bytes;

    for (int partition = 0; partition < kPartitions; ++partition) {
        auto parts = group.convertIndexToBuffer(
            /*layer_id=*/0, allocated[0], kPartitions, partition);
        ASSERT_EQ(parts.size(), 7u);  // SSM + two histories * Q/K/V.
        EXPECT_EQ(parts[0].size_bytes, ssm_bytes / kPartitions);
        EXPECT_EQ(static_cast<char*>(parts[0].addr) - base,
                  partition * static_cast<int64_t>(ssm_bytes / kPartitions));
        for (size_t history = 0; history < 2; ++history) {
            const size_t segment = 1 + history * 3;
            const size_t history_base = ssm_bytes + history * history_stride;
            EXPECT_EQ(static_cast<char*>(parts[segment].addr) - base,
                      static_cast<int64_t>(history_base + partition * (q_bytes / kPartitions)));
            EXPECT_EQ(static_cast<char*>(parts[segment + 1].addr) - base,
                      static_cast<int64_t>(history_base + q_bytes
                                           + partition * (k_bytes / kPartitions)));
            EXPECT_EQ(static_cast<char*>(parts[segment + 2].addr) - base,
                      static_cast<int64_t>(history_base + q_bytes + k_bytes
                                           + partition * (v_bytes / kPartitions)));
        }
    }
    block_pool->requestFree(allocated);
}

}  // namespace test
}  // namespace rtp_llm
