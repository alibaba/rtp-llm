#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"

namespace rtp_llm {
namespace test {

class LinearKVCacheGroupPartitionTest: public ::testing::TestWithParam<int> {};

TEST_P(LinearKVCacheGroupPartitionTest, PartitionSlicesEveryKdaHeadSegment) {
    auto spec                = std::make_shared<LinearKVCacheSpec>();
    spec->type               = KVCacheSpecType::LinearAttention;
    spec->dtype              = DataType::TYPE_FP16;
    spec->layer_num          = 1;
    spec->local_num_k_heads  = 96;
    spec->local_num_v_heads  = 96;
    spec->local_head_num_kv  = 96;
    spec->head_k_dim         = 2;
    spec->head_v_dim         = 2;
    spec->conv_kernel_dim    = 3;
    spec->ssm_state_dtype    = DataType::TYPE_FP32;
    spec->conv_state_dtype   = DataType::TYPE_FP16;
    spec->seq_size_per_block = 4;

    const size_t block_stride = spec->block_size_bytes();
    auto         pool_config  = BlockPoolConfigHelper::createConfig(
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

    const int partitions = GetParam();
    const size_t ssm_bytes = spec->k_block_size_bytes();
    const size_t q_bytes =
        static_cast<size_t>(spec->local_num_k_heads) * spec->head_k_dim * getTypeSize(spec->conv_state_dtype);
    const size_t k_bytes = q_bytes;
    const size_t v_bytes =
        static_cast<size_t>(spec->local_num_v_heads) * spec->head_v_dim * getTypeSize(spec->conv_state_dtype);
    const size_t history_stride = q_bytes + k_bytes + v_bytes;

    const std::array<size_t, 7> segment_bases = {
        0,
        ssm_bytes,
        ssm_bytes + q_bytes,
        ssm_bytes + q_bytes + k_bytes,
        ssm_bytes + history_stride,
        ssm_bytes + history_stride + q_bytes,
        ssm_bytes + history_stride + q_bytes + k_bytes,
    };
    const std::array<size_t, 7> segment_bytes = {
        ssm_bytes,
        q_bytes,
        k_bytes,
        v_bytes,
        q_bytes,
        k_bytes,
        v_bytes,
    };
    std::array<size_t, 7> covered_bytes{};

    for (int partition = 0; partition < partitions; ++partition) {
        const auto parts = group.convertIndexToBuffer(
            /*layer_id=*/0, allocated[0], partitions, partition);
        ASSERT_EQ(parts.size(), 7u);  // SSM + two histories * Q/K/V.
        for (size_t segment = 0; segment < parts.size(); ++segment) {
            const size_t partition_bytes = segment_bytes[segment] / static_cast<size_t>(partitions);
            EXPECT_EQ(parts[segment].size_bytes, partition_bytes);
            EXPECT_EQ(
                static_cast<char*>(parts[segment].addr) - base,
                static_cast<int64_t>(segment_bases[segment] + static_cast<size_t>(partition) * partition_bytes));
            covered_bytes[segment] += parts[segment].size_bytes;
        }
    }
    for (size_t segment = 0; segment < segment_bytes.size(); ++segment) {
        EXPECT_EQ(covered_bytes[segment], segment_bytes[segment]);
    }
    block_pool->requestFree(allocated);
}

INSTANTIATE_TEST_SUITE_P(PrefillTP, LinearKVCacheGroupPartitionTest, ::testing::Values(8, 16));

}  // namespace test
}  // namespace rtp_llm
