#include "gtest/gtest.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"

namespace rtp_llm {
namespace {

CacheGroup makeGroup(std::string tag, CacheGroupType type, uint32_t block_num, uint32_t physical_b, uint32_t kernel_b) {
    auto spec                       = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block        = physical_b;
    spec->kernel_seq_size_per_block = kernel_b;

    CacheGroup group;
    group.tag               = std::move(tag);
    group.spec              = std::move(spec);
    group.policy.group_type = type;
    group.block_num         = block_num;
    return group;
}

BlockPoolConfig makeHostBlockPoolConfig(uint32_t block_num) {
    constexpr uint32_t kLayerNum     = 1;
    constexpr size_t   kBlockStride  = 1024;
    constexpr size_t   kHalfKvStride = kBlockStride / 2;

    MemoryLayoutConfig layout;
    layout.layer_num                = kLayerNum;
    layout.block_num                = block_num;
    layout.dtype                    = DataType::TYPE_FP16;
    layout.kv_cache_offset_bytes    = 0;
    layout.kv_scale_offset_bytes    = kLayerNum * block_num * kBlockStride;
    layout.kv_block_stride_bytes    = kBlockStride;
    layout.k_block_stride_bytes     = kHalfKvStride;
    layout.v_block_stride_bytes     = kHalfKvStride;
    layout.kv_block_pool_size_bytes = kLayerNum * block_num * kBlockStride;
    layout.kv_scale_pool_size_bytes = 0;
    layout.total_size_bytes         = layout.kv_block_pool_size_bytes;

    BlockPoolConfig config;
    config.block_num        = block_num;
    config.total_size_bytes = layout.total_size_bytes;
    config.memory_layouts   = {layout};
    return config;
}

TEST(CacheBlockSafeDummyTest, SafeDummyBlockIsPhysicalBlockZeroReservedByBlockPool) {
    constexpr uint32_t kBlockNum = 4;
    const auto         group     = makeGroup("full", CacheGroupType::FULL, kBlockNum, /*physical_b=*/8, /*kernel_b=*/2);

    EXPECT_EQ(cudaGraphSafePhysicalBlockId(group), 0);
    EXPECT_EQ(cudaGraphSafeKernelBlockId(group), 0);

    BlockPool block_pool(makeHostBlockPoolConfig(kBlockNum), AllocationType::HOST);
    ASSERT_TRUE(block_pool.init());
    ASSERT_EQ(block_pool.freeBlocksNum(), kBlockNum - 1);
    const auto allocated = block_pool.malloc(kBlockNum - 1);
    ASSERT_EQ(allocated.size(), kBlockNum - 1);
    EXPECT_EQ(std::find(allocated.begin(), allocated.end(), 0), allocated.end());
}

TEST(CacheBlockSafeDummyTest, RejectsGroupWithoutReservedPhysicalBlock) {
    const auto group = makeGroup("full", CacheGroupType::FULL, /*block_num=*/0, /*physical_b=*/8, /*kernel_b=*/2);
    EXPECT_ANY_THROW(cudaGraphSafePhysicalBlockId(group));
    EXPECT_ANY_THROW(cudaGraphSafeKernelBlockId(group));
}

}  // namespace
}  // namespace rtp_llm
