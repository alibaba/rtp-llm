#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace rtp_llm::block_tree_cache {
namespace {

std::shared_ptr<HostBlockPoolConfig> makeConfig(size_t physical_block_count = 4,
                                                 size_t payload_bytes        = 1024,
                                                 size_t stride_bytes         = 4096,
                                                 bool   enable_pinned        = false,
                                                 size_t alignment            = 4096) {
    auto config                    = std::make_shared<HostBlockPoolConfig>();
    config->pool_type              = BlockPoolType::HOST;
    config->pool_name              = "host";
    config->physical_block_count   = physical_block_count;
    config->free_block_order_policy = FreeBlockOrderPolicy::ANY_ORDER;
    config->payload_bytes          = payload_bytes;
    config->stride_bytes           = stride_bytes;
    config->enable_pinned          = enable_pinned;
    config->alignment              = alignment;
    return config;
}

}  // namespace

TEST(HostBlockPoolTest, InitAllocatesHostBuffersAndSkipsBlockZero) {
    auto          config = makeConfig();
    HostBlockPool pool(config);

    ASSERT_TRUE(pool.init());
    EXPECT_FALSE(pool.isPinned());
    EXPECT_EQ(pool.totalBlocksNum(), 3u);
    EXPECT_EQ(pool.payloadBytes(), 1024u);
    EXPECT_EQ(pool.strideBytes(), 4096u);

    // block 0's slot is allocated as backing but is never handed out by malloc().
    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_NE(*block, 0);
}

TEST(HostBlockPoolTest, BlockBufferReturnsBasePlusBlockStride) {
    auto          config = makeConfig();
    HostBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto first = pool.malloc();
    ASSERT_TRUE(first.has_value());
    auto first_buffer = pool.blockBuffer(*first);
    EXPECT_EQ(first_buffer.block, *first);
    EXPECT_EQ(first_buffer.payload_bytes, 1024u);
    EXPECT_EQ(first_buffer.stride_bytes, 4096u);
    EXPECT_NE(first_buffer.addr, nullptr);

    auto second = pool.malloc();
    ASSERT_TRUE(second.has_value());
    auto second_buffer = pool.blockBuffer(*second);

    // addr = base_ptr + block * stride_bytes, regardless of which two distinct blocks
    // malloc() happens to hand back.
    const auto actual_diff =
        static_cast<uint8_t*>(second_buffer.addr) - static_cast<uint8_t*>(first_buffer.addr);
    const auto expected_diff =
        (static_cast<int64_t>(*second) - static_cast<int64_t>(*first)) * static_cast<int64_t>(4096);
    EXPECT_EQ(actual_diff, expected_diff);
}

TEST(HostBlockPoolTest, PinnedFallbackDoesNotFailInit) {
    auto          config = makeConfig(/*physical_block_count=*/4,
                             /*payload_bytes=*/1024,
                             /*stride_bytes=*/4096,
                             /*enable_pinned=*/true);
    HostBlockPool pool(config);

    // Whether or not this host actually supports CUDA pinned memory, init() must
    // succeed and isPinned() must truthfully reflect the resulting backing tensor.
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    auto buffer = pool.blockBuffer(*block);
    EXPECT_NE(buffer.addr, nullptr);
}

TEST(HostBlockPoolTest, LifecycleComesFromIBlockPool) {
    auto          config = makeConfig();
    HostBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 0u);

    pool.incRef(*block);
    EXPECT_EQ(pool.refCount(*block), 1u);
    pool.decRef(*block);
    EXPECT_EQ(pool.refCount(*block), 0u);

    pool.free(*block);
    EXPECT_FALSE(pool.isAllocated(*block));
}

}  // namespace rtp_llm::block_tree_cache
