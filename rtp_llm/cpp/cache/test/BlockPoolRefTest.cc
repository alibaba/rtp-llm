#include "rtp_llm/cpp/cache/BlockPool.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace test {
namespace {

BlockPoolConfig makeHostBlockPoolConfig() {
    constexpr uint32_t kLayerNum       = 1;
    constexpr uint32_t kBlockNum       = 6;
    constexpr size_t   kBlockSizeBytes = 16;

    MemoryLayoutConfig layout;
    layout.layer_num                = kLayerNum;
    layout.block_num                = kBlockNum;
    layout.dtype                    = DataType::TYPE_FP16;
    layout.kv_cache_offset_bytes    = 0;
    layout.kv_block_stride_bytes    = kBlockSizeBytes;
    layout.k_block_stride_bytes     = kBlockSizeBytes / 2;
    layout.v_block_stride_bytes     = kBlockSizeBytes / 2;
    layout.block_stride_bytes       = kBlockSizeBytes;
    layout.kv_block_pool_size_bytes = kLayerNum * kBlockNum * kBlockSizeBytes;
    layout.total_size_bytes         = layout.kv_block_pool_size_bytes;
    layout.local_head_num_kv        = 1;
    layout.seq_size_per_block       = 1;

    BlockPoolConfig config;
    config.pool_name        = "block_pool_ref_test";
    config.block_num        = kBlockNum;
    config.total_size_bytes = layout.total_size_bytes;
    config.memory_layouts   = {layout};
    return config;
}

class BlockPoolRefTest: public ::testing::Test {
protected:
    void SetUp() override {
        block_pool_ = std::make_shared<BlockPool>(makeHostBlockPoolConfig(), AllocationType::HOST);
        ASSERT_TRUE(block_pool_->init());
        total_blocks_ = block_pool_->freeBlocksNum();
    }

    std::shared_ptr<BlockPool> block_pool_;
    size_t                     total_blocks_ = 0;
};

TEST_F(BlockPoolRefTest, ConnectorFreeRespectsRemainingReferences) {
    // Releasing the connector reference must not free a request-owned block.
    auto request_owned_blocks = block_pool_->malloc(2);
    ASSERT_EQ(request_owned_blocks.size(), 2);
    block_pool_->connectorReference(request_owned_blocks);
    block_pool_->connectorFree(request_owned_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
    EXPECT_EQ(block_pool_->connectorRefBlocksNum(), 0);
    EXPECT_EQ(block_pool_->requestRefBlocksNum(), 2);
    block_pool_->requestFree(request_owned_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);

    // A block is reusable only after its final connector reference is released.
    auto connector_owned_blocks = block_pool_->malloc(2);
    ASSERT_EQ(connector_owned_blocks.size(), 2);
    block_pool_->connectorReference(connector_owned_blocks);
    block_pool_->connectorReference(connector_owned_blocks);
    block_pool_->requestFree(connector_owned_blocks);
    block_pool_->connectorFree(connector_owned_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
    EXPECT_EQ(block_pool_->connectorRefBlocksNum(), 2);
    block_pool_->connectorFree(connector_owned_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
    EXPECT_EQ(block_pool_->connectorRefBlocksNum(), 0);
}

TEST_F(BlockPoolRefTest, BlockCacheFreeRespectsRemainingReferences) {
    // Releasing the cache reference must not free a request-owned block.
    auto request_owned_blocks = block_pool_->malloc(2);
    ASSERT_EQ(request_owned_blocks.size(), 2);
    block_pool_->blockCacheReference(request_owned_blocks);
    block_pool_->blockCacheFree(request_owned_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
    EXPECT_EQ(block_pool_->blockCacheRefBlocksNum(), 0);
    EXPECT_EQ(block_pool_->requestRefBlocksNum(), 2);
    block_pool_->requestFree(request_owned_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);

    // A block is reusable only after its final cache reference is released.
    auto cache_owned_blocks = block_pool_->malloc(2);
    ASSERT_EQ(cache_owned_blocks.size(), 2);
    block_pool_->blockCacheReference(cache_owned_blocks);
    block_pool_->blockCacheReference(cache_owned_blocks);
    block_pool_->requestFree(cache_owned_blocks);
    block_pool_->blockCacheFree(cache_owned_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
    EXPECT_EQ(block_pool_->blockCacheRefBlocksNum(), 2);
    block_pool_->blockCacheFree(cache_owned_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
    EXPECT_EQ(block_pool_->blockCacheRefBlocksNum(), 0);
}

}  // namespace
}  // namespace test
}  // namespace rtp_llm
