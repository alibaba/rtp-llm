#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <set>
#include <torch/torch.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/cache_new/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

class BlockPoolTest: public ::testing::Test {
protected:
    void SetUp() override {
        device_ = createDevice();
    }

    void TearDown() override {
        block_pool_.reset();
    }

    rtp_llm::DeviceBase*       device_;
    std::shared_ptr<BlockPool> block_pool_;
};

// Initialization Test
TEST_F(BlockPoolTest, ConstructorAndInit) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    ASSERT_NE(block_pool_, nullptr);

    bool init_result = block_pool_->init();
    EXPECT_TRUE(init_result);

    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

// Allocation Test
TEST_F(BlockPoolTest, AllocSingleBlock) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    auto blocks = block_pool_->malloc(1);
    EXPECT_EQ(blocks.size(), 1);
    EXPECT_GE(blocks[0], 0);
    EXPECT_LT(blocks[0], static_cast<BlockIdxType>(config.block_num));
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 2);
}

TEST_F(BlockPoolTest, AllocMultipleBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    int  alloc_count = 5;
    auto blocks      = block_pool_->malloc(alloc_count);
    EXPECT_EQ(blocks.size(), alloc_count);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - alloc_count - 1);

    std::set<BlockIdxType> unique_blocks(blocks.begin(), blocks.end());
    EXPECT_EQ(unique_blocks.size(), alloc_count);
}

TEST_F(BlockPoolTest, AllocAllBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    auto blocks = block_pool_->malloc(config.block_num - 1);
    EXPECT_EQ(blocks.size(), config.block_num - 1);
    EXPECT_EQ(block_pool_->freeBlocksNum(), 0);
}

TEST_F(BlockPoolTest, AllocMoreThanAvailable) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    auto blocks1 = block_pool_->malloc(5);
    EXPECT_EQ(blocks1.size(), 5);

    auto blocks2 = block_pool_->malloc(10);
    EXPECT_EQ(blocks2.size(), 0);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 6);
}

// Free Test
TEST_F(BlockPoolTest, FreeBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    auto blocks = block_pool_->malloc(5);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 6);

    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

TEST_F(BlockPoolTest, FreePartialBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    auto blocks = block_pool_->malloc(5);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 6);

    std::vector<BlockIdxType> partial_blocks(blocks.begin(), blocks.begin() + 3);
    block_pool_->requestFree(partial_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 3);
}

TEST_F(BlockPoolTest, ReferenceAndFree) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    auto total_blocks = block_pool_->freeBlocksNum();

    {
        auto blocks = block_pool_->malloc(3);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->requestReference(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->requestFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->requestFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);
    }

    // Blocks referred to by the block cache do not affect the freeblocks count.
    // Blocks referred to by the block cache do not affect the available blocks count.
    {
        auto blocks2 = block_pool_->malloc(3);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->blockCacheReference(blocks2);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->requestFree(blocks2);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);

        block_pool_->blockCacheFree(blocks2);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);
    }
}

TEST_F(BlockPoolTest, MultipleReferencesAndFrees) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    auto blocks = block_pool_->malloc(2);

    block_pool_->requestReference(blocks);
    block_pool_->requestReference(blocks);
    block_pool_->requestReference(blocks);

    // free for 4 times (1 + 3)
    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 3);

    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 3);

    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 3);

    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

// Convert Index to Addr Test
TEST_F(BlockPoolTest, ConvertIndexToAddrLayerFirst) {
    auto config = createTestConfig(LAYER_FIRST);
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    for (int layer = 0; layer < static_cast<int>(config.layer_num); ++layer) {
        for (int block = 0; block < 3; ++block) {
            auto addr_info = block_pool_->convertIndexToAddr(layer, block);
            EXPECT_NE(addr_info.k_addr, nullptr);
            EXPECT_NE(addr_info.v_addr, nullptr);

            size_t diff = reinterpret_cast<size_t>(addr_info.v_addr) - reinterpret_cast<size_t>(addr_info.k_addr);
            EXPECT_EQ(diff, config.k_block_size);
        }
    }
}

TEST_F(BlockPoolTest, ConvertIndexToBuffer) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    int layer = 0;
    int block = 0;

    auto buffer_info = block_pool_->convertIndexToBuffer(layer, block);
    EXPECT_NE(buffer_info.k_addr, nullptr);
    EXPECT_NE(buffer_info.v_addr, nullptr);
}

// LayerCache Base Test
TEST_F(BlockPoolTest, LayerCacheBaseLayerFirst) {
    auto config = createTestConfig(LAYER_FIRST);
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    auto layer_tensors = block_pool_->layerCacheBase();
    EXPECT_EQ(layer_tensors.size(), config.layer_num);

    for (size_t i = 0; i < layer_tensors.size(); ++i) {
        EXPECT_TRUE(layer_tensors[i].defined());
        EXPECT_GT(layer_tensors[i].numel(), 0);
    }
}

// Boundary Condition Test
TEST_F(BlockPoolTest, AllocZeroBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    auto blocks = block_pool_->malloc(0);
    EXPECT_EQ(blocks.size(), 0);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

TEST_F(BlockPoolTest, FreeEmptyVector) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    std::vector<BlockIdxType> empty_blocks;
    block_pool_->requestFree(empty_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

TEST_F(BlockPoolTest, OutOfRangeLayerId) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    int  invalid_layer = config.layer_num + 10;
    auto addr_info     = block_pool_->convertIndexToAddr(invalid_layer, 0);
    EXPECT_EQ(addr_info.k_addr, nullptr);
    EXPECT_EQ(addr_info.v_addr, nullptr);
}

TEST_F(BlockPoolTest, AllocFreeAllocCycle) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    for (int i = 0; i < 5; ++i) {
        auto blocks = block_pool_->malloc(5);
        EXPECT_EQ(blocks.size(), 5);
        EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 6);

        block_pool_->requestFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
    }
}

TEST_F(BlockPoolTest, MixedAllocFreeOperations) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    std::vector<std::vector<BlockIdxType>> allocated_blocks;

    allocated_blocks.push_back(block_pool_->malloc(2));
    EXPECT_EQ(block_pool_->freeBlocksNum(), 7);

    allocated_blocks.push_back(block_pool_->malloc(3));
    EXPECT_EQ(block_pool_->freeBlocksNum(), 4);

    block_pool_->requestFree(allocated_blocks[0]);
    EXPECT_EQ(block_pool_->freeBlocksNum(), 6);

    allocated_blocks.push_back(block_pool_->malloc(4));
    EXPECT_EQ(block_pool_->freeBlocksNum(), 2);

    block_pool_->requestFree(allocated_blocks[1]);
    block_pool_->requestFree(allocated_blocks[2]);
    EXPECT_EQ(block_pool_->freeBlocksNum(), 9);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
