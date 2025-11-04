#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <set>
#include <torch/torch.h>
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

class BlockPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        torch::manual_seed(114514);
        
        rtp_llm::GptInitParameter gpt_init_params;
        gpt_init_params.device_resource_config.device_reserve_memory_bytes = 1024L * 1024 * 1024;  // 1GB
        gpt_init_params.device_resource_config.host_reserve_memory_bytes = 1024L * 1024 * 1024;    // 1GB
        rtp_llm::DeviceFactory::initDevices(gpt_init_params);
        device_ = rtp_llm::DeviceFactory::getDefaultDevice();
        
        ASSERT_NE(device_, nullptr);
    }

    void TearDown() override {
        block_pool_.reset();
    }

    BlockPoolConfig createTestConfig(MemoryLayout layout = LAYER_FIRST,
                                     size_t k_block_size = 512,
                                     size_t v_block_size = 512) {
        BlockPoolConfig config;
        config.layer_num = 4;
        config.block_num = 10;
        config.block_size = 1024;
        config.layout = layout;
        
        if (layout == KV_FIRST) {
            config.k_block_size = k_block_size;
            config.v_block_size = v_block_size;
            // K cache + V cache
            config.total_size = config.layer_num * config.block_num * 
                               (config.k_block_size + config.v_block_size);
        } else {
            config.total_size = config.layer_num * config.block_num * config.block_size;
        }
        
        return config;
    }

    rtp_llm::DeviceBase* device_;
    std::shared_ptr<BlockPool> block_pool_;
};

// Initialization Test
TEST_F(BlockPoolTest, ConstructorAndInit) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    ASSERT_NE(block_pool_, nullptr);
    
    bool init_result = block_pool_->init();
    EXPECT_TRUE(init_result);
    
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num);
}

TEST_F(BlockPoolTest, InitWithKVFirstLayout) {
    auto config = createTestConfig(KV_FIRST);
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    ASSERT_NE(block_pool_, nullptr);
    
    bool init_result = block_pool_->init();
    EXPECT_TRUE(init_result);
    
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num);
}


// Allocation Test
TEST_F(BlockPoolTest, AllocSingleBlock) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    auto blocks = block_pool_->alloc(1);
    EXPECT_EQ(blocks.size(), 1);
    EXPECT_GE(blocks[0], 0);
    EXPECT_LT(blocks[0], static_cast<BlockIdxType>(config.block_num));
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num - 1);
}

TEST_F(BlockPoolTest, AllocMultipleBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    int alloc_count = 5;
    auto blocks = block_pool_->alloc(alloc_count);
    EXPECT_EQ(blocks.size(), alloc_count);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num - alloc_count);
    
    std::set<BlockIdxType> unique_blocks(blocks.begin(), blocks.end());
    EXPECT_EQ(unique_blocks.size(), alloc_count);
}

TEST_F(BlockPoolTest, AllocAllBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    auto blocks = block_pool_->alloc(config.block_num);
    EXPECT_EQ(blocks.size(), config.block_num);
    EXPECT_EQ(block_pool_->freeBlockNums(), 0);
}

TEST_F(BlockPoolTest, AllocMoreThanAvailable) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    auto blocks1 = block_pool_->alloc(5);
    EXPECT_EQ(blocks1.size(), 5);
    
    auto blocks2 = block_pool_->alloc(10);
    EXPECT_EQ(blocks2.size(), 0);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num - 5);
}

// Free Test
TEST_F(BlockPoolTest, FreeBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    auto blocks = block_pool_->alloc(5);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num - 5);
    
    block_pool_->free(blocks);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num);
}

TEST_F(BlockPoolTest, FreePartialBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    auto blocks = block_pool_->alloc(5);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num - 5);
    
    std::vector<BlockIdxType> partial_blocks(blocks.begin(), blocks.begin() + 3);
    block_pool_->free(partial_blocks);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num - 2);
}

// Reference and Free Test
TEST_F(BlockPoolTest, ReferenceAndFree) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();

    auto blocks = block_pool_->alloc(3);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num - 3);
    
    block_pool_->reference(blocks);
    
    block_pool_->free(blocks);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num - 3);
    
    block_pool_->free(blocks);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num);
}

TEST_F(BlockPoolTest, MultipleReferencesAndFrees) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    auto blocks = block_pool_->alloc(2);
    
    block_pool_->reference(blocks);
    block_pool_->reference(blocks);
    block_pool_->reference(blocks);
    
    // free for 4 times (1 + 3)
    block_pool_->free(blocks);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num - 2);
    
    block_pool_->free(blocks);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num - 2);
    
    block_pool_->free(blocks);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num - 2);
    
    block_pool_->free(blocks);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num);
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
            EXPECT_EQ(addr_info.k_addr, addr_info.v_addr);
        }
    }
}

TEST_F(BlockPoolTest, ConvertIndexToAddrKVFirst) {
    auto config = createTestConfig(KV_FIRST);
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    for (int layer = 0; layer < static_cast<int>(config.layer_num); ++layer) {
        for (int block = 0; block < 3; ++block) {
            auto addr_info = block_pool_->convertIndexToAddr(layer, block);
            EXPECT_NE(addr_info.k_addr, nullptr);
            EXPECT_NE(addr_info.v_addr, nullptr);
            EXPECT_NE(addr_info.k_addr, addr_info.v_addr);
        }
    }
}

TEST_F(BlockPoolTest, GetKVCacheAddr) {
    auto config = createTestConfig(KV_FIRST);
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    int layer = 1;
    int block = 2;
    
    void* k_addr = block_pool_->getKCacheAddr(layer, block);
    void* v_addr = block_pool_->getVCacheAddr(layer, block);
    
    EXPECT_NE(k_addr, nullptr);
    EXPECT_NE(v_addr, nullptr);
    EXPECT_NE(k_addr, v_addr);
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

TEST_F(BlockPoolTest, LayerCacheBaseKVFirst) {
    auto config = createTestConfig(KV_FIRST);
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
    
    auto blocks = block_pool_->alloc(0);
    EXPECT_EQ(blocks.size(), 0);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num);
}

TEST_F(BlockPoolTest, FreeEmptyVector) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    std::vector<BlockIdxType> empty_blocks;
    block_pool_->free(empty_blocks);
    EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num);
}

TEST_F(BlockPoolTest, OutOfRangeLayerId) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    int invalid_layer = config.layer_num + 10;
    auto addr_info = block_pool_->convertIndexToAddr(invalid_layer, 0);
    EXPECT_EQ(addr_info.k_addr, nullptr);
    EXPECT_EQ(addr_info.v_addr, nullptr);
}

TEST_F(BlockPoolTest, AllocFreeAllocCycle) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    for (int i = 0; i < 5; ++i) {
        auto blocks = block_pool_->alloc(5);
        EXPECT_EQ(blocks.size(), 5);
        EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num - 5);
        
        block_pool_->free(blocks);
        EXPECT_EQ(block_pool_->freeBlockNums(), config.block_num);
    }
}

TEST_F(BlockPoolTest, MixedAllocFreeOperations) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config, device_);
    block_pool_->init();
    
    std::vector<std::vector<BlockIdxType>> allocated_blocks;
    
    allocated_blocks.push_back(block_pool_->alloc(2));
    EXPECT_EQ(block_pool_->freeBlockNums(), 8);
    
    allocated_blocks.push_back(block_pool_->alloc(3));
    EXPECT_EQ(block_pool_->freeBlockNums(), 5);

    block_pool_->free(allocated_blocks[0]);
    EXPECT_EQ(block_pool_->freeBlockNums(), 7);
    
    allocated_blocks.push_back(block_pool_->alloc(4));
    EXPECT_EQ(block_pool_->freeBlockNums(), 3);
    
    block_pool_->free(allocated_blocks[1]);
    block_pool_->free(allocated_blocks[2]);
    EXPECT_EQ(block_pool_->freeBlockNums(), 10);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

