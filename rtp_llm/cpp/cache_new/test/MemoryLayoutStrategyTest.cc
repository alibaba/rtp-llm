#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <torch/torch.h>
#include "rtp_llm/cpp/cache_new/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

class MemoryLayoutStrategyTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        torch::manual_seed(114514);

        rtp_llm::GptInitParameter gpt_init_params;
        gpt_init_params.device_resource_config.device_reserve_memory_bytes = 1024L * 1024 * 1024;  // 1GB
        gpt_init_params.device_resource_config.host_reserve_memory_bytes   = 1024L * 1024 * 1024;  // 1GB
        rtp_llm::DeviceFactory::initDevices(gpt_init_params);
        device_ = rtp_llm::DeviceFactory::getDefaultDevice();

        ASSERT_NE(device_, nullptr);
    }

    void TearDown() override {}

    BlockPoolConfig createTestConfig(MemoryLayout layout, size_t k_block_size = 512, size_t v_block_size = 512) {
        BlockPoolConfig config;
        config.layer_num  = 4;
        config.block_num  = 8;
        config.block_size = 1024;
        config.layout     = layout;
        config.total_size = config.layer_num * config.block_num * config.block_size;

        return config;
    }

    torch::Tensor createCacheBuffer(const BlockPoolConfig& config) {
        auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);

        return torch::zeros({static_cast<int64_t>(config.total_size)}, options);
    }

    rtp_llm::DeviceBase* device_;
};

// Factory Create Test
TEST_F(MemoryLayoutStrategyTest, FactoryCreateLayerFirst) {
    auto strategy = MemoryLayoutStrategyFactory::create(LAYER_FIRST);
    EXPECT_NE(strategy, nullptr);

    auto* layer_first = dynamic_cast<LayerFirstLayoutStrategy*>(strategy.get());
    EXPECT_NE(layer_first, nullptr);
}

// LayerFirstLayoutStrategy Test
class LayerFirstLayoutStrategyTest: public MemoryLayoutStrategyTest {};

TEST_F(LayerFirstLayoutStrategyTest, Initialization) {
    auto  config       = createTestConfig(LAYER_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy    = std::make_unique<LayerFirstLayoutStrategy>();
    bool init_result = strategy->init(config, cache_buffer, cache_ptr);

    EXPECT_TRUE(init_result);
}

TEST_F(LayerFirstLayoutStrategyTest, InitWithEmptyBuffer) {
    auto          config = createTestConfig(LAYER_FIRST);
    torch::Tensor empty_buffer;
    void*         cache_ptr = nullptr;

    auto strategy    = std::make_unique<LayerFirstLayoutStrategy>();
    bool init_result = strategy->init(config, empty_buffer, cache_ptr);

    EXPECT_FALSE(init_result);
}

TEST_F(LayerFirstLayoutStrategyTest, GetLayerCacheTensors) {
    auto  config       = createTestConfig(LAYER_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    auto layer_tensors = strategy->getLayerCacheTensors();
    EXPECT_EQ(layer_tensors.size(), config.layer_num);

    for (size_t i = 0; i < layer_tensors.size(); ++i) {
        EXPECT_TRUE(layer_tensors[i].defined());
        EXPECT_EQ(layer_tensors[i].dim(), 2);
        EXPECT_EQ(layer_tensors[i].size(0), config.block_num);
        EXPECT_EQ(layer_tensors[i].size(1), config.block_size);
    }
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToAddr) {
    auto  config       = createTestConfig(LAYER_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    for (int layer = 0; layer < static_cast<int>(config.layer_num); ++layer) {
        for (int block = 0; block < static_cast<int>(config.block_num); ++block) {
            auto addr_info = strategy->convertIndexToAddr(layer, block);

            EXPECT_NE(addr_info.k_addr, nullptr);
            EXPECT_NE(addr_info.v_addr, nullptr);
            EXPECT_EQ(addr_info.k_addr, addr_info.v_addr);
        }
    }
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToAddrOutOfRange) {
    auto  config       = createTestConfig(LAYER_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    auto addr_info = strategy->convertIndexToAddr(config.layer_num + 1, 0);
    EXPECT_EQ(addr_info.k_addr, nullptr);
    EXPECT_EQ(addr_info.v_addr, nullptr);
}

TEST_F(LayerFirstLayoutStrategyTest, GetKVCacheAddr) {
    auto  config       = createTestConfig(LAYER_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    int layer = 1;
    int block = 2;

    void* k_addr = strategy->getKCacheAddr(layer, block);
    void* v_addr = strategy->getVCacheAddr(layer, block);

    EXPECT_NE(k_addr, nullptr);
    EXPECT_NE(v_addr, nullptr);

    EXPECT_EQ(k_addr, v_addr);
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToBuffer) {
    auto  config       = createTestConfig(LAYER_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    int layer = 0;
    int block = 0;

    auto buffer_info = strategy->convertIndexToBuffer(layer, block);
    EXPECT_NE(buffer_info.k_addr, nullptr);
    EXPECT_NE(buffer_info.v_addr, nullptr);
}

TEST_F(LayerFirstLayoutStrategyTest, AddressSequentiality) {
    auto  config       = createTestConfig(LAYER_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    int  layer = 0;
    auto addr1 = strategy->convertIndexToAddr(layer, 0);
    auto addr2 = strategy->convertIndexToAddr(layer, 1);

    size_t addr1_val = reinterpret_cast<size_t>(addr1.k_addr);
    size_t addr2_val = reinterpret_cast<size_t>(addr2.k_addr);

    EXPECT_EQ(addr2_val - addr1_val, config.block_size);
}

// Layout Comparison Test
class LayoutComparisonTest: public MemoryLayoutStrategyTest {};

// Boundary Condition Test
TEST_F(MemoryLayoutStrategyTest, SingleLayerSingleBlock) {
    BlockPoolConfig config;
    config.layer_num  = 1;
    config.block_num  = 1;
    config.block_size = 512;
    config.layout     = LAYER_FIRST;
    config.total_size = config.layer_num * config.block_num * config.block_size;

    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    EXPECT_TRUE(strategy->init(config, cache_buffer, cache_ptr));

    auto layer_tensors = strategy->getLayerCacheTensors();
    EXPECT_EQ(layer_tensors.size(), 1);

    auto addr_info = strategy->convertIndexToAddr(0, 0);
    EXPECT_NE(addr_info.k_addr, nullptr);
}

TEST_F(MemoryLayoutStrategyTest, LargeConfiguration) {
    BlockPoolConfig config;
    config.layer_num  = 32;
    config.block_num  = 1024;
    config.block_size = 4096;
    config.layout     = LAYER_FIRST;
    config.total_size = config.layer_num * config.block_num * config.block_size;

    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    EXPECT_TRUE(strategy->init(config, cache_buffer, cache_ptr));

    auto layer_tensors = strategy->getLayerCacheTensors();
    EXPECT_EQ(layer_tensors.size(), config.layer_num);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
