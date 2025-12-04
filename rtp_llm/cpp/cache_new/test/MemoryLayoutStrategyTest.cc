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

        if (layout == KV_FIRST) {
            config.k_block_size = k_block_size;
            config.v_block_size = v_block_size;
            // K cache + V cache
            config.total_size = config.layer_num * config.block_num * (config.k_block_size + config.v_block_size);
        } else {
            config.total_size   = config.layer_num * config.block_num * config.block_size;
            config.k_block_size = k_block_size;
            config.v_block_size = v_block_size;
        }

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

TEST_F(MemoryLayoutStrategyTest, FactoryCreateKVFirst) {
    auto strategy = MemoryLayoutStrategyFactory::create(KV_FIRST);
    EXPECT_NE(strategy, nullptr);

    auto* kv_first = dynamic_cast<KVFirstLayoutStrategy*>(strategy.get());
    EXPECT_NE(kv_first, nullptr);
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

            size_t diff = reinterpret_cast<size_t>(addr_info.v_addr) - reinterpret_cast<size_t>(addr_info.k_addr);
            EXPECT_EQ(diff, config.k_block_size);
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

    size_t diff = reinterpret_cast<size_t>(v_addr) - reinterpret_cast<size_t>(k_addr);
    EXPECT_EQ(diff, config.k_block_size);
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

// KVFirstLayoutStrategy Test
class KVFirstLayoutStrategyTest: public MemoryLayoutStrategyTest {};

TEST_F(KVFirstLayoutStrategyTest, Initialization) {
    auto  config       = createTestConfig(KV_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy    = std::make_unique<KVFirstLayoutStrategy>();
    bool init_result = strategy->init(config, cache_buffer, cache_ptr);

    EXPECT_TRUE(init_result);
}

TEST_F(KVFirstLayoutStrategyTest, InitWithEmptyBuffer) {
    auto          config = createTestConfig(KV_FIRST);
    torch::Tensor empty_buffer;
    void*         cache_ptr = nullptr;

    auto strategy    = std::make_unique<KVFirstLayoutStrategy>();
    bool init_result = strategy->init(config, empty_buffer, cache_ptr);

    EXPECT_FALSE(init_result);
}

TEST_F(KVFirstLayoutStrategyTest, GetLayerCacheTensors) {
    auto  config       = createTestConfig(KV_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<KVFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    auto layer_tensors = strategy->getLayerCacheTensors();
    EXPECT_EQ(layer_tensors.size(), config.layer_num);

    for (size_t i = 0; i < layer_tensors.size(); ++i) {
        EXPECT_TRUE(layer_tensors[i].defined());
        EXPECT_EQ(layer_tensors[i].dim(), 2);
        EXPECT_EQ(layer_tensors[i].size(0), config.block_num);
        EXPECT_EQ(layer_tensors[i].size(1), config.k_block_size);
    }
}

TEST_F(KVFirstLayoutStrategyTest, ConvertIndexToAddr) {
    auto  config       = createTestConfig(KV_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<KVFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    for (int layer = 0; layer < static_cast<int>(config.layer_num); ++layer) {
        for (int block = 0; block < static_cast<int>(config.block_num); ++block) {
            auto addr_info = strategy->convertIndexToAddr(layer, block);

            EXPECT_NE(addr_info.k_addr, nullptr);
            EXPECT_NE(addr_info.v_addr, nullptr);

            EXPECT_NE(addr_info.k_addr, addr_info.v_addr);
        }
    }
}

TEST_F(KVFirstLayoutStrategyTest, ConvertIndexToAddrOutOfRange) {
    auto  config       = createTestConfig(KV_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<KVFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    // Out-of-range layer_id
    auto addr_info = strategy->convertIndexToAddr(config.layer_num + 1, 0);
    EXPECT_EQ(addr_info.k_addr, nullptr);
    EXPECT_EQ(addr_info.v_addr, nullptr);
}

TEST_F(KVFirstLayoutStrategyTest, GetKVCacheAddr) {
    auto  config       = createTestConfig(KV_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<KVFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    int layer = 1;
    int block = 2;

    void* k_addr = strategy->getKCacheAddr(layer, block);
    void* v_addr = strategy->getVCacheAddr(layer, block);

    EXPECT_NE(k_addr, nullptr);
    EXPECT_NE(v_addr, nullptr);

    EXPECT_NE(k_addr, v_addr);
}

TEST_F(KVFirstLayoutStrategyTest, ConvertIndexToBuffer) {
    auto  config       = createTestConfig(KV_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<KVFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    int layer = 0;
    int block = 0;

    auto buffer_info = strategy->convertIndexToBuffer(layer, block);
    EXPECT_NE(buffer_info.k_addr, nullptr);
    EXPECT_NE(buffer_info.v_addr, nullptr);
}

TEST_F(KVFirstLayoutStrategyTest, KVAddressSeparation) {
    auto  config       = createTestConfig(KV_FIRST);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<KVFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    int layer = 0;
    int block = 0;

    auto   addr_info      = strategy->convertIndexToAddr(layer, block);
    size_t k_addr_val     = reinterpret_cast<size_t>(addr_info.k_addr);
    size_t v_addr_val     = reinterpret_cast<size_t>(addr_info.v_addr);
    size_t cache_base_val = reinterpret_cast<size_t>(cache_ptr);

    EXPECT_LT(k_addr_val - cache_base_val, v_addr_val - cache_base_val);
}

TEST_F(KVFirstLayoutStrategyTest, KVBlockSequentiality) {
    auto  config       = createTestConfig(KV_FIRST, 512, 256);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<KVFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    int  layer = 0;
    auto addr1 = strategy->convertIndexToAddr(layer, 0);
    auto addr2 = strategy->convertIndexToAddr(layer, 1);

    size_t k_addr1_val = reinterpret_cast<size_t>(addr1.k_addr);
    size_t k_addr2_val = reinterpret_cast<size_t>(addr2.k_addr);

    size_t k_offset = k_addr2_val - k_addr1_val;
    EXPECT_EQ(k_offset, config.k_block_size);

    size_t v_addr1_val = reinterpret_cast<size_t>(addr1.v_addr);
    size_t v_addr2_val = reinterpret_cast<size_t>(addr2.v_addr);

    size_t v_offset = v_addr2_val - v_addr1_val;
    EXPECT_EQ(v_offset, config.v_block_size);

    auto layer0_block0 = strategy->convertIndexToAddr(0, 0);
    auto layer1_block0 = strategy->convertIndexToAddr(1, 0);

    size_t k_layer_offset =
        reinterpret_cast<size_t>(layer1_block0.k_addr) - reinterpret_cast<size_t>(layer0_block0.k_addr);
    size_t v_layer_offset =
        reinterpret_cast<size_t>(layer1_block0.v_addr) - reinterpret_cast<size_t>(layer0_block0.v_addr);

    size_t expected_k_layer_offset = config.block_num * config.k_block_size;
    size_t expected_v_layer_offset = config.block_num * config.v_block_size;

    EXPECT_EQ(k_layer_offset, expected_k_layer_offset);
    EXPECT_EQ(v_layer_offset, expected_v_layer_offset);
}

// Layout Comparison Test
class LayoutComparisonTest: public MemoryLayoutStrategyTest {};

TEST_F(LayoutComparisonTest, SameDataDifferentLayouts) {
    BlockPoolConfig layer_first_config = createTestConfig(LAYER_FIRST);
    BlockPoolConfig kv_first_config    = createTestConfig(KV_FIRST);

    EXPECT_EQ(layer_first_config.total_size, kv_first_config.total_size);

    auto lf_buffer = createCacheBuffer(layer_first_config);
    auto kv_buffer = createCacheBuffer(kv_first_config);

    auto lf_strategy = std::make_unique<LayerFirstLayoutStrategy>();
    auto kv_strategy = std::make_unique<KVFirstLayoutStrategy>();

    EXPECT_TRUE(lf_strategy->init(layer_first_config, lf_buffer, lf_buffer.data_ptr()));
    EXPECT_TRUE(kv_strategy->init(kv_first_config, kv_buffer, kv_buffer.data_ptr()));

    auto lf_tensors = lf_strategy->getLayerCacheTensors();
    auto kv_tensors = kv_strategy->getLayerCacheTensors();

    EXPECT_EQ(lf_tensors.size(), kv_tensors.size());
}

TEST_F(LayoutComparisonTest, AccessPatternsDiffer) {
    BlockPoolConfig layer_first_config = createTestConfig(LAYER_FIRST);
    BlockPoolConfig kv_first_config    = createTestConfig(KV_FIRST);

    auto lf_buffer = createCacheBuffer(layer_first_config);
    auto kv_buffer = createCacheBuffer(kv_first_config);

    auto lf_strategy = std::make_unique<LayerFirstLayoutStrategy>();
    auto kv_strategy = std::make_unique<KVFirstLayoutStrategy>();

    lf_strategy->init(layer_first_config, lf_buffer, lf_buffer.data_ptr());
    kv_strategy->init(kv_first_config, kv_buffer, kv_buffer.data_ptr());

    int layer = 0;
    int block = 0;

    auto lf_addr = lf_strategy->convertIndexToAddr(layer, block);
    auto kv_addr = kv_strategy->convertIndexToAddr(layer, block);

    // Both layouts expose valid and distinct K/V addresses, but with different memory layouts.
    EXPECT_NE(lf_addr.k_addr, nullptr);
    EXPECT_NE(lf_addr.v_addr, nullptr);
    EXPECT_NE(lf_addr.k_addr, lf_addr.v_addr);

    EXPECT_NE(kv_addr.k_addr, nullptr);
    EXPECT_NE(kv_addr.v_addr, nullptr);
    EXPECT_NE(kv_addr.k_addr, kv_addr.v_addr);
}

// Boundary Condition Test
TEST_F(MemoryLayoutStrategyTest, SingleLayerSingleBlock) {
    BlockPoolConfig config;
    config.layer_num    = 1;
    config.block_num    = 1;
    config.block_size   = 512;
    config.layout       = LAYER_FIRST;
    config.total_size   = config.layer_num * config.block_num * config.block_size;
    config.k_block_size = config.block_size / 2;
    config.v_block_size = config.block_size - config.k_block_size;

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
    config.layer_num    = 32;
    config.block_num    = 1024;
    config.block_size   = 4096;
    config.layout       = LAYER_FIRST;
    config.total_size   = config.layer_num * config.block_num * config.block_size;
    config.k_block_size = config.block_size / 2;
    config.v_block_size = config.block_size - config.k_block_size;

    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    EXPECT_TRUE(strategy->init(config, cache_buffer, cache_ptr));

    auto layer_tensors = strategy->getLayerCacheTensors();
    EXPECT_EQ(layer_tensors.size(), config.layer_num);
}

//  KV Block Size Tests
TEST_F(KVFirstLayoutStrategyTest, DifferentKVBlockSizes) {
    auto  config       = createTestConfig(KV_FIRST, 1024, 512);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy    = std::make_unique<KVFirstLayoutStrategy>();
    bool init_result = strategy->init(config, cache_buffer, cache_ptr);

    EXPECT_TRUE(init_result);

    for (int layer = 0; layer < static_cast<int>(config.layer_num); ++layer) {
        for (int block = 0; block < static_cast<int>(config.block_num); ++block) {
            auto addr_info = strategy->convertIndexToAddr(layer, block);
            EXPECT_NE(addr_info.k_addr, nullptr);
            EXPECT_NE(addr_info.v_addr, nullptr);
            EXPECT_NE(addr_info.k_addr, addr_info.v_addr);
        }
    }
}

TEST_F(KVFirstLayoutStrategyTest, SmallKBlockLargeVBlock) {
    auto  config       = createTestConfig(KV_FIRST, 256, 768);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy    = std::make_unique<KVFirstLayoutStrategy>();
    bool init_result = strategy->init(config, cache_buffer, cache_ptr);

    EXPECT_TRUE(init_result);

    int layer = 1;
    int block = 2;

    void* k_addr = strategy->getKCacheAddr(layer, block);
    void* v_addr = strategy->getVCacheAddr(layer, block);

    EXPECT_NE(k_addr, nullptr);
    EXPECT_NE(v_addr, nullptr);
    EXPECT_NE(k_addr, v_addr);

    size_t k_total_size    = config.layer_num * config.block_num * config.k_block_size;
    size_t block_offset    = (layer * config.block_num + block);
    size_t expected_offset = k_total_size - block_offset * config.k_block_size + block_offset * config.v_block_size;

    size_t actual_offset = reinterpret_cast<size_t>(v_addr) - reinterpret_cast<size_t>(k_addr);
    EXPECT_EQ(actual_offset, expected_offset);
}

TEST_F(KVFirstLayoutStrategyTest, EqualKVBlockSizes) {
    auto  config       = createTestConfig(KV_FIRST, 512, 512);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy    = std::make_unique<KVFirstLayoutStrategy>();
    bool init_result = strategy->init(config, cache_buffer, cache_ptr);

    EXPECT_TRUE(init_result);

    auto layer_tensors = strategy->getLayerCacheTensors();
    EXPECT_EQ(layer_tensors.size(), config.layer_num);

    for (size_t i = 0; i < layer_tensors.size(); ++i) {
        EXPECT_TRUE(layer_tensors[i].defined());
        EXPECT_EQ(layer_tensors[i].dim(), 2);
        EXPECT_EQ(layer_tensors[i].size(0), config.block_num);
        EXPECT_EQ(layer_tensors[i].size(1), config.k_block_size);
    }
}

TEST_F(KVFirstLayoutStrategyTest, BufferSizeMismatch) {
    auto config = createTestConfig(KV_FIRST, 512, 512);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);

    size_t        wrong_size   = config.total_size / 2;
    torch::Tensor wrong_buffer = torch::zeros({static_cast<int64_t>(wrong_size)}, options);
    void*         cache_ptr    = wrong_buffer.data_ptr();

    auto strategy    = std::make_unique<KVFirstLayoutStrategy>();
    bool init_result = strategy->init(config, wrong_buffer, cache_ptr);

    EXPECT_FALSE(init_result);
}

TEST_F(KVFirstLayoutStrategyTest, KVAddressOrdering) {
    auto  config       = createTestConfig(KV_FIRST, 512, 256);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<KVFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    void* last_k_addr  = strategy->getKCacheAddr(config.layer_num - 1, config.block_num - 1);
    void* first_v_addr = strategy->getVCacheAddr(0, 0);

    size_t last_k  = reinterpret_cast<size_t>(last_k_addr);
    size_t first_v = reinterpret_cast<size_t>(first_v_addr);

    size_t expected_offset = config.k_block_size;
    size_t actual_offset   = first_v - last_k;

    EXPECT_EQ(actual_offset, expected_offset);
    EXPECT_GT(first_v, last_k);
}

TEST_F(KVFirstLayoutStrategyTest, CompleteMemoryLayoutVerification) {
    auto  config       = createTestConfig(KV_FIRST, 1024, 512);
    auto  cache_buffer = createCacheBuffer(config);
    void* cache_ptr    = cache_buffer.data_ptr();

    auto strategy = std::make_unique<KVFirstLayoutStrategy>();
    strategy->init(config, cache_buffer, cache_ptr);

    size_t base_addr = reinterpret_cast<size_t>(cache_ptr);

    void* first_k = strategy->getKCacheAddr(0, 0);
    EXPECT_EQ(reinterpret_cast<size_t>(first_k), base_addr);

    for (int layer = 0; layer < 2; ++layer) {
        for (int block = 0; block < 3; ++block) {
            void*  k_addr            = strategy->getKCacheAddr(layer, block);
            size_t expected_k_offset = (layer * config.block_num + block) * config.k_block_size;
            size_t actual_k_offset   = reinterpret_cast<size_t>(k_addr) - base_addr;
            EXPECT_EQ(actual_k_offset, expected_k_offset)
                << "K block (" << layer << ", " << block << ") offset mismatch";
        }
    }

    size_t k_total_size    = config.layer_num * config.block_num * config.k_block_size;
    void*  first_v         = strategy->getVCacheAddr(0, 0);
    size_t expected_v_base = base_addr + k_total_size;
    EXPECT_EQ(reinterpret_cast<size_t>(first_v), expected_v_base);

    for (int layer = 0; layer < 2; ++layer) {
        for (int block = 0; block < 3; ++block) {
            void*  v_addr            = strategy->getVCacheAddr(layer, block);
            size_t expected_v_offset = k_total_size + (layer * config.block_num + block) * config.v_block_size;
            size_t actual_v_offset   = reinterpret_cast<size_t>(v_addr) - base_addr;
            EXPECT_EQ(actual_v_offset, expected_v_offset)
                << "V block (" << layer << ", " << block << ") offset mismatch";
        }
    }

    void*  last_v     = strategy->getVCacheAddr(config.layer_num - 1, config.block_num - 1);
    size_t last_v_end = reinterpret_cast<size_t>(last_v) + config.v_block_size;
    size_t buffer_end = base_addr + config.total_size;
    EXPECT_LE(last_v_end, buffer_end);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
