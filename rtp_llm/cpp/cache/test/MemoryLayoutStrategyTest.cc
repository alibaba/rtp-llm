#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <torch/torch.h>
#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

class MemoryLayoutStrategyTest: public ::testing::Test {
protected:
    enum class BufferInitMode {
        Zeros,
        Arange,
    };

    struct TestContext {
        BlockPoolConfig config;
        torch::Tensor   cache_buffer;
        void*           cache_ptr = nullptr;
    };

    void SetUp() override {
        rtp_llm::initLogger();
        torch::manual_seed(114514);

        rtp_llm::ParallelismConfig           parallelism_config;
        rtp_llm::ModelConfig                 model_config;
        rtp_llm::EPLBConfig                  eplb_config;
        rtp_llm::FMHAConfig                  fmha_config;
        rtp_llm::DeviceResourceConfig        device_resource_config;
        rtp_llm::MoeConfig                   moe_config;
        rtp_llm::SpeculativeExecutionConfig  sp_config;
        rtp_llm::MiscellaneousConfig         misc_config;
        rtp_llm::ProfilingDebugLoggingConfig profiling_debug_logging_config;
        rtp_llm::HWKernelConfig              hw_kernel_config;
        rtp_llm::ConcurrencyConfig           concurrency_config;
        rtp_llm::FfnDisAggregateConfig       ffn_disaggregate_config;
        rtp_llm::RuntimeConfig               runtime_config;

        // Provide safe defaults for fields that might be accessed during device initialization.
        model_config.max_seq_len                  = 128;
        model_config.hidden_size                  = 1;
        model_config.attn_config.head_num         = 1;
        model_config.attn_config.kv_head_num      = 1;
        model_config.attn_config.size_per_head    = 1;
        model_config.attn_config.tokens_per_block = 8;
        model_config.attn_config.q_lora_rank      = 0;
        model_config.attn_config.kv_lora_rank     = 0;
        model_config.attn_config.nope_head_dim    = 0;
        model_config.attn_config.rope_head_dim    = 0;
        model_config.attn_config.v_head_dim       = 0;

        device_resource_config.device_reserve_memory_bytes = 1024L * 1024 * 1024;  // 1GB
        device_resource_config.host_reserve_memory_bytes   = 1024L * 1024 * 1024;  // 1GB

        rtp_llm::DeviceFactory::initDevices(parallelism_config,
                                            model_config,
                                            eplb_config,
                                            fmha_config,
                                            device_resource_config,
                                            moe_config,
                                            sp_config,
                                            misc_config,
                                            profiling_debug_logging_config,
                                            hw_kernel_config,
                                            concurrency_config,
                                            ffn_disaggregate_config,
                                            runtime_config);
        device_ = rtp_llm::DeviceFactory::getDefaultDevice();

        ASSERT_NE(device_, nullptr);
    }

    void TearDown() override {}

    static BlockPoolConfig createTestConfig(
        MemoryLayout layout, uint32_t layer_num, uint32_t block_num, size_t k_block_bytes, size_t v_block_bytes) {
        BlockPoolConfig config;
        config.layer_num = layer_num;
        config.block_num = block_num;
        config.layout    = layout;

        // NOTE:
        // - This test uses int8 cache buffer, so "bytes" == "elements".
        // - LayerFirstLayoutStrategy reshapes cache_buffer by config.block_size_bytes.
        config.dtype = rtp_llm::TYPE_INT8;

        config.k_block_size = k_block_bytes;
        config.v_block_size = v_block_bytes;
        config.block_size   = k_block_bytes + v_block_bytes;

        config.k_block_stride       = config.k_block_size;
        config.v_block_stride       = config.v_block_size;
        config.block_stride         = config.block_size;
        config.k_block_stride_bytes = k_block_bytes;
        config.v_block_stride_bytes = v_block_bytes;
        config.block_stride_bytes   = config.block_size;

        config.k_block_size_bytes = k_block_bytes;
        config.v_block_size_bytes = v_block_bytes;
        config.block_size_bytes   = config.block_size;

        config.total_size =
            static_cast<size_t>(config.layer_num) * static_cast<size_t>(config.block_num) * config.block_size_bytes;

        // Make kvCacheBuffer() shape construction deterministic for tests.
        config.is_mla             = false;
        config.local_head_num_kv  = 1;
        config.seq_size_per_block = 1;
        config.k_token_size       = 1;
        config.v_token_size       = 1;

        return config;
    }

    static BlockPoolConfig
    createTestConfig(MemoryLayout layout, size_t k_block_bytes = 512, size_t v_block_bytes = 512) {
        return createTestConfig(layout, /*layer_num=*/4, /*block_num=*/8, k_block_bytes, v_block_bytes);
    }

    static torch::Tensor createCacheBuffer(const BlockPoolConfig& config,
                                           const torch::Device&   device    = torch::kCPU,
                                           BufferInitMode         init_mode = BufferInitMode::Zeros) {
        auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
        auto n       = static_cast<int64_t>(config.total_size);
        if (n <= 0) {
            return torch::Tensor();
        }

        if (init_mode == BufferInitMode::Arange) {
            return torch::arange(0, n, options).contiguous();
        }
        return torch::zeros({n}, options);
    }

    static TestContext createTestContext(BlockPoolConfig      config,
                                         const torch::Device& device    = torch::kCPU,
                                         BufferInitMode       init_mode = BufferInitMode::Zeros) {
        TestContext ctx;
        ctx.config       = std::move(config);
        ctx.cache_buffer = createCacheBuffer(ctx.config, device, init_mode);
        ctx.cache_ptr    = ctx.cache_buffer.defined() ? ctx.cache_buffer.data_ptr() : nullptr;
        return ctx;
    }

    static TestContext createTestContext(MemoryLayout         layout,
                                         size_t               k_block_bytes = 512,
                                         size_t               v_block_bytes = 512,
                                         const torch::Device& device        = torch::kCPU,
                                         BufferInitMode       init_mode     = BufferInitMode::Zeros) {
        return createTestContext(createTestConfig(layout, k_block_bytes, v_block_bytes), device, init_mode);
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
    auto ctx = createTestContext(LAYER_FIRST);

    auto strategy    = std::make_unique<LayerFirstLayoutStrategy>();
    bool init_result = strategy->init(ctx.config, ctx.cache_buffer, ctx.cache_ptr, ctx.config.dtype);

    EXPECT_TRUE(init_result);
}

TEST_F(LayerFirstLayoutStrategyTest, InitWithEmptyBuffer) {
    auto          config = createTestConfig(LAYER_FIRST);
    torch::Tensor empty_buffer;
    void*         cache_ptr = nullptr;

    auto strategy    = std::make_unique<LayerFirstLayoutStrategy>();
    bool init_result = strategy->init(config, empty_buffer, cache_ptr, config.dtype);

    EXPECT_FALSE(init_result);
}

TEST_F(LayerFirstLayoutStrategyTest, GetLayerCacheTensors) {
    auto ctx = createTestContext(LAYER_FIRST);

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    ASSERT_TRUE(strategy->init(ctx.config, ctx.cache_buffer, ctx.cache_ptr, ctx.config.dtype));

    auto layer_tensors = strategy->getLayerCacheTensors();
    EXPECT_EQ(layer_tensors.size(), ctx.config.layer_num);

    for (size_t i = 0; i < layer_tensors.size(); ++i) {
        EXPECT_TRUE(layer_tensors[i].defined());
        EXPECT_EQ(layer_tensors[i].dim(), 2);
        EXPECT_EQ(layer_tensors[i].size(0), ctx.config.block_num);
        EXPECT_EQ(layer_tensors[i].size(1), static_cast<int64_t>(ctx.config.block_size_bytes));
    }
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToAddr) {
    auto ctx = createTestContext(LAYER_FIRST);

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    ASSERT_TRUE(strategy->init(ctx.config, ctx.cache_buffer, ctx.cache_ptr, ctx.config.dtype));

    for (int layer = 0; layer < static_cast<int>(ctx.config.layer_num); ++layer) {
        for (int block = 0; block < static_cast<int>(ctx.config.block_num); ++block) {
            auto addr_info = strategy->convertIndexToAddr(layer, block);

            EXPECT_NE(addr_info.kv_addr, nullptr);
        }
    }
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToAddrOutOfRange) {
    auto ctx = createTestContext(LAYER_FIRST);

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    ASSERT_TRUE(strategy->init(ctx.config, ctx.cache_buffer, ctx.cache_ptr, ctx.config.dtype));

    auto addr_info = strategy->convertIndexToAddr(static_cast<int>(ctx.config.layer_num) + 1, 0);
    EXPECT_EQ(addr_info.kv_addr, nullptr);
}

TEST_F(LayerFirstLayoutStrategyTest, GetKVCacheAddr) {
    auto ctx = createTestContext(LAYER_FIRST);

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    ASSERT_TRUE(strategy->init(ctx.config, ctx.cache_buffer, ctx.cache_ptr, ctx.config.dtype));

    int layer = 1;
    int block = 2;

    void* k_addr = strategy->getKCacheAddr(layer, block);
    void* v_addr = strategy->getVCacheAddr(layer, block);

    EXPECT_NE(k_addr, nullptr);
    EXPECT_NE(v_addr, nullptr);

    size_t diff = reinterpret_cast<size_t>(v_addr) - reinterpret_cast<size_t>(k_addr);
    EXPECT_EQ(diff, 0);
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToBuffer) {
    auto ctx = createTestContext(LAYER_FIRST);

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    ASSERT_TRUE(strategy->init(ctx.config, ctx.cache_buffer, ctx.cache_ptr, ctx.config.dtype));

    int layer = 0;
    int block = 0;

    auto buffer_info = strategy->convertIndexToBuffer(layer, block);
    EXPECT_NE(buffer_info.kv_addr, nullptr);
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToBufferPartitionedByHead) {
    auto config               = createTestConfig(LAYER_FIRST, /*k_block_bytes=*/512, /*v_block_bytes=*/512);
    config.is_mla             = false;
    config.local_head_num_kv  = 8;
    config.seq_size_per_block = 64;
    config.k_token_size       = 1;
    config.v_token_size       = 1;

    auto ctx = createTestContext(std::move(config), torch::kCPU, BufferInitMode::Arange);

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    ASSERT_TRUE(strategy->init(ctx.config, ctx.cache_buffer, ctx.cache_ptr, ctx.config.dtype));

    const int layer = 1;
    const int block = 3;

    auto full_block_tensor = strategy->getLayerCacheTensors()[layer][block];
    ASSERT_TRUE(full_block_tensor.defined());
    ASSERT_EQ(full_block_tensor.dim(), 1);
    ASSERT_EQ(static_cast<size_t>(full_block_tensor.numel()), ctx.config.block_size_bytes);

    const uintptr_t base_ptr = reinterpret_cast<uintptr_t>(full_block_tensor.data_ptr());

    const int    partition_count = 4;
    const size_t k_total_bytes   = ctx.config.k_block_size;
    const size_t v_total_bytes   = ctx.config.v_block_size;
    const int    heads           = static_cast<int>(ctx.config.local_head_num_kv);

    ASSERT_EQ(k_total_bytes % static_cast<size_t>(heads), 0);
    ASSERT_EQ(v_total_bytes % static_cast<size_t>(heads), 0);
    ASSERT_EQ(heads % partition_count, 0);

    const size_t k_bytes_per_head = k_total_bytes / static_cast<size_t>(heads);
    const size_t v_bytes_per_head = v_total_bytes / static_cast<size_t>(heads);
    const int    head_cnt         = heads / partition_count;

    for (int partition_id = 0; partition_id < partition_count; ++partition_id) {
        auto buffers = strategy->convertIndexToBuffer(layer, block, partition_count, partition_id);
        ASSERT_EQ(buffers.size(), 2);
        ASSERT_NE(buffers[0], nullptr);
        ASSERT_NE(buffers[1], nullptr);

        const int    head_begin = partition_id * head_cnt;
        const size_t k_off      = static_cast<size_t>(head_begin) * k_bytes_per_head;
        const size_t v_off      = k_total_bytes + static_cast<size_t>(head_begin) * v_bytes_per_head;
        const size_t k_sz       = static_cast<size_t>(head_cnt) * k_bytes_per_head;
        const size_t v_sz       = static_cast<size_t>(head_cnt) * v_bytes_per_head;

        EXPECT_EQ(buffers[0]->sizeBytes(), k_sz);
        EXPECT_EQ(buffers[1]->sizeBytes(), v_sz);

        const uintptr_t k_ptr = reinterpret_cast<uintptr_t>(buffers[0]->data());
        const uintptr_t v_ptr = reinterpret_cast<uintptr_t>(buffers[1]->data());
        EXPECT_EQ(k_ptr, base_ptr + k_off);
        EXPECT_EQ(v_ptr, base_ptr + v_off);

        auto expected_k = full_block_tensor.narrow(0, static_cast<int64_t>(k_off), static_cast<int64_t>(k_sz));
        auto expected_v = full_block_tensor.narrow(0, static_cast<int64_t>(v_off), static_cast<int64_t>(v_sz));
        auto options    = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
        auto actual_k   = torch::from_blob(buffers[0]->data(), {static_cast<int64_t>(k_sz)}, options);
        auto actual_v   = torch::from_blob(buffers[1]->data(), {static_cast<int64_t>(v_sz)}, options);
        EXPECT_TRUE(torch::equal(expected_k, actual_k));
        EXPECT_TRUE(torch::equal(expected_v, actual_v));
    }
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToBufferPartitionedLayerOutOfRangeReturnsEmpty) {
    auto config               = createTestConfig(LAYER_FIRST, /*k_block_bytes=*/512, /*v_block_bytes=*/512);
    config.is_mla             = false;
    config.local_head_num_kv  = 8;
    config.seq_size_per_block = 64;
    config.k_token_size       = 1;
    config.v_token_size       = 1;
    auto ctx                  = createTestContext(std::move(config), torch::kCPU, BufferInitMode::Zeros);

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    ASSERT_TRUE(strategy->init(ctx.config, ctx.cache_buffer, ctx.cache_ptr, ctx.config.dtype));

    auto buffers = strategy->convertIndexToBuffer(static_cast<int>(ctx.config.layer_num) + 1, 0, 2, 0);
    EXPECT_TRUE(buffers.empty());
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToBufferPartitionedInvalidArgsThrow) {
    auto config               = createTestConfig(LAYER_FIRST, /*k_block_bytes=*/512, /*v_block_bytes=*/512);
    config.is_mla             = false;
    config.local_head_num_kv  = 8;
    config.seq_size_per_block = 64;
    config.k_token_size       = 1;
    config.v_token_size       = 1;
    auto ctx                  = createTestContext(std::move(config), torch::kCPU, BufferInitMode::Zeros);

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    ASSERT_TRUE(strategy->init(ctx.config, ctx.cache_buffer, ctx.cache_ptr, ctx.config.dtype));

    const int layer = 0;
    const int block = 0;

    EXPECT_THROW((void)strategy->convertIndexToBuffer(layer, block, /*partition_count=*/0, /*partition_id=*/0),
                 rtp_llm::RTPException);
    EXPECT_THROW((void)strategy->convertIndexToBuffer(layer, block, /*partition_count=*/2, /*partition_id=*/2),
                 rtp_llm::RTPException);
    EXPECT_THROW((void)strategy->convertIndexToBuffer(layer, block, /*partition_count=*/3, /*partition_id=*/0),
                 rtp_llm::RTPException);
}

TEST_F(LayerFirstLayoutStrategyTest, AddressSequentiality) {
    auto ctx = createTestContext(LAYER_FIRST);

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    ASSERT_TRUE(strategy->init(ctx.config, ctx.cache_buffer, ctx.cache_ptr, ctx.config.dtype));

    int  layer = 0;
    auto addr1 = strategy->convertIndexToAddr(layer, 0);
    auto addr2 = strategy->convertIndexToAddr(layer, 1);

    size_t addr1_val = reinterpret_cast<size_t>(addr1.kv_addr);
    size_t addr2_val = reinterpret_cast<size_t>(addr2.kv_addr);

    EXPECT_EQ(addr2_val - addr1_val, ctx.config.block_size_bytes);
}

// Layout Comparison Test
class LayoutComparisonTest: public MemoryLayoutStrategyTest {};

// Boundary Condition Test
TEST_F(MemoryLayoutStrategyTest, SingleLayerSingleBlock) {
    auto ctx = createTestContext(createTestConfig(LAYER_FIRST,
                                                  /*layer_num=*/1,
                                                  /*block_num=*/1,
                                                  /*k_block_bytes=*/256,
                                                  /*v_block_bytes=*/256));

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    EXPECT_TRUE(strategy->init(ctx.config, ctx.cache_buffer, ctx.cache_ptr, ctx.config.dtype));

    auto layer_tensors = strategy->getLayerCacheTensors();
    EXPECT_EQ(layer_tensors.size(), 1);

    auto addr_info = strategy->convertIndexToAddr(0, 0);
    EXPECT_NE(addr_info.kv_addr, nullptr);
}

TEST_F(MemoryLayoutStrategyTest, LargeConfiguration) {
    auto ctx = createTestContext(createTestConfig(LAYER_FIRST,
                                                  /*layer_num=*/32,
                                                  /*block_num=*/1024,
                                                  /*k_block_bytes=*/2048,
                                                  /*v_block_bytes=*/2048));

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    EXPECT_TRUE(strategy->init(ctx.config, ctx.cache_buffer, ctx.cache_ptr, ctx.config.dtype));

    auto layer_tensors = strategy->getLayerCacheTensors();
    EXPECT_EQ(layer_tensors.size(), ctx.config.layer_num);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
