#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <torch/torch.h>
#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
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
        MemoryLayoutConfig config;
        torch::Tensor      kv_cache_buffer;
        torch::Tensor      kv_scale_buffer;
        void*              cache_ptr = nullptr;
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

        // Keep tests stable on shared GPUs with low free memory:
        // - device_reserve_memory_bytes=1 => avoid DeviceFactory default (-512MB), i.e. avoid reserving (free - 512MB)
        // - host_reserve_memory_bytes=0   => don't reserve pinned host memory
        device_resource_config.device_reserve_memory_bytes = 2048000000;
        device_resource_config.host_reserve_memory_bytes   = 2048000000;

        rtp_llm::ModelSpecificConfig model_specific_config;
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
                                            runtime_config,
                                            model_specific_config);
        device_ = rtp_llm::DeviceFactory::getDefaultDevice();

        ASSERT_NE(device_, nullptr);
    }

    void TearDown() override {}

    static KVCacheSpecPtr createTestKvCacheSpec(uint32_t          layer_num,
                                                rtp_llm::DataType dtype,
                                                uint32_t          local_head_num_kv,
                                                uint32_t          seq_size_per_block,
                                                size_t            k_block_stride_bytes,
                                                size_t            v_block_stride_bytes) {
        const size_t type_sz = rtp_llm::getTypeSize(dtype);
        RTP_LLM_CHECK_WITH_INFO(type_sz > 0, "invalid dtype=%d", static_cast<int>(dtype));
        RTP_LLM_CHECK_WITH_INFO(k_block_stride_bytes % type_sz == 0,
                                "k_block_stride_bytes=%zu must be divisible by type size=%zu",
                                k_block_stride_bytes,
                                type_sz);
        RTP_LLM_CHECK_WITH_INFO(v_block_stride_bytes % type_sz == 0,
                                "v_block_stride_bytes=%zu must be divisible by type size=%zu",
                                v_block_stride_bytes,
                                type_sz);
        RTP_LLM_CHECK_WITH_INFO(local_head_num_kv > 0, "local_head_num_kv must be > 0");
        RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "seq_size_per_block must be > 0");

        const size_t k_elems = k_block_stride_bytes / type_sz;
        const size_t v_elems = v_block_stride_bytes / type_sz;
        const size_t denom   = static_cast<size_t>(local_head_num_kv) * static_cast<size_t>(seq_size_per_block);
        RTP_LLM_CHECK_WITH_INFO(k_elems % denom == 0, "k elems %zu must be divisible by heads*seq=%zu", k_elems, denom);
        RTP_LLM_CHECK_WITH_INFO(v_elems % denom == 0, "v elems %zu must be divisible by heads*seq=%zu", v_elems, denom);

        if (k_block_stride_bytes == v_block_stride_bytes) {
            auto spec                = std::make_shared<MHAKVCacheSpec>();
            spec->type               = KVCacheSpecType::MultiHeadAttention;
            spec->dtype              = dtype;
            spec->layer_num          = layer_num;
            spec->local_head_num_kv  = local_head_num_kv;
            spec->seq_size_per_block = seq_size_per_block;
            spec->size_per_head      = static_cast<uint32_t>(k_elems / denom);
            return spec;
        } else {
            auto spec                = std::make_shared<MLAKVCacheSpec>();
            spec->type               = KVCacheSpecType::MultiHeadLatentAttention;
            spec->dtype              = dtype;
            spec->layer_num          = layer_num;
            spec->local_head_num_kv  = local_head_num_kv;
            spec->seq_size_per_block = seq_size_per_block;
            spec->kv_lora_rank       = static_cast<uint32_t>(k_elems / denom);
            spec->rope_head_dim      = static_cast<uint32_t>(v_elems / denom);
            return spec;
        }
    }

    static MemoryLayoutConfig
    createTestConfig(uint32_t layer_num, uint32_t block_num, size_t k_block_bytes, size_t v_block_bytes) {

        // Keep tests using int8 raw tensor (bytes == elements) unless overridden later.
        auto spec = createTestKvCacheSpec(layer_num,
                                          /*dtype=*/rtp_llm::DataType::TYPE_INT8,
                                          /*local_head_num_kv=*/1,
                                          /*seq_size_per_block=*/1,
                                          /*k_block_stride_bytes=*/k_block_bytes,
                                          /*v_block_stride_bytes=*/v_block_bytes);

        auto pool_cfg   = BlockPoolConfigHelper::createLayerFirstConfig(layer_num, block_num, spec);
        auto layout_cfg = pool_cfg.memory_layouts[0];

        layout_cfg.enable_kv_scale = false;
        // layout_cfg.kv_scale_stride          = 0;
        layout_cfg.kv_scale_stride_bytes    = 0;
        layout_cfg.kv_scale_pool_size_bytes = 0;
        // layout_cfg.kv_scale_size            = 0;
        layout_cfg.kv_scale_size_bytes   = 0;
        layout_cfg.kv_scale_offset_bytes = layout_cfg.kv_cache_offset_bytes + layout_cfg.kv_block_pool_size_bytes;
        layout_cfg.total_size_bytes      = layout_cfg.kv_block_pool_size_bytes;
        // layout_cfg.block_stride             = layout_cfg.kv_block_stride;
        layout_cfg.block_stride_bytes = layout_cfg.kv_block_stride_bytes;
        // layout_cfg.block_size               = layout_cfg.kv_block_size;
        layout_cfg.block_size_bytes = layout_cfg.kv_block_size_bytes;

        return layout_cfg;
    }

    static MemoryLayoutConfig createTestConfig(size_t k_block_bytes = 512, size_t v_block_bytes = 512) {
        return createTestConfig(/*layer_num=*/4, /*block_num=*/8, k_block_bytes, v_block_bytes);
    }

    static torch::Tensor createKVCacheBuffer(const MemoryLayoutConfig& config,
                                             const torch::Device&      device    = torch::kCPU,
                                             BufferInitMode            init_mode = BufferInitMode::Zeros) {
        auto       options = torch::TensorOptions().dtype(torch::kInt8).device(device);
        const auto n       = static_cast<int64_t>(config.kv_block_pool_size_bytes);
        if (n <= 0 || config.kv_block_pool_size_bytes == 0) {
            return torch::Tensor();
        }

        if (init_mode == BufferInitMode::Arange) {
            return torch::arange(0, n, options).contiguous();
        }
        return torch::zeros({n}, options);
    }

    static torch::Tensor createKVScaleBuffer(const MemoryLayoutConfig& config,
                                             const torch::Device&      device = torch::kCPU) {
        if (!config.enable_kv_scale || config.kv_scale_pool_size_bytes == 0) {
            return torch::Tensor();
        }
        auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
        return torch::zeros({static_cast<int64_t>(config.kv_scale_pool_size_bytes)}, options);
    }

    static TestContext createTestContext(MemoryLayoutConfig   config,
                                         const torch::Device& device    = torch::kCPU,
                                         BufferInitMode       init_mode = BufferInitMode::Zeros) {
        TestContext ctx;
        ctx.config          = std::move(config);
        ctx.kv_cache_buffer = createKVCacheBuffer(ctx.config, device, init_mode);
        ctx.kv_scale_buffer = createKVScaleBuffer(ctx.config, device);
        ctx.cache_ptr       = ctx.kv_cache_buffer.defined() ? ctx.kv_cache_buffer.data_ptr() : nullptr;
        return ctx;
    }

    static TestContext createTestContext(size_t               k_block_bytes = 512,
                                         size_t               v_block_bytes = 512,
                                         const torch::Device& device        = torch::kCPU,
                                         BufferInitMode       init_mode     = BufferInitMode::Zeros) {
        return createTestContext(createTestConfig(k_block_bytes, v_block_bytes), device, init_mode);
    }

    rtp_llm::DeviceBase* device_;
};

// LayerFirstLayoutStrategy Test
class LayerFirstLayoutStrategyTest: public MemoryLayoutStrategyTest {};

TEST_F(LayerFirstLayoutStrategyTest, Initialization) {
    auto ctx = createTestContext();

    auto          strategy = std::make_unique<LayerFirstLayoutStrategy>();
    torch::Tensor empty_scale;
    bool          init_result = strategy->init(ctx.config, ctx.kv_cache_buffer, empty_scale, ctx.cache_ptr);

    EXPECT_TRUE(init_result);
}

TEST_F(LayerFirstLayoutStrategyTest, InitializationWithScaleTensor) {
    // Create an int8 config with kv-scale enabled (matches current production behavior).
    auto spec     = createTestKvCacheSpec(/*layer_num=*/4,
                                      /*dtype=*/rtp_llm::DataType::TYPE_INT8,
                                      /*local_head_num_kv=*/2,
                                      /*seq_size_per_block=*/4,
                                      /*k_block_stride_bytes=*/512,
                                      /*v_block_stride_bytes=*/512);
    auto pool_cfg = BlockPoolConfigHelper::createLayerFirstConfig(/*layer_num=*/4, /*block_num=*/8, spec);
    auto config   = pool_cfg.memory_layouts[0];  // keep enable_kv_scale=true

    auto  kv_cache_tensor = torch::zeros({static_cast<int64_t>(config.kv_block_pool_size_bytes)}, torch::kInt8);
    auto  kv_scale_tensor = torch::zeros({static_cast<int64_t>(config.kv_scale_pool_size_bytes)}, torch::kInt8);
    void* cache_ptr       = kv_cache_tensor.data_ptr();

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    ASSERT_TRUE(strategy->init(config, kv_cache_tensor, kv_scale_tensor, cache_ptr));

    auto addr_info = strategy->convertIndexToAddr(0, 0);
    EXPECT_NE(addr_info.kv_addr, nullptr);
    EXPECT_NE(addr_info.kv_scale_addr, nullptr);

    auto buf_info = strategy->convertIndexToBuffer(0, 0);
    ASSERT_EQ(buf_info.size(), 2u);
    EXPECT_NE(buf_info[0].addr, nullptr);
    EXPECT_NE(buf_info[1].addr, nullptr);
    EXPECT_EQ(buf_info[1].size_bytes, config.kv_scale_stride_bytes);
}

TEST_F(LayerFirstLayoutStrategyTest, InitWithEmptyBuffer) {
    auto          config       = createTestConfig();
    auto          options      = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
    torch::Tensor empty_buffer = torch::empty({0}, options);
    torch::Tensor empty_scale  = torch::empty({0}, options);
    void*         cache_ptr    = nullptr;

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    // init() reshapes the buffer to [layer, block, stride] unconditionally.
    // A 0-sized buffer triggers a torch exception during reshape; treat that as expected invalid-input behavior.
    EXPECT_ANY_THROW((void)strategy->init(config, empty_buffer, empty_scale, cache_ptr));
}

TEST_F(LayerFirstLayoutStrategyTest, GetLayerCacheTensors) {
    auto ctx = createTestContext();

    auto          strategy = std::make_unique<LayerFirstLayoutStrategy>();
    torch::Tensor empty_scale;
    ASSERT_TRUE(strategy->init(ctx.config, ctx.kv_cache_buffer, empty_scale, ctx.cache_ptr));

    auto layer_tensors = strategy->getLayerCacheTensors();
    EXPECT_EQ(layer_tensors.size(), ctx.config.layer_num);

    for (size_t i = 0; i < layer_tensors.size(); ++i) {
        EXPECT_TRUE(layer_tensors[i].defined());
        EXPECT_EQ(layer_tensors[i].dim(), 2);
        EXPECT_EQ(layer_tensors[i].size(0), ctx.config.block_num);
        EXPECT_EQ(layer_tensors[i].size(1), static_cast<int64_t>(ctx.config.kv_block_stride_bytes));
    }
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToAddr) {
    auto ctx = createTestContext();

    auto          strategy = std::make_unique<LayerFirstLayoutStrategy>();
    torch::Tensor empty_scale;
    ASSERT_TRUE(strategy->init(ctx.config, ctx.kv_cache_buffer, empty_scale, ctx.cache_ptr));

    for (int layer = 0; layer < static_cast<int>(ctx.config.layer_num); ++layer) {
        for (int block = 0; block < static_cast<int>(ctx.config.block_num); ++block) {
            auto addr_info = strategy->convertIndexToAddr(layer, block);

            EXPECT_NE(addr_info.kv_addr, nullptr);
        }
    }
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToAddrOutOfRange) {
    auto ctx = createTestContext();

    auto          strategy = std::make_unique<LayerFirstLayoutStrategy>();
    torch::Tensor empty_scale;
    ASSERT_TRUE(strategy->init(ctx.config, ctx.kv_cache_buffer, empty_scale, ctx.cache_ptr));

    EXPECT_THROW((void)strategy->convertIndexToAddr(static_cast<int>(ctx.config.layer_num) + 1, 0),
                 rtp_llm::RTPException);
}

TEST_F(LayerFirstLayoutStrategyTest, GetKVCacheAddr) {
    auto ctx = createTestContext();

    auto          strategy = std::make_unique<LayerFirstLayoutStrategy>();
    torch::Tensor empty_scale;
    ASSERT_TRUE(strategy->init(ctx.config, ctx.kv_cache_buffer, empty_scale, ctx.cache_ptr));

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
    auto ctx = createTestContext();

    auto          strategy = std::make_unique<LayerFirstLayoutStrategy>();
    torch::Tensor empty_scale;
    ASSERT_TRUE(strategy->init(ctx.config, ctx.kv_cache_buffer, empty_scale, ctx.cache_ptr));

    int layer = 0;
    int block = 0;

    auto buffer_info = strategy->convertIndexToBuffer(layer, block);
    ASSERT_EQ(buffer_info.size(), 1u);
    EXPECT_NE(buffer_info[0].addr, nullptr);
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToBufferPartitionedByHead) {
    auto config               = createTestConfig(/*k_block_bytes=*/512, /*v_block_bytes=*/512);
    config.is_mla             = false;
    config.local_head_num_kv  = 8;
    config.seq_size_per_block = 64;
    config.k_dim              = 1;
    config.v_dim              = 1;

    auto ctx = createTestContext(std::move(config), torch::kCPU, BufferInitMode::Arange);

    auto          strategy = std::make_unique<LayerFirstLayoutStrategy>();
    torch::Tensor empty_scale;
    ASSERT_TRUE(strategy->init(ctx.config, ctx.kv_cache_buffer, empty_scale, ctx.cache_ptr));

    const int layer = 1;
    const int block = 3;

    auto full_block_tensor = strategy->getLayerCacheTensors()[layer][block];
    ASSERT_TRUE(full_block_tensor.defined());
    ASSERT_EQ(full_block_tensor.dim(), 1);
    ASSERT_EQ(static_cast<size_t>(full_block_tensor.nbytes()), ctx.config.kv_block_stride_bytes);

    const uintptr_t base_ptr = reinterpret_cast<uintptr_t>(full_block_tensor.data_ptr());

    const int    partition_count = 4;
    const size_t k_total_bytes   = static_cast<size_t>(ctx.config.k_block_stride_bytes);
    const size_t v_total_bytes   = static_cast<size_t>(ctx.config.v_block_stride_bytes);
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
        ASSERT_NE(buffers[0].addr, nullptr);
        ASSERT_NE(buffers[1].addr, nullptr);

        const int    head_begin = partition_id * head_cnt;
        const size_t k_off      = static_cast<size_t>(head_begin) * k_bytes_per_head;
        const size_t v_off      = k_total_bytes + static_cast<size_t>(head_begin) * v_bytes_per_head;
        const size_t k_sz       = static_cast<size_t>(head_cnt) * k_bytes_per_head;
        const size_t v_sz       = static_cast<size_t>(head_cnt) * v_bytes_per_head;

        EXPECT_EQ(buffers[0].size_bytes, k_sz);
        EXPECT_EQ(buffers[1].size_bytes, v_sz);

        const uintptr_t k_ptr = reinterpret_cast<uintptr_t>(buffers[0].addr);
        const uintptr_t v_ptr = reinterpret_cast<uintptr_t>(buffers[1].addr);
        EXPECT_EQ(k_ptr, base_ptr + k_off);
        EXPECT_EQ(v_ptr, base_ptr + v_off);

        auto expected_k = full_block_tensor.narrow(0, static_cast<int64_t>(k_off), static_cast<int64_t>(k_sz));
        auto expected_v = full_block_tensor.narrow(0, static_cast<int64_t>(v_off), static_cast<int64_t>(v_sz));
        auto options    = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
        auto actual_k   = torch::from_blob(buffers[0].addr, {static_cast<int64_t>(k_sz)}, options);
        auto actual_v   = torch::from_blob(buffers[1].addr, {static_cast<int64_t>(v_sz)}, options);
        EXPECT_TRUE(torch::equal(expected_k, actual_k));
        EXPECT_TRUE(torch::equal(expected_v, actual_v));
    }
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToBufferPartitionedByHeadFp16UsesByteView) {
    // Regression test: splitKVPartition uses byte offsets; when dtype element size > 1 (e.g. FP16),
    // partitioned slicing must use byte-view tensors.
    auto spec     = createTestKvCacheSpec(/*layer_num=*/4,
                                      /*dtype=*/rtp_llm::DataType::TYPE_FP16,
                                      /*local_head_num_kv=*/8,
                                      /*seq_size_per_block=*/64,
                                      /*k_block_stride_bytes=*/1024,
                                      /*v_block_stride_bytes=*/1024);
    auto pool_cfg = BlockPoolConfigHelper::createLayerFirstConfig(/*layer_num=*/4, /*block_num=*/8, spec);
    auto config   = pool_cfg.memory_layouts[0];

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
    auto kv_cache_tensor =
        torch::arange(0, static_cast<int64_t>(config.kv_block_pool_size_bytes), options).contiguous();
    torch::Tensor empty_scale;
    void*         cache_ptr = kv_cache_tensor.data_ptr();

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    ASSERT_TRUE(strategy->init(config, kv_cache_tensor, empty_scale, cache_ptr));

    const int layer = 1;
    const int block = 3;

    // Compute base_ptr of the raw byte block region.
    const int64_t block_base_off =
        static_cast<int64_t>((layer * config.block_num + block) * config.kv_block_stride_bytes);
    auto full_block_bytes =
        kv_cache_tensor.narrow(0, block_base_off, static_cast<int64_t>(config.kv_block_stride_bytes));
    const uintptr_t base_ptr = reinterpret_cast<uintptr_t>(full_block_bytes.data_ptr());

    const int    partition_count = 4;
    const size_t k_total_bytes   = static_cast<size_t>(config.k_block_stride_bytes);
    const size_t v_total_bytes   = static_cast<size_t>(config.v_block_stride_bytes);
    const int    heads           = static_cast<int>(config.local_head_num_kv);

    ASSERT_EQ(k_total_bytes % static_cast<size_t>(heads), 0);
    ASSERT_EQ(v_total_bytes % static_cast<size_t>(heads), 0);
    ASSERT_EQ(heads % partition_count, 0);

    const size_t k_bytes_per_head = k_total_bytes / static_cast<size_t>(heads);
    const size_t v_bytes_per_head = v_total_bytes / static_cast<size_t>(heads);
    const int    head_cnt         = heads / partition_count;

    for (int partition_id = 0; partition_id < partition_count; ++partition_id) {
        auto buffers = strategy->convertIndexToBuffer(layer, block, partition_count, partition_id);
        ASSERT_EQ(buffers.size(), 2u);
        ASSERT_NE(buffers[0].addr, nullptr);
        ASSERT_NE(buffers[1].addr, nullptr);

        const int    head_begin = partition_id * head_cnt;
        const size_t k_off      = static_cast<size_t>(head_begin) * k_bytes_per_head;
        const size_t v_off      = k_total_bytes + static_cast<size_t>(head_begin) * v_bytes_per_head;
        const size_t k_sz       = static_cast<size_t>(head_cnt) * k_bytes_per_head;
        const size_t v_sz       = static_cast<size_t>(head_cnt) * v_bytes_per_head;

        EXPECT_EQ(buffers[0].size_bytes, k_sz);
        EXPECT_EQ(buffers[1].size_bytes, v_sz);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(buffers[0].addr), base_ptr + k_off);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(buffers[1].addr), base_ptr + v_off);
    }
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToBufferPartitionedByHeadWithScale) {
    // Create an int8 config with kv-scale enabled, and verify both kv-cache and kv-scale are partitioned.
    auto spec     = createTestKvCacheSpec(/*layer_num=*/4,
                                      /*dtype=*/rtp_llm::DataType::TYPE_INT8,
                                      /*local_head_num_kv=*/8,
                                      /*seq_size_per_block=*/64,
                                      /*k_block_stride_bytes=*/512,
                                      /*v_block_stride_bytes=*/512);
    auto pool_cfg = BlockPoolConfigHelper::createLayerFirstConfig(/*layer_num=*/4, /*block_num=*/8, spec);
    auto config   = pool_cfg.memory_layouts[0];  // keep enable_kv_scale=true

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
    auto kv_cache_tensor =
        torch::arange(0, static_cast<int64_t>(config.kv_block_pool_size_bytes), options).contiguous();
    auto kv_scale_tensor =
        torch::arange(0, static_cast<int64_t>(config.kv_scale_pool_size_bytes), options).contiguous();
    void* cache_ptr = kv_cache_tensor.data_ptr();

    auto strategy = std::make_unique<LayerFirstLayoutStrategy>();
    ASSERT_TRUE(strategy->init(config, kv_cache_tensor, kv_scale_tensor, cache_ptr));

    const int layer = 1;
    const int block = 3;

    auto full_block_tensor = strategy->getLayerCacheTensors()[layer][block];
    ASSERT_TRUE(full_block_tensor.defined());
    ASSERT_EQ(full_block_tensor.dim(), 1);
    ASSERT_EQ(static_cast<size_t>(full_block_tensor.nbytes()), config.kv_block_stride_bytes);

    auto full_scale_tensor = strategy->getLayerScaleCacheTensors()[layer][block];
    ASSERT_TRUE(full_scale_tensor.defined());
    ASSERT_EQ(full_scale_tensor.dim(), 1);
    ASSERT_EQ(static_cast<size_t>(full_scale_tensor.nbytes()), config.kv_scale_stride_bytes);

    const uintptr_t kv_base_ptr = reinterpret_cast<uintptr_t>(full_block_tensor.data_ptr());
    const uintptr_t sc_base_ptr = reinterpret_cast<uintptr_t>(full_scale_tensor.data_ptr());

    const int64_t scale_block_base_off =
        static_cast<int64_t>((layer * config.block_num + block) * config.kv_scale_stride_bytes);
    auto full_scale_bytes =
        kv_scale_tensor.narrow(0, scale_block_base_off, static_cast<int64_t>(config.kv_scale_stride_bytes));

    const int    partition_count = 4;
    const size_t k_total_bytes   = static_cast<size_t>(config.k_block_stride_bytes);
    const size_t v_total_bytes   = static_cast<size_t>(config.v_block_stride_bytes);
    const size_t sc_bytes        = static_cast<size_t>(config.k_scale_stride_bytes);  // K or V plane bytes (all heads)
    const int    heads           = static_cast<int>(config.local_head_num_kv);

    ASSERT_EQ(k_total_bytes % static_cast<size_t>(heads), 0);
    ASSERT_EQ(v_total_bytes % static_cast<size_t>(heads), 0);
    ASSERT_EQ(sc_bytes % static_cast<size_t>(heads), 0);
    ASSERT_EQ(heads % partition_count, 0);

    const size_t k_bytes_per_head  = k_total_bytes / static_cast<size_t>(heads);
    const size_t v_bytes_per_head  = v_total_bytes / static_cast<size_t>(heads);
    const size_t sc_bytes_per_head = sc_bytes / static_cast<size_t>(heads);
    const int    head_cnt          = heads / partition_count;

    for (int partition_id = 0; partition_id < partition_count; ++partition_id) {
        auto buffers = strategy->convertIndexToBuffer(layer, block, partition_count, partition_id);
        ASSERT_EQ(buffers.size(), 4u);
        ASSERT_NE(buffers[0].addr, nullptr);
        ASSERT_NE(buffers[1].addr, nullptr);
        ASSERT_NE(buffers[2].addr, nullptr);
        ASSERT_NE(buffers[3].addr, nullptr);

        const int    head_begin = partition_id * head_cnt;
        const size_t k_off      = static_cast<size_t>(head_begin) * k_bytes_per_head;
        const size_t v_off      = k_total_bytes + static_cast<size_t>(head_begin) * v_bytes_per_head;
        const size_t k_sz       = static_cast<size_t>(head_cnt) * k_bytes_per_head;
        const size_t v_sz       = static_cast<size_t>(head_cnt) * v_bytes_per_head;

        const size_t sc_k_off = static_cast<size_t>(head_begin) * sc_bytes_per_head;
        const size_t sc_v_off = sc_bytes + static_cast<size_t>(head_begin) * sc_bytes_per_head;
        const size_t sc_k_sz  = static_cast<size_t>(head_cnt) * sc_bytes_per_head;
        const size_t sc_v_sz  = static_cast<size_t>(head_cnt) * sc_bytes_per_head;

        EXPECT_EQ(buffers[0].size_bytes, k_sz);
        EXPECT_EQ(buffers[1].size_bytes, v_sz);
        EXPECT_EQ(buffers[2].size_bytes, sc_k_sz);
        EXPECT_EQ(buffers[3].size_bytes, sc_v_sz);

        EXPECT_EQ(reinterpret_cast<uintptr_t>(buffers[0].addr), kv_base_ptr + k_off);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(buffers[1].addr), kv_base_ptr + v_off);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(buffers[2].addr), sc_base_ptr + sc_k_off);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(buffers[3].addr), sc_base_ptr + sc_v_off);

        auto expected_k    = full_block_tensor.narrow(0, static_cast<int64_t>(k_off), static_cast<int64_t>(k_sz));
        auto expected_v    = full_block_tensor.narrow(0, static_cast<int64_t>(v_off), static_cast<int64_t>(v_sz));
        auto expected_sc_k = full_scale_bytes.narrow(0, static_cast<int64_t>(sc_k_off), static_cast<int64_t>(sc_k_sz));
        auto expected_sc_v = full_scale_bytes.narrow(0, static_cast<int64_t>(sc_v_off), static_cast<int64_t>(sc_v_sz));

        auto actual_k    = torch::from_blob(buffers[0].addr, {static_cast<int64_t>(k_sz)}, options);
        auto actual_v    = torch::from_blob(buffers[1].addr, {static_cast<int64_t>(v_sz)}, options);
        auto actual_sc_k = torch::from_blob(buffers[2].addr, {static_cast<int64_t>(sc_k_sz)}, options);
        auto actual_sc_v = torch::from_blob(buffers[3].addr, {static_cast<int64_t>(sc_v_sz)}, options);

        EXPECT_TRUE(torch::equal(expected_k, actual_k));
        EXPECT_TRUE(torch::equal(expected_v, actual_v));
        EXPECT_TRUE(torch::equal(expected_sc_k, actual_sc_k));
        EXPECT_TRUE(torch::equal(expected_sc_v, actual_sc_v));
    }
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToBufferPartitionedLayerOutOfRangeReturnsEmpty) {
    auto config               = createTestConfig(/*k_block_bytes=*/512, /*v_block_bytes=*/512);
    config.is_mla             = false;
    config.local_head_num_kv  = 8;
    config.seq_size_per_block = 64;
    config.k_dim              = 1;
    config.v_dim              = 1;
    auto ctx                  = createTestContext(std::move(config), torch::kCPU, BufferInitMode::Zeros);

    auto          strategy = std::make_unique<LayerFirstLayoutStrategy>();
    torch::Tensor empty_scale;
    ASSERT_TRUE(strategy->init(ctx.config, ctx.kv_cache_buffer, empty_scale, ctx.cache_ptr));

    EXPECT_THROW((void)strategy->convertIndexToBuffer(static_cast<int>(ctx.config.layer_num) + 1, 0, 2, 0),
                 rtp_llm::RTPException);
}

TEST_F(LayerFirstLayoutStrategyTest, ConvertIndexToBufferPartitionedInvalidArgsThrow) {
    auto config               = createTestConfig(/*k_block_bytes=*/512, /*v_block_bytes=*/512);
    config.is_mla             = false;
    config.local_head_num_kv  = 8;
    config.seq_size_per_block = 64;
    config.k_dim              = 1;
    config.v_dim              = 1;
    auto ctx                  = createTestContext(std::move(config), torch::kCPU, BufferInitMode::Zeros);

    auto          strategy = std::make_unique<LayerFirstLayoutStrategy>();
    torch::Tensor empty_scale;
    ASSERT_TRUE(strategy->init(ctx.config, ctx.kv_cache_buffer, empty_scale, ctx.cache_ptr));

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
    auto ctx = createTestContext();

    auto          strategy = std::make_unique<LayerFirstLayoutStrategy>();
    torch::Tensor empty_scale;
    ASSERT_TRUE(strategy->init(ctx.config, ctx.kv_cache_buffer, empty_scale, ctx.cache_ptr));

    int  layer = 0;
    auto addr1 = strategy->convertIndexToAddr(layer, 0);
    auto addr2 = strategy->convertIndexToAddr(layer, 1);

    size_t addr1_val = reinterpret_cast<size_t>(addr1.kv_addr);
    size_t addr2_val = reinterpret_cast<size_t>(addr2.kv_addr);

    EXPECT_EQ(addr2_val - addr1_val, ctx.config.kv_block_stride_bytes);
}

// Layout Comparison Test
class LayoutComparisonTest: public MemoryLayoutStrategyTest {};

// Boundary Condition Test
TEST_F(MemoryLayoutStrategyTest, SingleLayerSingleBlock) {
    auto ctx = createTestContext(createTestConfig(/*layer_num=*/1,
                                                  /*block_num=*/1,
                                                  /*k_block_bytes=*/256,
                                                  /*v_block_bytes=*/256));

    auto          strategy = std::make_unique<LayerFirstLayoutStrategy>();
    torch::Tensor empty_scale;
    EXPECT_TRUE(strategy->init(ctx.config, ctx.kv_cache_buffer, empty_scale, ctx.cache_ptr));

    auto layer_tensors = strategy->getLayerCacheTensors();
    EXPECT_EQ(layer_tensors.size(), 1);

    auto addr_info = strategy->convertIndexToAddr(0, 0);
    EXPECT_NE(addr_info.kv_addr, nullptr);
}

TEST_F(MemoryLayoutStrategyTest, LargeConfiguration) {
    auto ctx = createTestContext(createTestConfig(/*layer_num=*/32,
                                                  /*block_num=*/1024,
                                                  /*k_block_bytes=*/2048,
                                                  /*v_block_bytes=*/2048));

    auto          strategy = std::make_unique<LayerFirstLayoutStrategy>();
    torch::Tensor empty_scale;
    EXPECT_TRUE(strategy->init(ctx.config, ctx.kv_cache_buffer, empty_scale, ctx.cache_ptr));

    auto layer_tensors = strategy->getLayerCacheTensors();
    EXPECT_EQ(layer_tensors.size(), ctx.config.layer_num);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
