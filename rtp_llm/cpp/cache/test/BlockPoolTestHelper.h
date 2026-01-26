#pragma once

#include <cstdint>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

namespace rtp_llm {

inline KVCacheSpecPtr createTestKvCacheSpec(uint32_t          layer_num,
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
    RTP_LLM_CHECK_WITH_INFO(denom > 0, "invalid denom");
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
        // Use MLA spec to allow different K/V sizes.
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

inline BlockPoolConfig createTestConfig(size_t            k_block_stride_bytes = 512,
                                        size_t            v_block_stride_bytes = 512,
                                        rtp_llm::DataType dtype                = rtp_llm::DataType::TYPE_FP16,
                                        uint32_t          local_head_num_kv    = 1,
                                        uint32_t          seq_size_per_block   = 1) {
    constexpr uint32_t kLayerNum = 4;
    constexpr uint32_t kBlockNum = 10;

    auto spec = createTestKvCacheSpec(
        kLayerNum, dtype, local_head_num_kv, seq_size_per_block, k_block_stride_bytes, v_block_stride_bytes);
    return BlockPoolConfigHelper::createLayerFirstConfig(kLayerNum, kBlockNum, spec);
}

DeviceBase* createDevice() {
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
    rtp_llm::ModelSpecificConfig         model_specific_config;

    // Keep tests stable on shared GPUs with low free memory:
    // - device_reserve_memory_bytes=0 => use DeviceFactory default (-512MB), i.e. reserve (free - 512MB)
    // - host_reserve_memory_bytes=0  => don't reserve pinned host memory
    device_resource_config.device_reserve_memory_bytes = 0;
    device_resource_config.host_reserve_memory_bytes   = 0;

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
    return rtp_llm::DeviceFactory::getDefaultDevice();
}

BlockPoolPtr createBlockPool() {
    auto device     = createDevice();
    auto config     = createTestConfig();
    auto block_pool = std::make_shared<BlockPool>(config, device);
    return block_pool;
}

}  // namespace rtp_llm
