#pragma once

#include <cstdint>
#include <numeric>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

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
        spec->local_head_num_kv  = local_head_num_kv;
        spec->seq_size_per_block = seq_size_per_block;
        spec->size_per_head      = static_cast<uint32_t>(k_elems / denom);
        return spec;
    } else {
        // Use MLA spec to allow different K/V sizes.
        auto spec                = std::make_shared<MLAKVCacheSpec>();
        spec->type               = KVCacheSpecType::MultiHeadLatentAttention;
        spec->dtype              = dtype;
        spec->local_head_num_kv  = local_head_num_kv;
        spec->seq_size_per_block = seq_size_per_block;
        spec->kv_lora_rank       = static_cast<uint32_t>(k_elems / denom);
        spec->rope_head_dim      = static_cast<uint32_t>(v_elems / denom);
        return spec;
    }
}

inline DeviceBlockPoolConfig createTestConfig(size_t            k_block_stride_bytes = 512,
                                              size_t            v_block_stride_bytes = 512,
                                              size_t            k_scale_stride_bytes = 0,
                                              size_t            v_scale_stride_bytes = 0,
                                              DataType          dtype = DataType::TYPE_FP16,
                                              uint32_t          local_head_num_kv = 1,
                                              uint32_t          seq_size_per_block = 1) {
    constexpr uint32_t kLayerNum = 4;
    constexpr uint32_t kBlockNum = 10;
    auto spec = createTestKvCacheSpec(kLayerNum,
                                      dtype,
                                      local_head_num_kv,
                                      seq_size_per_block,
                                      k_block_stride_bytes,
                                      v_block_stride_bytes);
    CacheConfig cache_config;
    cache_config.layer_num             = kLayerNum;
    cache_config.layer_all_num         = kLayerNum;
    cache_config.block_num             = kBlockNum;
    cache_config.dtype                 = dtype;
    cache_config.seq_size_per_block    = seq_size_per_block;
    cache_config.kv_block_stride_bytes = k_block_stride_bytes + v_block_stride_bytes;
    cache_config.kv_scale_stride_bytes = k_scale_stride_bytes + v_scale_stride_bytes;
    std::vector<int> layer_ids(kLayerNum);
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    cache_config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});
    return DeviceBlockPoolConfigHelper::createConfig(cache_config);
}

inline void createDevice() {
    torch::manual_seed(114514);
    rtp_llm::initRuntime(/*device_id=*/0,
                         /*trace_memory=*/false,
                         /*enable_comm_overlap=*/false,
                         rtp_llm::MlaOpsType::AUTO);
}

// Build the DeviceBlockPool from the same test config, for the KVCacheGroup / allocator
// tests (single-count incRef/decRef pool).
inline DeviceBlockPoolPtr createDeviceBlockPool() {
    createDevice();
    auto device_config                     = std::make_shared<DeviceBlockPoolConfig>(createTestConfig());
    device_config->use_cuda_malloc_backing = false;
    std::shared_ptr<const DeviceBlockPoolConfig> const_config = device_config;
    return std::make_shared<DeviceBlockPool>(const_config);
}

}  // namespace rtp_llm
