#pragma once

#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "maga_transformer/cpp/cache/CacheConfigCreator.h"

namespace ft = fastertransformer;
namespace rtp_llm {

 CacheConfig CacheConfigCreator::createBasicConfig(const GptInitParameter& param) {
    // TODO(xinfei.sxf) tp_size from where
    size_t tp_size = 1;
    int local_head_num_kv = (param.head_num_kv_ > 1) ? param.head_num_kv_ / tp_size : param.head_num_kv_;
    auto dtype = param.int8_kv_cache_ ? ft::TYPE_INT8 : ft::TYPE_FP16;
    return CacheConfig((uint)param.num_layers_, (uint)0, (uint)local_head_num_kv, (uint)param.size_per_head_, (uint)param.seq_size_per_block_, dtype);
}

absl::StatusOr<int64_t> CacheConfigCreator::getKVCacheMemorySize(const GptInitParameter& param) {
    auto device = ft::DeviceFactory::getDefaultDevice();
    const auto memory_status = device->getDeviceStatus().device_memory_status;
    const auto free_bytes = memory_status.free_bytes;
    const auto total_bytes = memory_status.total_bytes;
    FT_LOG_INFO("free mem bytes: %lu", free_bytes);
    int64_t kv_cache_mem_size = (int64_t)free_bytes - (int64_t)param.reserve_runtime_mem_mb_ * 1024 * 1024;
    if (param.kv_cache_mem_mb_ > 0) {
        kv_cache_mem_size = (int64_t)param.kv_cache_mem_mb_ * 1024 * 1024;
    }
    if (kv_cache_mem_size <= 0) {
        FT_LOG_ERROR("kv cache mem size = %ld, it's <= 0\n", kv_cache_mem_size);
        return absl::InternalError("");
    }
    FT_LOG_INFO("kv cache mem size = %ld", kv_cache_mem_size);
    return kv_cache_mem_size;
}

absl::StatusOr<CacheConfig> CacheConfigCreator::createConfig(const GptInitParameter& param) {
    CacheConfig  config         = CacheConfigCreator::createBasicConfig(param);
    uint32_t     block_nums     = 0;
    if (param.block_nums_ > 0) {
        block_nums = param.block_nums_;
    } else {
        auto result = CacheConfigCreator::getKVCacheMemorySize(param);
        if (!result.ok()) {
            FT_LOG_ERROR("get kv cache memory size failed");
            return absl::InternalError("");
        }
        block_nums = result.value() / config.block_size / 2;
    }
    if (block_nums == 0) {
        FT_LOG_ERROR("kv cache block nums is 0");
        return absl::InternalError("");
    }
    config.block_nums = block_nums;
    FT_LOG_INFO("kv cache block nums is %u", block_nums);
    return config;
}

}  // namespace rtp_llm
