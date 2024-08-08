#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/cache/CacheConfigCreator.h"

namespace ft = fastertransformer;
namespace rtp_llm {

CacheConfig CacheConfigCreator::createBasicConfig(const ft::GptInitParameter& param) {
    int local_head_num_kv = (param.head_num_kv_ > 1) ? param.head_num_kv_ / param.tp_size_ : param.head_num_kv_;
    auto dtype = param.int8_kv_cache_ ? ft::TYPE_INT8 : ft::TYPE_FP16;
    return CacheConfig((uint)param.num_layers_, (uint)0, (uint)local_head_num_kv, (uint)param.size_per_head_, (uint)param.seq_size_per_block_, dtype);
}

absl::StatusOr<int64_t> CacheConfigCreator::getKVCacheMemorySize(const ft::GptInitParameter& param) {
    auto device = ft::DeviceFactory::getDefaultDevice();
    const auto memory_status = device->getDeviceStatus().device_memory_status;
    const auto free_bytes = memory_status.preserved_bytes;
    FT_LOG_INFO("kv cache available mem bytes: %lu", free_bytes);
    int64_t kv_cache_mem_size = (int64_t)free_bytes - (int64_t)param.reserve_runtime_mem_mb_ * 1024 * 1024;
    if (param.kv_cache_mem_mb_ > 0) {
        kv_cache_mem_size = (int64_t)param.kv_cache_mem_mb_ * 1024 * 1024;
    }
    if (kv_cache_mem_size <= 0) {
        return absl::InternalError("kv cache mem size = " + std::to_string(kv_cache_mem_size) + ", it's <= 0");
    }
    FT_LOG_INFO("kv cache mem size = %ld", kv_cache_mem_size);
    return kv_cache_mem_size;
}

absl::StatusOr<CacheConfig> CacheConfigCreator::createConfig(const ft::GptInitParameter& param) {
    CacheConfig  config         = CacheConfigCreator::createBasicConfig(param);
    uint32_t     block_nums     = 0;
    if (param.block_nums_ > 0) {
        block_nums = param.block_nums_;
    } else {
        CHECK_AND_RETURN_REF(result, CacheConfigCreator::getKVCacheMemorySize(param));
        block_nums = result / config.block_size;
    }
    if (block_nums == 0) {
        return absl::InternalError("kv cache block nums is 0");
    }
    config.block_nums = block_nums;
    config.reserve_runtime_mem_mb = param.reserve_runtime_mem_mb_;
    FT_LOG_INFO("kv cache block nums is %u", block_nums);
    return config;
}

}  // namespace rtp_llm
