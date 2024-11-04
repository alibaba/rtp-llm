#include "src/fastertransformer/devices/DeviceFactory.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/cache/CacheConfigCreator.h"

namespace ft = fastertransformer;
namespace rtp_llm {

CacheConfig CacheConfigCreator::createBasicConfig(const ft::GptInitParameter& param) {
    int local_head_num_kv = (param.head_num_kv_ > 1) ? param.head_num_kv_ / param.tp_size_ : param.head_num_kv_;
#if defined(__aarch64__)
    // Arm attention operator support FP32 data type only
    auto dtype = param.kv_cache_data_type_ == ft::DataType::TYPE_INT8 ? ft::TYPE_INT8 : ft::TYPE_FP32;
#else
    auto dtype = param.kv_cache_data_type_;
#endif
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
    if (param.is_multimodal_) {
        const auto runtime_left_mem_bytes = (int64_t)free_bytes - kv_cache_mem_size;
        const auto target_runtime_bytes = 2L * 1024 * 1024 * 1024; // 2 GiB
        if (runtime_left_mem_bytes < target_runtime_bytes) {
            const auto lack_mem_bytes = target_runtime_bytes - runtime_left_mem_bytes;
            kv_cache_mem_size -= lack_mem_bytes;
            FT_LOG_WARNING("multimodal needs at least 2 GiB memory for runtime, "
                           "but only %ld MB reserved memory left. "
                           "thus we shrink kv cache memory size by %ld MB to %ld MB",
                           runtime_left_mem_bytes / 1024 / 1024,
                           lack_mem_bytes / 1024 / 1024,
                           kv_cache_mem_size / 1024 / 1024);
        }
    }
    if (param.tp_size_ > 1) {
        const auto runtime_left_mem_bytes = (int64_t)free_bytes - kv_cache_mem_size;
        const auto minimal_runtime_bytes = 256L * 1024 * 1024 * std::min(4, (int)param.tp_size_);
        if (runtime_left_mem_bytes < minimal_runtime_bytes) {
            const auto lack_mem_bytes = minimal_runtime_bytes - runtime_left_mem_bytes;
            kv_cache_mem_size -= lack_mem_bytes;
            FT_LOG_WARNING("tp_size %d needs at least %d MiB memory for runtime, "
                           "but only %ld MB reserved memory left. "
                           "thus we shrink kv cache memory size by %ld MB to %ld MB",
                           param.tp_size_,
                           minimal_runtime_bytes / 1024 / 1024,
                           runtime_left_mem_bytes / 1024 / 1024,
                           lack_mem_bytes / 1024 / 1024,
                           kv_cache_mem_size / 1024 / 1024);
        }
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
        CHECK_AND_RETURN_REF(kv_cache_mem_size, CacheConfigCreator::getKVCacheMemorySize(param));
        block_nums = kv_cache_mem_size / config.block_size;
    }
    if (block_nums == 0) {
        return absl::InternalError("kv cache block nums is 0");
    }
    config.block_nums = block_nums;
    config.reserve_runtime_mem_mb = param.reserve_runtime_mem_mb_;
    FT_LOG_INFO("kv cache block nums is %u", block_nums);
    return config;
}

absl::StatusOr<std::tuple<CacheConfig, CacheConfig>> CacheConfigCreator::createSpConfig(const ft::GptInitParameter& score_param, const ft::GptInitParameter& propose_param) {
    CacheConfig  score_config         = CacheConfigCreator::createBasicConfig(score_param);
    CacheConfig  propose_config         = CacheConfigCreator::createBasicConfig(propose_param);
    size_t     block_nums     = 0;
    if (score_param.block_nums_ > 0) {
        block_nums = score_param.block_nums_;
    } else {
        CHECK_AND_RETURN_REF(kv_cache_mem_size, CacheConfigCreator::getKVCacheMemorySize(score_param));
        block_nums = kv_cache_mem_size / (score_config.block_size + propose_config.block_size);
    }
    if (block_nums == 0) {
        return absl::InternalError("kv cache block nums is 0");
    }
    score_config.block_nums = block_nums;
    propose_config.block_nums = block_nums;
    score_config.reserve_runtime_mem_mb = score_param.reserve_runtime_mem_mb_;
    propose_config.reserve_runtime_mem_mb = 0;
    FT_LOG_INFO("kv cache block nums is %u", block_nums);
    return std::make_tuple(score_config, propose_config);
}

}  // namespace rtp_llm
