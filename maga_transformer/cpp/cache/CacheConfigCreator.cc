#include "src/fastertransformer/devices/DeviceFactory.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/cache/CacheConfigCreator.h"

namespace ft = fastertransformer;
namespace rtp_llm {

CacheConfig CacheConfigCreator::createBasicConfig(const ft::GptInitParameter& param) {
    int local_head_num_kv = (param.head_num_kv_ > 1) ? param.head_num_kv_ / param.tp_size_ : param.head_num_kv_;
    const auto device_prop = ft::DeviceFactory::getDefaultDevice()->getDeviceProperties();
    auto dtype = param.kv_cache_data_type_;
    if (device_prop.type == ft::DeviceType::ArmCpu) {
        // Arm attention operator support FP32 data type only
        dtype = param.kv_cache_data_type_ == ft::DataType::TYPE_INT8 ? ft::TYPE_INT8 : ft::TYPE_FP32;
    }
    return CacheConfig((uint)param.num_layers_, (uint)0, (uint)local_head_num_kv, (uint)param.size_per_head_, (uint)param.seq_size_per_block_, dtype);
}

size_t CacheConfigCreator::getDefaultRuntimeMemorySize(const ft::GptInitParameter& params) {
    auto reserve_runtime_mem_bytes = params.reserve_runtime_mem_mb_ * 1024 * 1024;
    FT_LOG_INFO("GptInitParameter has reserve_runtime_mem_mb_=%ld", params.reserve_runtime_mem_mb_);

    const auto minimal_runtime_bytes = 256L * 1024 * 1024 * std::min(4, (int)params.tp_size_);
    if (reserve_runtime_mem_bytes < minimal_runtime_bytes) {
        reserve_runtime_mem_bytes = minimal_runtime_bytes;
        FT_LOG_INFO("tp_size %d needs at least %d MiB memory for runtime by default, "
                    "but only %ld MiB reserved memory set by config. adjust to minimal value.",
                    params.tp_size_,
                    minimal_runtime_bytes / 1024 / 1024,
                    reserve_runtime_mem_bytes / 1024 / 1024);
    }

    if (params.is_multimodal_) {
        const auto minimal_runtime_required = 2L * 1024 * 1024 * 1024; // 2 GiB
        if (reserve_runtime_mem_bytes < minimal_runtime_required) {
            reserve_runtime_mem_bytes = minimal_runtime_required;
            FT_LOG_INFO("multimodal needs at least %ld MiB memory for runtime by default, "
                        "but only %ld MiB memory reserved. adjust to minimal value.",
                        minimal_runtime_required / 1024 / 1024,
                        reserve_runtime_mem_bytes / 1024 / 1024);
        }
    }

    return reserve_runtime_mem_bytes;
}

size_t CacheConfigCreator::getKVCacheMemorySize(
        const ft::GptInitParameter& params,
        const std::optional<WarmUpResult>& warm_up_result)
{
    const auto device = ft::DeviceFactory::getDefaultDevice();
    size_t device_reserved_memory_bytes = device->getDeviceStatus().device_memory_status.preserved_bytes;
    size_t runtime_required_bytes = 0;

    if (params.kv_cache_mem_mb_ > 0) {
        FT_LOG_INFO("GptInitParameter explicitly specified kv cache memory size %ld MiB",
                    params.kv_cache_mem_mb_);
        return params.kv_cache_mem_mb_ * 1024 * 1024;
    }

    if (warm_up_result) {
        if (device_reserved_memory_bytes != warm_up_result->device_reserved_bytes) {
            FT_LOG_WARNING("device reserved memory bytes %ld when create config does not equal to "
                           "the amount when warm up %ld. take min value.",
                            device_reserved_memory_bytes,
                            warm_up_result->device_reserved_bytes);
            device_reserved_memory_bytes = std::min(device_reserved_memory_bytes, warm_up_result->device_reserved_bytes);
        }

        FT_LOG_INFO("devices reserved %ld MiB memory, warm up consumed %ld MiB max memory",
                    device_reserved_memory_bytes / 1024 / 1024,
                    warm_up_result->max_used_memory / 1024 / 1024);
        runtime_required_bytes = warm_up_result->max_used_memory;
    } else {
        runtime_required_bytes = getDefaultRuntimeMemorySize(params);
        FT_LOG_INFO("warm up result not available, use default runtime memory size %ld MiB",
                    runtime_required_bytes / 1024 / 1024);
    }

    size_t sample_need_mem = (size_t)params.max_generate_batch_size_ * params.vocab_size_ * 4 * 8; // just estimated value
    FT_LOG_INFO("sampler needs %ld MiB memory, model runtime needs %ld MiB memory, take max value.",
                sample_need_mem / 1024 / 1024,
                runtime_required_bytes / 1024 / 1024);
    runtime_required_bytes = std::max(sample_need_mem, runtime_required_bytes);

    FT_CHECK_WITH_INFO(device_reserved_memory_bytes > runtime_required_bytes,
                       "device reserved memory %ld  MiB is less than runtime required memory %ld MiB",
                       device_reserved_memory_bytes / 1024 / 1024,
                       runtime_required_bytes / 1024 / 1024);

    const auto kv_cache_mem_size = device_reserved_memory_bytes - runtime_required_bytes;
    FT_LOG_INFO("cache config final decided kv cache memory size %ld MiB",
                kv_cache_mem_size / 1024 / 1024);
    return kv_cache_mem_size;
}

CacheConfig CacheConfigCreator::createConfig(
        const ft::GptInitParameter& param,
        const std::optional<WarmUpResult>& warm_up_result)
{
    CacheConfig  config         = CacheConfigCreator::createBasicConfig(param);
    uint32_t     block_nums     = 0;

    if (param.block_nums_ > 0) {
        FT_LOG_INFO("GptInitParameter explicitly specified kv cache block num %d", param.block_nums_);
        block_nums = param.block_nums_;
    } else {
        const auto kv_cache_mem_size = getKVCacheMemorySize(param, warm_up_result);
        block_nums = kv_cache_mem_size / config.block_size;
    }
    FT_CHECK_WITH_INFO(block_nums > 0,
                       "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                       block_nums, config.block_size / 1024 / 1024);

    const auto kv_cache_seq_len = block_nums * config.seq_size_per_block;
    config.block_nums = block_nums;
    config.reserve_runtime_mem_mb = param.reserve_runtime_mem_mb_;
    FT_LOG_INFO("kv cache block nums is %u, allows storing %ld tokens", block_nums, kv_cache_seq_len);
    if (kv_cache_seq_len < param.max_seq_len_) {
        FT_LOG_WARNING("kv cache block nums %u can only store %ld tokens, less than max_seq_len %ld, "
                       "this is dangerous, consider decrease max_seq_len",
                       block_nums, kv_cache_seq_len, param.max_seq_len_);
    }
    return config;
}

std::tuple<CacheConfig, CacheConfig> CacheConfigCreator::createSpConfig(
        const ft::GptInitParameter& score_param, const ft::GptInitParameter& propose_param)
{
    CacheConfig  score_config         = CacheConfigCreator::createBasicConfig(score_param);
    CacheConfig  propose_config         = CacheConfigCreator::createBasicConfig(propose_param);
    size_t     block_nums     = 0;
    if (score_param.block_nums_ > 0) {
        block_nums = score_param.block_nums_;
    } else {
        const auto kv_cache_mem_size = CacheConfigCreator::getKVCacheMemorySize(score_param);
        block_nums = kv_cache_mem_size / (score_config.block_size + propose_config.block_size);
    }
    FT_CHECK_WITH_INFO(block_nums > 0, "kv cache needs at least 1 block but %ld", block_nums);

    score_config.block_nums = block_nums;
    propose_config.block_nums = block_nums;
    score_config.reserve_runtime_mem_mb = score_param.reserve_runtime_mem_mb_;
    propose_config.reserve_runtime_mem_mb = 0;
    FT_LOG_INFO("kv cache block nums is %u", block_nums);
    return std::make_tuple(score_config, propose_config);
}

}  // namespace rtp_llm
