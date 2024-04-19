#pragma once

#include <cuda_runtime.h>
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

std::tuple<bool, int64_t> CacheConfigCreator::getKVCacheMemorySize(const GptInitParameter& param) {
    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes)) {
        FT_LOG_ERROR("cudaMemGetInfo failed");
        return {false, 0};
    }
    FT_LOG_INFO("free kv cache mem size: %d", free_bytes);
    int64_t kv_cache_mem_size = (int64_t)free_bytes - param.reserve_runtime_mem_mb_ * 1024 * 1024;
    if (param.kv_cache_mem_mb_ > 0) {
        kv_cache_mem_size = param.kv_cache_mem_mb_ * 1024 * 1024;
    }
    if (kv_cache_mem_size <= 0) {
        FT_LOG_ERROR("kv cache mem size = %ld, it's less than 0\n", kv_cache_mem_size);
        return {false, 0};
    }
    FT_LOG_INFO("kv cache mem size = %ld", kv_cache_mem_size);
    return {true, kv_cache_mem_size};
}

std::tuple<bool, CacheConfig> CacheConfigCreator::createConfig(const GptInitParameter& param) {
    CacheConfig  config  = CacheConfigCreator::createBasicConfig(param);
    uint     block_nums  = 0;
    char* block_num_env  = std::getenv("TEST_BLOCK_NUM"); // for test
    if (block_num_env) {
        try {
            block_nums = std::stoi(block_num_env);
        } catch (std::invalid_argument const &e) {
            FT_LOG_ERROR("Invalid argument: block_num_env = %s", block_num_env);
            return {false, {}};
        } catch (std::out_of_range const &e) {
            FT_LOG_ERROR("Out of range: block_num_env = %s", block_num_env);
            return {false, {}};
        }
    } else {
        auto [success, kv_cache_mem_size] = CacheConfigCreator::getKVCacheMemorySize(param);
        if (!success) {
            FT_LOG_ERROR("getKVCacheMemorySize failed");
            return {false, {}};
        }
        block_nums = kv_cache_mem_size / config.block_size / 2;
    }
    if (block_nums == 0) {
        FT_LOG_ERROR("block_nums = 0");
        return {false, {}};
    }
    config.block_nums = block_nums;
    return {true, config};
}

}  // namespace rtp_llm
