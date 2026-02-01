#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm::test {

// A tiny helper for unit tests to construct a minimal MultiHeadAttention KV cache config.
//
// NOTE:
// - This is test-only code. Keep it small and dependency-light.
// - The returned CacheConfig is fully initialized (no uninitialized fundamental fields).
inline CacheConfig makeSimpleMhaCacheConfig(int               layer_num,
                                            int               block_num,
                                            size_t            tokens_per_block,
                                            rtp_llm::DataType dtype,
                                            uint32_t          local_head_num_kv = 1,
                                            uint32_t          size_per_head     = 1) {
    CacheConfig config;
    config.dtype              = dtype;
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.block_num          = static_cast<uint32_t>(block_num);
    config.seq_size_per_block = tokens_per_block;

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->type               = KVCacheSpecType::MultiHeadAttention;
    spec->dtype              = dtype;
    spec->seq_size_per_block = static_cast<uint32_t>(tokens_per_block);
    spec->layer_num          = static_cast<uint32_t>(layer_num);
    spec->local_head_num_kv  = local_head_num_kv;
    spec->size_per_head      = size_per_head;
    config.cache_specs.push_back(spec);

    std::vector<int> layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        layer_ids[i] = i;
    }
    config.layer_ids.push_back(layer_ids);
    config.global_layer_ids.push_back(layer_ids);
    config.layer_to_group_id.assign(layer_num, 0);

    // config.kv_block_stride       = static_cast<size_t>(spec->block_size());
    config.kv_block_stride_bytes = spec->block_size_bytes();
    // config.kv_block_size         = static_cast<size_t>(spec->block_size() * spec->layer_num);
    config.kv_block_size_bytes = static_cast<size_t>(spec->block_size_bytes() * spec->layer_num);

    if (dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3) {
        const size_t kv_scale_kv_stride       = static_cast<size_t>(spec->local_head_num_kv) * tokens_per_block;
        const size_t kv_scale_kv_stride_bytes = kv_scale_kv_stride * sizeof(float);
        // config.kv_scale_stride                = 2 * kv_scale_kv_stride;
        config.kv_scale_stride_bytes = 2 * kv_scale_kv_stride_bytes;
        // config.kv_scale_size                  = static_cast<size_t>(layer_num) * config.kv_scale_stride;
        config.kv_scale_size_bytes = static_cast<size_t>(layer_num) * config.kv_scale_stride_bytes;
    }

    // config.block_stride       = config.kv_block_stride + config.kv_scale_stride;
    // config.block_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    // config.block_size         = config.kv_block_size + config.kv_scale_size;
    config.block_size_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    return config;
}

}  // namespace rtp_llm::test
