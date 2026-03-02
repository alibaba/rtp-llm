#pragma once

#include <algorithm>
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

    config.kv_block_stride_bytes = spec->block_size_bytes();
    config.kv_block_size_bytes   = static_cast<size_t>(spec->block_size_bytes() * spec->layer_num);

    if (dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3) {
        const size_t kv_scale_kv_stride       = static_cast<size_t>(spec->local_head_num_kv) * tokens_per_block;
        const size_t kv_scale_kv_stride_bytes = kv_scale_kv_stride * sizeof(float);
        config.kv_scale_stride_bytes          = 2 * kv_scale_kv_stride_bytes;
        config.kv_scale_size_bytes            = static_cast<size_t>(layer_num) * config.kv_scale_stride_bytes;
    }

    config.block_size_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    // Per-layer block stride (kv + scale).
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    return config;
}

// A tiny helper for unit tests to construct a minimal "hybrid attention" KV cache config where `groupNums() > 1`.
//
// Notes:
// - This is a multi-group config that will trigger
//   HybridTypeKVCacheAllocator path (KVCacheManager selects hybrid allocator when groupNums()>1).
// - Groups are mixed: the first group is LinearAttention, the remaining groups are MultiHeadAttention.
// - For correctness with current hybrid allocator/memory layout, require `layer_num % group_layer_num == 0` and
//   `layer_num / group_layer_num >= 2` (at least 2 groups).
inline CacheConfig makeSimpleHybridMhaCacheConfig(int               layer_num,
                                                  int               block_num,
                                                  size_t            tokens_per_block,
                                                  rtp_llm::DataType dtype,
                                                  int               group_layer_num   = 2,
                                                  uint32_t          local_head_num_kv = 1,
                                                  uint32_t          size_per_head     = 1) {
    CacheConfig config;
    config.dtype              = dtype;
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.block_num          = static_cast<uint32_t>(block_num);
    config.seq_size_per_block = tokens_per_block;
    config.group_layer_num    = std::max(group_layer_num, 1);
    config.linear_step        = 2;

    // If the split is not even or cannot form >=2 groups, fall back to a single-group MHA config (keeps config valid).
    // Tests that need `groupNums()>1` should pass `layer_num % group_layer_num == 0` and
    // `layer_num/group_layer_num>=2`.
    if (layer_num <= 0 || (layer_num % config.group_layer_num) != 0 || (layer_num / config.group_layer_num) < 2) {
        return makeSimpleMhaCacheConfig(
            layer_num, block_num, tokens_per_block, dtype, local_head_num_kv, size_per_head);
    }

    const int group_cnt     = layer_num / config.group_layer_num;
    const int linear_groups = 1;
    const int full_groups   = group_cnt - 1;

    // Specs.
    auto linear_spec                = std::make_shared<LinearKVCacheSpec>();
    linear_spec->type               = KVCacheSpecType::LinearAttention;
    linear_spec->dtype              = dtype;
    linear_spec->layer_num          = static_cast<uint32_t>(config.group_layer_num);
    linear_spec->local_num_k_heads  = 1;
    linear_spec->local_num_v_heads  = 1;
    linear_spec->head_k_dim         = 1;
    linear_spec->head_v_dim         = 1;
    linear_spec->conv_kernel_dim    = 2;
    linear_spec->local_head_num_kv  = 1;
    linear_spec->seq_size_per_block = static_cast<uint32_t>(tokens_per_block);

    auto full_spec                = std::make_shared<MHAKVCacheSpec>();
    full_spec->type               = KVCacheSpecType::MultiHeadAttention;
    full_spec->dtype              = dtype;
    full_spec->seq_size_per_block = static_cast<uint32_t>(tokens_per_block);
    full_spec->layer_num          = static_cast<uint32_t>(config.group_layer_num);
    full_spec->local_head_num_kv  = local_head_num_kv;
    full_spec->size_per_head      = size_per_head;

    config.layer_ids.clear();
    config.global_layer_ids.clear();
    config.linear_groups.clear();
    config.full_groups.clear();
    config.cache_specs.clear();
    config.group_types.clear();

    config.layer_to_group_id.assign(static_cast<size_t>(layer_num), 0);

    // Build groups: gid=0 linear, gid>=1 full.
    for (int gid = 0; gid < group_cnt; ++gid) {
        std::vector<int> group_layers;
        group_layers.reserve(static_cast<size_t>(config.group_layer_num));
        for (int local = 0; local < config.group_layer_num; ++local) {
            const int layer_id = gid * config.group_layer_num + local;
            group_layers.push_back(layer_id);
            config.layer_to_group_id[static_cast<size_t>(layer_id)] = gid;
        }
        config.layer_ids.push_back(group_layers);
        config.global_layer_ids.push_back(group_layers);

        if (gid == 0) {
            config.cache_specs.push_back(linear_spec);
            config.group_types.push_back(CacheGroupType::LINEAR);
            config.linear_groups.push_back(group_layers);
        } else {
            config.cache_specs.push_back(full_spec);
            config.group_types.push_back(CacheGroupType::FULL);
            config.full_groups.push_back(group_layers);
        }
    }

    config.linear_group_num = linear_groups;
    config.full_group_num   = full_groups;

    // Physical sizes for hybrid memory layout: one group (group_layer_num) worth of layers.
    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = full_spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    // Per-layer block stride (kv + scale).
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));
    return config;
}

}  // namespace rtp_llm::test
