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
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->type               = KVCacheSpecType::MultiHeadAttention;
    spec->dtype              = dtype;
    spec->seq_size_per_block = static_cast<uint32_t>(tokens_per_block);
    spec->layer_num          = static_cast<uint32_t>(layer_num);
    spec->local_head_num_kv  = local_head_num_kv;
    spec->size_per_head      = size_per_head;

    std::vector<int> layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        layer_ids[i] = i;
    }

    const size_t kv_block_stride_bytes = spec->block_size_bytes();
    const size_t kv_block_size_bytes   = kv_block_stride_bytes * static_cast<size_t>(layer_num);
    size_t       kv_scale_stride_bytes = 0;
    size_t       kv_scale_size_bytes   = 0;
    if (dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3) {
        const size_t kv_scale_kv_stride_bytes =
            static_cast<size_t>(local_head_num_kv) * tokens_per_block * sizeof(float);
        kv_scale_stride_bytes = 2 * kv_scale_kv_stride_bytes;
        kv_scale_size_bytes   = static_cast<size_t>(layer_num) * kv_scale_stride_bytes;
    }
    const size_t block_size_bytes       = kv_block_size_bytes + kv_scale_size_bytes;
    const size_t per_layer_stride_bytes = kv_block_stride_bytes + kv_scale_stride_bytes;

    // Build KVCacheAllocatorConfig with all per-model fields.
    KVCacheAllocatorConfig alloc;
    alloc.model_id    = 0;
    alloc.layer_num   = static_cast<uint32_t>(layer_num);
    alloc.dtype       = dtype;
    alloc.use_mla     = false;
    alloc.is_sparse   = false;
    alloc.cache_specs = {spec};
    alloc.group_types = {CacheGroupType::FULL};
    alloc.layer_ids   = {layer_ids};
    alloc.layer_to_group_id.assign(layer_num, 0);
    alloc.layer_to_block_stride_bytes.assign(static_cast<size_t>(layer_num), static_cast<int>(per_layer_stride_bytes));
    alloc.block_num             = static_cast<uint32_t>(block_num);
    alloc.seq_size_per_block    = tokens_per_block;
    alloc.kv_block_size_bytes   = kv_block_size_bytes;
    alloc.kv_scale_size_bytes   = kv_scale_size_bytes;
    alloc.block_size_bytes      = block_size_bytes;
    alloc.kv_block_stride_bytes = kv_block_stride_bytes;
    alloc.kv_scale_stride_bytes = kv_scale_stride_bytes;
    alloc.linear_step           = 1;
    alloc.group_layer_num       = layer_num;
    alloc.linear_group_num      = 0;
    alloc.full_group_num        = 1;

    // Build CacheConfig with cross-model fields only.
    CacheConfig config;
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.seq_size_per_block = tokens_per_block;
    config.layer_to_group_id.assign(layer_num, 0);
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(layer_num), static_cast<int>(per_layer_stride_bytes));
    config.allocator_configs.push_back(std::move(alloc));

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
    const int eff_group_layer_num = std::max(group_layer_num, 1);

    // If the split is not even or cannot form >=2 groups, fall back to a single-group MHA config (keeps config valid).
    if (layer_num <= 0 || (layer_num % eff_group_layer_num) != 0 || (layer_num / eff_group_layer_num) < 2) {
        return makeSimpleMhaCacheConfig(
            layer_num, block_num, tokens_per_block, dtype, local_head_num_kv, size_per_head);
    }

    const int group_cnt     = layer_num / eff_group_layer_num;
    const int linear_groups = 1;
    const int full_groups   = group_cnt - 1;

    // Specs.
    auto linear_spec                = std::make_shared<LinearKVCacheSpec>();
    linear_spec->type               = KVCacheSpecType::LinearAttention;
    linear_spec->dtype              = dtype;
    linear_spec->layer_num          = static_cast<uint32_t>(eff_group_layer_num);
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
    full_spec->layer_num          = static_cast<uint32_t>(eff_group_layer_num);
    full_spec->local_head_num_kv  = local_head_num_kv;
    full_spec->size_per_head      = size_per_head;

    std::vector<KVCacheSpecPtr>   cache_specs;
    std::vector<CacheGroupType>   group_types;
    std::vector<int>              layer_to_group_id(layer_num, 0);
    std::vector<std::vector<int>> layer_ids_for_groups;

    for (int gid = 0; gid < group_cnt; ++gid) {
        std::vector<int> group_layers;
        group_layers.reserve(static_cast<size_t>(eff_group_layer_num));
        for (int local = 0; local < eff_group_layer_num; ++local) {
            const int layer_id = gid * eff_group_layer_num + local;
            group_layers.push_back(layer_id);
            layer_to_group_id[static_cast<size_t>(layer_id)] = gid;
        }
        layer_ids_for_groups.push_back(group_layers);

        if (gid == 0) {
            cache_specs.push_back(linear_spec);
            group_types.push_back(CacheGroupType::LINEAR);
        } else {
            cache_specs.push_back(full_spec);
            group_types.push_back(CacheGroupType::FULL);
        }
    }

    // Physical sizes for hybrid memory layout.
    const size_t kv_block_stride_bytes  = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    const size_t kv_block_size_bytes    = static_cast<size_t>(eff_group_layer_num) * kv_block_stride_bytes;
    const size_t kv_scale_stride_bytes  = full_spec->scale_block_size_bytes();
    const size_t kv_scale_size_bytes    = static_cast<size_t>(eff_group_layer_num) * kv_scale_stride_bytes;
    const size_t block_size_bytes       = kv_block_size_bytes + kv_scale_size_bytes;
    const size_t per_layer_stride_bytes = kv_block_stride_bytes + kv_scale_stride_bytes;

    // Build KVCacheAllocatorConfig with all per-model fields.
    KVCacheAllocatorConfig alloc;
    alloc.model_id          = 0;
    alloc.layer_num         = static_cast<uint32_t>(layer_num);
    alloc.dtype             = dtype;
    alloc.use_mla           = false;
    alloc.is_sparse         = false;
    alloc.cache_specs       = cache_specs;
    alloc.group_types       = group_types;
    alloc.layer_ids         = layer_ids_for_groups;
    alloc.layer_to_group_id = layer_to_group_id;
    alloc.layer_to_block_stride_bytes.assign(static_cast<size_t>(layer_num), static_cast<int>(per_layer_stride_bytes));
    alloc.block_num             = static_cast<uint32_t>(block_num);
    alloc.seq_size_per_block    = tokens_per_block;
    alloc.kv_block_size_bytes   = kv_block_size_bytes;
    alloc.kv_scale_size_bytes   = kv_scale_size_bytes;
    alloc.block_size_bytes      = block_size_bytes;
    alloc.kv_block_stride_bytes = kv_block_stride_bytes;
    alloc.kv_scale_stride_bytes = kv_scale_stride_bytes;
    alloc.linear_step           = 2;
    alloc.group_layer_num       = eff_group_layer_num;
    alloc.linear_group_num      = linear_groups;
    alloc.full_group_num        = full_groups;

    // Build CacheConfig with cross-model fields only.
    CacheConfig config;
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.seq_size_per_block = tokens_per_block;
    config.layer_to_group_id  = layer_to_group_id;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(layer_num), static_cast<int>(per_layer_stride_bytes));
    config.allocator_configs.push_back(std::move(alloc));

    return config;
}

}  // namespace rtp_llm::test
