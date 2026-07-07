#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm::test {

inline CacheConfig makeSimpleMhaCacheConfig(int               layer_num,
                                            int               block_num,
                                            size_t            tokens_per_block,
                                            rtp_llm::DataType dtype,
                                            uint32_t          local_head_num_kv = 1,
                                            uint32_t          size_per_head     = 1) {
    CacheConfig config;
    config.dtype                     = dtype;
    config.layer_num                 = static_cast<uint32_t>(layer_num);
    config.layer_all_num             = static_cast<uint32_t>(layer_num);
    config.block_num                 = static_cast<uint32_t>(block_num);
    config.seq_size_per_block        = tokens_per_block;
    config.kernel_seq_size_per_block = tokens_per_block;

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = "default";
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
    config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});
    config.group_seq_size_per_block = {tokens_per_block};

    config.kv_block_stride_bytes = spec->block_size_bytes();
    config.kv_block_size_bytes   = static_cast<size_t>(spec->block_size_bytes() * spec->layer_num);

    if (dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3) {
        const size_t kv_scale_kv_stride       = static_cast<size_t>(spec->local_head_num_kv) * tokens_per_block;
        const size_t kv_scale_kv_stride_bytes = kv_scale_kv_stride * sizeof(float);
        config.kv_scale_stride_bytes          = 2 * kv_scale_kv_stride_bytes;
        config.kv_scale_size_bytes            = static_cast<size_t>(layer_num) * config.kv_scale_stride_bytes;
    }

    config.block_size_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    return config;
}

inline CacheConfig makeSimpleHybridMhaCacheConfig(int               layer_num,
                                                  int               block_num,
                                                  size_t            tokens_per_block,
                                                  rtp_llm::DataType dtype,
                                                  int               group_layer_num   = 2,
                                                  uint32_t          local_head_num_kv = 1,
                                                  uint32_t          size_per_head     = 1) {
    CacheConfig config;
    config.dtype                     = dtype;
    config.layer_num                 = static_cast<uint32_t>(layer_num);
    config.layer_all_num             = static_cast<uint32_t>(layer_num);
    config.block_num                 = static_cast<uint32_t>(block_num);
    config.seq_size_per_block        = tokens_per_block;
    config.kernel_seq_size_per_block = tokens_per_block;
    config.group_layer_num           = std::max(group_layer_num, 1);
    config.linear_step               = 2;

    if (layer_num <= 0 || (layer_num % config.group_layer_num) != 0 || (layer_num / config.group_layer_num) < 2) {
        return makeSimpleMhaCacheConfig(
            layer_num, block_num, tokens_per_block, dtype, local_head_num_kv, size_per_head);
    }

    const int group_cnt = layer_num / config.group_layer_num;

    auto linear_spec                = std::make_shared<LinearKVCacheSpec>();
    linear_spec->tag                = "linear";
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
    full_spec->tag                = "full";
    full_spec->type               = KVCacheSpecType::MultiHeadAttention;
    full_spec->dtype              = dtype;
    full_spec->seq_size_per_block = static_cast<uint32_t>(tokens_per_block);
    full_spec->layer_num          = static_cast<uint32_t>(config.group_layer_num);
    full_spec->local_head_num_kv  = local_head_num_kv;
    full_spec->size_per_head      = size_per_head;

    std::vector<KVCacheSpecPtr>   specs;
    std::vector<std::vector<int>> layers_by_group;
    std::vector<CacheGroupType>   types;
    std::vector<std::string>      tags;
    specs.reserve(static_cast<size_t>(group_cnt));
    layers_by_group.reserve(static_cast<size_t>(group_cnt));
    types.reserve(static_cast<size_t>(group_cnt));
    tags.reserve(static_cast<size_t>(group_cnt));

    for (int gid = 0; gid < group_cnt; ++gid) {
        std::vector<int> group_layers;
        group_layers.reserve(static_cast<size_t>(config.group_layer_num));
        for (int local = 0; local < config.group_layer_num; ++local) {
            group_layers.push_back(gid * config.group_layer_num + local);
        }
        if (gid == 0) {
            specs.push_back(linear_spec);
            types.push_back(CacheGroupType::LINEAR);
            tags.push_back("linear");
        } else {
            specs.push_back(full_spec);
            types.push_back(CacheGroupType::FULL);
            tags.push_back("full" + std::to_string(gid));
        }
        layers_by_group.push_back(std::move(group_layers));
    }
    config.fromGroupedSpecs(specs, layers_by_group, types, tags);
    config.group_seq_size_per_block.assign(static_cast<size_t>(group_cnt), tokens_per_block);

    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = full_spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));
    return config;
}

}  // namespace rtp_llm::test
