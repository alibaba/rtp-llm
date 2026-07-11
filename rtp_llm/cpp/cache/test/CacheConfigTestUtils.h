#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

namespace rtp_llm::test {

inline KVCacheSpecPtr makeMhaSpec(const std::string& tag,
                                  size_t             tokens_per_block,
                                  rtp_llm::DataType  dtype,
                                  uint32_t           local_head_num_kv,
                                  uint32_t           size_per_head) {
    AttentionConfigs attn_config;
    attn_config.kv_head_num      = local_head_num_kv;
    attn_config.size_per_head    = size_per_head;
    attn_config.tokens_per_block = static_cast<uint32_t>(tokens_per_block);

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = static_cast<uint32_t>(tokens_per_block);
    ctx.attn_config             = &attn_config;
    ctx.parallelism_config      = &parallelism_config;
    return SpecBuilder::build(desc, ctx);
}

inline KVCacheSpecPtr makeLinearSpec(const std::string& tag,
                                     size_t             tokens_per_block,
                                     rtp_llm::DataType  dtype,
                                     uint32_t           local_head_num_kv,
                                     uint32_t           size_per_head) {
    LinearAttentionConfig linear_config;
    linear_config.linear_conv_kernel_dim = 2;
    linear_config.linear_key_head_dim    = static_cast<int>(size_per_head);
    linear_config.linear_value_head_dim  = static_cast<int>(size_per_head);
    linear_config.linear_num_key_heads   = static_cast<int>(local_head_num_kv);
    linear_config.linear_num_value_heads = static_cast<int>(local_head_num_kv);

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::LinearAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = static_cast<uint32_t>(tokens_per_block);
    ctx.linear_attention_config = &linear_config;
    ctx.parallelism_config      = &parallelism_config;
    return SpecBuilder::build(desc, ctx);
}

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

    auto spec = makeMhaSpec("default", tokens_per_block, dtype, local_head_num_kv, size_per_head);

    std::vector<int> layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        layer_ids[i] = i;
    }
    config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});

    config.kv_block_stride_bytes = spec->block_size_bytes();
    config.kv_block_size_bytes   = static_cast<size_t>(layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;

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

    auto linear_spec = makeLinearSpec("linear", tokens_per_block, dtype, local_head_num_kv, size_per_head);
    auto full_spec   = makeMhaSpec("full", tokens_per_block, dtype, local_head_num_kv, size_per_head);

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
