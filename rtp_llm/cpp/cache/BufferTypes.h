#pragma once

#include <vector>
#include <string>

#include <torch/extension.h>
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

struct KVCachePoolMetricsSnapshot {
    std::string storage;
    size_t capacity_bytes = 0;
    size_t occupied_bytes = 0;
    size_t indexer_bytes = 0;
    size_t working_set_bytes = 0;
    size_t pool_index           = 0;
    size_t free_blocks          = 0;
    size_t available_blocks     = 0;
    size_t request_ref_blocks   = 0;
    size_t connector_ref_blocks = 0;
    size_t total_blocks         = 0;
    float  used_ratio           = 0.0f;
};

struct BlockBufferPtrInfo {
    torch::Tensor kv_addr;
    torch::Tensor kv_scale_addr;
};

struct CacheLayerLayout {
    size_t        dsa_mla_resident_tokens = 0;
    size_t        dsa_mla_hbm_blocks = 0;
    std::vector<torch::Tensor> mla_hbm_cache_by_layer;
    torch::Tensor block_generations;
    std::vector<int>              layer_to_groups;
    std::vector<std::vector<int>> layer_to_group_ids;
    std::vector<std::vector<int>> layer_region_to_group_id;
    std::vector<CacheGroupType>   group_types;
    std::vector<KVCacheRegionName>  group_region_names;
    std::vector<size_t>             group_seq_size_per_block;
    std::vector<CacheGroupType>   layer_group_types;
    std::vector<torch::Tensor>              layers_to_kv_buffer_ptrs;
    std::vector<torch::Tensor>              layers_to_scale_buffer_ptrs;
    std::vector<std::vector<torch::Tensor>> layers_to_kv_buffer_ptrs_by_attn;
    std::vector<std::vector<torch::Tensor>> layers_to_scale_buffer_ptrs_by_attn;
};

struct KVCacheBuffer {
    torch::Tensor kv_blocks;
    torch::Tensor kv_scale_blocks;
};

}  // namespace rtp_llm
