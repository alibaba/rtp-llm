#pragma once

#include <cstddef>
#include <optional>
#include <vector>
#include <string>

#include <torch/extension.h>
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

struct KVCachePoolMetricsSnapshot {
    std::string storage;
    size_t      capacity_bytes       = 0;
    size_t      occupied_bytes       = 0;
    size_t      indexer_bytes        = 0;
    size_t      working_set_bytes    = 0;
    size_t      pool_index           = 0;
    size_t      free_blocks          = 0;
    size_t      available_blocks     = 0;
    size_t      request_ref_blocks   = 0;
    size_t      connector_ref_blocks = 0;
    size_t      total_blocks         = 0;
    float       used_ratio           = 0.0f;
};

struct BlockBufferPtrInfo {
    torch::Tensor kv_addr;
    torch::Tensor kv_scale_addr;
};

struct CacheLayerLayout {
    size_t                        dsa_mla_resident_tokens = 0;
    size_t                        dsa_mla_hbm_blocks      = 0;
    std::vector<torch::Tensor>    mla_hbm_cache_by_layer;
    torch::Tensor                 block_generations;
    std::vector<torch::Tensor>    block_generations_by_layer;
    std::vector<int>              layer_to_groups;
    std::vector<std::vector<int>> layer_to_group_ids;
    // layer_region_to_group_id addresses the process-wide physical cache pools.
    // MTP models additionally use local group ids to index their compact
    // group_region_names/group_seq_size_per_block vectors.
    std::vector<std::vector<int>> layer_region_to_group_id;
    std::vector<std::vector<int>> local_layer_region_to_group_id;
    std::vector<CacheGroupType>   group_types;
    std::vector<KVCacheRegionName>  group_region_names;
    std::vector<size_t>             group_seq_size_per_block;
    std::vector<CacheGroupType>   layer_group_types;
    std::vector<torch::Tensor>              layers_to_kv_buffer_ptrs;
    std::vector<torch::Tensor>              layers_to_scale_buffer_ptrs;
    std::vector<std::vector<torch::Tensor>> layers_to_kv_buffer_ptrs_by_attn;
    std::vector<std::vector<torch::Tensor>> layers_to_scale_buffer_ptrs_by_attn;

    std::optional<int> resolvePhysicalGroupId(size_t local_layer_id, KVCacheRegionName region_name) const {
        if (local_layer_id >= layer_to_groups.size()) {
            return std::nullopt;
        }
        if (region_name == KVCacheRegionName::DEFAULT) {
            const int group_id = layer_to_groups[local_layer_id];
            return group_id >= 0 ? std::optional<int>(group_id) : std::nullopt;
        }
        const size_t region_id = static_cast<size_t>(region_name);
        if (local_layer_id >= layer_region_to_group_id.size()
            || region_id >= layer_region_to_group_id[local_layer_id].size()) {
            return std::nullopt;
        }
        const int group_id = layer_region_to_group_id[local_layer_id][region_id];
        return group_id >= 0 ? std::optional<int>(group_id) : std::nullopt;
    }
};

struct KVCacheBuffer {
    torch::Tensor kv_blocks;
    torch::Tensor kv_scale_blocks;
};

}  // namespace rtp_llm
