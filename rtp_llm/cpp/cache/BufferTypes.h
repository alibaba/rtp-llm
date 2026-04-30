#pragma once

#include <vector>

#include <torch/extension.h>
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

struct BlockBufferPtrInfo {
    torch::Tensor kv_addr;
    torch::Tensor kv_scale_addr;
};

struct CacheLayerLayout {
    std::vector<int>              layer_to_groups;
    std::vector<std::vector<int>> layer_to_group_ids;
    std::vector<std::vector<int>> layer_region_to_group_id;
    std::vector<CacheGroupType>   group_types;
    std::vector<KVCacheRegionName>  group_region_names;
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
