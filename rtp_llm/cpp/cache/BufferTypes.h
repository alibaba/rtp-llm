#pragma once

#include <map>
#include <string>
#include <vector>

#include <torch/extension.h>
#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"

namespace rtp_llm {

struct BlockBufferPtrInfo {
    torch::Tensor kv_addr;
    torch::Tensor kv_scale_addr;
};

struct CacheLayerLayout {
    std::vector<std::vector<int>> layer_to_group_ids;
    std::vector<CacheGroupType>   group_types;
    std::vector<std::string>        group_tags;
    std::vector<std::map<std::string, int>> layer_tag_to_group_id;
    std::vector<size_t>             group_seq_size_per_block;
    std::vector<CacheGroupType>   layer_group_types;
    std::vector<torch::Tensor>              layers_to_kv_buffer_ptrs;
    std::vector<torch::Tensor>              layers_to_scale_buffer_ptrs;
    std::vector<std::vector<torch::Tensor>> layers_to_kv_buffer_ptrs_by_group;
    std::vector<std::vector<torch::Tensor>> layers_to_scale_buffer_ptrs_by_group;
};

struct KVCacheBuffer {
    torch::Tensor kv_blocks;
    torch::Tensor kv_scale_blocks;
};

}  // namespace rtp_llm
