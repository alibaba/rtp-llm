#pragma once

#include <cstddef>
#include <vector>
#include <torch/torch.h>

namespace rtp_llm {

struct KvCacheInfo {
    int layer_num;
    // [batch_size, block_nums], KV-cache block offsets.
    torch::Tensor kv_cache_block_id;
    // Per-group block tables for hybrid cache, each [batch_size, block_nums].
    std::vector<torch::Tensor> kv_cache_block_ids_by_group;
    // Base K-buffer; the V address is derived from the configured stride.
    torch::Tensor kv_cache_buffer;
    // Optional quantization scale buffer with the same block layout.
    torch::Tensor kv_scale_buffer;
};

}  // namespace rtp_llm
