#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/cache_new/KVCacheSpec.h"
#include "rtp_llm/cpp/core/Types.h"

namespace rtp_llm {

enum struct KVCacheType {
    FULL,
    LINEAR,
};

struct KVCacheSpec {
    std::vector<int> layer_ids_;
    KVCacheGroupType type_;
    uint             block_stride = 1; // record block Every 'block_stride' blocks
};

struct MHAKVCacheSpec : public KVCacheSpec {
    uint              layer_num;
    uint              block_nums;
    uint              local_head_num_kv;
    uint              size_per_head;
    uint              seq_size_per_block = 1;
    rtp_llm::DataType dtype;
};

struct MLAKVCacheSpec : public KVCacheSpec {
    uint              layer_num;
    uint              block_nums;
    uint              kv_lora_rank;
    uint              rope_head_dim;
    uint              seq_size_per_block = 1;
    rtp_llm::DataType dtype;
};


struct LinearKVCacheSpec : public KVCacheSpec {
    uint              conv_state_size;
    uint              temporal_state_size;
    rtp_llm::DataType dtype;
};

}  // namespace rtp_llm
