#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

struct PdKvWritebackSnapshot {
    int64_t                       request_id = 0;
    std::string                   request_key;
    int32_t                       seq_size_per_block  = 0;
    int64_t                       final_token_count   = 0;
    int64_t                       prefill_token_count = 0;
    CacheKeysType                 cache_keys;
    std::vector<BlockIndicesType> group_block_ids;
    std::vector<std::vector<int>> mm_intervals;
};

struct PdKvWritebackManifest {
    int64_t                       request_id = 0;
    std::string                   request_key;
    int32_t                       seq_size_per_block   = 0;
    int64_t                       final_token_count    = 0;
    int64_t                       start_block_index    = 0;
    int64_t                       reusable_block_count = 0;
    CacheKeysType                 cache_keys;
    std::vector<BlockIndicesType> group_block_ids;
};

absl::StatusOr<PdKvWritebackManifest> buildPdKvWritebackManifest(const PdKvWritebackSnapshot& snapshot);

}  // namespace rtp_llm
