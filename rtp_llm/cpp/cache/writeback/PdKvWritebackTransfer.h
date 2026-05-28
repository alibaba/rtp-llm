#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"

namespace rtp_llm {

enum class PdKvWritebackBlockSide {
    DecodeSource,
    PrefillDestination,
};

struct PdKvWritebackTransferPlan {
    int64_t request_id  = 0;
    int64_t deadline_ms = 0;

    std::string request_key;

    int32_t layer_count    = 0;
    int32_t group_count    = 1;
    int32_t remote_tp_size = 1;

    CacheKeysType                 cache_keys;
    std::vector<BlockIndicesType> decode_group_block_ids;
    std::vector<BlockIndicesType> prefill_group_block_ids;
    std::vector<int32_t>          layer_to_group_id;

    std::vector<std::pair<std::string, uint32_t>> prefill_transfer_servers;
};

class PdKvWritebackTransferClient {
public:
    virtual ~PdKvWritebackTransferClient()                               = default;
    virtual absl::Status transfer(const PdKvWritebackTransferPlan& plan) = 0;
};

absl::StatusOr<KVCacheResourcePtr> buildPdKvWritebackResource(const PdKvWritebackTransferPlan& plan,
                                                              PdKvWritebackBlockSide           side);

std::vector<BlockIndicesType> extractPdKvWritebackGroupBlockIds(const BatchKVCacheResourcePtr& resource);

std::vector<std::pair<std::string, uint32_t>>
parsePdKvWritebackTransferServers(const std::vector<std::string>& worker_addrs);

}  // namespace rtp_llm
