#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

inline absl::StatusOr<std::vector<size_t>> selectCacheTransferBlockPositions(CacheGroupType         group_type,
                                                                             const BlockIndicesType& block_ids,
                                                                             size_t                  cache_key_count,
                                                                             int64_t reuse_block_count) {
    if (reuse_block_count < 0) {
        return absl::InvalidArgumentError("reuse block count must not be negative");
    }
    if (cache_key_count > block_ids.size()) {
        return absl::InvalidArgumentError("cache key count exceeds allocated block count");
    }
    const auto reuse_count = static_cast<size_t>(reuse_block_count);
    if (reuse_count > cache_key_count) {
        return absl::InvalidArgumentError("reuse block count exceeds cache key count");
    }
    if (cache_key_count == 0 || reuse_count == cache_key_count) {
        return std::vector<size_t>{};
    }

    std::vector<size_t> positions;
    if (group_type == CacheGroupType::LINEAR) {
        positions.push_back(cache_key_count - 1);
    } else if (group_type == CacheGroupType::FULL) {
        positions.reserve(cache_key_count - reuse_count);
        for (size_t position = reuse_count; position < cache_key_count; ++position) {
            positions.push_back(position);
        }
    } else {
        return absl::InvalidArgumentError("unknown cache group type");
    }

    for (const size_t position : positions) {
        if (isNullBlockIdx(block_ids[position])) {
            return absl::FailedPreconditionError("selected cache block has no physical allocation");
        }
    }
    return positions;
}

}  // namespace rtp_llm
