#pragma once

#include <cstddef>
#include <limits>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/model_rpc/RemoteLoadFence.h"

namespace rtp_llm {

inline absl::StatusOr<KVCacheResource> makeCacheTransferLeaseResource(int                  configured_group_count,
                                                                      const GroupBlockIds& block_ids_by_group,
                                                                      size_t               cache_key_count,
                                                                      size_t               max_block_id) {
    if (configured_group_count <= 0) {
        return absl::InvalidArgumentError("cache transfer configuration has invalid group count");
    }
    const auto group_count = static_cast<size_t>(configured_group_count);
    if (block_ids_by_group.size() != group_count) {
        return absl::InvalidArgumentError("cache transfer group count does not match cache configuration");
    }
    if (cache_key_count > static_cast<size_t>(std::numeric_limits<CacheKeyType>::max())) {
        return absl::InvalidArgumentError("cache transfer key count exceeds supported range");
    }

    KVCacheResource resource;
    auto&           lease_groups = resource.groupBlocks();
    lease_groups.reserve(group_count);

    for (size_t group_id = 0; group_id < group_count; ++group_id) {
        const auto& block_ids = block_ids_by_group[group_id];
        if (block_ids == nullptr) {
            return absl::InvalidArgumentError("cache transfer group has no block ids");
        }
        if (block_ids->blocksNum() < cache_key_count) {
            return absl::InvalidArgumentError("cache transfer group has fewer blocks than cache keys");
        }
        for (size_t position = 0; position < cache_key_count; ++position) {
            const auto block_id = block_ids->blocks()[position];
            if (!isNullBlockIdx(block_id)
                && (block_id <= 0 || static_cast<size_t>(block_id) > max_block_id)) {
                return absl::InvalidArgumentError("cache transfer group has an invalid physical block id");
            }
        }
        auto lease_block_ids = std::make_shared<BlockIds>();
        lease_block_ids->assign(block_ids->blocks());
        lease_groups.push_back(std::move(lease_block_ids));
    }

    auto& lease_keys = resource.cacheKeys();
    lease_keys.reserve(cache_key_count);
    for (size_t position = 0; position < cache_key_count; ++position) {
        lease_keys.push_back(static_cast<CacheKeyType>(position));
    }
    return resource;
}

inline std::shared_ptr<void> makeCacheTransferAddress(const std::shared_ptr<KVCacheResource>& lease, void* address) {
    if (lease == nullptr || address == nullptr) {
        return nullptr;
    }
    return std::shared_ptr<void>(lease, address);
}

struct CacheTransferLifetime {
    std::shared_ptr<KVCacheResource>          block_lease;
    RemoteLoadFenceRegistry::Operation operation;
};

inline std::shared_ptr<CacheTransferLifetime>
makeCacheTransferLifetime(std::shared_ptr<KVCacheResource>          block_lease,
                          RemoteLoadFenceRegistry::Operation operation) {
    if (block_lease == nullptr || operation == nullptr) {
        return nullptr;
    }
    return std::make_shared<CacheTransferLifetime>(
        CacheTransferLifetime{std::move(block_lease), std::move(operation)});
}

inline std::shared_ptr<void> makeCacheTransferAddress(const std::shared_ptr<CacheTransferLifetime>& lifetime,
                                                      void*                                         address) {
    if (lifetime == nullptr || address == nullptr) {
        return nullptr;
    }
    return std::shared_ptr<void>(lifetime, address);
}

}  // namespace rtp_llm
