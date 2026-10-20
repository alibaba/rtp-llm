#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

namespace rtp_llm {

enum class KVCacheEventType {
    BLOCK_ADD,
    BLOCK_DELETE,
};

struct KVCacheEvent {
    KVCacheEventType type      = KVCacheEventType::BLOCK_ADD;
    int64_t          block_key = 0;
    // Index into the immutable publication group/spec order.
    size_t           group_index = 0;
    // Publisher-local transport metadata assigned after an ingress slot is
    // reserved. Accepted events are strictly increasing in committed queue
    // publication order, not producer call-start or cache-version order.
    // Consumers must tolerate gaps after discard/reconciliation and resets
    // when a publisher instance is recreated.
    uint64_t sequence = 0;
};

struct KVCacheSnapshot {
    int64_t version = -1;
    // One entry per key, containing every currently resident component.
    std::map<int64_t, std::vector<size_t>> block_groups;
};

using KVCacheSnapshotProvider = std::function<KVCacheSnapshot()>;

}  // namespace rtp_llm
