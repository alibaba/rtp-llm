#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace rtp_llm {

enum class KVCacheEventType {
    BLOCK_ADD,
    BLOCK_DELETE,
};

struct KVCacheEvent {
    KVCacheEventType type      = KVCacheEventType::BLOCK_ADD;
    int64_t          block_key = 0;
    uint64_t         sequence  = 0;
};

struct KVCacheSnapshot {
    int64_t              version = -1;
    std::vector<int64_t> block_keys;
};

using KVCacheSnapshotProvider = std::function<KVCacheSnapshot()>;

}  // namespace rtp_llm
