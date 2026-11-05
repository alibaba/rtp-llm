#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <map>
#include <vector>

namespace rtp_llm {

enum class KVCacheEventMedium : uint8_t {
    HBM,
    HOST,
    DISK
};
inline constexpr std::array<KVCacheEventMedium, 3> kKVCacheEventMediums{
    KVCacheEventMedium::HBM, KVCacheEventMedium::HOST, KVCacheEventMedium::DISK};

inline const char* cacheEventMediumName(KVCacheEventMedium medium) {
    switch (medium) {
        case KVCacheEventMedium::HBM:
            return "hbm";
        case KVCacheEventMedium::HOST:
            return "mem";
        case KVCacheEventMedium::DISK:
            return "disk";
    }
    return "";
}

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
    // Consumers must tolerate gaps after drops and registration recovery, and resets
    // when a publisher instance is recreated.
    uint64_t sequence = 0;
    KVCacheEventMedium medium   = KVCacheEventMedium::HBM;
};

struct KVCacheSnapshot {
    int64_t version = -1;
    using BlockGroups = std::map<int64_t, std::vector<size_t>>;
    // Keep the original HBM field for existing snapshot providers.
    BlockGroups block_groups;
    BlockGroups host_block_groups{};
    BlockGroups disk_block_groups{};

    const BlockGroups& groups(KVCacheEventMedium medium) const {
        switch (medium) {
            case KVCacheEventMedium::HBM:
                return block_groups;
            case KVCacheEventMedium::HOST:
                return host_block_groups;
            case KVCacheEventMedium::DISK:
                return disk_block_groups;
        }
        return block_groups;
    }

    BlockGroups& groups(KVCacheEventMedium medium) {
        return const_cast<BlockGroups&>(static_cast<const KVCacheSnapshot&>(*this).groups(medium));
    }
};

using KVCacheSnapshotProvider = std::function<KVCacheSnapshot()>;

}  // namespace rtp_llm
