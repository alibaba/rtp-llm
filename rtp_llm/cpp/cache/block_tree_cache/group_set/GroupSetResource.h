#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/cache/BlockIndex.h"

namespace rtp_llm {

// Sorting metadata for the current serving tier.
struct CandidateMeta {
    uint64_t last_access_seq{0};  // LRU: logical clock of the last real match
    uint64_t admission_seq{0};    // FIFO: logical clock of entering the current tier
    uint64_t hit_count{0};        // LFU: cumulative real hit count
    int64_t  tier_last_hit_time_us{0};
    int64_t  last_access_time_us{0};
    int64_t  tier_enter_time_us{0};
};

// Real data-transfer state; a resource is excluded from all heaps while != IDLE.
enum class GroupSetTransferState : uint8_t {
    IDLE,
    EVICTING,      // Device -> Host, or Host -> Disk
    LOAD_PENDING,  // Host/Disk source reserved by a deferred load ticket
    LOADING        // Host/Disk -> Device
};

inline const char* transferStateName(GroupSetTransferState state) {
    switch (state) {
        case GroupSetTransferState::IDLE:
            return "IDLE";
        case GroupSetTransferState::EVICTING:
            return "EVICTING";
        case GroupSetTransferState::LOAD_PENDING:
            return "LOAD_PENDING";
        case GroupSetTransferState::LOADING:
            return "LOADING";
    }
    return "UNKNOWN";
}

// Per-GroupSet data location across storage tiers.
// Each GroupSetResource corresponds to one GroupSet on one TreeNode.
struct GroupSetResource {
    // L1: GPU Device — one block per independent DeviceBlockPool
    std::vector<BlockIdxType> device_blocks;
    // L2: CPU Host — one packed block
    BlockIdxType host_block{NULL_BLOCK_IDX};
    // L3: Disk — one disk resource
    BlockIdxType disk_block{NULL_BLOCK_IDX};

    // Async migration state and the single sorting-metadata copy (current serving tier).
    GroupSetTransferState transfer_state{GroupSetTransferState::IDLE};
    // The source remains owned by the in-flight operation, but its target must no longer be installed.
    bool          transfer_detached{false};
    CandidateMeta candidate_meta;
    uint64_t      eviction_access_seq{0};
    uint64_t      generation{0};
    bool          retained_host_replica{false};
    bool          retained_disk_replica{false};

    bool hasTier(Tier tier) const {
        switch (tier) {
            case Tier::DEVICE:
                return std::any_of(device_blocks.begin(), device_blocks.end(), [](BlockIdxType block) {
                    return !isNullBlockIdx(block);
                });
            case Tier::HOST:
                return !isNullBlockIdx(host_block);
            case Tier::DISK:
                return !isNullBlockIdx(disk_block);
            default:
                return false;
        }
    }
    size_t servingTierCount() const {
        return static_cast<size_t>(hasTier(Tier::DEVICE)) + static_cast<size_t>(hasTier(Tier::HOST))
               + static_cast<size_t>(hasTier(Tier::DISK));
    }
    bool isValidSteadyState() const {
        const size_t host_copy = retained_host_replica && hasCompleteDeviceValue() && hasTier(Tier::HOST);
        const size_t disk_copy = retained_disk_replica && hasTier(Tier::DISK)
                                 && (hasCompleteDeviceValue() || hasTier(Tier::HOST));
        return transfer_state == GroupSetTransferState::IDLE && servingTierCount() <= 1 + host_copy + disk_copy;
    }
    bool is_empty() const {
        return !hasTier(Tier::DEVICE) && !hasTier(Tier::HOST) && !hasTier(Tier::DISK);
    }
    bool is_removable() const {
        return transfer_state == GroupSetTransferState::IDLE && is_empty();
    }
    bool isMatchUsable() const {
        return !transfer_detached && (hasCompleteDeviceValue() || hasTier(Tier::HOST) || hasTier(Tier::DISK));
    }
    bool hasCompleteDeviceValue() const {
        return !device_blocks.empty()
               && std::all_of(device_blocks.begin(), device_blocks.end(), [](BlockIdxType block) {
                      return !isNullBlockIdx(block);
                  });
    }
    void                      evictFromTier(Tier tier);
    std::vector<BlockIdxType> getBlocks(Tier tier) const;
    void                      setBlocks(Tier tier, const std::vector<BlockIdxType>& blocks);
    void                      setHostReplica(BlockIdxType block);
    Tier                      getTopTier() const {
        if (hasTier(Tier::DEVICE)) {
            return Tier::DEVICE;
        }
        if (hasTier(Tier::HOST)) {
            return Tier::HOST;
        }
        if (hasTier(Tier::DISK)) {
            return Tier::DISK;
        }
        return Tier::NONE;
    }
};

}  // namespace rtp_llm
