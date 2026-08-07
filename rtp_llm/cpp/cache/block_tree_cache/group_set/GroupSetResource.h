#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

// Sorting metadata for a candidate node. A single copy per GroupSetResource follows the
// data to its current serving tier (steady-state single-tier-service invariant).
struct CandidateMeta {
    uint64_t last_access_seq{0};  // LRU: logical clock of the last real match
    uint64_t admission_seq{0};    // FIFO: logical clock of entering the current tier
    uint64_t hit_count{0};        // LFU: cumulative real hit count
    int64_t  insert_time_us{0};
    int64_t  last_access_time_us{0};
    int64_t  tier_enter_time_us{0};
};

// Real data-transfer state; a resource is excluded from all heaps while != IDLE.
enum class GroupSetTransferState : uint8_t {
    IDLE,
    DEMOTING,      // Device -> Host, or Host -> Disk
    LOAD_PENDING,  // Host/Disk source reserved by a deferred load ticket
    LOADING        // Host/Disk -> Device
};

// Per-GroupSet data location across storage tiers.
// Each GroupSetResource corresponds to one GroupSet on one TreeNode.
struct GroupSetResource {
    // L1: GPU Device — one block per independent DeviceBlockPool
    std::vector<BlockIdxType> device_blocks;
    // L2: CPU Host — one packed block
    BlockIdxType host_block{NULL_BLOCK_IDX};
    // L3: Disk — one disk resource
    BlockIdxType disk_slot{NULL_BLOCK_IDX};

    // Async migration state and the single sorting-metadata copy (current serving tier).
    GroupSetTransferState transfer_state{GroupSetTransferState::IDLE};
    // The source remains owned by the in-flight operation, but its target must no longer be installed.
    bool                  transfer_detached{false};
    CandidateMeta         candidate_meta;

    bool hasTier(Tier tier) const {
        switch (tier) {
            case Tier::DEVICE:
                return std::any_of(device_blocks.begin(), device_blocks.end(), [](BlockIdxType block) {
                    return !isNullBlockIdx(block);
                });
            case Tier::HOST:
                return !isNullBlockIdx(host_block);
            case Tier::DISK:
                return !isNullBlockIdx(disk_slot);
            default:
                return false;
        }
    }
    size_t servingTierCount() const {
        return static_cast<size_t>(hasTier(Tier::DEVICE)) + static_cast<size_t>(hasTier(Tier::HOST))
               + static_cast<size_t>(hasTier(Tier::DISK));
    }
    bool isValidSteadyState() const {
        return transfer_state == GroupSetTransferState::IDLE && servingTierCount() <= 1;
    }
    bool is_empty() const {
        return !hasTier(Tier::DEVICE) && !hasTier(Tier::HOST) && !hasTier(Tier::DISK);
    }
    bool is_removable() const {
        return transfer_state == GroupSetTransferState::IDLE && is_empty();
    }
    bool isMatchUsable() const {
        return transfer_state == GroupSetTransferState::IDLE || transfer_state == GroupSetTransferState::LOADING;
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
