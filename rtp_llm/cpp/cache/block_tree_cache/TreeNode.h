#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

// Storage tier enumeration for multi-tier cache management.
enum class Tier : int8_t {
    DEVICE = 0,  // L1: GPU
    HOST   = 1,  // L2: CPU memory
    DISK   = 2,  // L3: Local disk
    REMOTE = 3,  // L4: Remote storage
    NONE   = 4,  // No tier (direct release)
};

inline const char* tierName(Tier tier) {
    switch (tier) {
        case Tier::DEVICE:
            return "DEVICE";
        case Tier::HOST:
            return "HOST";
        case Tier::DISK:
            return "DISK";
        case Tier::REMOTE:
            return "REMOTE";
        case Tier::NONE:
            return "NONE";
    }
    return "UNKNOWN";
}

// Sorting metadata for a candidate node. A single copy per GroupSetResource follows the
// data to its current serving tier (steady-state single-tier-service invariant).
struct CandidateMeta {
    uint64_t last_access_seq{0};  // LRU: logical clock of the last real match
    uint64_t admission_seq{0};    // FIFO: logical clock of entering the current tier
    uint64_t hit_count{0};        // LFU: cumulative real hit count
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
    CandidateMeta         candidate_meta;

    bool hasTier(Tier tier) const {
        switch (tier) {
            case Tier::DEVICE:
                return std::any_of(
                    device_blocks.begin(), device_blocks.end(), [](BlockIdxType b) { return b != NULL_BLOCK_IDX; });
            case Tier::HOST:
                return host_block != NULL_BLOCK_IDX;
            case Tier::DISK:
                return disk_slot != NULL_BLOCK_IDX;
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
    // A match may consume or join this resource only while no exclusive migration
    // owns it: IDLE serves normally, LOADING is joinable, while DEMOTING and
    // LOAD_PENDING sources are reserved by an in-flight transfer.
    bool isMatchUsable() const {
        return transfer_state == GroupSetTransferState::IDLE || transfer_state == GroupSetTransferState::LOADING;
    }
};

// Tree node in the BlockTree radix tree.
// Each node represents one block-aligned cache_key.
struct TreeNode {
    // Tree structure
    CacheKeyType                                cache_key{0};
    std::vector<int>                            token_ids;  // debug/validation only
    std::unordered_map<CacheKeyType, TreeNode*> children;
    TreeNode*                                   parent{nullptr};

    // Multi-tier data locations, indexed by group_set_id
    std::vector<GroupSetResource> group_set_resources;
};

}  // namespace rtp_llm
