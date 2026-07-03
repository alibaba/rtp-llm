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

// Memory block layout descriptor for Host/Disk layers.
// Describes the position and size of a (layer, group) slot within a packed memory block.
// Mirrors the semantics of KVCacheMemoryConnector::layerTagSlots().
struct MemoryBlockLayerTagSlot {
    int         layer_id{-1};
    std::string tag;              // group tag, e.g. "csa_kv", "hca_kv", "swa_kv"
    size_t      stride_bytes{0};  // bytes this slot occupies in the memory block
};

// Per-ComponentGroup data location across storage tiers.
// Each GroupSlot corresponds to one ComponentGroup on one TreeNode.
struct GroupSlot {
    // L1: GPU Device — one block per independent Device BlockPool
    std::vector<BlockIdxType> device_blocks;
    // L2: CPU Host — one packed block (REUSABLE groups only)
    BlockIdxType host_block{NULL_BLOCK_IDX};
    // L3: Disk — one disk slot (REUSABLE groups only)
    BlockIdxType disk_slot{NULL_BLOCK_IDX};

    // Heap membership flags (per-group, eviction is group-granular)
    bool in_device_heap{false};
    bool in_host_heap{false};
    bool in_disk_heap{false};

    bool has_device_value() const {
        return std::any_of(
            device_blocks.begin(), device_blocks.end(), [](BlockIdxType b) { return b != NULL_BLOCK_IDX; });
    }
    bool has_host_value() const {
        return host_block != NULL_BLOCK_IDX;
    }
    bool has_disk_value() const {
        return disk_slot != NULL_BLOCK_IDX;
    }
    bool has_any_value() const {
        return has_device_value() || has_host_value() || has_disk_value();
    }
    bool is_empty() const {
        return !has_device_value() && !has_host_value() && !has_disk_value();
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

    // Multi-tier data locations, indexed by component_group_id
    std::vector<GroupSlot> group_slots;
};

}  // namespace rtp_llm
