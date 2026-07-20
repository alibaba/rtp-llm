#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"

namespace rtp_llm {

class HostBlockPool;
class BlockTreeDiskBlockPool;

// Constructor-time snapshot of one component's device layout. Shared internal schema for the
// copy_engine facade and its executors; executors never re-parse ComponentGroup.
struct ResolvedComponentLayout {
    int                                  component_index{-1};
    int                                  device_pool_index{-1};
    DeviceBlockPoolPtr                   device_pool;
    std::vector<MemoryBlockLayerTagSlot> layer_slots;
};

// Constructor-time snapshot of one component group. Holds shared ownership of the host/disk
// pools so the cached schema keeps them alive independently of the ComponentGroup.
struct ResolvedGroupLayout {
    int                                     component_group_id{-1};
    std::shared_ptr<HostBlockPool>          host_pool;
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool;
    std::vector<ResolvedComponentLayout>    components;
    size_t                                  layout_bytes{0};
    bool                                    has_device_host_layout{false};
};

inline bool layoutHasAnyLayerSlot(const std::vector<ResolvedComponentLayout>& layouts) {
    for (const auto& layout : layouts) {
        if (!layout.layer_slots.empty()) {
            return true;
        }
    }
    return false;
}

// layout_bytes = sum of stride_bytes across every resolved component layer slot.
inline size_t computeLayoutsBlockSize(const std::vector<ResolvedComponentLayout>& layouts) {
    size_t total = 0;
    for (const auto& layout : layouts) {
        for (const auto& slot : layout.layer_slots) {
            total += slot.stride_bytes;
        }
    }
    return total;
}

}  // namespace rtp_llm
