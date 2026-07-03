#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/EvictionHeap.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TransferDescriptor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"

namespace rtp_llm {

class AsyncContext;

// Pure-descriptor component: describes one device pool's layout.
struct Component {
    int                                  component_id{-1};
    int                                  component_group_id{-1};
    CacheGroupType                       type{CacheGroupType::FULL};
    std::vector<MemoryBlockLayerTagSlot> memory_block_layer_tag_slots;
    int                                  device_pool_index{-1};
};

// Match result returned by BlockTreeCache::match().
struct BlockTreeMatchResult {
    TreeNode*        matched_node{nullptr};
    size_t           matched_blocks{0};
    BlockIndicesType block_indices;

    std::shared_ptr<AsyncContext> async_context;
    size_t                        load_back_blocks{0};
    size_t                        host_load_back_blocks{0};
    size_t                        disk_load_back_blocks{0};
    size_t                        remote_load_back_blocks{0};
};

// Match validator interface.
class MatchValidator {
public:
    virtual ~MatchValidator()                                          = default;
    virtual bool validate(const TreeNode* node, const GroupSlot& slot) = 0;
};

// Transfer type for buildTransfer.
enum class TransferType {
    DEVICE_TO_HOST,
    HOST_TO_DEVICE,
    HOST_TO_DISK,
    DISK_TO_HOST,
    DEVICE_TO_REMOTE,
    HOST_TO_REMOTE,
    REMOTE_TO_DEVICE,
};

// Eviction result from ComponentGroup::driveEviction().
struct EvictionResult {
    TreeNode*                 node{nullptr};
    int                       component_group_id{-1};
    Tier                      source_tier{Tier::NONE};
    Tier                      target_tier{Tier::NONE};
    TransferDescriptor        transfer;
    std::vector<BlockIdxType> blocks_to_release;
    BlockIdxType              target_block{NULL_BLOCK_IDX};
};

// Predicate: returns true if all blocks in the group are evictable (refcount == 1).
using IsBlockEvictableFn = std::function<bool(BlockIdxType)>;

// ComponentGroup: active management entity.
class ComponentGroup {
public:
    virtual ~ComponentGroup() = default;

    // ---- Static metadata ----
    int              component_group_id{-1};
    CacheGroupType   group_type{CacheGroupType::FULL};
    std::vector<int> component_indices;
    size_t           host_block_size{0};

    // ---- Three-tier eviction heaps ----
    std::unique_ptr<EvictionHeap> device_heap;
    std::unique_ptr<EvictionHeap> host_heap;
    std::unique_ptr<EvictionHeap> disk_heap;

    // ---- Match ----
    virtual std::unique_ptr<MatchValidator> createMatchValidator() = 0;
    virtual void                            finalizeMatchResult(BlockTreeMatchResult& result) {}

    // ---- Insert (base class provides default; subclasses may override) ----
    virtual void commitInsertData(TreeNode* node, GroupSlot& slot, const std::vector<BlockIdxType>& block_indices);
    virtual void updateOnInsertOverlap(TreeNode* node, GroupSlot& slot);

    // ---- Evict (base class provides default; subclasses may override) ----
    virtual void                          evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier);
    virtual std::optional<EvictionResult> driveEviction(int num_blocks, Tier tier);

    // ---- Transfer (base class provides default; subclasses may override) ----
    virtual TransferDescriptor buildTransfer(TreeNode* node, TransferType type);

    // ---- Heap management ----
    virtual void tryAddToDeviceHeap(TreeNode* node) = 0;

    // Shared across all subclasses (implemented in ComponentGroup.cc).
    // Virtual: FullComponentGroup overrides to add Leaf checks.
    virtual void tryAddToHostHeap(TreeNode* node);
    virtual void tryAddToDiskHeap(TreeNode* node);

    // ---- Leaf checks (used by FullComponentGroup and base tryAddTo*Heap) ----
    bool isLeafAtTier(const TreeNode* node, int group_id, Tier tier) const;

    // ---- Reference count for path lock (per-group strategy) ----
    // Returns number of nodes from path tail to lock.
    // Default = matched_blocks (full path); SWA overrides for window lock.
    virtual size_t computeReferenceCount(size_t matched_blocks, const std::vector<TreeNode*>& path) const {
        return matched_blocks;
    }

    // ---- Reference counting callback (injected by BlockTreeCache) ----
    void setIsBlockEvictable(IsBlockEvictableFn fn) {
        is_block_evictable_ = std::move(fn);
    }

    // ---- Heap access helpers ----
    EvictionHeap* heapForTier(Tier tier) {
        switch (tier) {
            case Tier::DEVICE:
                return device_heap.get();
            case Tier::HOST:
                return host_heap.get();
            case Tier::DISK:
                return disk_heap.get();
            default:
                return nullptr;
        }
    }

protected:
    IsBlockEvictableFn is_block_evictable_;
};

using ComponentGroupPtr = std::shared_ptr<ComponentGroup>;

}  // namespace rtp_llm
