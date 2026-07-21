#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"

namespace rtp_llm {

namespace block_tree_cache_test {
class LoadBackShutdownTestPeer;
}

class AsyncContext;

struct PendingLoadBackItem {
    TreeNode*                 node{nullptr};
    int                       group_id{-1};
    size_t                    path_index{0};
    Tier                      source_tier{Tier::NONE};
    std::vector<BlockIdxType> source_blocks;
    // DEVICE denotes an already-resident logical coordinate that lies outside
    // the public ready boundary. It is ticket-owned and settled asynchronously
    // without a copy; target_device_blocks must preserve source identity.
    // Allocator-facing per-tag group ids, ordered exactly like this component
    // group's device pools. The allocator fills target_device_blocks from the
    // request block table before committing the ticket.
    std::vector<int>          device_group_ids;
    std::vector<BlockIdxType> target_device_blocks;
};

class LoadBackTicketRegistry;

class LoadBackTicket {
public:
    ~LoadBackTicket();

    LoadBackTicket(const LoadBackTicket&)            = delete;
    LoadBackTicket& operator=(const LoadBackTicket&) = delete;
    LoadBackTicket(LoadBackTicket&&)                 = delete;
    LoadBackTicket& operator=(LoadBackTicket&&)      = delete;

    // Submits copies into allocator-owned targets. Every target must already be
    // present in the request block table; BlockTreeCache never allocates a second
    // private target set. Returns null on synchronous validation/submission failure
    // and on repeated or empty commits.
    std::shared_ptr<AsyncContext> commit();

    bool empty() const {
        return items_.empty();
    }

    size_t logicalMatchedBlocks() const {
        return logical_matched_blocks_;
    }
    size_t logicalMatchedBlocks(Tier tier) const;

    std::vector<PendingLoadBackItem>& items() {
        return items_;
    }
    const std::vector<PendingLoadBackItem>& items() const {
        return items_;
    }

private:
    friend class LoadBackTicketRegistry;

    LoadBackTicket(std::shared_ptr<LoadBackTicketRegistry> registry,
                   uint64_t                                ticket_id,
                   std::vector<PendingLoadBackItem>        items,
                   size_t                                  logical_matched_blocks);

    std::shared_ptr<LoadBackTicketRegistry> registry_;
    uint64_t                                ticket_id_{0};
    std::vector<PendingLoadBackItem>        items_;
    const size_t                            logical_matched_blocks_{0};
    std::array<size_t, 3>                   logical_matched_blocks_by_tier_{};
};

class LoadBackTicketRegistry: public std::enable_shared_from_this<LoadBackTicketRegistry> {
public:
    using CommitCallback = std::function<std::shared_ptr<AsyncContext>(const std::vector<PendingLoadBackItem>& items)>;
    using AbortCallback  = std::function<void(const std::vector<PendingLoadBackItem>& items)>;

    LoadBackTicketRegistry(CommitCallback commit_callback, AbortCallback abort_callback);

    std::shared_ptr<LoadBackTicket> createTicket(const std::vector<PendingLoadBackItem>& items,
                                                 size_t                                  logical_matched_blocks = 0);
    void                            shutdown();

private:
    friend class LoadBackTicket;
    friend class block_tree_cache_test::LoadBackShutdownTestPeer;

    class ActiveCallbackLease {
    public:
        explicit ActiveCallbackLease(LoadBackTicketRegistry* registry): registry_(registry) {}
        ~ActiveCallbackLease() {
            registry_->retireActiveCallback();
        }

        ActiveCallbackLease(const ActiveCallbackLease&)            = delete;
        ActiveCallbackLease& operator=(const ActiveCallbackLease&) = delete;

    private:
        LoadBackTicketRegistry* registry_;
    };

    std::shared_ptr<AsyncContext> commit(uint64_t ticket_id, const std::vector<PendingLoadBackItem>& items);
    void                          abort(uint64_t ticket_id);
    void                          retireActiveCallback();

    std::mutex                                                     mutex_;
    std::condition_variable                                        cv_;
    bool                                                           accepting_{true};
    uint64_t                                                       next_ticket_id_{1};
    size_t                                                         active_callbacks_{0};
    std::unordered_map<uint64_t, std::vector<PendingLoadBackItem>> pending_tickets_;
    CommitCallback                                                 commit_callback_;
    AbortCallback                                                  abort_callback_;
    // Installed only by the shutdown test peer; production keeps this empty.
    std::function<void()> shutdown_wait_observer_for_test_;
};

struct Component {
    int                 component_id{-1};
    int                 component_group_id{-1};
    CacheGroupType      type{CacheGroupType::FULL};
    std::string         tag;
    std::vector<int>    model_layer_ids;
    std::vector<size_t> layer_bytes;

    size_t layerCount() const {
        return layer_bytes.size();
    }
    size_t layerBytes(size_t layer_idx) const {
        return layer_bytes[layer_idx];
    }
};

struct GroupBlockSet {
    int                                    component_group_id{-1};
    Tier                                   tier{Tier::DEVICE};
    std::vector<std::vector<BlockIdxType>> per_node;
    // Optional: tree nodes aligned with per_node, populated for match-protection
    // sets so release can drive candidate refresh. Empty when not needed.
    std::vector<TreeNode*> nodes;
};

struct BlockTreeMatchResult {
    TreeNode* matched_node{nullptr};
    size_t    matched_blocks{0};

    // Keyed by allocator-facing per-tag gid.
    std::unordered_map<int, BlockIndicesType> group_block_indices;

    std::vector<GroupBlockSet> matched_block_sets;

    std::shared_ptr<AsyncContext> async_context;
    size_t                        load_back_blocks{0};
    size_t                        host_load_back_blocks{0};
    size_t                        disk_load_back_blocks{0};
    size_t                        remote_load_back_blocks{0};

    std::shared_ptr<LoadBackTicket> load_back_ticket;
};

class MatchValidator {
public:
    virtual ~MatchValidator()                                          = default;
    virtual bool validate(const TreeNode* node, const GroupSlot& slot) = 0;
};

enum class TransferType {
    DEVICE_TO_HOST,
    HOST_TO_DEVICE,
    HOST_TO_DISK,
    DISK_TO_HOST,
    DEVICE_TO_REMOTE,
    HOST_TO_REMOTE,
    REMOTE_TO_DEVICE,
};

struct EvictionMove {
    TreeNode*                 node{nullptr};
    int                       component_group_id{-1};
    Tier                      source_tier{Tier::NONE};
    Tier                      target_tier{Tier::NONE};
    int64_t                   source_tier_enter_time_us{0};
    std::vector<BlockIdxType> source_blocks;
    std::vector<BlockIdxType> target_blocks;
};

class ComponentGroupLayout {
public:
    struct Slice {
        size_t component_idx{0};
        size_t layer_idx{0};  // Component-local, not model-global.
        size_t offset_bytes{0};
    };

    static std::optional<ComponentGroupLayout> create(const std::vector<std::vector<size_t>>& component_layer_bytes);

    const std::vector<Slice>& slices() const {
        return slices_;
    }
    // Slices are emitted in canonical component order, so the last slice's
    // component_idx + 1 is the component count. Not stored separately.
    size_t componentCount() const {
        return slices_.empty() ? 0 : slices_.back().component_idx + 1;
    }
    size_t payloadBytes() const {
        return payload_bytes_;
    }

private:
    std::vector<Slice> slices_;
    size_t             payload_bytes_{0};
};

class ComponentGroup {
public:
    virtual ~ComponentGroup() = default;

    int            component_group_id{-1};
    CacheGroupType group_type{CacheGroupType::FULL};

    bool finalizeLayout(std::vector<int> component_indices, const std::vector<Component>& components);

    const std::vector<int>& componentIndices() const {
        return component_indices_;
    }

    bool hasLayout() const {
        return layout_.has_value();
    }
    const ComponentGroupLayout& layout() const;

    virtual std::unique_ptr<MatchValidator> createMatchValidator() = 0;

    virtual void evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier);

    virtual TransferDescriptor buildTransfer(TreeNode* node, TransferType type);

    bool isLeafAtTier(const TreeNode* node, int group_id, Tier tier) const;

    virtual size_t computeReuseBlockCount(size_t matched_block_count, const std::vector<TreeNode*>& path) const = 0;

    void setDevicePools(std::vector<DeviceBlockPoolPtr> pools) {
        device_pools_ = std::move(pools);
    }
    void setHostPool(std::shared_ptr<HostBlockPool> pool) {
        host_pool_ = std::move(pool);
    }
    void setDiskPool(std::shared_ptr<DiskBlockPool> pool) {
        disk_pool_ = std::move(pool);
    }

    const std::vector<DeviceBlockPoolPtr>& devicePools() const {
        return device_pools_;
    }
    std::shared_ptr<HostBlockPool> hostPool() const {
        return host_pool_;
    }
    std::shared_ptr<DiskBlockPool> diskPool() const {
        return disk_pool_;
    }

    size_t devicePoolCount() const {
        return device_pools_.size();
    }

    bool anyDevicePoolExceedsRatio(double ratio) const {
        for (const auto& pool : device_pools_) {
            if (!pool)
                continue;
            size_t capacity = pool->totalBlocksNum();
            if (capacity == 0)
                continue;
            size_t used      = capacity - pool->freeBlocksNum();
            size_t threshold = static_cast<size_t>(capacity * ratio);
            if (used > threshold)
                return true;
        }
        return false;
    }

    size_t devicePoolMaxExcess(double ratio) const {
        size_t max_excess = 0;
        for (const auto& pool : device_pools_) {
            if (!pool)
                continue;
            size_t capacity = pool->totalBlocksNum();
            if (capacity == 0)
                continue;
            size_t used      = capacity - pool->freeBlocksNum();
            size_t threshold = static_cast<size_t>(capacity * ratio);
            if (used > threshold)
                max_excess = std::max(max_excess, used - threshold);
        }
        return max_excess;
    }

    // A tree-node eviction releases one block from every physical device pool
    // in this component group. Return the exact number of node evictions needed
    // so every pool retains at least min_free_blocks (clamped to its capacity).
    size_t devicePoolMaxExcessForMinFree(size_t min_free_blocks) const {
        size_t max_excess = 0;
        for (const auto& pool : device_pools_) {
            if (!pool) {
                continue;
            }
            const size_t capacity = pool->totalBlocksNum();
            if (capacity == 0) {
                continue;
            }
            const size_t used      = capacity - pool->freeBlocksNum();
            const size_t min_free  = std::min(min_free_blocks, capacity);
            const size_t threshold = capacity - min_free;
            if (used > threshold) {
                max_excess = std::max(max_excess, used - threshold);
            }
        }
        return max_excess;
    }

    // Host/Disk pool usage queries for watermark checking
    size_t hostPoolUsed() const {
        return host_pool_ ? (host_pool_->totalBlocksNum() - host_pool_->freeBlocksNum()) : 0;
    }
    size_t hostPoolCapacity() const {
        return host_pool_ ? host_pool_->totalBlocksNum() : 0;
    }
    size_t diskPoolUsed() const {
        return disk_pool_ ? (disk_pool_->totalBlocksNum() - disk_pool_->freeBlocksNum()) : 0;
    }
    size_t diskPoolCapacity() const {
        return disk_pool_ ? disk_pool_->totalBlocksNum() : 0;
    }

    GroupBlockSet allocateBlocks(Tier tier, size_t count, BlockRefType ref_type);
    void          referenceBlocks(const GroupBlockSet& set, BlockRefType ref_type) const;
    void          unreferenceBlocks(const GroupBlockSet& set, BlockRefType ref_type) const;

    BlockIdxType allocateSingleBlock(Tier tier, BlockRefType ref_type);
    void         releaseSingleBlock(Tier tier, BlockIdxType block, BlockRefType ref_type) const;

    std::vector<BlockIdxType> getBlocks(const GroupSlot& slot, Tier tier) const;
    void                      setBlocks(GroupSlot& slot, Tier tier, const std::vector<BlockIdxType>& blocks);
    // Highest tier (DEVICE > HOST > DISK) holding this slot's data, else NONE.
    Tier getTopTier(const GroupSlot& slot) const;

    virtual bool isSlotEvictable(const TreeNode& node, Tier tier) const;

protected:
    std::vector<DeviceBlockPoolPtr> device_pools_;
    std::shared_ptr<HostBlockPool>  host_pool_;
    std::shared_ptr<DiskBlockPool>  disk_pool_;

private:
    std::vector<int>                    component_indices_;
    std::optional<ComponentGroupLayout> layout_;
};

using ComponentGroupPtr = std::shared_ptr<ComponentGroup>;

}  // namespace rtp_llm
