#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"

namespace rtp_llm {

namespace block_tree_cache_test {
class LoadShutdownTestPeer;
}

class AsyncContext;
class LoadAsyncContext;
class LoadTicketRegistry;

class LoadTicket {
public:
    struct PendingLoadItem {
        TreeNode*                 node{nullptr};
        size_t                    group_set_id{0};
        size_t                    path_index{0};
        Tier                      source_tier{Tier::NONE};
        std::vector<BlockIdxType> source_blocks;
        // This ticket joins an existing transfer and does not own its source or state.
        bool joined_load{false};
        // DEVICE denotes an already-resident logical coordinate that lies outside
        // the public ready boundary. It is ticket-owned and settled asynchronously
        // without a copy; target_device_blocks must preserve source identity.
        std::vector<BlockIdxType> target_device_blocks;
    };
    using PendingLoadItems = std::vector<PendingLoadItem>;

    ~LoadTicket();

    LoadTicket(const LoadTicket&)            = delete;
    LoadTicket& operator=(const LoadTicket&) = delete;
    LoadTicket(LoadTicket&&)                 = delete;
    LoadTicket& operator=(LoadTicket&&)      = delete;

    // Submits copies into allocator-owned targets. Every target must already be
    // present in the request block table; BlockTreeLoader never allocates a second
    // private target set. Returns null on synchronous preparation/submission failure
    // and on repeated or empty commits.
    std::shared_ptr<AsyncContext> commit();

    bool empty() const {
        return items_.empty();
    }

    size_t logicalMatchedBlocks() const {
        return logical_matched_blocks_;
    }
    size_t logicalMatchedBlocks(Tier tier) const;

    // Expose immutable planning metadata without publishing the ticket's item container.
    size_t itemCount() const {
        return items_.size();
    }

    size_t groupSetId(size_t item_index) const {
        return items_.at(item_index).group_set_id;
    }

    size_t pathIndex(size_t item_index) const {
        return items_.at(item_index).path_index;
    }

    Tier sourceTier(size_t item_index) const {
        return items_.at(item_index).source_tier;
    }

    const std::vector<BlockIdxType>& sourceBlocks(size_t item_index) const {
        return items_.at(item_index).source_blocks;
    }

    bool joinedLoad(size_t item_index) const {
        return items_.at(item_index).joined_load;
    }

    const std::vector<BlockIdxType>& targetDeviceBlocks(size_t item_index) const {
        return items_.at(item_index).target_device_blocks;
    }

    bool bindTargetDeviceBlocks(size_t item_index, std::vector<BlockIdxType> target_device_blocks) {
        if (item_index >= items_.size()) {
            return false;
        }
        PendingLoadItem& item = items_[item_index];
        if (item.joined_load || (item.source_tier == Tier::DEVICE && item.source_blocks != target_device_blocks)) {
            return false;
        }
        item.target_device_blocks = std::move(target_device_blocks);
        return true;
    }

    const PendingLoadItems& items() const {
        return items_;
    }

    const std::shared_ptr<LoadAsyncContext>& context() const {
        return context_;
    }

private:
    friend class LoadTicketRegistry;

    LoadTicket(std::shared_ptr<LoadTicketRegistry> registry,
               uint64_t                            ticket_id,
               PendingLoadItems                    items,
               size_t                              logical_matched_blocks,
               std::shared_ptr<LoadAsyncContext>   context);

    std::shared_ptr<LoadTicketRegistry>     registry_;
    uint64_t                                ticket_id_{0};
    PendingLoadItems                        items_;
    const size_t                            logical_matched_blocks_{0};
    std::array<size_t, 3>                   logical_matched_blocks_by_tier_{};
    std::shared_ptr<LoadAsyncContext>       context_;
};

class LoadTicketRegistry: public std::enable_shared_from_this<LoadTicketRegistry> {
public:
    using CommitCallback = std::function<std::shared_ptr<AsyncContext>(const LoadTicket& ticket)>;
    using AbortCallback  = std::function<void(const LoadTicket& ticket)>;

    LoadTicketRegistry(CommitCallback commit_callback, AbortCallback abort_callback);

    void shutdown();

private:
    friend class BlockTreeLoader;
    friend class LoadTicket;
    friend class block_tree_cache_test::LoadShutdownTestPeer;

    std::shared_ptr<LoadTicket> createTicket(const LoadTicket::PendingLoadItems&      items,
                                             size_t                                   logical_matched_blocks,
                                             const std::shared_ptr<LoadAsyncContext>& context);

    class ActiveCallbackLease {
    public:
        explicit ActiveCallbackLease(LoadTicketRegistry* registry): registry_(registry) {}
        ~ActiveCallbackLease() {
            registry_->retireActiveCallback();
        }

        ActiveCallbackLease(const ActiveCallbackLease&)            = delete;
        ActiveCallbackLease& operator=(const ActiveCallbackLease&) = delete;

    private:
        LoadTicketRegistry* registry_;
    };

    std::shared_ptr<AsyncContext> commit(uint64_t ticket_id, const LoadTicket& ticket);
    void                          abort(uint64_t ticket_id);
    void                          retireActiveCallback();

    struct PendingTicket {
        LoadTicket::PendingLoadItems      items;
        std::shared_ptr<LoadAsyncContext> context;
    };

    std::mutex                                  mutex_;
    std::condition_variable                     cv_;
    bool                                        accepting_{true};
    uint64_t                                    next_ticket_id_{1};
    size_t                                      active_callbacks_{0};
    std::unordered_map<uint64_t, PendingTicket> pending_tickets_;
    CommitCallback                              commit_callback_;
    AbortCallback                               abort_callback_;
    // Installed only by the shutdown test peer; production keeps this empty.
    std::function<void()> shutdown_wait_observer_for_test_;
};

}  // namespace rtp_llm
