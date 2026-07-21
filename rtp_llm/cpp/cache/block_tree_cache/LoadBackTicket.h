#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"

namespace rtp_llm {

namespace block_tree_cache_test {
class LoadBackShutdownTestPeer;
}

class AsyncContext;
class BlockTreeCache;
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

    // Expose immutable planning metadata without publishing PendingLoadBackItem as a cross-module DTO.
    size_t itemCount() const {
        return items_.size();
    }

    int groupId(size_t item_index) const {
        return items_.at(item_index).group_id;
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

    const std::vector<std::string>& deviceGroupTags(size_t item_index) const {
        return items_.at(item_index).device_group_tags;
    }

    bool bindTargetDeviceBlocks(size_t item_index, std::vector<BlockIdxType> target_device_blocks) {
        if (item_index >= items_.size()) {
            return false;
        }
        items_[item_index].target_device_blocks = std::move(target_device_blocks);
        return true;
    }

private:
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
        std::vector<std::string>  device_group_tags;
        std::vector<BlockIdxType> target_device_blocks;
    };
    using PendingLoadBackItems = std::vector<PendingLoadBackItem>;

    PendingLoadBackItems& items() {
        return items_;
    }
    const PendingLoadBackItems& items() const {
        return items_;
    }

    friend class BlockTreeCache;
    friend class LoadBackTicketRegistry;

    LoadBackTicket(std::shared_ptr<LoadBackTicketRegistry> registry,
                   uint64_t                                ticket_id,
                   PendingLoadBackItems                    items,
                   size_t                                  logical_matched_blocks);

    std::shared_ptr<LoadBackTicketRegistry> registry_;
    uint64_t                                ticket_id_{0};
    PendingLoadBackItems                    items_;
    const size_t                            logical_matched_blocks_{0};
    std::array<size_t, 3>                   logical_matched_blocks_by_tier_{};
};

class LoadBackTicketRegistry: public std::enable_shared_from_this<LoadBackTicketRegistry> {
public:
    using CommitCallback = std::function<std::shared_ptr<AsyncContext>(const LoadBackTicket& ticket)>;
    using AbortCallback  = std::function<void(const LoadBackTicket& ticket)>;

    LoadBackTicketRegistry(CommitCallback commit_callback, AbortCallback abort_callback);

    void shutdown();

private:
    friend class BlockTreeCache;
    friend class LoadBackTicket;
    friend class block_tree_cache_test::LoadBackShutdownTestPeer;

    std::shared_ptr<LoadBackTicket> createTicket(const LoadBackTicket::PendingLoadBackItems& items,
                                                 size_t logical_matched_blocks = 0);

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

    std::shared_ptr<AsyncContext> commit(uint64_t ticket_id, const LoadBackTicket& ticket);
    void                          abort(uint64_t ticket_id);
    void                          retireActiveCallback();

    std::mutex                                                               mutex_;
    std::condition_variable                                                  cv_;
    bool                                                                     accepting_{true};
    uint64_t                                                                 next_ticket_id_{1};
    size_t                                                                   active_callbacks_{0};
    std::unordered_map<uint64_t, LoadBackTicket::PendingLoadBackItems>           pending_tickets_;
    CommitCallback                                                           commit_callback_;
    AbortCallback                                                            abort_callback_;
    // Installed only by the shutdown test peer; production keeps this empty.
    std::function<void()> shutdown_wait_observer_for_test_;
};

}  // namespace rtp_llm
