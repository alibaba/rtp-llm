#include "rtp_llm/cpp/cache/block_tree_cache/LoadBackTicket.h"

namespace rtp_llm {

LoadBackTicket::LoadBackTicket(std::shared_ptr<LoadBackTicketRegistry> registry,
                               uint64_t                                ticket_id,
                               LoadBackTicket::PendingLoadBackItems    items,
                               size_t                                  logical_matched_blocks):
    registry_(std::move(registry)),
    ticket_id_(ticket_id),
    items_(std::move(items)),
    logical_matched_blocks_(logical_matched_blocks) {
    std::unordered_map<size_t, Tier> reuse_tier_by_path;
    for (const PendingLoadBackItem& item : items_) {
        if (item.source_tier < Tier::DEVICE || item.source_tier > Tier::DISK) {
            continue;
        }
        const std::pair<std::unordered_map<size_t, Tier>::iterator, bool> insert_result =
            reuse_tier_by_path.emplace(item.path_index, item.source_tier);
        if (!insert_result.second
            && (item.source_tier == Tier::DISK
                || (item.source_tier == Tier::HOST && insert_result.first->second == Tier::DEVICE))) {
            insert_result.first->second = item.source_tier;
        }
    }
    for (const std::pair<const size_t, Tier>& reuse_tier : reuse_tier_by_path) {
        ++logical_matched_blocks_by_tier_[static_cast<size_t>(reuse_tier.second)];
    }
}

LoadBackTicket::~LoadBackTicket() {
    if (registry_ != nullptr && ticket_id_ != 0) {
        registry_->abort(ticket_id_);
    }
}

std::shared_ptr<AsyncContext> LoadBackTicket::commit() {
    if (registry_ == nullptr || ticket_id_ == 0 || items_.empty()) {
        return nullptr;
    }
    return registry_->commit(ticket_id_, *this);
}

size_t LoadBackTicket::logicalMatchedBlocks(Tier tier) const {
    if (tier < Tier::DEVICE || tier > Tier::DISK) {
        return 0;
    }
    return logical_matched_blocks_by_tier_[static_cast<size_t>(tier)];
}

LoadBackTicketRegistry::LoadBackTicketRegistry(CommitCallback commit_callback, AbortCallback abort_callback):
    commit_callback_(std::move(commit_callback)), abort_callback_(std::move(abort_callback)) {}

std::shared_ptr<LoadBackTicket> LoadBackTicketRegistry::createTicket(const LoadBackTicket::PendingLoadBackItems& items,
                                                                     size_t logical_matched_blocks) {
    std::shared_ptr<LoadBackTicket> ticket(new LoadBackTicket(
        shared_from_this(), /*ticket_id=*/0, LoadBackTicket::PendingLoadBackItems(items), logical_matched_blocks));
    if (items.empty()) {
        return ticket;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!accepting_ || next_ticket_id_ == 0) {
        return nullptr;
    }

    const uint64_t ticket_id = next_ticket_id_;
    pending_tickets_.emplace(ticket_id, items);
    ticket->ticket_id_ = ticket_id;
    ++next_ticket_id_;
    return ticket;
}

std::shared_ptr<AsyncContext> LoadBackTicketRegistry::commit(uint64_t ticket_id, const LoadBackTicket& ticket) {
    CommitCallback commit_callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  pending_it = pending_tickets_.find(ticket_id);
        if (!accepting_ || pending_it == pending_tickets_.end()) {
            return nullptr;
        }
        commit_callback = commit_callback_;
        pending_tickets_.erase(pending_it);
        ++active_callbacks_;
    }

    ActiveCallbackLease active_callback(this);
    return commit_callback ? commit_callback(ticket) : nullptr;
}

void LoadBackTicketRegistry::abort(uint64_t ticket_id) {
    AbortCallback                        abort_callback;
    LoadBackTicket::PendingLoadBackItems abort_payload;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  pending_it = pending_tickets_.find(ticket_id);
        if (!accepting_ || pending_it == pending_tickets_.end()) {
            return;
        }
        abort_callback = abort_callback_;
        abort_payload  = std::move(pending_it->second);
        pending_tickets_.erase(pending_it);
        ++active_callbacks_;
    }

    ActiveCallbackLease active_callback(this);
    if (abort_callback) {
        LoadBackTicket abort_ticket(nullptr, /*ticket_id=*/0, std::move(abort_payload), /*logical_matched_blocks=*/0);
        abort_callback(abort_ticket);
    }
}

void LoadBackTicketRegistry::retireActiveCallback() {
    std::lock_guard<std::mutex> lock(mutex_);
    --active_callbacks_;
    if (active_callbacks_ == 0) {
        cv_.notify_all();
    }
}

void LoadBackTicketRegistry::shutdown() {
    AbortCallback                                     abort_callback;
    std::vector<LoadBackTicket::PendingLoadBackItems> detached_payloads;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (accepting_) {
            abort_callback = abort_callback_;
            detached_payloads.reserve(pending_tickets_.size());
            accepting_ = false;
            for (auto& [ticket_id, abort_payload] : pending_tickets_) {
                (void)ticket_id;
                detached_payloads.push_back(std::move(abort_payload));
            }
            pending_tickets_.clear();
            if (!detached_payloads.empty()) {
                // Keep the complete detached batch visible to every overlapping shutdown caller.
                ++active_callbacks_;
            }
        }
    }

    if (!detached_payloads.empty()) {
        ActiveCallbackLease active_callback(this);
        if (abort_callback) {
            for (auto& abort_payload : detached_payloads) {
                LoadBackTicket abort_ticket(
                    nullptr, /*ticket_id=*/0, std::move(abort_payload), /*logical_matched_blocks=*/0);
                abort_callback(abort_ticket);
            }
        }
    }

    std::unique_lock<std::mutex> lock(mutex_);
    bool                         wait_observer_invoked = false;
    cv_.wait(lock, [this, &wait_observer_invoked] {
        if (active_callbacks_ != 0 && !wait_observer_invoked) {
            wait_observer_invoked                      = true;
            const auto shutdown_wait_observer_for_test = shutdown_wait_observer_for_test_;
            if (shutdown_wait_observer_for_test) {
                shutdown_wait_observer_for_test();
            }
        }
        return active_callbacks_ == 0;
    });
    commit_callback_ = {};
    abort_callback_  = {};
}

}  // namespace rtp_llm
