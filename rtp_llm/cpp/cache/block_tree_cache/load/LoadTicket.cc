#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTicket.h"

namespace rtp_llm {

LoadTicket::LoadTicket(std::shared_ptr<LoadTicketRegistry> registry,
                       uint64_t                            ticket_id,
                       PendingLoadItems                    items,
                       size_t                              logical_matched_blocks,
                       std::shared_ptr<LoadAsyncContext>   context):
    registry_(std::move(registry)),
    ticket_id_(ticket_id),
    items_(std::move(items)),
    logical_matched_blocks_(logical_matched_blocks),
    context_(std::move(context)) {
    std::unordered_map<size_t, Tier> reuse_tier_by_path;
    for (const PendingLoadItem& item : items_) {
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

LoadTicket::~LoadTicket() {
    if (registry_ != nullptr && ticket_id_ != 0) {
        registry_->abort(ticket_id_);
    }
}

std::shared_ptr<AsyncContext> LoadTicket::commit() {
    if (registry_ == nullptr || ticket_id_ == 0 || items_.empty()) {
        return nullptr;
    }
    return registry_->commit(ticket_id_, *this);
}

size_t LoadTicket::logicalMatchedBlocks(Tier tier) const {
    if (tier < Tier::DEVICE || tier > Tier::DISK) {
        return 0;
    }
    return logical_matched_blocks_by_tier_[static_cast<size_t>(tier)];
}

LoadTicketRegistry::LoadTicketRegistry(CommitCallback commit_callback, AbortCallback abort_callback):
    commit_callback_(std::move(commit_callback)), abort_callback_(std::move(abort_callback)) {}

std::shared_ptr<LoadTicket> LoadTicketRegistry::createTicket(const LoadTicket::PendingLoadItems& items,
                                                             size_t                              logical_matched_blocks,
                                                             const std::shared_ptr<LoadAsyncContext>& context) {
    std::shared_ptr<LoadTicket> ticket(new LoadTicket(shared_from_this(),
                                                      /*ticket_id=*/0,
                                                      LoadTicket::PendingLoadItems(items),
                                                      logical_matched_blocks,
                                                      context));
    if (items.empty()) {
        return ticket;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!accepting_ || next_ticket_id_ == 0) {
        return nullptr;
    }

    const uint64_t ticket_id    = next_ticket_id_;
    pending_tickets_[ticket_id] = PendingTicket{items, context};
    ticket->ticket_id_          = ticket_id;
    ++next_ticket_id_;
    return ticket;
}

std::shared_ptr<AsyncContext> LoadTicketRegistry::commit(uint64_t ticket_id, const LoadTicket& ticket) {
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

void LoadTicketRegistry::abort(uint64_t ticket_id) {
    AbortCallback abort_callback;
    PendingTicket abort_payload;
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
        LoadTicket abort_ticket(nullptr,
                                /*ticket_id=*/0,
                                std::move(abort_payload.items),
                                /*logical_matched_blocks=*/0,
                                std::move(abort_payload.context));
        abort_callback(abort_ticket);
    }
}

void LoadTicketRegistry::retireActiveCallback() {
    std::lock_guard<std::mutex> lock(mutex_);
    --active_callbacks_;
    if (active_callbacks_ == 0) {
        cv_.notify_all();
    }
}

void LoadTicketRegistry::shutdown() {
    AbortCallback              abort_callback;
    std::vector<PendingTicket> detached_payloads;
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
            for (PendingTicket& abort_payload : detached_payloads) {
                LoadTicket abort_ticket(nullptr,
                                        /*ticket_id=*/0,
                                        std::move(abort_payload.items),
                                        /*logical_matched_blocks=*/0,
                                        std::move(abort_payload.context));
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
