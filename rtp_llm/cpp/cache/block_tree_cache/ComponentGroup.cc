#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"

namespace rtp_llm {

LoadBackTicket::LoadBackTicket(std::shared_ptr<LoadBackTicketRegistry> registry,
                               uint64_t                                ticket_id,
                               std::vector<PendingLoadBackItem>        items,
                               size_t                                  logical_matched_blocks):
    registry_(std::move(registry)),
    ticket_id_(ticket_id),
    items_(std::move(items)),
    logical_matched_blocks_(logical_matched_blocks) {}

LoadBackTicket::~LoadBackTicket() {
    if (registry_ != nullptr && ticket_id_ != 0) {
        registry_->abort(ticket_id_);
    }
}

std::shared_ptr<AsyncContext> LoadBackTicket::commit() {
    if (registry_ == nullptr || ticket_id_ == 0 || items_.empty()) {
        return nullptr;
    }
    return registry_->commit(ticket_id_, items_);
}

LoadBackTicketRegistry::LoadBackTicketRegistry(CommitCallback commit_callback, AbortCallback abort_callback):
    commit_callback_(std::move(commit_callback)), abort_callback_(std::move(abort_callback)) {}

std::shared_ptr<LoadBackTicket> LoadBackTicketRegistry::createTicket(const std::vector<PendingLoadBackItem>& items,
                                                                     size_t logical_matched_blocks) {
    std::shared_ptr<LoadBackTicket> ticket(new LoadBackTicket(
        shared_from_this(), /*ticket_id=*/0, std::vector<PendingLoadBackItem>(items), logical_matched_blocks));
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

std::shared_ptr<AsyncContext> LoadBackTicketRegistry::commit(uint64_t                                ticket_id,
                                                             const std::vector<PendingLoadBackItem>& items) {
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
    return commit_callback ? commit_callback(items) : nullptr;
}

void LoadBackTicketRegistry::abort(uint64_t ticket_id) {
    AbortCallback                    abort_callback;
    std::vector<PendingLoadBackItem> abort_payload;
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
        abort_callback(abort_payload);
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
    AbortCallback                                 abort_callback;
    std::vector<std::vector<PendingLoadBackItem>> detached_payloads;
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
            for (const auto& abort_payload : detached_payloads) {
                abort_callback(abort_payload);
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

// ---- Base class default implementations (shared by Full/SWA/Linear) ----

void ComponentGroup::evictFromTier(TreeNode* node, GroupSlot& slot, Tier tier) {
    // Clear only the tier's block fields; heap membership is owned by BlockTreeEvictor.
    switch (tier) {
        case Tier::DEVICE:
            for (auto& block : slot.device_blocks) {
                block = NULL_BLOCK_IDX;
            }
            break;
        case Tier::HOST:
            slot.host_block = NULL_BLOCK_IDX;
            break;
        case Tier::DISK:
            slot.disk_slot = NULL_BLOCK_IDX;
            break;
        default:
            break;
    }
}

TransferDescriptor ComponentGroup::buildTransfer(TreeNode* node, TransferType type) {
    auto& slot = node->group_slots[static_cast<size_t>(component_group_id)];

    switch (type) {
        case TransferType::DEVICE_TO_HOST:
            return TransferDescriptor::deviceToHost(component_group_id, slot.device_blocks, NULL_BLOCK_IDX);
        case TransferType::HOST_TO_DEVICE:
            return TransferDescriptor::hostToDevice(component_group_id, slot.host_block, slot.device_blocks);
        case TransferType::HOST_TO_DISK:
            return TransferDescriptor::hostToDisk(component_group_id, slot.host_block, NULL_BLOCK_IDX);
        case TransferType::DISK_TO_HOST:
            return TransferDescriptor::diskToHost(component_group_id, slot.disk_slot, NULL_BLOCK_IDX);
        default:
            return {};
    }
}

bool ComponentGroup::isLeafAtTier(const TreeNode* node, int group_id, Tier tier) const {
    if (node == nullptr)
        return false;
    auto& slot = node->group_slots[static_cast<size_t>(group_id)];

    bool has_value = false;
    switch (tier) {
        case Tier::DEVICE:
            has_value = slot.has_value(Tier::DEVICE);
            break;
        case Tier::HOST:
            has_value = slot.has_value(Tier::HOST);
            break;
        case Tier::DISK:
            has_value = slot.has_value(Tier::DISK);
            break;
        default:
            return false;
    }
    if (!has_value)
        return false;

    for (const auto& [key, child] : node->children) {
        auto& child_slot = child->group_slots[static_cast<size_t>(group_id)];
        switch (tier) {
            case Tier::DEVICE:
                if (child_slot.has_value(Tier::DEVICE))
                    return false;
                break;
            case Tier::HOST:
                if (child_slot.has_value(Tier::HOST))
                    return false;
                break;
            case Tier::DISK:
                if (child_slot.has_value(Tier::DISK))
                    return false;
                break;
            default:
                break;
        }
    }
    return true;
}

// ---- Unified structured block lifecycle ----

GroupBlockSet ComponentGroup::allocateBlocks(Tier tier, size_t count) {
    GroupBlockSet set{component_group_id, tier};
    if (tier == Tier::DEVICE) {
        set.per_node.assign(count, std::vector<BlockIdxType>(device_pools_.size(), NULL_BLOCK_IDX));
        for (size_t p = 0; p < device_pools_.size(); ++p) {
            if (!device_pools_[p]) {
                unreferenceBlocks(set);
                return {};
            }
            auto blocks = device_pools_[p]->malloc(count);
            if (!blocks.has_value()) {
                unreferenceBlocks(set);
                return {};
            }
            device_pools_[p]->incRef(*blocks);
            for (size_t k = 0; k < count; ++k) {
                set.per_node[k][p] = (*blocks)[k];
            }
        }
        return set;
    }

    set.per_node.resize(count);
    for (size_t k = 0; k < count; ++k) {
        BlockIdxType b = allocateSingleBlock(tier);
        if (isNullBlockIdx(b)) {
            unreferenceBlocks(set);
            return {};
        }
        set.per_node[k] = {b};
    }
    return set;
}

void ComponentGroup::referenceBlocks(const GroupBlockSet& set) const {
    switch (set.tier) {
        case Tier::DEVICE:
            for (const auto& node_blocks : set.per_node) {
                for (size_t p = 0; p < node_blocks.size() && p < device_pools_.size(); ++p) {
                    if (device_pools_[p] && !isNullBlockIdx(node_blocks[p])) {
                        device_pools_[p]->incRef(node_blocks[p]);
                    }
                }
            }
            break;
        case Tier::HOST:
            if (host_pool_) {
                for (const auto& node_blocks : set.per_node)
                    for (auto b : node_blocks)
                        if (!isNullBlockIdx(b))
                            host_pool_->incRef(b);
            }
            break;
        case Tier::DISK:
            if (disk_pool_) {
                for (const auto& node_blocks : set.per_node)
                    for (auto b : node_blocks)
                        if (!isNullBlockIdx(b))
                            disk_pool_->incRef(b);
            }
            break;
        default:
            break;
    }
}

void ComponentGroup::unreferenceBlocks(const GroupBlockSet& set) const {
    switch (set.tier) {
        case Tier::DEVICE:
            for (const auto& node_blocks : set.per_node) {
                for (size_t p = 0; p < node_blocks.size() && p < device_pools_.size(); ++p) {
                    if (device_pools_[p] && !isNullBlockIdx(node_blocks[p])) {
                        device_pools_[p]->decRef(node_blocks[p]);
                    }
                }
            }
            break;
        case Tier::HOST:
            if (host_pool_) {
                for (const auto& node_blocks : set.per_node)
                    for (auto b : node_blocks)
                        if (!isNullBlockIdx(b))
                            host_pool_->decRef(b);
            }
            break;
        case Tier::DISK:
            if (disk_pool_) {
                for (const auto& node_blocks : set.per_node)
                    for (auto b : node_blocks)
                        if (!isNullBlockIdx(b))
                            disk_pool_->decRef(b);
            }
            break;
        default:
            break;
    }
}

BlockIdxType ComponentGroup::allocateSingleBlock(Tier tier) {
    // DEVICE spans multiple pools and has no scalar block: use allocateBlocks.
    IBlockPool* pool = nullptr;
    if (tier == Tier::HOST) {
        pool = host_pool_.get();
    } else if (tier == Tier::DISK) {
        pool = disk_pool_.get();
    }
    if (!pool)
        return NULL_BLOCK_IDX;
    auto b = pool->malloc();
    if (!b.has_value())
        return NULL_BLOCK_IDX;
    pool->incRef(*b);
    return *b;
}

void ComponentGroup::releaseSingleBlock(Tier tier, BlockIdxType block) const {
    if (isNullBlockIdx(block))
        return;
    if (tier == Tier::HOST) {
        if (host_pool_)
            host_pool_->decRef(block);
    } else if (tier == Tier::DISK) {
        if (disk_pool_)
            disk_pool_->decRef(block);
    }
}

std::vector<BlockIdxType> ComponentGroup::getBlocks(const GroupSlot& slot, Tier tier) const {
    switch (tier) {
        case Tier::DEVICE:
            return slot.has_value(Tier::DEVICE) ? slot.device_blocks : std::vector<BlockIdxType>{};
        case Tier::HOST:
            return slot.has_value(Tier::HOST) ? std::vector<BlockIdxType>{slot.host_block} :
                                                std::vector<BlockIdxType>{};
        case Tier::DISK:
            return slot.has_value(Tier::DISK) ? std::vector<BlockIdxType>{slot.disk_slot} : std::vector<BlockIdxType>{};
        default:
            return {};
    }
}

Tier ComponentGroup::getTopTier(const GroupSlot& slot) const {
    if (slot.has_value(Tier::DEVICE)) {
        return Tier::DEVICE;
    }
    if (slot.has_value(Tier::HOST)) {
        return Tier::HOST;
    }
    if (slot.has_value(Tier::DISK)) {
        return Tier::DISK;
    }
    return Tier::NONE;
}

void ComponentGroup::setBlocks(GroupSlot& slot, Tier tier, const std::vector<BlockIdxType>& blocks) {
    switch (tier) {
        case Tier::DEVICE:
            slot.device_blocks = blocks;
            break;
        case Tier::HOST:
            slot.host_block = blocks.empty() ? NULL_BLOCK_IDX : blocks[0];
            break;
        case Tier::DISK:
            slot.disk_slot = blocks.empty() ? NULL_BLOCK_IDX : blocks[0];
            break;
        default:
            break;
    }
}

bool ComponentGroup::isSlotEvictable(const TreeNode& node, Tier tier) const {
    if (component_group_id < 0 || static_cast<size_t>(component_group_id) >= node.group_slots.size()) {
        return false;
    }
    const auto& slot = node.group_slots[static_cast<size_t>(component_group_id)];

    // A block is evictable only when its sole holder is the cache reference
    // (refCount == 1). When no pool owns the block, treat it as evictable.
    auto pool_evictable = [](const auto& pool, BlockIdxType block) {
        if (isNullBlockIdx(block) || !pool) {
            return true;
        }
        return pool->isAllocated(block) && pool->refCount(block) == 1;
    };

    switch (tier) {
        case Tier::DEVICE:
            if (!slot.has_value(Tier::DEVICE)) {
                return false;
            }
            for (size_t i = 0; i < slot.device_blocks.size(); ++i) {
                const auto& pool = i < device_pools_.size() ? device_pools_[i] : nullptr;
                if (!pool_evictable(pool, slot.device_blocks[i])) {
                    return false;
                }
            }
            return true;
        case Tier::HOST:
            return slot.has_value(Tier::HOST) && pool_evictable(host_pool_, slot.host_block);
        case Tier::DISK:
            return slot.has_value(Tier::DISK) && pool_evictable(disk_pool_, slot.disk_slot);
        default:
            return false;
    }
}

}  // namespace rtp_llm
