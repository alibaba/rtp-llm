#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"

#include <limits>
#include <unordered_set>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

LoadBackTicket::LoadBackTicket(std::shared_ptr<LoadBackTicketRegistry> registry,
                               uint64_t                                ticket_id,
                               std::vector<PendingLoadBackItem>        items,
                               size_t                                  logical_matched_blocks):
    registry_(std::move(registry)),
    ticket_id_(ticket_id),
    items_(std::move(items)),
    logical_matched_blocks_(logical_matched_blocks) {
    std::vector<Tier> logical_tiers(logical_matched_blocks_, Tier::DEVICE);
    for (const PendingLoadBackItem& item : items_) {
        if (item.path_index >= logical_tiers.size() || item.source_tier < Tier::DEVICE
            || item.source_tier > Tier::DISK) {
            continue;
        }
        Tier& logical_tier = logical_tiers[item.path_index];
        if (static_cast<int>(item.source_tier) > static_cast<int>(logical_tier)) {
            logical_tier = item.source_tier;
        }
    }
    for (Tier tier : logical_tiers) {
        logical_matched_blocks_by_tier_[static_cast<size_t>(tier)]++;
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
    return registry_->commit(ticket_id_, items_);
}

size_t LoadBackTicket::logicalMatchedBlocks(Tier tier) const {
    if (tier < Tier::DEVICE || tier > Tier::DISK) {
        return 0;
    }
    return logical_matched_blocks_by_tier_[static_cast<size_t>(tier)];
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

std::optional<ComponentGroupLayout>
ComponentGroupLayout::create(const std::vector<std::vector<size_t>>& component_layer_bytes) {
    if (component_layer_bytes.empty()) {
        RTP_LLM_LOG_WARNING("ComponentGroupLayout: components must not be empty");
        return std::nullopt;
    }

    ComponentGroupLayout layout;
    size_t               offset = 0;
    for (size_t component_idx = 0; component_idx < component_layer_bytes.size(); ++component_idx) {
        const auto& layer_bytes = component_layer_bytes[component_idx];
        if (layer_bytes.empty()) {
            RTP_LLM_LOG_WARNING("ComponentGroupLayout: component=%zu has no layer binding", component_idx);
            return std::nullopt;
        }
        if (layer_bytes.size() > std::numeric_limits<size_t>::max() - layout.slices_.size()) {
            RTP_LLM_LOG_WARNING("ComponentGroupLayout: total layer count overflow at component=%zu", component_idx);
            return std::nullopt;
        }
        layout.slices_.reserve(layout.slices_.size() + layer_bytes.size());
        for (size_t layer_idx = 0; layer_idx < layer_bytes.size(); ++layer_idx) {
            const size_t bytes = layer_bytes[layer_idx];
            if (bytes == 0) {
                RTP_LLM_LOG_WARNING(
                    "ComponentGroupLayout: component=%zu layer=%zu has zero packed bytes", component_idx, layer_idx);
                return std::nullopt;
            }
            if (bytes > std::numeric_limits<size_t>::max() - offset) {
                RTP_LLM_LOG_WARNING("ComponentGroupLayout: payload offset overflow at component=%zu layer=%zu",
                                    component_idx,
                                    layer_idx);
                return std::nullopt;
            }
            layout.slices_.push_back(Slice{component_idx, layer_idx, offset});
            offset += bytes;
        }
    }

    layout.payload_bytes_ = offset;
    return layout;
}

bool ComponentGroup::finalizeLayout(std::vector<int> component_indices, const std::vector<Component>& components) {
    if (layout_.has_value()) {
        RTP_LLM_LOG_ERROR("ComponentGroup::finalizeLayout: group %d layout is already sealed", component_group_id);
        return false;
    }

    std::unordered_set<std::string>  tags;
    std::vector<std::vector<size_t>> component_layer_bytes;
    component_layer_bytes.reserve(component_indices.size());
    for (int component_index : component_indices) {
        if (component_index < 0 || static_cast<size_t>(component_index) >= components.size()) {
            RTP_LLM_LOG_ERROR("ComponentGroup::finalizeLayout: invalid component_index=%d group=%d registry_size=%zu",
                              component_index,
                              component_group_id,
                              components.size());
            return false;
        }
        const Component& component = components[static_cast<size_t>(component_index)];
        if (component.component_group_id != component_group_id) {
            RTP_LLM_LOG_ERROR("ComponentGroup::finalizeLayout: component[%d] belongs to group %d, expected %d",
                              component_index,
                              component.component_group_id,
                              component_group_id);
            return false;
        }
        if (component.tag.empty() || !tags.insert(component.tag).second) {
            RTP_LLM_LOG_ERROR("ComponentGroup::finalizeLayout: component[%d] has empty or duplicate tag=%s",
                              component_index,
                              component.tag.c_str());
            return false;
        }
        if (component.model_layer_ids.size() != component.layer_bytes.size()) {
            RTP_LLM_LOG_ERROR(
                "ComponentGroup::finalizeLayout: component[%d] layer id count %zu != layer bytes count %zu",
                component_index,
                component.model_layer_ids.size(),
                component.layer_bytes.size());
            return false;
        }
        component_layer_bytes.push_back(component.layer_bytes);
    }

    auto layout = ComponentGroupLayout::create(component_layer_bytes);
    if (!layout.has_value()) {
        RTP_LLM_LOG_ERROR("ComponentGroup::finalizeLayout: schema validation failed for group %d", component_group_id);
        return false;
    }
    // Commit membership and layout together; neither is observable on failure.
    component_indices_ = std::move(component_indices);
    layout_            = std::move(*layout);
    return true;
}

const ComponentGroupLayout& ComponentGroup::layout() const {
    RTP_LLM_CHECK_WITH_INFO(layout_.has_value(), "ComponentGroup %d layout has not been finalized", component_group_id);
    return *layout_;
}

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

GroupBlockSet ComponentGroup::allocateBlocks(Tier tier, size_t count, BlockRefType ref_type) {
    GroupBlockSet set{component_group_id, tier};
    if (tier == Tier::DEVICE) {
        set.per_node.assign(count, std::vector<BlockIdxType>(device_pools_.size(), NULL_BLOCK_IDX));
        for (size_t p = 0; p < device_pools_.size(); ++p) {
            if (!device_pools_[p]) {
                unreferenceBlocks(set, ref_type);
                return {};
            }
            const std::optional<BlockIdList> blocks = device_pools_[p]->malloc(count);
            if (!blocks.has_value()) {
                unreferenceBlocks(set, ref_type);
                return {};
            }
            device_pools_[p]->incRef(*blocks, ref_type);
            for (size_t k = 0; k < count; ++k) {
                set.per_node[k][p] = (*blocks)[k];
            }
        }
        return set;
    }

    set.per_node.resize(count);
    for (size_t k = 0; k < count; ++k) {
        BlockIdxType b = allocateSingleBlock(tier, ref_type);
        if (isNullBlockIdx(b)) {
            unreferenceBlocks(set, ref_type);
            return {};
        }
        set.per_node[k] = {b};
    }
    return set;
}

void ComponentGroup::referenceBlocks(const GroupBlockSet& set, BlockRefType ref_type) const {
    switch (set.tier) {
        case Tier::DEVICE:
            for (const auto& node_blocks : set.per_node) {
                for (size_t p = 0; p < node_blocks.size() && p < device_pools_.size(); ++p) {
                    if (device_pools_[p] && !isNullBlockIdx(node_blocks[p])) {
                        device_pools_[p]->incRef(node_blocks[p], ref_type);
                    }
                }
            }
            break;
        case Tier::HOST:
            if (host_pool_) {
                for (const auto& node_blocks : set.per_node)
                    for (auto b : node_blocks)
                        if (!isNullBlockIdx(b))
                            host_pool_->incRef(b, ref_type);
            }
            break;
        case Tier::DISK:
            if (disk_pool_) {
                for (const auto& node_blocks : set.per_node)
                    for (auto b : node_blocks)
                        if (!isNullBlockIdx(b))
                            disk_pool_->incRef(b, ref_type);
            }
            break;
        default:
            break;
    }
}

void ComponentGroup::unreferenceBlocks(const GroupBlockSet& set, BlockRefType ref_type) const {
    switch (set.tier) {
        case Tier::DEVICE:
            for (const auto& node_blocks : set.per_node) {
                for (size_t p = 0; p < node_blocks.size() && p < device_pools_.size(); ++p) {
                    if (device_pools_[p] && !isNullBlockIdx(node_blocks[p])) {
                        device_pools_[p]->decRef(node_blocks[p], ref_type);
                    }
                }
            }
            break;
        case Tier::HOST:
            if (host_pool_) {
                for (const auto& node_blocks : set.per_node)
                    for (auto b : node_blocks)
                        if (!isNullBlockIdx(b))
                            host_pool_->decRef(b, ref_type);
            }
            break;
        case Tier::DISK:
            if (disk_pool_) {
                for (const auto& node_blocks : set.per_node)
                    for (auto b : node_blocks)
                        if (!isNullBlockIdx(b))
                            disk_pool_->decRef(b, ref_type);
            }
            break;
        default:
            break;
    }
}

BlockIdxType ComponentGroup::allocateSingleBlock(Tier tier, BlockRefType ref_type) {
    // DEVICE spans multiple pools and has no scalar block: use allocateBlocks.
    IBlockPool* pool = nullptr;
    if (tier == Tier::HOST) {
        pool = host_pool_.get();
    } else if (tier == Tier::DISK) {
        pool = disk_pool_.get();
    }
    if (!pool)
        return NULL_BLOCK_IDX;
    const std::optional<BlockIdxType> b = pool->malloc();
    if (!b.has_value())
        return NULL_BLOCK_IDX;
    pool->incRef(*b, ref_type);
    return *b;
}

void ComponentGroup::releaseSingleBlock(Tier tier, BlockIdxType block, BlockRefType ref_type) const {
    if (isNullBlockIdx(block))
        return;
    if (tier == Tier::HOST) {
        if (host_pool_)
            host_pool_->decRef(block, ref_type);
    } else if (tier == Tier::DISK) {
        if (disk_pool_)
            disk_pool_->decRef(block, ref_type);
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
