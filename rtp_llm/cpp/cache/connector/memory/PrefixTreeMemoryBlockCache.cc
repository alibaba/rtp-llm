#include "rtp_llm/cpp/cache/connector/memory/PrefixTreeMemoryBlockCache.h"

#include <algorithm>
#include <mutex>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

size_t PrefixTreeMemoryBlockCache::kindIndex(CacheBlockKind kind) {
    RTP_LLM_CHECK_WITH_INFO(validKind(kind), "invalid prefix-tree memory kind %d", static_cast<int>(kind));
    return kind == CacheBlockKind::COMPRESSED_KV ? 0 : 1;
}

bool PrefixTreeMemoryBlockCache::validKind(CacheBlockKind kind) {
    return kind == CacheBlockKind::COMPRESSED_KV || kind == CacheBlockKind::STATE_SWA_KV;
}

bool PrefixTreeMemoryBlockCache::slotMaskCovers(const std::vector<uint8_t>& stored,
                                                const std::vector<uint8_t>& required) {
    for (size_t i = 0; i < required.size(); ++i) {
        if (required[i] == 0) {
            continue;
        }
        if (i >= stored.size() || stored[i] == 0) {
            return false;
        }
    }
    return true;
}

bool PrefixTreeMemoryBlockCache::contains(CacheKeyType cache_key, CacheBlockKind kind) const {
    static const std::vector<uint8_t> empty_required_mask;
    return contains(cache_key, kind, empty_required_mask);
}

bool PrefixTreeMemoryBlockCache::contains(CacheKeyType                 cache_key,
                                          CacheBlockKind               kind,
                                          const std::vector<uint8_t>& required_slot_mask) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto                                it = nodes_.find(cache_key);
    if (it == nodes_.end() || !validKind(kind)) {
        return false;
    }
    const auto& state = it->second.kinds[kindIndex(kind)];
    return state.has_value && !state.detached && slotMaskCovers(state.slot_valid_mask, required_slot_mask);
}

PrefixTreeMemoryBlockCache::MatchResult
PrefixTreeMemoryBlockCache::match(CacheKeyType cache_key, CacheBlockKind kind) {
    static const std::vector<uint8_t> empty_required_mask;
    return match(cache_key, kind, empty_required_mask);
}

PrefixTreeMemoryBlockCache::MatchResult
PrefixTreeMemoryBlockCache::match(CacheKeyType                 cache_key,
                                  CacheBlockKind               kind,
                                  const std::vector<uint8_t>& required_slot_mask) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = nodes_.find(cache_key);
    if (it == nodes_.end() || !validKind(kind)) {
        return {};
    }
    auto& state = it->second.kinds[kindIndex(kind)];
    if (!state.has_value || state.detached || !slotMaskCovers(state.slot_valid_mask, required_slot_mask)) {
        return {};
    }
    touchLocked(it->second, kind);
    return {true,
            state.backing_type,
            state.block_index,
            state.disk_slot,
            state.block_size,
            state.generation,
            state.created_time_us,
            state.slot_valid_mask};
}

PrefixTreeMemoryBlockCache::MatchResult
PrefixTreeMemoryBlockCache::matchAndMarkInFlight(CacheKeyType cache_key, CacheBlockKind kind) {
    static const std::vector<uint8_t> empty_required_mask;
    return matchAndMarkInFlight(cache_key, kind, empty_required_mask);
}

PrefixTreeMemoryBlockCache::MatchResult
PrefixTreeMemoryBlockCache::matchAndMarkInFlight(CacheKeyType                 cache_key,
                                                 CacheBlockKind               kind,
                                                 const std::vector<uint8_t>& required_slot_mask) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = nodes_.find(cache_key);
    if (it == nodes_.end() || !validKind(kind)) {
        return {};
    }
    auto& state = it->second.kinds[kindIndex(kind)];
    if (!state.has_value || state.detached || !slotMaskCovers(state.slot_valid_mask, required_slot_mask)) {
        return {};
    }
    touchLocked(it->second, kind);
    state.in_flight_ref++;
    eraseEvictKeyLocked(it->second, kind);
    return {true,
            state.backing_type,
            state.block_index,
            state.disk_slot,
            state.block_size,
            state.generation,
            state.created_time_us,
            state.slot_valid_mask};
}

std::pair<bool, std::optional<PrefixTreeMemoryBlockCache::CacheItem>>
PrefixTreeMemoryBlockCache::putCommitted(CacheKeyType            cache_key,
                                          const BlockDependency&  dependency,
                                          const CacheItem&        input_item) {
    RTP_LLM_CHECK_WITH_INFO(validKind(input_item.kind), "invalid prefix-tree memory kind");
    RTP_LLM_CHECK_WITH_INFO(input_item.cache_key == cache_key, "cache key mismatch");
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto&                               node  = upsertNodeLocked(cache_key, dependency);
    auto&                               state = node.kinds[kindIndex(input_item.kind)];
    std::optional<CacheItem> old_item;
    if (state.has_value && !state.detached) {
        if (slotMaskCovers(state.slot_valid_mask, input_item.slot_valid_mask)) {
            return {false, std::nullopt};
        }
        if (!slotMaskCovers(input_item.slot_valid_mask, state.slot_valid_mask)) {
            return {false, std::nullopt};
        }
        old_item = toItemLocked(node, input_item.kind);
        eraseEvictKeyLocked(node, input_item.kind);
        if (state.in_flight_ref > 0 && old_item.has_value()) {
            node.retired_items[kindIndex(input_item.kind)].push_back(RetiredItem{*old_item, state.in_flight_ref});
            old_item.reset();
        }
    } else {
        incrementAncestorsLocked(cache_key, input_item.kind);
    }

    state.has_value       = true;
    state.detached        = false;
    state.backing_type    = input_item.backing_type;
    state.block_index     = input_item.block_index;
    state.disk_slot       = input_item.disk_slot;
    state.block_size      = input_item.block_size;
    state.is_resident     = input_item.is_resident;
    state.generation      = ++generation_seq_;
    state.last_access_seq = ++access_seq_;
    state.created_time_us = input_item.created_time_us > 0 ? input_item.created_time_us : currentTimeUs();
    state.in_flight_ref   = 0;
    state.slot_valid_mask = input_item.slot_valid_mask;
    insertEvictKeyLocked(node, input_item.kind);
    return {true, old_item};
}

std::optional<PrefixTreeMemoryBlockCache::CacheItem>
PrefixTreeMemoryBlockCache::detachIfMatch(CacheKeyType     cache_key,
                                           CacheBlockKind   kind,
                                           CacheBackingType backing_type,
                                           BlockIdxType     expected_block_index,
                                           int32_t          expected_disk_slot,
                                           uint64_t         expected_generation) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = nodes_.find(cache_key);
    if (it == nodes_.end() || !validKind(kind)) {
        return std::nullopt;
    }
    auto& state = it->second.kinds[kindIndex(kind)];
    if (!state.has_value || state.detached || state.backing_type != backing_type
        || state.generation != expected_generation) {
        return std::nullopt;
    }
    if (backing_type == CacheBackingType::MEMORY && state.block_index != expected_block_index) {
        return std::nullopt;
    }
    if (backing_type == CacheBackingType::DISK && state.disk_slot != expected_disk_slot) {
        return std::nullopt;
    }
    auto item = toItemLocked(it->second, kind);
    if (!item.has_value()) {
        return std::nullopt;
    }
    eraseEvictKeyLocked(it->second, kind);
    state.detached = true;
    decrementAncestorsLocked(cache_key, kind);
    const auto descendant_ref_count = state.subtree_ref_count;
    if (state.in_flight_ref == 0) {
        state = KindState{};
        state.subtree_ref_count = descendant_ref_count;
        pruneLocked(cache_key);
        return item;
    }
    it->second.retired_items[kindIndex(kind)].push_back(RetiredItem{*item, state.in_flight_ref});
    state = KindState{};
    state.subtree_ref_count = descendant_ref_count;
    pruneLocked(cache_key);
    return std::nullopt;
}

std::optional<PrefixTreeMemoryBlockCache::CacheItem>
PrefixTreeMemoryBlockCache::releaseInFlight(CacheKeyType     cache_key,
                                            CacheBlockKind   kind,
                                            CacheBackingType backing_type,
                                            BlockIdxType     block_index,
                                            int32_t          disk_slot,
                                            uint64_t         generation) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = nodes_.find(cache_key);
    if (it == nodes_.end() || !validKind(kind)) {
        return std::nullopt;
    }
    auto& state = it->second.kinds[kindIndex(kind)];
    if (!state.has_value || state.backing_type != backing_type || state.generation != generation) {
        auto& retired_items = it->second.retired_items[kindIndex(kind)];
        for (auto retired_it = retired_items.begin(); retired_it != retired_items.end(); ++retired_it) {
            auto& item = retired_it->item;
            if (item.backing_type != backing_type || item.generation != generation) {
                continue;
            }
            if (backing_type == CacheBackingType::MEMORY && item.block_index != block_index) {
                continue;
            }
            if (backing_type == CacheBackingType::DISK && item.disk_slot != disk_slot) {
                continue;
            }
            if (retired_it->in_flight_ref > 0) {
                retired_it->in_flight_ref--;
            }
            if (retired_it->in_flight_ref == 0) {
                auto released = item;
                retired_items.erase(retired_it);
                pruneLocked(cache_key);
                return released;
            }
            return std::nullopt;
        }
        return std::nullopt;
    }
    if (backing_type == CacheBackingType::MEMORY && state.block_index != block_index) {
        return std::nullopt;
    }
    if (backing_type == CacheBackingType::DISK && state.disk_slot != disk_slot) {
        return std::nullopt;
    }
    if (state.in_flight_ref > 0) {
        state.in_flight_ref--;
    }
    if (state.detached && state.in_flight_ref == 0) {
        auto released = toItemLocked(it->second, kind);
        state = KindState{};
        pruneLocked(cache_key);
        return released;
    } else if (!state.detached && state.in_flight_ref == 0) {
        refreshEvictKeyLocked(it->second, kind);
    }
    return std::nullopt;
}

std::optional<PrefixTreeMemoryBlockCache::CacheItem>
PrefixTreeMemoryBlockCache::popOldestEvictable(CacheBlockKind kind) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!validKind(kind)) {
        return std::nullopt;
    }
    auto& lru = leaf_lru_[kindIndex(kind)];
    for (auto it = lru.begin(); it != lru.end();) {
        auto node_it = nodes_.find(it->cache_key);
        if (node_it == nodes_.end()) {
            it = lru.erase(it);
            continue;
        }
        auto& state = node_it->second.kinds[kindIndex(kind)];
        if (!state.has_value || state.detached || state.last_access_seq != it->last_access_seq
            || state.generation != it->generation) {
            it = lru.erase(it);
            continue;
        }
        if (state.is_resident || state.in_flight_ref > 0 || !isKindLeafLocked(node_it->second, kind)) {
            ++it;
            continue;
        }
        auto item = toItemLocked(node_it->second, kind);
        it        = lru.erase(it);
        state     = KindState{};
        decrementAncestorsLocked(item->cache_key, kind);
        pruneLocked(item->cache_key);
        return item;
    }
    return std::nullopt;
}

std::optional<PrefixTreeMemoryBlockCache::CacheItem>
PrefixTreeMemoryBlockCache::popOldestEvictable(CacheBlockKind kind, CacheBackingType backing_type) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!validKind(kind)) {
        return std::nullopt;
    }
    auto& lru = leaf_lru_[kindIndex(kind)];
    for (auto it = lru.begin(); it != lru.end();) {
        auto node_it = nodes_.find(it->cache_key);
        if (node_it == nodes_.end()) {
            it = lru.erase(it);
            continue;
        }
        auto& state = node_it->second.kinds[kindIndex(kind)];
        if (!state.has_value || state.detached || state.last_access_seq != it->last_access_seq
            || state.generation != it->generation) {
            it = lru.erase(it);
            continue;
        }
        if (state.backing_type != backing_type || state.is_resident || state.in_flight_ref > 0
            || !isKindLeafLocked(node_it->second, kind)) {
            ++it;
            continue;
        }
        auto item = toItemLocked(node_it->second, kind);
        it        = lru.erase(it);
        state     = KindState{};
        decrementAncestorsLocked(item->cache_key, kind);
        pruneLocked(item->cache_key);
        return item;
    }
    return std::nullopt;
}

std::vector<PrefixTreeMemoryBlockCache::CacheItem>
PrefixTreeMemoryBlockCache::popOldestStateOrChainEvictable(CacheBackingType backing_type) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    std::vector<CacheKeyType>           leaf_keys;
    const auto&                         state_lru = leaf_lru_[kindIndex(CacheBlockKind::STATE_SWA_KV)];
    leaf_keys.reserve(state_lru.size());
    for (const auto& evict_key : state_lru) {
        leaf_keys.push_back(evict_key.cache_key);
    }

    for (const auto leaf_key : leaf_keys) {
        auto item = popStateOnlyFromChainLocked(leaf_key, backing_type);
        if (item.has_value()) {
            return {*item};
        }
    }
    for (const auto leaf_key : leaf_keys) {
        auto items = popChainLocked(leaf_key, backing_type);
        if (!items.empty()) {
            return items;
        }
    }
    return {};
}

std::vector<CacheKeyType> PrefixTreeMemoryBlockCache::cacheKeys() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<std::pair<uint64_t, CacheKeyType>> entries;
    for (const auto& [key, node] : nodes_) {
        uint64_t latest = 0;
        for (const auto& state : node.kinds) {
            if (state.has_value && !state.detached) {
                latest = std::max(latest, state.last_access_seq);
            }
        }
        if (latest > 0) {
            entries.emplace_back(latest, key);
        }
    }
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first > rhs.first;
        }
        return lhs.second < rhs.second;
    });
    std::vector<CacheKeyType> keys;
    keys.reserve(entries.size());
    for (const auto& [_, key] : entries) {
        keys.push_back(key);
    }
    return keys;
}

std::vector<CacheKeyType> PrefixTreeMemoryBlockCache::cacheKeysUnorderedForStatus() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<CacheKeyType>           keys;
    keys.reserve(nodes_.size());
    for (const auto& [key, node] : nodes_) {
        for (const auto& state : node.kinds) {
            if (state.has_value && !state.detached) {
                keys.push_back(key);
                break;
            }
        }
    }
    return keys;
}

size_t PrefixTreeMemoryBlockCache::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& [_, node] : nodes_) {
        for (const auto& state : node.kinds) {
            if (state.has_value && !state.detached) {
                ++count;
            }
        }
    }
    return count;
}

PrefixTreeMemoryBlockCache::Node&
PrefixTreeMemoryBlockCache::upsertNodeLocked(CacheKeyType cache_key, const BlockDependency& dependency) {
    auto it = nodes_.find(cache_key);
    if (it == nodes_.end()) {
        Node node;
        node.cache_key  = cache_key;
        node.parent_key = dependency.parent_key;
        node.has_parent = dependency.has_parent && dependency.parent_key != cache_key;
        node.ordinal    = dependency.ordinal;
        auto [inserted_it, _] = nodes_.emplace(cache_key, std::move(node));
        it = inserted_it;
    } else {
        if (it->second.has_parent
            && (it->second.parent_key != dependency.parent_key || !dependency.has_parent
                || dependency.parent_key == cache_key)) {
            auto old_parent_it = nodes_.find(it->second.parent_key);
            if (old_parent_it != nodes_.end()) {
                subtractSubtreeRefsFromAncestorsLocked(old_parent_it->first, it->second);
                old_parent_it->second.children.erase(cache_key);
            } else {
                detachPendingChildLocked(it->second.parent_key, cache_key);
            }
        }
        it->second.parent_key = dependency.parent_key;
        it->second.has_parent = dependency.has_parent && dependency.parent_key != cache_key;
        it->second.ordinal    = dependency.ordinal;
    }
    if (it->second.has_parent) {
        auto parent_it = nodes_.find(it->second.parent_key);
        if (parent_it != nodes_.end()) {
            auto [_, inserted] = parent_it->second.children.insert(cache_key);
            if (inserted) {
                detachPendingChildLocked(it->second.parent_key, cache_key);
                addSubtreeRefsToAncestorsLocked(parent_it->first, it->second);
            }
        } else {
            pending_children_by_parent_[it->second.parent_key].insert(cache_key);
        }
    }
    attachPendingChildrenLocked(it->second);
    return it->second;
}

void PrefixTreeMemoryBlockCache::incrementAncestorsLocked(CacheKeyType cache_key, CacheBlockKind kind) {
    CacheKeyType cur = cache_key;
    while (true) {
        auto it = nodes_.find(cur);
        if (it == nodes_.end()) {
            break;
        }
        it->second.kinds[kindIndex(kind)].subtree_ref_count++;
        refreshEvictKeyLocked(it->second, kind);
        if (!it->second.has_parent) {
            break;
        }
        cur = it->second.parent_key;
    }
}

void PrefixTreeMemoryBlockCache::decrementAncestorsLocked(CacheKeyType cache_key, CacheBlockKind kind) {
    CacheKeyType cur = cache_key;
    while (true) {
        auto it = nodes_.find(cur);
        if (it == nodes_.end()) {
            break;
        }
        auto& count = it->second.kinds[kindIndex(kind)].subtree_ref_count;
        if (count > 0) {
            count--;
        }
        refreshEvictKeyLocked(it->second, kind);
        if (!it->second.has_parent) {
            break;
        }
        cur = it->second.parent_key;
    }
}

void PrefixTreeMemoryBlockCache::addSubtreeRefsToAncestorsLocked(CacheKeyType ancestor_key, const Node& child) {
    CacheKeyType cur = ancestor_key;
    while (true) {
        auto it = nodes_.find(cur);
        if (it == nodes_.end()) {
            break;
        }
        for (size_t kind_idx = 0; kind_idx < kKindCount; ++kind_idx) {
            const auto delta = child.kinds[kind_idx].subtree_ref_count;
            if (delta == 0) {
                continue;
            }
            auto kind = kind_idx == 0 ? CacheBlockKind::COMPRESSED_KV : CacheBlockKind::STATE_SWA_KV;
            eraseEvictKeyLocked(it->second, kind);
            it->second.kinds[kind_idx].subtree_ref_count += delta;
            insertEvictKeyLocked(it->second, kind);
        }
        if (!it->second.has_parent) {
            break;
        }
        cur = it->second.parent_key;
    }
}

void PrefixTreeMemoryBlockCache::subtractSubtreeRefsFromAncestorsLocked(CacheKeyType ancestor_key, const Node& child) {
    CacheKeyType cur = ancestor_key;
    while (true) {
        auto it = nodes_.find(cur);
        if (it == nodes_.end()) {
            break;
        }
        for (size_t kind_idx = 0; kind_idx < kKindCount; ++kind_idx) {
            const auto delta = child.kinds[kind_idx].subtree_ref_count;
            if (delta == 0) {
                continue;
            }
            auto kind = kind_idx == 0 ? CacheBlockKind::COMPRESSED_KV : CacheBlockKind::STATE_SWA_KV;
            eraseEvictKeyLocked(it->second, kind);
            auto& count = it->second.kinds[kind_idx].subtree_ref_count;
            count       = count > delta ? count - delta : 0;
            insertEvictKeyLocked(it->second, kind);
        }
        if (!it->second.has_parent) {
            break;
        }
        cur = it->second.parent_key;
    }
}

void PrefixTreeMemoryBlockCache::detachPendingChildLocked(CacheKeyType parent_key, CacheKeyType child_key) {
    auto pending_it = pending_children_by_parent_.find(parent_key);
    if (pending_it == pending_children_by_parent_.end()) {
        return;
    }
    pending_it->second.erase(child_key);
    if (pending_it->second.empty()) {
        pending_children_by_parent_.erase(pending_it);
    }
}

void PrefixTreeMemoryBlockCache::attachPendingChildrenLocked(Node& node) {
    auto pending_it = pending_children_by_parent_.find(node.cache_key);
    if (pending_it == pending_children_by_parent_.end()) {
        return;
    }
    auto pending_children = std::move(pending_it->second);
    pending_children_by_parent_.erase(pending_it);
    for (const auto child_key : pending_children) {
        auto child_it = nodes_.find(child_key);
        if (child_it == nodes_.end() || !child_it->second.has_parent || child_it->second.parent_key != node.cache_key) {
            continue;
        }
        auto [_, inserted] = node.children.insert(child_key);
        if (inserted) {
            addSubtreeRefsToAncestorsLocked(node.cache_key, child_it->second);
        }
    }
}

void PrefixTreeMemoryBlockCache::touchLocked(Node& node, CacheBlockKind kind) {
    eraseEvictKeyLocked(node, kind);
    auto& state = node.kinds[kindIndex(kind)];
    state.last_access_seq = ++access_seq_;
    insertEvictKeyLocked(node, kind);
}

void PrefixTreeMemoryBlockCache::insertEvictKeyLocked(const Node& node, CacheBlockKind kind) {
    const auto& state = node.kinds[kindIndex(kind)];
    if (!state.has_value || state.detached || state.is_resident || state.in_flight_ref > 0
        || !isKindLeafLocked(node, kind)) {
        return;
    }
    leaf_lru_[kindIndex(kind)].insert(EvictKey{state.last_access_seq, node.cache_key, state.generation});
}

void PrefixTreeMemoryBlockCache::eraseEvictKeyLocked(const Node& node, CacheBlockKind kind) {
    const auto& state = node.kinds[kindIndex(kind)];
    leaf_lru_[kindIndex(kind)].erase(EvictKey{state.last_access_seq, node.cache_key, state.generation});
}

void PrefixTreeMemoryBlockCache::refreshEvictKeyLocked(const Node& node, CacheBlockKind kind) {
    eraseEvictKeyLocked(node, kind);
    insertEvictKeyLocked(node, kind);
}

void PrefixTreeMemoryBlockCache::pruneLocked(CacheKeyType cache_key) {
    auto it = nodes_.find(cache_key);
    while (it != nodes_.end()) {
        bool has_state = false;
        for (const auto& state : it->second.kinds) {
            if (state.has_value) {
                has_state = true;
                break;
            }
        }
        if (!has_state) {
            for (const auto& retired_items : it->second.retired_items) {
                if (!retired_items.empty()) {
                    has_state = true;
                    break;
                }
            }
        }
        if (has_state || !it->second.children.empty()) {
            break;
        }
        const bool has_parent = it->second.has_parent;
        const auto parent_key = it->second.parent_key;
        nodes_.erase(it);
        if (!has_parent) {
            break;
        }
        auto parent_it = nodes_.find(parent_key);
        if (parent_it == nodes_.end()) {
            detachPendingChildLocked(parent_key, cache_key);
            break;
        }
        parent_it->second.children.erase(cache_key);
        cache_key = parent_key;
        it        = parent_it;
    }
}

std::optional<PrefixTreeMemoryBlockCache::CacheItem>
PrefixTreeMemoryBlockCache::toItemLocked(const Node& node, CacheBlockKind kind) const {
    if (!validKind(kind)) {
        return std::nullopt;
    }
    const auto& state = node.kinds[kindIndex(kind)];
    if (!state.has_value) {
        return std::nullopt;
    }
    return CacheItem{
        node.cache_key, kind, state.backing_type, state.block_index, state.disk_slot, state.block_size,
        state.is_resident, state.generation, state.created_time_us, state.slot_valid_mask};
}

bool PrefixTreeMemoryBlockCache::isKindLeafLocked(const Node& node, CacheBlockKind kind) const {
    const auto& state = node.kinds[kindIndex(kind)];
    if (!state.has_value || state.detached) {
        return false;
    }
    return state.subtree_ref_count <= 1;
}

std::optional<PrefixTreeMemoryBlockCache::CacheItem>
PrefixTreeMemoryBlockCache::popStateOnlyFromChainLocked(const CacheKeyType& leaf_key, CacheBackingType backing_type) {
    auto leaf_it = nodes_.find(leaf_key);
    if (leaf_it == nodes_.end()) {
        return std::nullopt;
    }
    std::vector<CacheKeyType> chain;
    CacheKeyType              cur = leaf_key;
    while (true) {
        auto node_it = nodes_.find(cur);
        if (node_it == nodes_.end()) {
            break;
        }
        chain.push_back(cur);
        if (!node_it->second.has_parent) {
            break;
        }
        auto parent_it = nodes_.find(node_it->second.parent_key);
        if (parent_it == nodes_.end() || parent_it->second.children.size() != 1) {
            break;
        }
        cur = parent_it->first;
    }
    if (chain.size() <= 1) {
        return std::nullopt;
    }
    for (size_t idx = 1; idx < chain.size(); ++idx) {
        auto node_it = nodes_.find(chain[idx]);
        if (node_it == nodes_.end()) {
            continue;
        }
        auto& state = node_it->second.kinds[kindIndex(CacheBlockKind::STATE_SWA_KV)];
        if (!state.has_value || state.detached || state.backing_type != backing_type || state.is_resident
            || state.in_flight_ref > 0) {
            continue;
	        }
	        auto item = toItemLocked(node_it->second, CacheBlockKind::STATE_SWA_KV);
	        eraseEvictKeyLocked(node_it->second, CacheBlockKind::STATE_SWA_KV);
	        state.detached = true;
	        decrementAncestorsLocked(item->cache_key, CacheBlockKind::STATE_SWA_KV);
	        const auto descendant_ref_count = state.subtree_ref_count;
	        state                           = KindState{};
	        state.subtree_ref_count         = descendant_ref_count;
        pruneLocked(item->cache_key);
        return item;
    }
    return std::nullopt;
}

std::vector<PrefixTreeMemoryBlockCache::CacheItem>
PrefixTreeMemoryBlockCache::popChainLocked(const CacheKeyType& leaf_key, CacheBackingType backing_type) {
    std::vector<CacheItem> items;
    auto                  leaf_it = nodes_.find(leaf_key);
    if (leaf_it == nodes_.end()) {
        return items;
    }
    std::vector<CacheKeyType> chain;
    CacheKeyType              cur = leaf_key;
    while (true) {
        auto node_it = nodes_.find(cur);
        if (node_it == nodes_.end()) {
            break;
        }
        chain.push_back(cur);
        if (!node_it->second.has_parent) {
            break;
        }
        auto parent_it = nodes_.find(node_it->second.parent_key);
        if (parent_it == nodes_.end() || parent_it->second.children.size() != 1) {
            break;
        }
        cur = parent_it->first;
    }

    bool has_target_state = false;
    for (const auto key : chain) {
        auto node_it = nodes_.find(key);
        if (node_it == nodes_.end()) {
            continue;
        }
        const auto& state = node_it->second.kinds[kindIndex(CacheBlockKind::STATE_SWA_KV)];
        if (state.has_value && !state.detached && state.backing_type == backing_type && !state.is_resident
            && state.in_flight_ref == 0) {
            has_target_state = true;
            break;
        }
    }
    if (!has_target_state) {
        return items;
    }

    for (auto chain_it = chain.begin(); chain_it != chain.end(); ++chain_it) {
        auto node_it = nodes_.find(*chain_it);
        if (node_it == nodes_.end()) {
            continue;
        }
        for (auto kind : {CacheBlockKind::COMPRESSED_KV, CacheBlockKind::STATE_SWA_KV}) {
            auto& state = node_it->second.kinds[kindIndex(kind)];
            if (!state.has_value || state.detached || state.backing_type != backing_type || state.is_resident
                || state.in_flight_ref > 0) {
                continue;
	            }
	            auto item = toItemLocked(node_it->second, kind);
	            if (!item.has_value()) {
	                continue;
	            }
	            eraseEvictKeyLocked(node_it->second, kind);
	            state.detached = true;
	            decrementAncestorsLocked(item->cache_key, kind);
	            const auto descendant_ref_count = state.subtree_ref_count;
	            state                           = KindState{};
	            state.subtree_ref_count         = descendant_ref_count;
            items.push_back(*item);
        }
        pruneLocked(*chain_it);
    }
    return items;
}

}  // namespace rtp_llm
