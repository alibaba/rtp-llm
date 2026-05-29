#include "rtp_llm/cpp/cache/connector/memory/PrefixTreeMemoryBlockCache.h"

#include <algorithm>
#include <mutex>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

size_t PrefixTreeMemoryBlockCache::kindIndex(CacheBlockKind kind) {
    RTP_LLM_CHECK_WITH_INFO(validKind(kind), "invalid prefix-tree memory kind %d", static_cast<int>(kind));
    return kind == CacheBlockKind::COMPRESSED_KV ? 0 : 1;
}

bool PrefixTreeMemoryBlockCache::validKind(CacheBlockKind kind) {
    return kind == CacheBlockKind::COMPRESSED_KV || kind == CacheBlockKind::STATE_SWA_KV;
}

bool PrefixTreeMemoryBlockCache::contains(CacheKeyType cache_key, CacheBlockKind kind) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto                                it = nodes_.find(cache_key);
    if (it == nodes_.end() || !validKind(kind)) {
        return false;
    }
    const auto& state = it->second.kinds[kindIndex(kind)];
    return state.has_value && !state.detached;
}

PrefixTreeMemoryBlockCache::MatchResult
PrefixTreeMemoryBlockCache::match(CacheKeyType cache_key, CacheBlockKind kind) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = nodes_.find(cache_key);
    if (it == nodes_.end() || !validKind(kind)) {
        return {};
    }
    auto& state = it->second.kinds[kindIndex(kind)];
    if (!state.has_value || state.detached) {
        return {};
    }
    touchLocked(it->second, kind);
    return {true, state.backing_type, state.block_index, state.disk_slot, state.block_size, state.generation};
}

PrefixTreeMemoryBlockCache::MatchResult
PrefixTreeMemoryBlockCache::matchAndMarkInFlight(CacheKeyType cache_key, CacheBlockKind kind) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = nodes_.find(cache_key);
    if (it == nodes_.end() || !validKind(kind)) {
        return {};
    }
    auto& state = it->second.kinds[kindIndex(kind)];
    if (!state.has_value || state.detached) {
        return {};
    }
    touchLocked(it->second, kind);
    state.in_flight_ref++;
    return {true, state.backing_type, state.block_index, state.disk_slot, state.block_size, state.generation};
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
    if (state.has_value && !state.detached) {
        return {false, std::nullopt};
    }

    std::optional<CacheItem> old_item;
    incrementAncestorsLocked(cache_key, input_item.kind);

    state.has_value       = true;
    state.detached        = false;
    state.backing_type    = input_item.backing_type;
    state.block_index     = input_item.block_index;
    state.disk_slot       = input_item.disk_slot;
    state.block_size      = input_item.block_size;
    state.is_resident     = input_item.is_resident;
    state.generation      = ++generation_seq_;
    state.last_access_seq = ++access_seq_;
    state.in_flight_ref   = 0;
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
    eraseEvictKeyLocked(it->second, kind);
    state.detached = true;
    decrementAncestorsLocked(cache_key, kind);
    if (state.in_flight_ref == 0) {
        state = KindState{};
        pruneLocked(cache_key);
    }
    return item;
}

void PrefixTreeMemoryBlockCache::releaseInFlight(CacheKeyType     cache_key,
                                                 CacheBlockKind   kind,
                                                 CacheBackingType backing_type,
                                                 BlockIdxType     block_index,
                                                 int32_t          disk_slot,
                                                 uint64_t         generation) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = nodes_.find(cache_key);
    if (it == nodes_.end() || !validKind(kind)) {
        return;
    }
    auto& state = it->second.kinds[kindIndex(kind)];
    if (!state.has_value || state.backing_type != backing_type || state.generation != generation) {
        return;
    }
    if (backing_type == CacheBackingType::MEMORY && state.block_index != block_index) {
        return;
    }
    if (backing_type == CacheBackingType::DISK && state.disk_slot != disk_slot) {
        return;
    }
    if (state.in_flight_ref > 0) {
        state.in_flight_ref--;
    }
    if (state.detached && state.in_flight_ref == 0) {
        state = KindState{};
        pruneLocked(cache_key);
    }
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
                old_parent_it->second.children.erase(cache_key);
            }
        }
        it->second.parent_key = dependency.parent_key;
        it->second.has_parent = dependency.has_parent && dependency.parent_key != cache_key;
        it->second.ordinal    = dependency.ordinal;
    }
    if (it->second.has_parent) {
        auto parent_it = nodes_.find(it->second.parent_key);
        if (parent_it != nodes_.end()) {
            parent_it->second.children.insert(cache_key);
        }
    }
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
        if (!it->second.has_parent) {
            break;
        }
        cur = it->second.parent_key;
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
        state.is_resident, state.generation};
}

bool PrefixTreeMemoryBlockCache::isKindLeafLocked(const Node& node, CacheBlockKind kind) const {
    const auto& state = node.kinds[kindIndex(kind)];
    if (!state.has_value || state.detached) {
        return false;
    }
    return state.subtree_ref_count <= 1;
}

}  // namespace rtp_llm
