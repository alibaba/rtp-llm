#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

void SharedBlockCache::init(int group_num, const std::vector<BlockPoolPtr>& group_pools) {
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(static_cast<int>(group_pools.size()) == group_num,
                            "group_pools size %zu != group_num %d",
                            group_pools.size(),
                            group_num);
    group_num_   = group_num;
    group_pools_ = group_pools;
}

void SharedBlockCache::put(CacheKeyType cache_key, const std::vector<BlockIdxType>& group_slots, bool is_resident) {
    BlockDependency dependency;
    put(cache_key, group_slots, is_resident, kDefaultNamespace, dependency);
}

void SharedBlockCache::put(CacheKeyType                     cache_key,
                           const std::vector<BlockIdxType>& group_slots,
                           bool                             is_resident,
                           NamespaceId                      namespace_id,
                           const BlockDependency&           dependency,
                           const std::vector<bool>&         matchable_slots) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    if (lru_cache_.contains(cache_key)) {
        auto [success, existing_item] = lru_cache_.get(cache_key);
        if (success) {
            const bool resident = existing_item.is_resident || is_resident;
            if (resident != existing_item.is_resident) {
                existing_item.is_resident = resident;
            }
            const bool dependency_updated = updateItemDependencyLocked(existing_item, namespace_id, dependency);
            bool updated = false;
            for (size_t gid = 0; gid < group_slots.size(); ++gid) {
                if (isNullBlockIdx(group_slots[gid])) {
                    continue;
                }
                if (gid >= existing_item.slots.size()) {
                    existing_item.slots.resize(gid + 1, NULL_BLOCK_IDX);
                }
                if (gid >= existing_item.matchable_slots.size()) {
                    existing_item.matchable_slots.resize(gid + 1, true);
                }
                if (isNullBlockIdx(existing_item.slots[gid])) {
                    existing_item.slots[gid] = group_slots[gid];
                    existing_item.matchable_slots[gid] =
                        matchable_slots.empty() || gid >= matchable_slots.size() ? true : matchable_slots[gid];
                    updated                  = true;
                    if (static_cast<int>(gid) < group_num_) {
                        group_pools_[gid]->blockCacheReference(group_slots[gid]);
                    }
                } else if (!matchable_slots.empty() && gid < matchable_slots.size() && matchable_slots[gid]
                           && !existing_item.matchable_slots[gid]) {
                    existing_item.matchable_slots[gid] = true;
                    updated                            = true;
                }
            }
            if (updated || existing_item.is_resident || dependency_updated) {
                lru_cache_.put(cache_key, existing_item);
                ++version_;
            }
            if (existing_item.is_resident) {
                markAllTreeAliasesResidentLocked(cache_key);
            }
            upsertTreeNodeLocked(cache_key, namespace_id, dependency, existing_item.is_resident);
        }
        return;
    }

    UnifiedCacheItem item;
    item.cache_key   = cache_key;
    item.is_resident = is_resident;
    item.slots       = group_slots;
    item.matchable_slots.resize(group_slots.size(), true);
    for (size_t gid = 0; gid < group_slots.size() && gid < matchable_slots.size(); ++gid) {
        item.matchable_slots[gid] = matchable_slots[gid];
    }
    updateItemDependencyLocked(item, namespace_id, dependency);

    lru_cache_.put(cache_key, item);
    ++version_;
    upsertTreeNodeLocked(cache_key, namespace_id, dependency, item.is_resident);

    for (int gid = 0; gid < static_cast<int>(group_slots.size()) && gid < group_num_; ++gid) {
        if (!isNullBlockIdx(group_slots[gid])) {
            group_pools_[gid]->blockCacheReference(group_slots[gid]);
        }
    }
}

SharedBlockCache::MatchResult SharedBlockCache::match(CacheKeyType cache_key) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    auto [success, item] = lru_cache_.get(cache_key);
    if (!success) {
        return {false, {}};
    }
    touchTreeAliasesLocked(cache_key);
    return {true, item.slots};
}

BlockIdxType SharedBlockCache::matchGroup(CacheKeyType cache_key, int group_id) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    auto [success, item] = lru_cache_.get(cache_key);
    if (!success) {
        return NULL_BLOCK_IDX;
    }
    touchTreeAliasesLocked(cache_key);
    if (group_id < 0 || static_cast<size_t>(group_id) >= item.slots.size()) {
        return NULL_BLOCK_IDX;
    }
    if (!slotMatchable(item, static_cast<size_t>(group_id))) {
        return NULL_BLOCK_IDX;
    }
    const auto block = item.slots[group_id];
    return block;
}

SharedBlockCache::EvictResult SharedBlockCache::selectAndEvict(size_t min_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    EvictResult result;
    if (lru_cache_.empty() || min_blocks == 0) {
        return result;
    }

    if (prefix_tree_enabled_ && !leaf_lru_.empty()) {
        size_t selected_blocks = 0;
        while (selected_blocks < min_blocks && !leaf_lru_.empty()) {
            const auto leaf     = *leaf_lru_.begin();
            const auto leaf_key = NamespacedKey{leaf.namespace_id, leaf.cache_key};
            auto       chain    = collectEvictChainLocked(leaf_key);
            if (chain.empty()) {
                removeTreeAliasLocked(leaf_key);
                continue;
            }
            std::vector<NamespacedKey> ordered_chain(chain.rbegin(), chain.rend());
            for (const auto& tree_key : ordered_chain) {
                UnifiedCacheItem removed_item;
                if (!lru_cache_.remove(tree_key.cache_key, &removed_item)) {
                    removeAllTreeAliasesForCacheKeyLocked(tree_key.cache_key);
                    continue;
                }
                if (result.evicted_slots.find(tree_key.cache_key) == result.evicted_slots.end()) {
                    result.evicted_keys.push_back(tree_key.cache_key);
                    result.evicted_slots[tree_key.cache_key] = removed_item.slots;
                    result.evicted_namespaces[tree_key.cache_key] =
                        removed_item.has_dependency ? removed_item.dependency_namespace : tree_key.namespace_id;
                    if (removed_item.has_dependency) {
                        result.evicted_dependencies[tree_key.cache_key] = removed_item.dependency;
                    }
                    for (const auto& slot : removed_item.slots) {
                        if (!isNullBlockIdx(slot)) {
                            selected_blocks++;
                        }
                    }
                }
                removeAllTreeAliasesForCacheKeyLocked(tree_key.cache_key);
            }
        }
        return result;
    }

    std::unordered_set<CacheKeyType> resident_keys;
    for (const auto& [key, item] : lru_cache_.items()) {
        if (item.is_resident) {
            resident_keys.insert(item.cache_key);
        }
    }

    std::vector<CacheKeyType> lru_keys;
    for (auto it = lru_cache_.items().rbegin(); it != lru_cache_.items().rend(); ++it) {
        const auto& item = it->second;
        if (item.is_resident || resident_keys.count(item.cache_key)) {
            continue;
        }
        lru_keys.push_back(item.cache_key);
    }

    size_t selected_blocks = 0;
    for (const auto cache_key : lru_keys) {
        UnifiedCacheItem removed_item;
        if (!lru_cache_.remove(cache_key, &removed_item)) {
            continue;
        }
        removeAllTreeAliasesForCacheKeyLocked(cache_key);

        result.evicted_keys.push_back(cache_key);
        result.evicted_slots[cache_key] = removed_item.slots;
        result.evicted_namespaces[cache_key] =
            removed_item.has_dependency ? removed_item.dependency_namespace : kDefaultNamespace;
        if (removed_item.has_dependency) {
            result.evicted_dependencies[cache_key] = removed_item.dependency;
        }

        for (const auto& slot : removed_item.slots) {
            if (!isNullBlockIdx(slot)) {
                selected_blocks++;
            }
        }
        if (selected_blocks >= min_blocks) {
            break;
        }
    }

    return result;
}

size_t SharedBlockCache::evictAndFree(size_t min_blocks) {
    RTP_LLM_PROFILE_FUNCTION();

    auto evict_result = selectAndEvict(min_blocks);
    if (evict_result.evicted_keys.empty()) {
        return 0;
    }

    size_t freed = 0;
    for (size_t i = 0; i < evict_result.evicted_keys.size(); ++i) {
        const auto  cache_key = evict_result.evicted_keys[i];
        const auto& slots     = evict_result.evicted_slots.at(cache_key);

        for (int gid = 0; gid < static_cast<int>(slots.size()) && gid < group_num_; ++gid) {
            if (!isNullBlockIdx(slots[gid])) {
                group_pools_[gid]->blockCacheFree(slots[gid]);
                freed++;
            }
        }
    }
    return freed;
}

std::optional<SharedBlockCache::UnifiedCacheItem> SharedBlockCache::remove(CacheKeyType cache_key) {
    std::lock_guard<std::mutex> lock(mu_);

    UnifiedCacheItem removed_item;
    if (!lru_cache_.remove(cache_key, &removed_item)) {
        return std::nullopt;
    }
    removeAllTreeAliasesForCacheKeyLocked(cache_key);
    return removed_item;
}

bool SharedBlockCache::contains(CacheKeyType cache_key) const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.contains(cache_key);
}

bool SharedBlockCache::empty() const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.empty();
}

size_t SharedBlockCache::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.size();
}

std::vector<CacheKeyType> SharedBlockCache::allCacheKeys() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<CacheKeyType>   keys;
    keys.reserve(lru_cache_.size());
    for (const auto& [key, item] : lru_cache_.items()) {
        keys.push_back(key);
    }
    return keys;
}

int64_t SharedBlockCache::version() const {
    std::lock_guard<std::mutex> lock(mu_);
    return version_;
}

void SharedBlockCache::setPrefixTreeEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mu_);
    prefix_tree_enabled_ = enabled;
}

bool SharedBlockCache::prefixTreeEnabled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return prefix_tree_enabled_;
}

void SharedBlockCache::upsertTreeNodeLocked(CacheKeyType               cache_key,
                                            NamespaceId                namespace_id,
                                            const BlockDependency&     dependency,
                                            bool                       is_resident) {
    if (!prefix_tree_enabled_) {
        return;
    }
    const NamespacedKey key{namespace_id, cache_key};
    const bool          has_parent = dependency.has_parent && dependency.parent_key != cache_key;
    const NamespacedKey parent{namespace_id, dependency.parent_key};
    auto                it = tree_nodes_.find(key);
    if (it == tree_nodes_.end()) {
        PrefixTreeNode node;
        node.key        = key;
        node.parent     = parent;
        node.has_parent = has_parent;
        node.ordinal    = dependency.ordinal;
        node.resident   = is_resident;
        node.last_access_seq = ++tree_access_seq_;
        auto [inserted_it, _] = tree_nodes_.emplace(key, std::move(node));
        it = inserted_it;
        aliases_by_cache_key_[cache_key].insert(key);
    } else {
        eraseLeafLocked(it->second);
        if (it->second.has_parent && (it->second.parent == parent) == false) {
            if (auto parent_it = tree_nodes_.find(it->second.parent); parent_it != tree_nodes_.end()) {
                parent_it->second.children.erase(key);
                refreshLeafLocked(parent_it->first);
            } else {
                detachPendingChildLocked(it->second.parent, key);
            }
        }
        it->second.parent     = parent;
        it->second.has_parent = has_parent;
        it->second.ordinal    = dependency.ordinal;
        it->second.resident   = it->second.resident || is_resident;
        it->second.last_access_seq = ++tree_access_seq_;
    }

    if (has_parent) {
        auto parent_it = tree_nodes_.find(parent);
        if (parent_it != tree_nodes_.end()) {
            eraseLeafLocked(parent_it->second);
            parent_it->second.children.insert(key);
        } else {
            pending_children_by_parent_[parent].insert(key);
        }
    }
    attachPendingChildrenLocked(it->second);
    insertLeafIfEligibleLocked(it->second);
}

void SharedBlockCache::detachPendingChildLocked(const NamespacedKey& parent, const NamespacedKey& child) {
    auto pending_it = pending_children_by_parent_.find(parent);
    if (pending_it == pending_children_by_parent_.end()) {
        return;
    }
    pending_it->second.erase(child);
    if (pending_it->second.empty()) {
        pending_children_by_parent_.erase(pending_it);
    }
}

void SharedBlockCache::attachPendingChildrenLocked(PrefixTreeNode& node) {
    auto pending_it = pending_children_by_parent_.find(node.key);
    if (pending_it == pending_children_by_parent_.end()) {
        return;
    }
    for (const auto& child_key : pending_it->second) {
        auto child_it = tree_nodes_.find(child_key);
        if (child_it != tree_nodes_.end() && child_it->second.has_parent && child_it->second.parent == node.key) {
            eraseLeafLocked(node);
            node.children.insert(child_key);
        }
    }
    pending_children_by_parent_.erase(pending_it);
}

void SharedBlockCache::touchTreeAliasesLocked(CacheKeyType cache_key) {
    if (!prefix_tree_enabled_) {
        return;
    }
    auto aliases_it = aliases_by_cache_key_.find(cache_key);
    if (aliases_it == aliases_by_cache_key_.end()) {
        return;
    }
    std::vector<NamespacedKey> aliases(aliases_it->second.begin(), aliases_it->second.end());
    for (const auto& key : aliases) {
        auto node_it = tree_nodes_.find(key);
        if (node_it != tree_nodes_.end()) {
            touchTreeNodeLocked(node_it->second);
        }
    }
}

void SharedBlockCache::touchTreeNodeLocked(PrefixTreeNode& node) {
    eraseLeafLocked(node);
    node.last_access_seq = ++tree_access_seq_;
    insertLeafIfEligibleLocked(node);
}

void SharedBlockCache::eraseLeafLocked(const PrefixTreeNode& node) {
    leaf_lru_.erase(LeafKey{node.last_access_seq, node.key.namespace_id, node.key.cache_key});
}

void SharedBlockCache::insertLeafIfEligibleLocked(const PrefixTreeNode& node) {
    if (node.resident || !node.children.empty() || !hasFlatItemLocked(node.key.cache_key)
        || isFlatItemResidentLocked(node.key.cache_key)) {
        return;
    }
    leaf_lru_.insert(LeafKey{node.last_access_seq, node.key.namespace_id, node.key.cache_key});
}

void SharedBlockCache::refreshLeafLocked(const NamespacedKey& key) {
    auto it = tree_nodes_.find(key);
    if (it == tree_nodes_.end()) {
        return;
    }
    eraseLeafLocked(it->second);
    insertLeafIfEligibleLocked(it->second);
}

void SharedBlockCache::removeTreeAliasLocked(const NamespacedKey& key) {
    auto it = tree_nodes_.find(key);
    if (it == tree_nodes_.end()) {
        return;
    }
    PrefixTreeNode node = it->second;
    eraseLeafLocked(node);
    if (node.has_parent) {
        auto parent_it = tree_nodes_.find(node.parent);
        if (parent_it != tree_nodes_.end()) {
            parent_it->second.children.erase(key);
            refreshLeafLocked(parent_it->first);
        } else {
            detachPendingChildLocked(node.parent, key);
        }
    }
    for (const auto& child : node.children) {
        auto child_it = tree_nodes_.find(child);
        if (child_it != tree_nodes_.end() && child_it->second.parent == key) {
            child_it->second.has_parent = false;
        }
    }
    auto aliases_it = aliases_by_cache_key_.find(key.cache_key);
    if (aliases_it != aliases_by_cache_key_.end()) {
        aliases_it->second.erase(key);
        if (aliases_it->second.empty()) {
            aliases_by_cache_key_.erase(aliases_it);
        }
    }
    tree_nodes_.erase(it);
}

void SharedBlockCache::removeAllTreeAliasesForCacheKeyLocked(CacheKeyType cache_key) {
    auto aliases_it = aliases_by_cache_key_.find(cache_key);
    if (aliases_it == aliases_by_cache_key_.end()) {
        return;
    }
    std::vector<NamespacedKey> aliases(aliases_it->second.begin(), aliases_it->second.end());
    for (const auto& key : aliases) {
        removeTreeAliasLocked(key);
    }
}

void SharedBlockCache::markAllTreeAliasesResidentLocked(CacheKeyType cache_key) {
    auto aliases_it = aliases_by_cache_key_.find(cache_key);
    if (aliases_it == aliases_by_cache_key_.end()) {
        return;
    }
    for (const auto& key : aliases_it->second) {
        auto node_it = tree_nodes_.find(key);
        if (node_it == tree_nodes_.end() || node_it->second.resident) {
            continue;
        }
        eraseLeafLocked(node_it->second);
        node_it->second.resident = true;
    }
}

bool SharedBlockCache::updateItemDependencyLocked(UnifiedCacheItem&      item,
                                                  NamespaceId            namespace_id,
                                                  const BlockDependency& dependency) const {
    if (item.has_dependency && item.dependency_namespace == kGpuCpCanonicalNamespace
        && namespace_id != kGpuCpCanonicalNamespace) {
        return false;
    }
    if (item.has_dependency && item.dependency_namespace == namespace_id
        && item.dependency.has_parent == dependency.has_parent && item.dependency.parent_key == dependency.parent_key
        && item.dependency.ordinal == dependency.ordinal) {
        return false;
    }
    item.dependency           = dependency;
    item.dependency_namespace = namespace_id;
    item.has_dependency       = true;
    return true;
}

bool SharedBlockCache::slotMatchable(const UnifiedCacheItem& item, size_t group_id) {
    return group_id >= item.matchable_slots.size() || item.matchable_slots[group_id];
}

std::vector<SharedBlockCache::NamespacedKey>
SharedBlockCache::collectEvictChainLocked(const NamespacedKey& leaf_key) const {
    std::vector<NamespacedKey> chain;
    auto                       it = tree_nodes_.find(leaf_key);
    if (it == tree_nodes_.end() || it->second.resident || !it->second.children.empty()
        || !hasFlatItemLocked(it->second.key.cache_key) || isFlatItemResidentLocked(it->second.key.cache_key)) {
        return chain;
    }

    NamespacedKey cur = leaf_key;
    while (true) {
        auto node_it = tree_nodes_.find(cur);
        if (node_it == tree_nodes_.end() || node_it->second.resident || !hasFlatItemLocked(cur.cache_key)
            || isFlatItemResidentLocked(cur.cache_key)) {
            break;
        }
        chain.push_back(cur);
        if (!node_it->second.has_parent) {
            break;
        }
        auto parent_it = tree_nodes_.find(node_it->second.parent);
        if (parent_it == tree_nodes_.end() || parent_it->second.resident
            || isFlatItemResidentLocked(parent_it->first.cache_key)) {
            break;
        }
        if (parent_it->second.children.size() != 1) {
            break;
        }
        cur = parent_it->first;
    }
    return chain;
}

bool SharedBlockCache::hasFlatItemLocked(CacheKeyType cache_key) const {
    return lru_cache_.contains(cache_key);
}

bool SharedBlockCache::isFlatItemResidentLocked(CacheKeyType cache_key) const {
    for (const auto& [key, item] : lru_cache_.items()) {
        if (key == cache_key) {
            return item.is_resident;
        }
    }
    return false;
}

}  // namespace rtp_llm
