#include "rtp_llm/cpp/cache/SharedBlockCache.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

void SharedBlockCache::init(const TaggedBlockPools& group_pools) {
    std::lock_guard<std::mutex> lock(mu_);
    group_tags_.clear();
    tag_to_group_index_.clear();
    group_pools_.clear();
    group_tags_.reserve(group_pools.size());
    tag_to_group_index_.reserve(group_pools.size());
    group_pools_.reserve(group_pools.size());
    for (const auto& [tag, pool] : group_pools) {
        const size_t group_index = group_tags_.size();
        group_tags_.push_back(tag);
        const auto [it, inserted] = tag_to_group_index_.emplace(tag, group_index);
        RTP_LLM_CHECK_WITH_INFO(inserted, "SharedBlockCache duplicate tag=%s", tag.c_str());
        (void)it;
        group_pools_.push_back(pool);
    }
}

void SharedBlockCache::put(CacheKeyType cache_key, const TaggedBlockIds& group_block_ids, bool is_resident) {
    BlockDependency dependency;
    put(cache_key, group_block_ids, is_resident, kDefaultNamespace, dependency);
}

void SharedBlockCache::put(CacheKeyType           cache_key,
                           const TaggedBlockIds&  group_block_ids,
                           bool                   is_resident,
                           NamespaceId            namespace_id,
                           const BlockDependency& dependency,
                           const TaggedMatches&   matchable_groups) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& [tag, block_id] : group_block_ids) {
        (void)block_id;
        ensureGroupIndexLocked(tag);
    }
    IndexedBlockIds indexed_block_ids(group_tags_.size(), NULL_BLOCK_IDX);
    IndexedMatches  indexed_matches(group_tags_.size(), 0);
    for (const auto& [tag, block_id] : group_block_ids) {
        const size_t group_index       = groupIndex(tag);
        indexed_block_ids[group_index] = block_id;
        const auto match_it            = matchable_groups.find(tag);
        indexed_matches[group_index]   = match_it == matchable_groups.end() || match_it->second;
    }
    putIndexedLocked(cache_key, indexed_block_ids, is_resident, namespace_id, dependency, indexed_matches);
}

void SharedBlockCache::putIndexed(CacheKeyType           cache_key,
                                  const IndexedBlockIds& group_block_ids,
                                  bool                   is_resident,
                                  NamespaceId            namespace_id,
                                  const BlockDependency& dependency,
                                  const IndexedMatches&  matchable_groups) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(group_block_ids.size() == group_tags_.size(),
                            "SharedBlockCache indexed block size=%zu differs from group size=%zu",
                            group_block_ids.size(),
                            group_tags_.size());
    RTP_LLM_CHECK_WITH_INFO(matchable_groups.empty() || matchable_groups.size() == group_tags_.size(),
                            "SharedBlockCache indexed match size=%zu differs from group size=%zu",
                            matchable_groups.size(),
                            group_tags_.size());
    putIndexedLocked(cache_key, group_block_ids, is_resident, namespace_id, dependency, matchable_groups);
}

void SharedBlockCache::putIndexedLocked(CacheKeyType           cache_key,
                                        const IndexedBlockIds& group_block_ids,
                                        bool                   is_resident,
                                        NamespaceId            namespace_id,
                                        const BlockDependency& dependency,
                                        const IndexedMatches&  matchable_groups) {

    if (lru_cache_.contains(cache_key)) {
        auto [success, existing_item] = lru_cache_.get(cache_key);
        if (success) {
            const auto now_us   = currentTimeUs();
            const bool resident = existing_item.is_resident || is_resident;
            if (resident != existing_item.is_resident) {
                existing_item.is_resident = resident;
            }
            const bool dependency_updated = updateItemDependencyLocked(existing_item, namespace_id, dependency);
            bool       updated            = false;
            for (size_t gid = 0; gid < group_block_ids.size(); ++gid) {
                if (isNullBlockIdx(group_block_ids[gid])) {
                    continue;
                }
                if (gid >= existing_item.group_block_ids.size()) {
                    existing_item.group_block_ids.resize(gid + 1, NULL_BLOCK_IDX);
                }
                if (gid >= existing_item.matchable_groups.size()) {
                    existing_item.matchable_groups.resize(gid + 1, true);
                }
                if (gid >= existing_item.group_block_created_time_us.size()) {
                    existing_item.group_block_created_time_us.resize(gid + 1, 0);
                }
                if (isNullBlockIdx(existing_item.group_block_ids[gid])) {
                    existing_item.group_block_ids[gid]             = group_block_ids[gid];
                    existing_item.group_block_created_time_us[gid] = now_us;
                    existing_item.matchable_groups[gid] =
                        matchable_groups.empty() || gid >= matchable_groups.size() ? true : matchable_groups[gid];
                    updated = true;
                    if (gid < group_pools_.size() && group_pools_[gid]) {
                        group_pools_[gid]->blockCacheReference(group_block_ids[gid]);
                    }
                } else if (!matchable_groups.empty() && gid < matchable_groups.size() && matchable_groups[gid]
                           && !existing_item.matchable_groups[gid]) {
                    existing_item.matchable_groups[gid] = true;
                    updated                             = true;
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
            refreshAllTreeAliasesLocked(cache_key);
        }
        return;
    }

    UnifiedCacheItem item;
    const auto       now_us = currentTimeUs();
    item.cache_key          = cache_key;
    item.is_resident        = is_resident;
    item.group_block_ids    = group_block_ids;
    item.created_time_us    = now_us;
    item.matchable_groups.resize(group_block_ids.size(), true);
    item.group_block_created_time_us.resize(group_block_ids.size(), 0);
    for (size_t gid = 0; gid < group_block_ids.size() && gid < matchable_groups.size(); ++gid) {
        item.matchable_groups[gid] = matchable_groups[gid];
    }
    for (size_t gid = 0; gid < group_block_ids.size(); ++gid) {
        if (!isNullBlockIdx(group_block_ids[gid])) {
            item.group_block_created_time_us[gid] = now_us;
        }
    }
    updateItemDependencyLocked(item, namespace_id, dependency);

    lru_cache_.put(cache_key, item);
    ++version_;
    upsertTreeNodeLocked(cache_key, namespace_id, dependency, item.is_resident);
    refreshAllTreeAliasesLocked(cache_key);

    for (int gid = 0; gid < static_cast<int>(group_block_ids.size()) && gid < static_cast<int>(group_pools_.size());
         ++gid) {
        if (!isNullBlockIdx(group_block_ids[gid]) && group_pools_[gid]) {
            group_pools_[gid]->blockCacheReference(group_block_ids[gid]);
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
    return {true, taggedBlockIds(item.group_block_ids)};
}

BlockIdxType SharedBlockCache::matchGroup(CacheKeyType cache_key, std::string_view tag) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    const auto                  group_it = tag_to_group_index_.find(std::string(tag));
    if (group_it == tag_to_group_index_.end()) {
        return NULL_BLOCK_IDX;
    }
    const size_t group_index = group_it->second;

    auto [success, item] = lru_cache_.get(cache_key);
    if (!success) {
        return NULL_BLOCK_IDX;
    }
    touchTreeAliasesLocked(cache_key);
    if (group_index >= item.group_block_ids.size()) {
        return NULL_BLOCK_IDX;
    }
    if (!groupMatchable(item, group_index)) {
        return NULL_BLOCK_IDX;
    }
    const auto block = item.group_block_ids[group_index];
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
                if (result.evicted_group_block_ids.find(tree_key.cache_key) == result.evicted_group_block_ids.end()) {
                    result.evicted_keys.push_back(tree_key.cache_key);
                    result.evicted_group_block_ids[tree_key.cache_key] = taggedBlockIds(removed_item.group_block_ids);
                    result.evicted_lifetime_ms[tree_key.cache_key] =
                        std::max<int64_t>(0, (currentTimeUs() - removed_item.created_time_us) / 1000);
                    result.evicted_namespaces[tree_key.cache_key] =
                        removed_item.has_dependency ? removed_item.dependency_namespace : tree_key.namespace_id;
                    if (removed_item.has_dependency) {
                        result.evicted_dependencies[tree_key.cache_key] = removed_item.dependency;
                    }
                    for (const auto& block_id : removed_item.group_block_ids) {
                        if (!isNullBlockIdx(block_id)) {
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
        result.evicted_group_block_ids[cache_key] = taggedBlockIds(removed_item.group_block_ids);
        result.evicted_lifetime_ms[cache_key] =
            std::max<int64_t>(0, (currentTimeUs() - removed_item.created_time_us) / 1000);
        result.evicted_namespaces[cache_key] =
            removed_item.has_dependency ? removed_item.dependency_namespace : kDefaultNamespace;
        if (removed_item.has_dependency) {
            result.evicted_dependencies[cache_key] = removed_item.dependency;
        }

        for (const auto& block_id : removed_item.group_block_ids) {
            if (!isNullBlockIdx(block_id)) {
                selected_blocks++;
            }
        }
        if (selected_blocks >= min_blocks) {
            break;
        }
    }

    return result;
}

SharedBlockCache::EvictResult SharedBlockCache::selectAndEvictForGroup(std::string_view tag, size_t min_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    if (min_blocks == 0) {
        return {};
    }

    std::lock_guard<std::mutex> lock(mu_);
    const auto                  group_it = tag_to_group_index_.find(std::string(tag));
    if (group_it == tag_to_group_index_.end()) {
        return {};
    }
    const size_t group_index = group_it->second;
    EvictResult  result;
    if (independent_group_eviction_enabled_ && prefix_tree_enabled_ && isIndependentEvictionGroupLocked(group_index)) {
        if (selectIndependentGroupEvictionsLocked(group_index, min_blocks, result)) {
            return result;
        }
    }
    if (!result.evicted_keys.empty()) {
        return result;
    }

    if (lru_cache_.empty()) {
        return result;
    }
    if (prefix_tree_enabled_ && !leaf_lru_.empty()) {
        size_t selected_blocks = 0;
        bool   made_progress   = true;
        while (selected_blocks < min_blocks && made_progress && !leaf_lru_.empty()) {
            made_progress = false;
            std::vector<LeafKey> leaves(leaf_lru_.begin(), leaf_lru_.end());
            for (const auto& leaf : leaves) {
                if (selected_blocks >= min_blocks) {
                    break;
                }
                const auto leaf_key = NamespacedKey{leaf.namespace_id, leaf.cache_key};
                auto       chain    = collectEvictChainLocked(leaf_key);
                if (chain.empty()) {
                    removeTreeAliasLocked(leaf_key);
                    made_progress = true;
                    continue;
                }
                const bool chain_has_target = chainHasUsableGroupLocked(chain, group_index);
                if (!chain_has_target && !chainHasReachableAncestorGroupLocked(chain, group_index)) {
                    continue;
                }
                std::vector<NamespacedKey> ordered_chain(chain.rbegin(), chain.rend());
                for (const auto& tree_key : ordered_chain) {
                    UnifiedCacheItem removed_item;
                    if (!lru_cache_.remove(tree_key.cache_key, &removed_item)) {
                        removeAllTreeAliasesForCacheKeyLocked(tree_key.cache_key);
                        continue;
                    }
                    made_progress = true;
                    if (result.evicted_group_block_ids.find(tree_key.cache_key)
                        == result.evicted_group_block_ids.end()) {
                        result.evicted_keys.push_back(tree_key.cache_key);
                        result.evicted_group_block_ids[tree_key.cache_key] =
                            taggedBlockIds(removed_item.group_block_ids);
                        result.evicted_lifetime_ms[tree_key.cache_key] =
                            std::max<int64_t>(0, (currentTimeUs() - removed_item.created_time_us) / 1000);
                        result.evicted_namespaces[tree_key.cache_key] =
                            removed_item.has_dependency ? removed_item.dependency_namespace : tree_key.namespace_id;
                        if (removed_item.has_dependency) {
                            result.evicted_dependencies[tree_key.cache_key] = removed_item.dependency;
                        }
                        if (hasUsableGroup(removed_item, group_index)) {
                            selected_blocks++;
                        }
                    }
                    removeAllTreeAliasesForCacheKeyLocked(tree_key.cache_key);
                }
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
        const auto*      item             = lru_cache_.find(cache_key);
        bool             has_target_group = item && hasUsableGroup(*item, group_index);
        if (!has_target_group) {
            continue;
        }
        if (!lru_cache_.remove(cache_key, &removed_item)) {
            continue;
        }
        removeAllTreeAliasesForCacheKeyLocked(cache_key);

        result.evicted_keys.push_back(cache_key);
        result.evicted_group_block_ids[cache_key] = taggedBlockIds(removed_item.group_block_ids);
        result.evicted_lifetime_ms[cache_key] =
            std::max<int64_t>(0, (currentTimeUs() - removed_item.created_time_us) / 1000);
        result.evicted_namespaces[cache_key] =
            removed_item.has_dependency ? removed_item.dependency_namespace : kDefaultNamespace;
        if (removed_item.has_dependency) {
            result.evicted_dependencies[cache_key] = removed_item.dependency;
        }

        if (hasUsableGroup(removed_item, group_index)) {
            selected_blocks++;
        }
        if (selected_blocks >= min_blocks) {
            break;
        }
    }

    return result;
}

size_t SharedBlockCache::evictAndFree(size_t min_blocks) {
    RTP_LLM_PROFILE_FUNCTION();

    auto   evict_result = selectAndEvict(min_blocks);
    size_t freed        = 0;
    for (const auto cache_key : evict_result.evicted_keys) {
        const auto& group_block_ids = evict_result.evicted_group_block_ids.at(cache_key);
        for (const auto& [tag, block_id] : group_block_ids) {
            const auto pool = groupPool(tag);
            if (!isNullBlockIdx(block_id) && pool) {
                pool->blockCacheFree(block_id);
                ++freed;
            }
        }
    }
    return freed;
}

size_t SharedBlockCache::evictAndFreeForGroup(std::string_view tag, size_t min_blocks, EvictResult* evict_result_out) {
    RTP_LLM_PROFILE_FUNCTION();

    auto   evict_result = selectAndEvictForGroup(tag, min_blocks);
    size_t freed        = 0;
    for (const auto cache_key : evict_result.evicted_keys) {
        const auto& group_block_ids = evict_result.evicted_group_block_ids.at(cache_key);
        for (const auto& [group_tag, block_id] : group_block_ids) {
            const auto pool = groupPool(group_tag);
            if (!isNullBlockIdx(block_id) && pool) {
                pool->blockCacheFree(block_id);
                if (group_tag == tag) {
                    ++freed;
                }
            }
        }
    }
    if (evict_result_out) {
        *evict_result_out = std::move(evict_result);
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

void SharedBlockCache::setIndependentGroupEviction(bool enabled, const std::vector<std::string>& tags) {
    std::lock_guard<std::mutex> lock(mu_);
    independent_group_eviction_enabled_ = enabled;
    independent_eviction_group_indices_.clear();
    for (const auto& tag : tags) {
        independent_eviction_group_indices_.insert(ensureGroupIndexLocked(tag));
    }
}

void SharedBlockCache::upsertTreeNodeLocked(CacheKeyType           cache_key,
                                            NamespaceId            namespace_id,
                                            const BlockDependency& dependency,
                                            bool                   is_resident) {
    if (!prefix_tree_enabled_) {
        return;
    }
    const NamespacedKey key{namespace_id, cache_key};
    const bool          has_parent = dependency.has_parent && dependency.parent_key != cache_key;
    const NamespacedKey parent{namespace_id, dependency.parent_key};
    auto                it = tree_nodes_.find(key);
    if (it == tree_nodes_.end()) {
        PrefixTreeNode node;
        node.key              = key;
        node.parent           = parent;
        node.has_parent       = has_parent;
        node.ordinal          = dependency.ordinal;
        node.resident         = is_resident;
        node.last_access_seq  = ++tree_access_seq_;
        auto [inserted_it, _] = tree_nodes_.emplace(key, std::move(node));
        it                    = inserted_it;
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
        it->second.parent          = parent;
        it->second.has_parent      = has_parent;
        it->second.ordinal         = dependency.ordinal;
        it->second.resident        = it->second.resident || is_resident;
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
    if (node.key.namespace_id != kGpuCpCanonicalNamespace && flatItemHasCanonicalDependencyLocked(node.key.cache_key)) {
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

void SharedBlockCache::refreshAllTreeAliasesLocked(CacheKeyType cache_key) {
    auto aliases_it = aliases_by_cache_key_.find(cache_key);
    if (aliases_it == aliases_by_cache_key_.end()) {
        return;
    }
    std::vector<NamespacedKey> aliases(aliases_it->second.begin(), aliases_it->second.end());
    for (const auto& key : aliases) {
        refreshLeafLocked(key);
    }
}

bool SharedBlockCache::flatItemHasCanonicalDependencyLocked(CacheKeyType cache_key) const {
    const auto* item = lru_cache_.find(cache_key);
    return item && item->has_dependency && item->dependency_namespace == kGpuCpCanonicalNamespace;
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

bool SharedBlockCache::groupMatchable(const UnifiedCacheItem& item, size_t group_index) {
    return group_index >= item.matchable_groups.size() || item.matchable_groups[group_index];
}

bool SharedBlockCache::hasUsableGroup(const UnifiedCacheItem& item, size_t group_index) {
    return group_index < item.group_block_ids.size() && !isNullBlockIdx(item.group_block_ids[group_index]);
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

bool SharedBlockCache::chainHasUsableGroupLocked(const std::vector<NamespacedKey>& chain, size_t group_index) const {
    for (const auto& key : chain) {
        const auto* item = lru_cache_.find(key.cache_key);
        if (item && hasUsableGroup(*item, group_index)) {
            return true;
        }
    }
    return false;
}

bool SharedBlockCache::chainHasReachableAncestorGroupLocked(const std::vector<NamespacedKey>& chain,
                                                            size_t                            group_index) const {
    if (chain.empty()) {
        return false;
    }
    auto node_it = tree_nodes_.find(chain.back());
    while (node_it != tree_nodes_.end() && node_it->second.has_parent) {
        auto parent_it = tree_nodes_.find(node_it->second.parent);
        if (parent_it == tree_nodes_.end() || parent_it->second.resident
            || !hasFlatItemLocked(parent_it->first.cache_key) || isFlatItemResidentLocked(parent_it->first.cache_key)) {
            return false;
        }
        const auto* parent_item             = lru_cache_.find(parent_it->first.cache_key);
        bool        parent_has_target_group = parent_item && hasUsableGroup(*parent_item, group_index);
        if (parent_has_target_group) {
            bool all_children_evictable = true;
            for (const auto& child : parent_it->second.children) {
                if (!subtreeEvictableForAncestorGroupLocked(child)) {
                    all_children_evictable = false;
                    break;
                }
            }
            if (all_children_evictable) {
                return true;
            }
        }
        node_it = parent_it;
    }
    return false;
}

bool SharedBlockCache::subtreeEvictableForAncestorGroupLocked(const NamespacedKey& key) const {
    auto node_it = tree_nodes_.find(key);
    if (node_it == tree_nodes_.end() || node_it->second.resident || !hasFlatItemLocked(key.cache_key)
        || isFlatItemResidentLocked(key.cache_key)) {
        return false;
    }
    for (const auto& child : node_it->second.children) {
        if (!subtreeEvictableForAncestorGroupLocked(child)) {
            return false;
        }
    }
    return true;
}

bool SharedBlockCache::selectIndependentGroupEvictionsLocked(size_t       group_index,
                                                             size_t       min_blocks,
                                                             EvictResult& result) {
    if (group_index >= group_pools_.size() || min_blocks == 0) {
        return false;
    }
    size_t               selected_blocks = 0;
    std::vector<LeafKey> leaves(leaf_lru_.begin(), leaf_lru_.end());
    for (const auto& leaf : leaves) {
        if (selected_blocks >= min_blocks) {
            break;
        }
        const auto leaf_key = NamespacedKey{leaf.namespace_id, leaf.cache_key};
        auto       chain    = collectEvictChainLocked(leaf_key);
        if (chain.size() <= 1) {
            continue;
        }
        for (size_t chain_idx = 1; chain_idx < chain.size(); ++chain_idx) {
            const auto& key      = chain[chain_idx];
            auto [success, item] = lru_cache_.get(key.cache_key);
            if (!success || item.is_resident || group_index >= item.group_block_ids.size()
                || isNullBlockIdx(item.group_block_ids[group_index])) {
                continue;
            }
            removeGroupFromItemLocked(key.cache_key, group_index, result);
            ++selected_blocks;
            break;
        }
    }
    return selected_blocks >= min_blocks;
}

void SharedBlockCache::removeGroupFromItemLocked(CacheKeyType cache_key, size_t group_index, EvictResult& result) {
    UnifiedCacheItem item;
    if (!lru_cache_.remove(cache_key, &item)) {
        return;
    }
    if (group_index >= item.group_block_ids.size() || isNullBlockIdx(item.group_block_ids[group_index])) {
        lru_cache_.put(cache_key, item);
        return;
    }

    TaggedBlockIds evicted_group_block_ids{{group_tags_.at(group_index), item.group_block_ids[group_index]}};
    result.evicted_keys.push_back(cache_key);
    result.evicted_group_block_ids[cache_key] = std::move(evicted_group_block_ids);
    result.evicted_namespaces[cache_key] =
        item.has_dependency ? item.dependency_namespace : SharedBlockCache::kGpuLogicalNamespace;
    if (item.has_dependency) {
        result.evicted_dependencies[cache_key] = item.dependency;
    }
    const int64_t created_time_us               = group_index < item.group_block_created_time_us.size() ?
                                                      item.group_block_created_time_us[group_index] :
                                                      item.created_time_us;
    result.evicted_lifetime_ms[cache_key]       = std::max<int64_t>(0, (currentTimeUs() - created_time_us) / 1000);
    result.evicted_independent_group[cache_key] = group_tags_.at(group_index);

    item.group_block_ids[group_index] = NULL_BLOCK_IDX;
    if (group_index < item.matchable_groups.size()) {
        item.matchable_groups[group_index] = false;
    }
    if (group_index < item.group_block_created_time_us.size()) {
        item.group_block_created_time_us[group_index] = 0;
    }

    const bool has_any_group = std::any_of(item.group_block_ids.begin(),
                                           item.group_block_ids.end(),
                                           [](BlockIdxType block_id) { return !isNullBlockIdx(block_id); });
    if (has_any_group) {
        lru_cache_.put(cache_key, item);
        refreshAllTreeAliasesLocked(cache_key);
    } else {
        removeAllTreeAliasesForCacheKeyLocked(cache_key);
    }
    ++version_;
}

bool SharedBlockCache::hasFlatItemLocked(CacheKeyType cache_key) const {
    return lru_cache_.contains(cache_key);
}

bool SharedBlockCache::isFlatItemResidentLocked(CacheKeyType cache_key) const {
    const auto* item = lru_cache_.find(cache_key);
    return item && item->is_resident;
}

bool SharedBlockCache::isIndependentEvictionGroupLocked(size_t group_index) const {
    return independent_eviction_group_indices_.find(group_index) != independent_eviction_group_indices_.end();
}

size_t SharedBlockCache::ensureGroupIndexLocked(std::string_view tag) {
    const auto value = std::string(tag);
    const auto it    = tag_to_group_index_.find(value);
    if (it != tag_to_group_index_.end()) {
        return it->second;
    }
    const size_t group_index = group_tags_.size();
    group_tags_.push_back(value);
    tag_to_group_index_.emplace(value, group_index);
    group_pools_.push_back(nullptr);
    return group_index;
}

size_t SharedBlockCache::groupIndex(std::string_view tag) const {
    const auto value = std::string(tag);
    const auto it    = tag_to_group_index_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != tag_to_group_index_.end(), "SharedBlockCache missing tag=%s", value.c_str());
    return it->second;
}

BlockPoolPtr SharedBlockCache::groupPool(std::string_view tag) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto                  it = tag_to_group_index_.find(std::string(tag));
    if (it == tag_to_group_index_.end() || it->second >= group_pools_.size()) {
        return nullptr;
    }
    return group_pools_[it->second];
}

SharedBlockCache::TaggedBlockIds SharedBlockCache::taggedBlockIds(const IndexedBlockIds& block_ids) const {
    TaggedBlockIds result;
    const size_t   group_num = std::min(group_tags_.size(), block_ids.size());
    for (size_t group_index = 0; group_index < group_num; ++group_index) {
        if (!isNullBlockIdx(block_ids[group_index])) {
            result.emplace(group_tags_[group_index], block_ids[group_index]);
        }
    }
    return result;
}

}  // namespace rtp_llm
