#include "rtp_llm/cpp/cache/SharedBlockCache.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

void SharedBlockCache::init(const CacheConfig& config, const std::map<std::string, BlockPoolPtr>& group_pools) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto&                 groups = config.groups();
    RTP_LLM_CHECK_WITH_INFO(group_pools.size() == groups.size(),
                            "group_pools size %zu != cache group count %zu",
                            group_pools.size(),
                            groups.size());

    std::map<std::string, BlockPoolPtr> staged_group_pools;
    std::vector<std::string>            staged_group_tags_in_order;
    staged_group_tags_in_order.reserve(groups.size());
    for (const auto& group : groups) {
        const auto& tag = group.tag;
        RTP_LLM_CHECK_WITH_INFO(!tag.empty(), "SharedBlockCache cache group tag must not be empty");
        RTP_LLM_CHECK_WITH_INFO(
            staged_group_pools.emplace(tag, nullptr).second, "duplicate SharedBlockCache config tag=%s", tag.c_str());
        staged_group_tags_in_order.push_back(tag);
    }
    for (const auto& [tag, pool] : group_pools) {
        RTP_LLM_CHECK_WITH_INFO(!tag.empty(), "SharedBlockCache pool tag must not be empty");
        const auto staged_it = staged_group_pools.find(tag);
        RTP_LLM_CHECK_WITH_INFO(
            staged_it != staged_group_pools.end(), "unknown SharedBlockCache pool tag=%s", tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(pool != nullptr, "null SharedBlockCache pool for tag=%s", tag.c_str());
        staged_it->second = pool;
    }
    for (const auto& [tag, pool] : staged_group_pools) {
        RTP_LLM_CHECK_WITH_INFO(pool != nullptr, "missing SharedBlockCache pool for tag=%s", tag.c_str());
    }
    group_pools_         = std::move(staged_group_pools);
    group_tags_in_order_ = std::move(staged_group_tags_in_order);
}

void SharedBlockCache::put(CacheKeyType                               cache_key,
                           const std::map<std::string, BlockIdxType>& blocks_by_group,
                           bool                                       is_resident) {
    BlockDependency dependency;
    put(cache_key, blocks_by_group, {}, is_resident, kDefaultNamespace, dependency);
}

void SharedBlockCache::put(CacheKeyType                               cache_key,
                           const std::map<std::string, BlockIdxType>& blocks_by_group,
                           const std::map<std::string, bool>&         group_matchable,
                           bool                                       is_resident,
                           NamespaceId                                namespace_id,
                           const BlockDependency&                     dependency) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& [tag, block_id] : blocks_by_group) {
        validateTagLocked(tag);
    }
    for (const auto& [tag, matchable] : group_matchable) {
        validateTagLocked(tag);
    }

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
            for (const auto& tag : group_tags_in_order_) {
                const auto block_it = blocks_by_group.find(tag);
                if (block_it == blocks_by_group.end()) {
                    continue;
                }
                const auto block_id = block_it->second;
                if (isNullBlockIdx(block_id)) {
                    continue;
                }
                auto binding_it = existing_item.bindings_by_group.find(tag);
                if (binding_it == existing_item.bindings_by_group.end()) {
                    existing_item.bindings_by_group.emplace(
                        tag,
                        SharedGroupBinding{block_id,
                                           group_matchable.find(tag) == group_matchable.end()
                                               || group_matchable.at(tag),
                                           now_us});
                    updated = true;
                    blockCacheReferenceForGroup(tag, block_id);
                } else if (const auto matchable_it = group_matchable.find(tag); matchable_it != group_matchable.end()
                                                                                && matchable_it->second
                                                                                && !binding_it->second.matchable) {
                    binding_it->second.matchable = true;
                    updated                      = true;
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
    item.is_resident        = is_resident;
    item.created_time_us    = now_us;
    for (const auto& [tag, block_id] : blocks_by_group) {
        if (!isNullBlockIdx(block_id)) {
            item.bindings_by_group.emplace(
                tag,
                SharedGroupBinding{
                    block_id, group_matchable.find(tag) == group_matchable.end() || group_matchable.at(tag), now_us});
        }
    }
    updateItemDependencyLocked(item, namespace_id, dependency);

    lru_cache_.put(cache_key, item);
    ++version_;
    upsertTreeNodeLocked(cache_key, namespace_id, dependency, item.is_resident);
    refreshAllTreeAliasesLocked(cache_key);

    for (const auto& tag : group_tags_in_order_) {
        const auto block_it = blocks_by_group.find(tag);
        if (block_it == blocks_by_group.end()) {
            continue;
        }
        const auto block_id = block_it->second;
        if (!isNullBlockIdx(block_id)) {
            blockCacheReferenceForGroup(tag, block_id);
        }
    }
}

BlockIdxType SharedBlockCache::matchGroup(CacheKeyType cache_key, std::string_view tag) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    validateTagLocked(tag);

    auto [success, item] = lru_cache_.get(cache_key);
    if (!success) {
        return NULL_BLOCK_IDX;
    }
    touchTreeAliasesLocked(cache_key);
    const auto binding_it = item.bindings_by_group.find(std::string(tag));
    if (binding_it == item.bindings_by_group.end() || !binding_it->second.matchable) {
        return NULL_BLOCK_IDX;
    }
    return binding_it->second.pool_block_id;
}

EvictResult SharedBlockCache::selectAndEvict(size_t min_blocks) {
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
                result.evictions.push_back(
                    makeWholeItemEvictionLocked(tree_key.cache_key, removed_item, tree_key.namespace_id));
                selected_blocks += removed_item.bindings_by_group.size();
                removeAllTreeAliasesForCacheKeyLocked(tree_key.cache_key);
            }
        }
        return result;
    }

    std::unordered_set<CacheKeyType> resident_keys;
    for (const auto& [key, item] : lru_cache_.items()) {
        if (item.is_resident) {
            resident_keys.insert(key);
        }
    }

    std::vector<CacheKeyType> lru_keys;
    for (auto it = lru_cache_.items().rbegin(); it != lru_cache_.items().rend(); ++it) {
        const auto  cache_key = it->first;
        const auto& item      = it->second;
        if (item.is_resident || resident_keys.count(cache_key)) {
            continue;
        }
        lru_keys.push_back(cache_key);
    }

    size_t selected_blocks = 0;
    for (const auto cache_key : lru_keys) {
        UnifiedCacheItem removed_item;
        if (!lru_cache_.remove(cache_key, &removed_item)) {
            continue;
        }
        removeAllTreeAliasesForCacheKeyLocked(cache_key);

        result.evictions.push_back(makeWholeItemEvictionLocked(cache_key, removed_item, kDefaultNamespace));
        selected_blocks += removed_item.bindings_by_group.size();
        if (selected_blocks >= min_blocks) {
            break;
        }
    }

    return result;
}

EvictResult SharedBlockCache::selectAndEvictForGroup(std::string_view tag, size_t min_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    if (min_blocks == 0) {
        return {};
    }

    std::lock_guard<std::mutex> lock(mu_);
    EvictResult                 result;
    validateTagLocked(tag);
    if (independent_group_eviction_enabled_ && prefix_tree_enabled_ && isIndependentEvictionGroupLocked(tag)) {
        if (selectIndependentGroupEvictionsLocked(tag, min_blocks, result)) {
            return result;
        }
    }
    if (!result.evictions.empty()) {
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
                const bool chain_has_target = chainHasUsableGroupLocked(chain, tag);
                if (!chain_has_target && !chainHasReachableAncestorGroupLocked(chain, tag)) {
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
                    result.evictions.push_back(
                        makeWholeItemEvictionLocked(tree_key.cache_key, removed_item, tree_key.namespace_id));
                    if (hasUsableGroup(removed_item, tag)) {
                        selected_blocks++;
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
            resident_keys.insert(key);
        }
    }

    std::vector<CacheKeyType> lru_keys;
    for (auto it = lru_cache_.items().rbegin(); it != lru_cache_.items().rend(); ++it) {
        const auto  cache_key = it->first;
        const auto& item      = it->second;
        if (item.is_resident || resident_keys.count(cache_key)) {
            continue;
        }
        lru_keys.push_back(cache_key);
    }

    size_t selected_blocks = 0;
    for (const auto cache_key : lru_keys) {
        UnifiedCacheItem removed_item;
        const auto*      item             = lru_cache_.find(cache_key);
        bool             has_target_group = item && hasUsableGroup(*item, tag);
        if (!has_target_group) {
            continue;
        }
        if (!lru_cache_.remove(cache_key, &removed_item)) {
            continue;
        }
        removeAllTreeAliasesForCacheKeyLocked(cache_key);

        result.evictions.push_back(makeWholeItemEvictionLocked(cache_key, removed_item, kDefaultNamespace));

        if (hasUsableGroup(removed_item, tag)) {
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

    auto evict_result = selectAndEvict(min_blocks);
    if (evict_result.evictions.empty()) {
        return 0;
    }

    size_t freed = 0;
    for (const auto& eviction : evict_result.evictions) {
        for (const auto& tag : group_tags_in_order_) {
            const auto block_it = eviction.blocks_by_group.find(tag);
            if (block_it != eviction.blocks_by_group.end()) {
                blockCacheFreeForGroup(tag, block_it->second);
                freed++;
            }
        }
    }
    return freed;
}

size_t SharedBlockCache::evictAndFreeForGroup(std::string_view tag, size_t min_blocks, EvictResult* evict_result_out) {
    RTP_LLM_PROFILE_FUNCTION();

    auto evict_result = selectAndEvictForGroup(tag, min_blocks);
    validateTagLocked(tag);
    if (evict_result.evictions.empty()) {
        if (evict_result_out) {
            *evict_result_out = std::move(evict_result);
        }
        return 0;
    }

    size_t freed = 0;
    for (const auto& eviction : evict_result.evictions) {
        for (const auto& group_tag : group_tags_in_order_) {
            const auto block_it = eviction.blocks_by_group.find(group_tag);
            if (block_it != eviction.blocks_by_group.end()) {
                blockCacheFreeForGroup(group_tag, block_it->second);
                if (group_tag == tag) {
                    freed++;
                }
            }
        }
    }
    if (evict_result_out) {
        *evict_result_out = std::move(evict_result);
    }
    return freed;
}

std::optional<UnifiedCacheItem> SharedBlockCache::remove(CacheKeyType cache_key) {
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
    independent_eviction_group_tags_.clear();
    for (const auto& tag : tags) {
        validateTagLocked(tag);
        independent_eviction_group_tags_.insert(tag);
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

CacheEviction SharedBlockCache::makeWholeItemEvictionLocked(CacheKeyType            cache_key,
                                                            const UnifiedCacheItem& item,
                                                            NamespaceId             fallback_namespace) const {
    CacheEviction eviction;
    eviction.cache_key            = cache_key;
    eviction.dependency           = item.dependency;
    eviction.dependency_namespace = item.has_dependency ? item.dependency_namespace : fallback_namespace;
    eviction.has_dependency       = item.has_dependency;
    eviction.lifetime_ms          = std::max<int64_t>(0, (currentTimeUs() - item.created_time_us) / 1000);
    for (const auto& [tag, binding] : item.bindings_by_group) {
        eviction.blocks_by_group.emplace(tag, binding.pool_block_id);
    }
    return eviction;
}

void SharedBlockCache::validateTagLocked(std::string_view tag) const {
    RTP_LLM_CHECK_WITH_INFO(!tag.empty(), "SharedBlockCache group tag must not be empty");
    const auto it = group_pools_.find(std::string(tag));
    RTP_LLM_CHECK_WITH_INFO(
        it != group_pools_.end(), "unknown SharedBlockCache group tag=%s", std::string(tag).c_str());
}

void SharedBlockCache::blockCacheReferenceForGroup(std::string_view tag, BlockIdxType block_id) {
    group_pools_.at(std::string(tag))->blockCacheReference(block_id);
}

void SharedBlockCache::blockCacheFreeForGroup(std::string_view tag, BlockIdxType block_id) {
    group_pools_.at(std::string(tag))->blockCacheFree(block_id);
}

bool SharedBlockCache::hasUsableGroup(const UnifiedCacheItem& item, std::string_view tag) {
    return item.bindings_by_group.find(std::string(tag)) != item.bindings_by_group.end();
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

bool SharedBlockCache::chainHasUsableGroupLocked(const std::vector<NamespacedKey>& chain, std::string_view tag) const {
    for (const auto& key : chain) {
        const auto* item = lru_cache_.find(key.cache_key);
        if (item && hasUsableGroup(*item, tag)) {
            return true;
        }
    }
    return false;
}

bool SharedBlockCache::chainHasReachableAncestorGroupLocked(const std::vector<NamespacedKey>& chain,
                                                            std::string_view                  tag) const {
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
        bool        parent_has_target_group = parent_item && hasUsableGroup(*parent_item, tag);
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

bool SharedBlockCache::selectIndependentGroupEvictionsLocked(std::string_view tag,
                                                             size_t           min_blocks,
                                                             EvictResult&     result) {
    if (group_pools_.find(std::string(tag)) == group_pools_.end() || min_blocks == 0) {
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
            if (!success || item.is_resident || !hasUsableGroup(item, tag)) {
                continue;
            }
            removeGroupFromItemLocked(key.cache_key, tag, result);
            ++selected_blocks;
            break;
        }
    }
    return selected_blocks >= min_blocks;
}

void SharedBlockCache::removeGroupFromItemLocked(CacheKeyType cache_key, std::string_view tag, EvictResult& result) {
    UnifiedCacheItem item;
    if (!lru_cache_.remove(cache_key, &item)) {
        return;
    }
    if (!hasUsableGroup(item, tag)) {
        lru_cache_.put(cache_key, item);
        return;
    }

    const std::string group_tag(tag);
    const auto        binding_it = item.bindings_by_group.find(group_tag);
    RTP_LLM_CHECK_WITH_INFO(binding_it != item.bindings_by_group.end(),
                            "missing SharedBlockCache binding for cache_key=%ld tag=%s",
                            cache_key,
                            group_tag.c_str());

    CacheEviction eviction;
    eviction.cache_key            = cache_key;
    eviction.blocks_by_group      = {{group_tag, binding_it->second.pool_block_id}};
    eviction.dependency           = item.dependency;
    eviction.dependency_namespace = item.has_dependency ? item.dependency_namespace : kGpuLogicalNamespace;
    eviction.has_dependency       = item.has_dependency;
    eviction.lifetime_ms          = std::max<int64_t>(0, (currentTimeUs() - binding_it->second.created_time_us) / 1000);
    eviction.kind                 = EvictionKind::IndependentGroup;
    eviction.group_tag            = group_tag;
    result.evictions.push_back(std::move(eviction));

    item.bindings_by_group.erase(binding_it);

    if (!item.bindings_by_group.empty()) {
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

bool SharedBlockCache::isIndependentEvictionGroupLocked(std::string_view tag) const {
    return independent_eviction_group_tags_.find(std::string(tag)) != independent_eviction_group_tags_.end();
}

}  // namespace rtp_llm
