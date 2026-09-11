#include "rtp_llm/cpp/cache/connector/memory/PrefixTreeMemoryBlockCache.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <set>

#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {
bool sameDependency(const BlockDependency& left, const BlockDependency& right) {
    return left.has_parent == right.has_parent && left.ordinal == right.ordinal
           && (!left.has_parent || left.parent_key == right.parent_key);
}
bool validJointBacking(const PrefixTreeMemoryBlockCache::CacheItem& item, CacheBlockKind kind) {
    const auto&  mask           = item.slot_valid_mask;
    const size_t required_slots = kind == CacheBlockKind::COMPRESSED_KV ? 8 : 46;
    return item.kind == kind && item.backing_type == CacheBackingType::MEMORY && item.block_index > 0
           && item.disk_slot == -1 && item.block_size > 0 && mask.size() == 54
           && std::all_of(mask.begin(), mask.end(), [](uint8_t valid) { return valid <= 1; })
           && static_cast<size_t>(std::count(mask.begin(), mask.end(), 1)) == required_slots;
}
}  // namespace

PrefixTreeMemoryBlockCache::DSV41Commit PrefixTreeMemoryBlockCache::putDsv41Committed(
    const DSV41CacheIdentity&                                        identity,
    const std::vector<DSV41Entry>&                                   entries,
    const std::function<bool(const DSV41Entry&, const DSV41Entry&)>& same_bytes) {
    identity.validate();
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (entries.empty())
        return {};
    std::set<CacheKeyType> seen;
    std::optional<size_t>  reuse_unit;
    for (size_t index = 0; index < entries.size(); ++index) {
        const auto& entry      = entries[index];
        const auto& dependency = entry.dependency;
        if (!validJointBacking(entry.global, CacheBlockKind::COMPRESSED_KV)
            || !seen.insert(entry.global.cache_key).second || dependency.ordinal != index
            || dependency.has_parent != (index > 0)
            || (index > 0 && dependency.parent_key != entries[index - 1].global.cache_key)
            || entry.swa.has_value() != entry.checkpoint.has_value()
            || entry.global.slot_valid_mask != entries.front().global.slot_valid_mask)
            return {};
        if (entry.swa
            && (!validJointBacking(*entry.swa, CacheBlockKind::STATE_SWA_KV)
                || entry.swa->cache_key != entry.global.cache_key || !(entry.checkpoint->identity == identity)))
            return {};
        if (entry.checkpoint) {
            const auto end = entry.checkpoint->materialized_end;
            if (end <= 0 || end % (index + 1) != 0)
                return {};
            const size_t unit = end / (index + 1);
            if (reuse_unit && *reuse_unit != unit)
                return {};
            reuse_unit = unit;
            for (size_t slot = 0; slot < entry.global.slot_valid_mask.size(); ++slot) {
                if (entry.global.slot_valid_mask[slot] + entry.swa->slot_valid_mask[slot] != 1)
                    return {};
            }
            try {
                entry.checkpoint->validate(unit);
            } catch (const std::exception&) {
                return {};
            }
        }
        const auto previous = dsv41_nodes_.find({identity, entry.global.cache_key});
        if (previous != dsv41_nodes_.end()) {
            const auto& stored = previous->second.entry;
            if (!sameDependency(stored.dependency, dependency) || !same_bytes || !same_bytes(stored, entry)
                || (stored.checkpoint && entry.checkpoint && !(*stored.checkpoint == *entry.checkpoint)))
                return {};
        }
    }
    if (!entries.back().checkpoint)
        return {};

    DSV41Commit result;
    result.success = true;
    result.retained.reserve(entries.size() * 2);
    const uint64_t                generation = ++generation_seq_;
    std::map<DSV41Key, DSV41Node> prepared;
    for (const auto& entry : entries) {
        const DSV41Key key{identity, entry.global.cache_key};
        const auto     existing = dsv41_nodes_.find(key);
        const bool     inserted = existing == dsv41_nodes_.end();
        auto&          node     = prepared[key];
        if (!inserted)
            node = existing->second;
        if (inserted) {
            node.entry                        = entry;
            node.root_key                     = entries.front().global.cache_key;
            node.generation                   = generation;
            node.entry.global.generation      = generation;
            node.entry.global.created_time_us = currentTimeUs();
            result.retained.push_back(node.entry.global);
            if (node.entry.swa) {
                node.entry.swa->generation      = generation;
                node.entry.swa->created_time_us = node.entry.global.created_time_us;
                result.retained.push_back(*node.entry.swa);
            }
        } else if (entry.swa && !node.entry.swa) {
            node.entry.swa                  = entry.swa;
            node.entry.swa->generation      = node.generation;
            node.entry.swa->created_time_us = currentTimeUs();
            node.entry.checkpoint           = entry.checkpoint;
            result.retained.push_back(*node.entry.swa);
        }
        node.last_access_seq = ++access_seq_;
    }
    // All allocation and validation above completes before any node is visible.
    for (auto& [key, node] : prepared) {
        const auto existing = dsv41_nodes_.find(key);
        if (existing != dsv41_nodes_.end())
            std::swap(existing->second, node);
    }
    dsv41_nodes_.merge(prepared);
    return result;
}

PrefixTreeMemoryBlockCache::DSV41Match
PrefixTreeMemoryBlockCache::matchDsv41AndMarkInFlight(const DSV41CacheIdentity&    identity,
                                                      const CacheKeysType&         keys,
                                                      const BlockDependenciesType& dependencies,
                                                      size_t                       limit) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    DSV41Match                          result;
    limit = std::min({limit, keys.size(), dependencies.size()});
    for (size_t index = 0; index < limit; ++index) {
        const auto it = dsv41_nodes_.find({identity, keys[index]});
        if (it == dsv41_nodes_.end() || !sameDependency(it->second.entry.dependency, dependencies[index]))
            break;
        const auto& entry = it->second.entry;
        result.chain.push_back(entry);
        if (entry.swa && entry.checkpoint)
            result.matched_blocks = index + 1;
    }
    result.chain.resize(result.matched_blocks);
    for (const auto& entry : result.chain) {
        auto& node = dsv41_nodes_.at({identity, entry.global.cache_key});
        ++node.in_flight_ref;
        node.last_access_seq = ++access_seq_;
    }
    return result;
}

void PrefixTreeMemoryBlockCache::releaseDsv41InFlight(const DSV41CacheIdentity& identity, const DSV41Match& match) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (const auto& entry : match.chain) {
        const auto it = dsv41_nodes_.find({identity, entry.global.cache_key});
        if (it == dsv41_nodes_.end() || it->second.generation != entry.global.generation
            || it->second.in_flight_ref == 0) {
            throw std::logic_error("V4.1 memory lease generation or reference mismatch");
        }
        --it->second.in_flight_ref;
    }
}

std::vector<PrefixTreeMemoryBlockCache::CacheItem> PrefixTreeMemoryBlockCache::popOldestDsv41JointEvictable() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    struct TreeState {
        bool     protected_tree{false};
        uint64_t last_access{0};
    };
    std::map<DSV41Key, TreeState> trees;
    for (const auto& [key, node] : dsv41_nodes_) {
        auto& tree = trees[{key.identity, node.root_key}];
        tree.protected_tree |=
            node.in_flight_ref > 0 || node.entry.global.is_resident || (node.entry.swa && node.entry.swa->is_resident);
        tree.last_access = std::max(tree.last_access, node.last_access_seq);
    }
    auto oldest = trees.end();
    for (auto it = trees.begin(); it != trees.end(); ++it) {
        if (!it->second.protected_tree
            && (oldest == trees.end() || it->second.last_access < oldest->second.last_access))
            oldest = it;
    }
    if (oldest == trees.end())
        return {};
    std::vector<CacheItem> released;
    // Evict a whole dependency tree. This deliberately trades eviction
    // granularity for preserving every SWA/global pair and all parent chains.
    for (auto it = dsv41_nodes_.begin(); it != dsv41_nodes_.end();) {
        if (it->first.identity == oldest->first.identity && it->second.root_key == oldest->first.key) {
            released.push_back(it->second.entry.global);
            if (it->second.entry.swa)
                released.push_back(*it->second.entry.swa);
            it = dsv41_nodes_.erase(it);
        } else
            ++it;
    }
    return released;
}

std::vector<CacheKeyType> PrefixTreeMemoryBlockCache::dsv41CacheKeys() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<CacheKeyType>           keys;
    for (const auto& [key, node] : dsv41_nodes_) {
        if (node.entry.swa)
            keys.push_back(key.key);
    }
    return keys;
}

}  // namespace rtp_llm
