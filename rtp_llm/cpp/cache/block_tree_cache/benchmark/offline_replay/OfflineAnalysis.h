#pragma once

#include <algorithm>
#include <fstream>
#include <map>
#include <limits>
#include <vector>

#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/offline_replay/StateIO.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm::benchmark {

inline std::shared_ptr<const CacheTopology> blockTreeKeyTopology(const CacheConfig& cache_config, size_t cp_size) {
    auto cache_topology = cache_config.topologyPtr();
    if (cp_size > 1) {
        auto groups = cache_topology->groups();
        RTP_LLM_CHECK_WITH_INFO(cache_config.seq_size_per_block <= std::numeric_limits<size_t>::max() / cp_size,
                                "canonical CP cache key stride overflow");
        for (auto& group : groups) {
            auto spec                    = group.spec->clone();
            spec->cache_key_token_stride = cache_config.seq_size_per_block * cp_size;
            group.spec                   = std::move(spec);
        }
        cache_topology = CacheTopology::create(std::move(groups), cache_topology->layers());
    }
    return cache_topology;
}

struct OfflineMatchLengths {
    size_t full_blocks{0};
    size_t joint_blocks{0};
};

// The serial driver calls this before malloc, after all previous workflows settle.
// Validators are native BTC validators; this read does not touch candidate metadata.
inline OfflineMatchLengths observeMatch(const BlockTree&        tree,
                                        const CacheConfig&      config,
                                        const CPSlotMapper&     mapper,
                                        const CompleteTokenIds& tokens,
                                        const CacheKeysType&    original_keys) {
    auto       keys = mapper.isSharded() ? mapper.canonicalCacheKeys(original_keys) : original_keys;
    const auto unit = mapper.reuseBlockTokens(config);
    stateRequire(unit > 0, "invalid native reuse block span");
    const auto max_keys = static_cast<size_t>(config.maxPrefixReuseTokens(tokens.seqLength())) / unit;
    keys.resize(std::min(keys.size(), max_keys));
    while (!keys.empty() && !tokens.isValidReuseLength(keys.size() * unit, config.cache_min_replay_tokens))
        keys.pop_back();
    const auto                                   path = tree.findNode(keys);
    std::vector<std::unique_ptr<MatchValidator>> validators;
    for (const auto& group : tree.groupSets())
        validators.push_back(group->createMatchValidator());
    OfflineMatchLengths result;
    bool                full_prefix_valid = true;
    for (size_t i = 0; i < path.size(); ++i) {
        bool joint_valid = true;
        for (size_t g = 0; g < validators.size(); ++g) {
            const bool valid = validators[g]->validate(path[i]->group_set_resources[g]);
            joint_valid      = joint_valid && valid;
            if (tree.groupSets()[g]->groupType() == CacheGroupType::FULL)
                full_prefix_valid = full_prefix_valid && valid;
        }
        if (tokens.isValidReuseLength((i + 1) * unit)) {
            if (full_prefix_valid)
                result.full_blocks = i + 1;
            if (joint_valid)
                result.joint_blocks = i + 1;
        }
    }
    return result;
}

// Static topology, not historical insertion endpoints. Called only at a settled boundary.
inline void exportTreeStructure(const BlockTree& tree, const std::string& directory) {
    std::ofstream rows(directory + "/tree-structure.csv");
    rows << "key,parent_key,depth_blocks,full_children,is_branch\n";
    size_t                                          branches = 0, leaves = 0, nodes = 0;
    std::vector<std::pair<const TreeNode*, size_t>> pending{{tree.root(), 0}};
    while (!pending.empty()) {
        const auto [node, depth] = pending.back();
        pending.pop_back();
        if (node != tree.root()) {
            ++nodes;
            branches += node->children.size() > 1;
            leaves += node->children.empty();
            rows << node->cache_key << ',' << node->parent->cache_key << ',' << depth << ',' << node->children.size()
                 << ',' << (node->children.size() > 1) << '\n';
        }
        std::map<CacheKeyType, const TreeNode*> children(node->children.begin(), node->children.end());
        for (auto it = children.rbegin(); it != children.rend(); ++it)
            pending.emplace_back(it->second, depth + 1);
    }
    std::ofstream summary(directory + "/tree-structure.json");
    summary << "{\"scope\":\"settled_live_tree\",\"branch_definition\":\"children.size()>1; root excluded\""
            << ",\"nodes\":" << nodes << ",\"branch_nodes\":" << branches << ",\"leaf_nodes\":" << leaves
            << ",\"root_children\":" << tree.root()->children.size() << ",\"historical_branch_statistics\":null"
            << ",\"historical_branch_statistics_status\":\"not_collected\"}\n";
    stateRequire(bool(rows.flush()) && bool(summary.flush()), "tree structure export failed");
}

}  // namespace rtp_llm::benchmark
