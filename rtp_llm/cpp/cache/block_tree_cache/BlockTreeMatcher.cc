#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeMatcher.h"

#include <algorithm>
#include <memory>
#include <unordered_set>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTreeMatcher::BlockTreeMatcher(BlockTree*                    tree,
                                   std::vector<GroupSetPtr>&     group_sets,
                                   const ReusableGroupLocations& reusable_group_locations,
                                   BlockTreeEvictor&             evictor):
    tree_(tree), group_sets_(group_sets), reusable_group_locations_(reusable_group_locations), evictor_(evictor) {}

std::pair<BlockTreeMatchResult, std::vector<TreeNode*>> BlockTreeMatcher::matchLocked(const CacheKeysType& cache_keys) {
    BlockTreeMatchResult result;
    BlockTreeFindResult  tree_find_result = tree_->findNode(cache_keys);
    if (tree_find_result.matched_node == nullptr) {
        RTP_LLM_LOG_DEBUG("no match found for %zu cache_keys", cache_keys.size());
        return {std::move(result), {}};
    }

    for (TreeNode* path_node : tree_find_result.path) {
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            const GroupSetResource& resource = path_node->group_set_resources[group_set_id];
            RTP_LLM_CHECK_WITH_INFO(!resource.hasTier(Tier::DEVICE)
                                        || group_sets_[group_set_id]->hasCompleteDeviceValue(resource),
                                    "BlockTreeCache partial DEVICE resource: node_key=%ld group_set_id=%zu "
                                    "device_width=%zu expected_width=%zu",
                                    path_node->cache_key,
                                    group_set_id,
                                    resource.device_blocks.size(),
                                    group_sets_[group_set_id]->devicePoolCount());
        }
    }

    size_t            valid_matched_block_count = 0;
    std::vector<bool> candidate_logically_valid;
    candidate_logically_valid.reserve(tree_find_result.path.size());
    std::vector<std::unique_ptr<MatchValidator>> match_validators;
    match_validators.reserve(group_sets_.size());
    for (const GroupSetPtr& group_set : group_sets_) {
        match_validators.push_back(group_set->createMatchValidator());
    }
    for (size_t i = 0; i < tree_find_result.path.size(); ++i) {
        TreeNode* path_node        = tree_find_result.path[i];
        bool      all_groups_valid = true;
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            GroupSetResource& group_set_resource = path_node->group_set_resources[group_set_id];
            const bool        group_valid = match_validators[group_set_id]->validate(path_node, group_set_resource);
            if (!group_valid) {
                all_groups_valid = false;
            }
        }
        if (all_groups_valid) {
            valid_matched_block_count = i + 1;
        }
        candidate_logically_valid.push_back(all_groups_valid);
    }

    std::vector<TreeNode*> logical_matched_path(tree_find_result.path.begin(),
                                                tree_find_result.path.begin()
                                                    + static_cast<ptrdiff_t>(valid_matched_block_count));
    candidate_logically_valid.resize(valid_matched_block_count);
    prepareReadyMatchedResourcesLocked(logical_matched_path, candidate_logically_valid, result);

    RTP_LLM_LOG_DEBUG("matched %zu blocks, cache_keys=%zu, tree_nodes=%zu",
                      result.matched_blocks,
                      cache_keys.size(),
                      tree_->nodeCount());
    return {std::move(result), std::move(logical_matched_path)};
}

void BlockTreeMatcher::releaseMatchedResourcesLocked(const std::vector<MultiNodeResource>& resources) {
    std::unordered_set<size_t> seen_group_set_ids;
    for (const auto& resource : resources) {
        RTP_LLM_CHECK_WITH_INFO(seen_group_set_ids.emplace(resource.group_set_id).second,
                                "releaseMatchedResources duplicate group_set_id=%zu",
                                resource.group_set_id);
        validateMatchedResource(resource);
    }
    for (const auto& resource : resources) {
        group_sets_[resource.group_set_id]->unreferenceBlocks(resource, BlockRefType::REQUEST);
        evictor_.refreshCandidatesAfterRelease(resource);
    }
}

BlockIndicesType
BlockTreeMatcher::matchedBlocksForGroup(size_t                                group_id,
                                        const std::vector<MultiNodeResource>& matched_resources) const {
    const auto location_it = reusable_group_locations_.find(group_id);
    if (location_it == reusable_group_locations_.end()) {
        return {};
    }
    const ReusableGroupLocation& location = location_it->second;
    for (const auto& resource : matched_resources) {
        if (resource.group_set_id != location.group_set_id) {
            continue;
        }
        validateMatchedResource(resource);
        BlockIndicesType blocks;
        blocks.reserve(resource.per_node.size());
        for (const auto& node_blocks : resource.per_node) {
            blocks.push_back(node_blocks[location.member_index]);
        }
        return blocks;
    }
    return {};
}

void BlockTreeMatcher::validateMatchedResource(const MultiNodeResource& resource) const {
    RTP_LLM_CHECK_WITH_INFO(resource.group_set_id < group_sets_.size(),
                            "invalid matched group_set_id=%zu group_set_count=%zu",
                            resource.group_set_id,
                            group_sets_.size());
    RTP_LLM_CHECK_WITH_INFO(resource.tier == Tier::DEVICE,
                            "matched resource requires DEVICE tier, group_set_id=%zu tier=%s",
                            resource.group_set_id,
                            tierName(resource.tier));

    const GroupSetPtr& group_set = group_sets_[resource.group_set_id];
    for (const auto& node_blocks : resource.per_node) {
        RTP_LLM_CHECK_WITH_INFO(node_blocks.size() == group_set->devicePoolCount()
                                    && std::all_of(node_blocks.begin(),
                                                   node_blocks.end(),
                                                   [](BlockIdxType block) { return !isNullBlockIdx(block); }),
                                "malformed matched DEVICE blocks, group_set_id=%zu expected_width=%zu actual_width=%zu",
                                resource.group_set_id,
                                group_set->devicePoolCount(),
                                node_blocks.size());
    }
    RTP_LLM_CHECK_WITH_INFO(resource.tree_nodes.empty()
                                || (resource.tree_nodes.size() == resource.per_node.size()
                                    && std::all_of(resource.tree_nodes.begin(),
                                                   resource.tree_nodes.end(),
                                                   [](const TreeNode* node) { return node != nullptr; })),
                            "malformed matched tree-node alignment, group_set_id=%zu nodes=%zu blocks=%zu",
                            resource.group_set_id,
                            resource.tree_nodes.size(),
                            resource.per_node.size());
}

void BlockTreeMatcher::prepareReadyMatchedResourcesLocked(const std::vector<TreeNode*>& matched_path,
                                                          const std::vector<bool>&      candidate_logically_valid,
                                                          BlockTreeMatchResult&         result) {
    const size_t logical_matched_block_count = matched_path.size();
    if (logical_matched_block_count == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(candidate_logically_valid.size() == logical_matched_block_count,
                            "candidate validity size mismatch, path=%zu valid=%zu",
                            logical_matched_block_count,
                            candidate_logically_valid.size());

    const size_t ready_matched_block_count = computeReadyMatchedBlockCount(matched_path, candidate_logically_valid);
    if (ready_matched_block_count > 0) {
        result.matched_node   = matched_path[ready_matched_block_count - 1];
        result.matched_blocks = ready_matched_block_count;
        evictor_.onMatched(std::vector<TreeNode*>(
            matched_path.begin(), matched_path.begin() + static_cast<ptrdiff_t>(ready_matched_block_count)));
    }

    for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
        const GroupSetPtr& group_set = group_sets_[group_set_id];
        MultiNodeResource  matched_device_blocks{group_set_id, Tier::DEVICE};

        const size_t ready_reuse_count = std::min(
            group_set->computeReuseBlockCount(ready_matched_block_count, matched_path), ready_matched_block_count);
        const size_t ready_reuse_begin = ready_matched_block_count - ready_reuse_count;
        for (size_t i = ready_reuse_begin; i < ready_matched_block_count; ++i) {
            TreeNode*                       path_node          = matched_path[i];
            GroupSetResource&               group_set_resource = path_node->group_set_resources[group_set_id];
            const std::vector<BlockIdxType> device_blocks      = group_set->getBlocks(group_set_resource, Tier::DEVICE);
            matched_device_blocks.per_node.push_back(device_blocks);
            matched_device_blocks.tree_nodes.push_back(path_node);
        }

        if (!matched_device_blocks.per_node.empty()) {
            group_set->referenceBlocks(matched_device_blocks, BlockRefType::REQUEST);
            result.matched_resources.push_back(std::move(matched_device_blocks));
        }
    }
}

size_t BlockTreeMatcher::computeReadyMatchedBlockCount(const std::vector<TreeNode*>& matched_path,
                                                       const std::vector<bool>&      candidate_logically_valid) const {
    for (size_t candidate_count = matched_path.size(); candidate_count > 0; --candidate_count) {
        if (!candidate_logically_valid[candidate_count - 1]) {
            continue;
        }
        bool all_groups_ready = true;
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            const GroupSetPtr& group_set = group_sets_[group_set_id];
            const size_t       reuse_count =
                std::min(group_set->computeReuseBlockCount(candidate_count, matched_path), candidate_count);
            for (size_t path_index = candidate_count - reuse_count; path_index < candidate_count; ++path_index) {
                TreeNode*               path_node = matched_path[path_index];
                const GroupSetResource& resource  = path_node->group_set_resources[group_set_id];
                // A DEMOTING resource still carries device blocks but is owned by an
                // in-flight transfer; ready reuse must only consume usable resources.
                if (!resource.isMatchUsable() || !group_set->hasCompleteDeviceValue(resource)) {
                    all_groups_ready = false;
                    break;
                }
            }
            if (!all_groups_ready) {
                break;
            }
        }
        if (all_groups_ready) {
            return candidate_count;
        }
    }
    return 0;
}

}  // namespace rtp_llm
