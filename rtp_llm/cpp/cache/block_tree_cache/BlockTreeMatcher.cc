#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeMatcher.h"

#include <algorithm>
#include <memory>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTreeMatcher::BlockTreeMatcher(BlockTree* tree, BlockTreeEvictor& evictor): tree_(tree), evictor_(evictor) {}

std::pair<BlockTreeMatchResult, std::vector<TreeNode*>> BlockTreeMatcher::matchLocked(const CacheKeysType& cache_keys) {
    BlockTreeMatchResult   result;
    std::vector<TreeNode*> matched_path = tree_->findNode(cache_keys);
    if (matched_path.empty()) {
        RTP_LLM_LOG_DEBUG("no match found for %zu cache_keys", cache_keys.size());
        return {std::move(result), {}};
    }

    size_t            valid_matched_block_count = 0;
    std::vector<bool> candidate_valid;
    candidate_valid.reserve(matched_path.size());
    std::vector<std::unique_ptr<MatchValidator>> match_validators;
    match_validators.reserve(tree_->groupSets().size());
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        match_validators.push_back(group_set->createMatchValidator());
    }
    for (size_t i = 0; i < matched_path.size(); ++i) {
        TreeNode* path_node        = matched_path[i];
        bool      all_groups_valid = true;
        for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
            GroupSetResource& group_set_resource = path_node->group_set_resources[group_set_id];
            const bool        group_valid = match_validators[group_set_id]->validate(group_set_resource);
            if (!group_valid) {
                all_groups_valid = false;
            }
        }
        if (all_groups_valid) {
            valid_matched_block_count = i + 1;
        }
        candidate_valid.push_back(all_groups_valid);
    }
    if (valid_matched_block_count == 0) {
        RTP_LLM_LOG_DEBUG("no valid match found for %zu cache_keys", cache_keys.size());
        return {std::move(result), {}};
    }

    matched_path.resize(valid_matched_block_count);
    candidate_valid.resize(valid_matched_block_count);
    prepareReadyMatchedResourcesLocked(matched_path, candidate_valid, result);

    RTP_LLM_LOG_DEBUG("matched %zu blocks, cache_keys=%zu, tree_nodes=%zu",
                      result.matched_blocks,
                      cache_keys.size(),
                      tree_->nodes().size());
    return {std::move(result), std::move(matched_path)};
}

void BlockTreeMatcher::releaseMatchedResourcesLocked(const std::vector<MultiNodeResource>& resources) {
    for (const auto& resource : resources) {
        tree_->groupSets()[resource.group_set_id]->unreferenceBlocks(resource, BlockRefType::REQUEST);
        evictor_.refreshCandidatesAfterRelease(resource);
    }
}

BlockIndicesType
BlockTreeMatcher::matchedBlocksForGroup(size_t                                group_id,
                                        const std::vector<MultiNodeResource>& matched_resources) const {
    const ReusableGroupLocation* location = tree_->reusableGroupLocation(group_id);
    if (location == nullptr) {
        return {};
    }
    for (const auto& resource : matched_resources) {
        if (resource.group_set_id != location->group_set_id) {
            continue;
        }
        BlockIndicesType blocks;
        blocks.reserve(resource.node_blocks.size());
        for (const auto& [_, node_blocks] : resource.node_blocks) {
            blocks.push_back(node_blocks[location->member_group_id]);
        }
        return blocks;
    }
    return {};
}

void BlockTreeMatcher::prepareReadyMatchedResourcesLocked(const std::vector<TreeNode*>& matched_path,
                                                          const std::vector<bool>&      candidate_valid,
                                                          BlockTreeMatchResult&         result) {
    for (size_t candidate_count = matched_path.size(); candidate_count > 0; --candidate_count) {
        if (!candidate_valid[candidate_count - 1]) {
            continue;
        }
        bool all_groups_ready = true;
        for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
            const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
            const size_t       reuse_count =
                std::min(group_set->computeReuseBlockCount(candidate_count), candidate_count);
            for (size_t path_index = candidate_count - reuse_count; path_index < candidate_count; ++path_index) {
                TreeNode*               path_node = matched_path[path_index];
                const GroupSetResource& resource  = path_node->group_set_resources[group_set_id];
                if (!resource.isMatchUsable() || !resource.hasCompleteDeviceValue()) {
                    all_groups_ready = false;
                    break;
                }
            }
            if (!all_groups_ready) {
                break;
            }
        }
        if (all_groups_ready) {
            result.matched_node   = matched_path[candidate_count - 1];
            result.matched_blocks = candidate_count;
            evictor_.onMatched(std::vector<TreeNode*>(
                matched_path.begin(), matched_path.begin() + static_cast<ptrdiff_t>(candidate_count)));
            break;
        }
    }

    for (size_t group_set_id = 0; group_set_id < tree_->groupSets().size(); ++group_set_id) {
        const GroupSetPtr& group_set = tree_->groupSets()[group_set_id];
        MultiNodeResource  matched_device_blocks{group_set_id, Tier::DEVICE};

        const size_t ready_reuse_count =
            std::min(group_set->computeReuseBlockCount(result.matched_blocks), result.matched_blocks);
        const size_t ready_reuse_begin = result.matched_blocks - ready_reuse_count;
        for (size_t i = ready_reuse_begin; i < result.matched_blocks; ++i) {
            TreeNode*                       path_node          = matched_path[i];
            GroupSetResource&               group_set_resource = path_node->group_set_resources[group_set_id];
            const std::vector<BlockIdxType> device_blocks      = group_set_resource.getBlocks(Tier::DEVICE);
            matched_device_blocks.node_blocks.emplace_back(path_node, device_blocks);
        }

        if (!matched_device_blocks.node_blocks.empty()) {
            group_set->referenceBlocks(matched_device_blocks, BlockRefType::REQUEST);
            result.matched_resources.push_back(std::move(matched_device_blocks));
        }
    }
}

}  // namespace rtp_llm
