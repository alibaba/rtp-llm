#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadJoinRegistry.h"

#include <algorithm>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool LoadJoinRegistry::start(TreeNode*                                node,
                             size_t                                   group_set_id,
                             const std::vector<BlockIdxType>&         target_blocks,
                             const std::shared_ptr<LoadAsyncContext>& context) {
    if (node == nullptr || target_blocks.empty() || context == nullptr
        || std::any_of(
            target_blocks.begin(), target_blocks.end(), [](BlockIdxType block) { return isNullBlockIdx(block); })) {
        return false;
    }

    const std::pair<decltype(records_)::iterator, bool> insert_result =
        records_.emplace(Key{node, group_set_id}, Record{target_blocks, {context}});
    return insert_result.second;
}

std::optional<std::vector<BlockIdxType>>
LoadJoinRegistry::join(TreeNode* node, size_t group_set_id, const std::shared_ptr<LoadAsyncContext>& context) {
    if (node == nullptr || context == nullptr) {
        return std::nullopt;
    }

    const auto record_it = records_.find(Key{node, group_set_id});
    if (record_it == records_.end()) {
        return std::nullopt;
    }
    for (const auto& registered_context : record_it->second.contexts) {
        if (registered_context == context) {
            return record_it->second.target_blocks;
        }
    }
    record_it->second.contexts.push_back(context);
    return record_it->second.target_blocks;
}

bool LoadJoinRegistry::finish(TreeNode* node, size_t group_set_id, bool success) {
    if (node == nullptr) {
        return false;
    }

    const auto record_it = records_.find(Key{node, group_set_id});
    if (record_it == records_.end()) {
        return false;
    }
    auto contexts = std::move(record_it->second.contexts);
    records_.erase(record_it);

    bool all_completed = true;
    for (const auto& context : contexts) {
        if (context == nullptr || !context->completeOne(success)) {
            all_completed = false;
            RTP_LLM_LOG_WARNING("failed to complete joined load context, group_set=%zu", group_set_id);
        }
    }
    return all_completed;
}

bool LoadJoinRegistry::eraseForContext(TreeNode*                                node,
                                       size_t                                   group_set_id,
                                       const std::shared_ptr<LoadAsyncContext>& context) {
    if (node == nullptr || context == nullptr) {
        return false;
    }

    const auto record_it = records_.find(Key{node, group_set_id});
    if (record_it == records_.end()) {
        return false;
    }
    const auto context_it = std::find(record_it->second.contexts.begin(), record_it->second.contexts.end(), context);
    if (context_it == record_it->second.contexts.end()) {
        return false;
    }
    record_it->second.contexts.erase(context_it);
    if (record_it->second.contexts.empty()) {
        records_.erase(record_it);
    }
    return true;
}

}  // namespace rtp_llm
