#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadJoinRegistry.h"

#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool LoadJoinRegistry::getTargetBlocks(TreeNode*                  node,
                                       size_t                     group_set_id,
                                       std::vector<BlockIdxType>& target_blocks) const {
    target_blocks.clear();
    const auto record_it = records_.find(Key{node, group_set_id});
    if (record_it == records_.end()) {
        return false;
    }
    target_blocks = record_it->second.target_blocks;
    return true;
}

bool LoadJoinRegistry::start(TreeNode*                                node,
                             size_t                                   group_set_id,
                             const std::vector<BlockIdxType>&         target_blocks,
                             const std::shared_ptr<LoadAsyncContext>& context) {
    const std::pair<decltype(records_)::iterator, bool> insert_result =
        records_.emplace(Key{node, group_set_id}, Record{target_blocks, {{context->contextId(), context}}});
    return insert_result.second;
}

bool LoadJoinRegistry::join(TreeNode*                                node,
                            size_t                                   group_set_id,
                            const std::shared_ptr<LoadAsyncContext>& context,
                            std::vector<BlockIdxType>&               target_blocks) {
    target_blocks.clear();
    const auto record_it = records_.find(Key{node, group_set_id});
    if (record_it == records_.end()) {
        return false;
    }
    const uint64_t                     context_id = context->contextId();
    const Record::ContextMap::iterator context_it = record_it->second.contexts.find(context_id);
    if (context_it == record_it->second.contexts.end()) {
        record_it->second.contexts[context_id] = context;
    }
    target_blocks = record_it->second.target_blocks;
    return true;
}

bool LoadJoinRegistry::finish(TreeNode* node, size_t group_set_id, bool success) {
    const auto record_it = records_.find(Key{node, group_set_id});
    if (record_it == records_.end()) {
        return false;
    }
    Record::ContextMap contexts = std::move(record_it->second.contexts);
    records_.erase(record_it);

    bool all_completed = true;
    for (const std::pair<const uint64_t, std::weak_ptr<LoadAsyncContext>>& context_entry : contexts) {
        const std::shared_ptr<LoadAsyncContext> context = context_entry.second.lock();
        if (context == nullptr) {
            continue;
        }
        if (!context->completeOne(success)) {
            all_completed = false;
            RTP_LLM_LOG_WARNING("failed to complete joined load context, group_set=%zu", group_set_id);
        }
    }
    return all_completed;
}

bool LoadJoinRegistry::eraseForContext(TreeNode* node, size_t group_set_id, uint64_t context_id) {
    const auto record_it = records_.find(Key{node, group_set_id});
    if (record_it == records_.end()) {
        return false;
    }
    const size_t erased_count = record_it->second.contexts.erase(context_id);
    if (erased_count != 1) {
        return false;
    }
    if (record_it->second.contexts.empty()) {
        records_.erase(record_it);
    }
    return true;
}

}  // namespace rtp_llm
