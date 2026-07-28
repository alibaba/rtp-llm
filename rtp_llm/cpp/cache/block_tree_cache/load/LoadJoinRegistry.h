#pragma once

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"

namespace rtp_llm {

class LoadJoinRegistry {
public:
    bool start(TreeNode*                                node,
               size_t                                   group_set_id,
               const std::vector<BlockIdxType>&         target_blocks,
               const std::shared_ptr<LoadAsyncContext>& context);
    std::optional<std::vector<BlockIdxType>>
         join(TreeNode* node, size_t group_set_id, const std::shared_ptr<LoadAsyncContext>& context);
    bool finish(TreeNode* node, size_t group_set_id, bool success);
    bool eraseForContext(TreeNode* node, size_t group_set_id, const std::shared_ptr<LoadAsyncContext>& context);

private:
    struct Key {
        TreeNode* node;
        size_t    group_set_id;

        bool operator==(const Key& other) const {
            return node == other.node && group_set_id == other.group_set_id;
        }
    };

    struct KeyHash {
        size_t operator()(const Key& key) const {
            const size_t node_hash  = std::hash<TreeNode*>{}(key.node);
            const size_t group_hash = std::hash<size_t>{}(key.group_set_id);
            return node_hash ^ (group_hash << 1);
        }
    };

    struct Record {
        std::vector<BlockIdxType>                      target_blocks;
        std::vector<std::shared_ptr<LoadAsyncContext>> contexts;
    };

    std::unordered_map<Key, Record, KeyHash> records_;
};

}  // namespace rtp_llm
