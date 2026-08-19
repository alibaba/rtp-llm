#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"

namespace rtp_llm {

class LoadJoinRegistry {
public:
    explicit LoadJoinRegistry(BlockTree* tree): tree_(tree) {}

    bool start(TreeNode*                                node,
               size_t                                   group_set_id,
               const std::vector<BlockIdxType>&         target_blocks,
               const std::shared_ptr<LoadAsyncContext>& context);
    bool join(const std::shared_ptr<LoadAsyncContext>& context);
    bool finish(TreeNode* node, size_t group_set_id, bool success);
    bool eraseForContext(TreeNode* node, size_t group_set_id, uint64_t context_id);

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
        using ContextMap = std::unordered_map<uint64_t, std::weak_ptr<LoadAsyncContext>>;

        std::vector<BlockIdxType> target_blocks;
        // Joining a load must not extend its context lifetime; the context owns RAII abort.
        ContextMap contexts;
    };

    using RecordMap = std::unordered_map<Key, Record, KeyHash>;

    BlockTree* tree_;
    RecordMap  records_;
};

}  // namespace rtp_llm
