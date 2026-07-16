#pragma once

#include <cstdint>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"

namespace rtp_llm {

enum class EvictionPolicy : int8_t {
    LRU,
    LFU,
    FIFO,
};

struct EvictionEntry {
    TreeNode* node{nullptr};
    int       component_group_id{-1};

    uint64_t last_access_time{0};
    uint64_t hit_count{0};
    uint64_t insert_seq{0};
};

class EvictionHeap {
public:
    explicit EvictionHeap(EvictionPolicy policy);

    void push(TreeNode* node, int group_id);

    std::optional<EvictionEntry> pop();

    void invalidate(TreeNode* node);

    void onAccess(TreeNode* node);

    bool contains(TreeNode* node) const;

    std::vector<TreeNode*> nodes() const {
        std::vector<TreeNode*> out;
        out.reserve(entry_map_.size());
        for (const auto& kv : entry_map_) {
            out.push_back(kv.first);
        }
        return out;
    }

    bool   empty() const;
    size_t size() const;

    EvictionPolicy policy() const {
        return policy_;
    }

private:
    struct EntryComparator {
        EvictionPolicy policy;
        bool           operator()(const EvictionEntry& a, const EvictionEntry& b) const;
    };

    using PriorityQueue = std::priority_queue<EvictionEntry, std::vector<EvictionEntry>, EntryComparator>;

    EvictionPolicy                               policy_;
    PriorityQueue                                heap_;
    std::unordered_map<TreeNode*, EvictionEntry> entry_map_;
    uint64_t                                     insert_seq_counter_{0};
};

}  // namespace rtp_llm
