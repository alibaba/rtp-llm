#pragma once

#include <cstdint>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"

namespace rtp_llm {

enum class EvictionPolicy : int8_t {
    LRU,   // Least recently used (by last_access_seq ascending)
    LFU,   // Least frequently used (by hit_count ascending)
    FIFO,  // First in first out (by admission_seq ascending)
};

struct EvictionEntry {
    TreeNode*    node{nullptr};
    uint64_t     primary_key{0};
    uint64_t     secondary_key{0};
    CacheKeyType cache_key{0};
};

// Exact-update eviction heap backed by std::set + node->iterator index.
// At most one entry per node; physical size equals the current ready-candidate
// count. upsert/erase/takeBest/contains are all O(log N) with no stale entries.
class EvictionHeap {
public:
    explicit EvictionHeap(EvictionPolicy policy);

    // Insert or replace a node's ordered entry from its candidate meta.
    void upsert(TreeNode* node, const CandidateMeta& meta);
    // Remove a node's entry if present. Idempotent.
    void erase(TreeNode* node);
    // Pop the best victim (smallest key); removes it from both containers.
    std::optional<EvictionEntry> takeBest();

    // Collect all current nodes. Used for read-only capacity queries.
    std::vector<TreeNode*> nodes() const {
        std::vector<TreeNode*> out;
        out.reserve(index_.size());
        for (const auto& kv : index_) {
            out.push_back(kv.first);
        }
        return out;
    }

    bool contains(TreeNode* node) const;
    bool empty() const {
        return ordered_.empty();
    }
    size_t size() const;

    EvictionPolicy policy() const {
        return policy_;
    }

private:
    struct EntryLess {
        bool operator()(const EvictionEntry& a, const EvictionEntry& b) const;
    };
    using OrderedSet = std::set<EvictionEntry, EntryLess>;

    // Build the ordered entry (primary/secondary keys) from meta per policy.
    EvictionEntry makeEntry(TreeNode* node, const CandidateMeta& meta) const;

    EvictionPolicy                                      policy_;
    OrderedSet                                          ordered_;
    std::unordered_map<TreeNode*, OrderedSet::iterator> index_;
};

}  // namespace rtp_llm
