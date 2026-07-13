#pragma once

#include <cstdint>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"

namespace rtp_llm {

// Eviction policy for the heap.
enum class EvictionPolicy : int8_t {
    LRU,       // Least recently used (by last_access_time ascending)
    LFU,       // Least frequently used (by hit_count ascending)
    FIFO,      // First in first out (by insert_seq ascending)
    PRIORITY,  // Custom priority (by priority ascending)
};

// Entry in the eviction heap. Contains fields for all policies;
// each policy uses a subset of these fields.
struct EvictionEntry {
    TreeNode* node{nullptr};
    int       component_group_id{-1};

    uint64_t last_access_time{0};  // LRU: steady_clock nanoseconds
    int      hit_count{0};         // LFU: access count
    int      priority{0};          // PRIORITY: custom priority
    uint64_t insert_seq{0};        // FIFO: insertion sequence number
};

// Eviction heap with configurable policy and lazy deletion.
// Uses std::priority_queue internally; invalidated entries are
// skipped during pop() via an auxiliary entry_map_.
class EvictionHeap {
public:
    explicit EvictionHeap(EvictionPolicy policy);

    // Add a node to the heap. Sets insert_seq automatically.
    void push(TreeNode* node, int group_id);

    // Pop the best eviction candidate (skips invalidated entries).
    // Returns nullopt if the heap is empty.
    std::optional<EvictionEntry> pop();

    // Mark a node as invalidated (logical deletion).
    // The entry remains in the priority queue but is skipped on pop().
    void invalidate(TreeNode* node);

    // Called when a node is accessed (match/insert).
    // - LRU: updates last_access_time and re-pushes
    // - LFU: increments hit_count
    // - FIFO/PRIORITY: no-op
    void onAccess(TreeNode* node);

    // Query whether a node is in the heap (and not invalidated).
    bool contains(TreeNode* node) const;

    // Collect all currently-valid (non-invalidated) nodes. Used for read-only
    // capacity queries (e.g. counting evictable blocks per group).
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
    // Comparison: returns true if lhs should be evicted AFTER rhs
    // (i.e., lhs is "more valuable" and should stay in heap longer).
    // priority_queue is a max-heap, so the "worst" candidate (to evict first)
    // needs to be at the top.
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
