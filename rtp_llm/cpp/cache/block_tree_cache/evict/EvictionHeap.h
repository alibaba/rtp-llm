#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
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

// One ordered entry per candidate, indexed by node. A force-drop batch may
// remember a rejected prefix without changing heap membership.
class EvictionHeap {
private:
    struct EntryLess {
        bool operator()(const EvictionEntry& a, const EvictionEntry& b) const noexcept;
    };
    using OrderedSet = std::set<EvictionEntry, EntryLess>;

public:
    using const_iterator = OrderedSet::const_iterator;

    // A batch-local proof that the current ordered prefix is rejected. Only a
    // copied key survives a lookup; erasing its boundary invalidates the proof.
    class RejectedPrefixScope {
    public:
        RejectedPrefixScope(EvictionHeap& heap, const std::unordered_set<TreeNode*>& rejected);
        ~RejectedPrefixScope() noexcept;
        RejectedPrefixScope(const RejectedPrefixScope&) = delete;
        RejectedPrefixScope& operator=(const RejectedPrefixScope&) = delete;
        RejectedPrefixScope(RejectedPrefixScope&&) = delete;
        RejectedPrefixScope& operator=(RejectedPrefixScope&&) = delete;

        // Every false result must leave the node in the monotonic rejection set.
        // The predicate cannot mutate heap membership or metadata.
        std::optional<EvictionEntry> best(const std::function<bool(TreeNode*)>& can_evict);

    private:
        friend class EvictionHeap;
        void onAdmission(const_iterator position);
        void onErase(TreeNode* node) noexcept;
        EvictionHeap&                        heap_;
        const std::unordered_set<TreeNode*>& rejected_;
        std::optional<EvictionEntry>         fence_;
    };

    explicit EvictionHeap(EvictionPolicy policy);
    EvictionHeap(const EvictionHeap&)            = delete;
    EvictionHeap& operator=(const EvictionHeap&) = delete;
    EvictionHeap(EvictionHeap&&)                 = delete;
    EvictionHeap& operator=(EvictionHeap&&)      = delete;

    // Insert or replace a node's ordered entry from its candidate meta.
    void upsert(TreeNode* node, const CandidateMeta& meta);
    // Replace an existing entry without admitting a missing node.
    bool updateIfPresent(TreeNode* node, const CandidateMeta& meta);
    // Remove a node's entry if present. Idempotent.
    void erase(TreeNode* node);
    // Read the best matching victim without changing heap membership.
    std::optional<EvictionEntry> best(const std::function<bool(TreeNode*)>& can_evict) const;
    // Match the batch by rejection set identity. Reject overlapping batches
    // and a prefix-enabled batch sent to a heap without its scope.
    RejectedPrefixScope* rejectedPrefixFor(const std::unordered_set<TreeNode*>& rejected,
                                           bool expect_rejected_prefix) const;
    // Ordered read-only traversal under the cache mutex. Advance before erasing
    // the current entry; no other entry may be erased, inserted or reordered.
    // Iterators must not survive unlock, rollback or a cascading direct drop.
    const_iterator begin() const {
        return ordered_.cbegin();
    }
    const_iterator end() const {
        return ordered_.cend();
    }
    bool isSkippingRejectedPrefix() const noexcept {
        return rejected_prefix_scope_ != nullptr;
    }

    // Collect all current nodes. Used for read-only capacity queries.
    std::vector<TreeNode*> nodes() const {
        std::vector<TreeNode*> out;
        out.reserve(index_.size());
        for (const auto& kv : index_) {
            out.push_back(kv.first);
        }
        return out;
    }

    bool   contains(TreeNode* node) const;
    size_t size() const;

private:
    // Build the ordered entry (primary/secondary keys) from meta per policy.
    EvictionEntry makeEntry(TreeNode* node, const CandidateMeta& meta) const;

    EvictionPolicy                                      policy_;
    OrderedSet                                          ordered_;
    std::unordered_map<TreeNode*, OrderedSet::iterator> index_;
    RejectedPrefixScope*                                rejected_prefix_scope_{nullptr};
};

}  // namespace rtp_llm
