#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionHeap.h"

#include <cassert>
#include <functional>
#include <iterator>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

bool EvictionHeap::EntryLess::operator()(const EvictionEntry& a, const EvictionEntry& b) const noexcept {
    // Ascending order: the smallest entry is the best victim (evicted first).
    if (a.primary_key != b.primary_key)
        return a.primary_key < b.primary_key;
    if (a.secondary_key != b.secondary_key)
        return a.secondary_key < b.secondary_key;
    if (a.cache_key != b.cache_key)
        return a.cache_key < b.cache_key;
    return std::less<TreeNode*>{}(a.node, b.node);  // process-local deterministic tie-breaker
}

EvictionHeap::EvictionHeap(EvictionPolicy policy): policy_(policy) {}

EvictionHeap::RejectedPrefixScope::RejectedPrefixScope(EvictionHeap& heap,
                                                       const std::unordered_set<TreeNode*>& rejected):
    heap_(heap), rejected_(rejected) {
    RTP_LLM_CHECK_WITH_INFO(!heap_.isSkippingRejectedPrefix(), "nested rejected-prefix selection on the same heap");
    heap_.rejected_prefix_scope_ = this;
}

EvictionHeap::RejectedPrefixScope::~RejectedPrefixScope() noexcept {
    heap_.rejected_prefix_scope_ = nullptr;
}

void EvictionHeap::RejectedPrefixScope::onAdmission(const_iterator position) {
    const EvictionEntry& entry = *position;
    // A non-rejected insertion or reorder inside the prefix only invalidates
    // it from that position onward. The predecessor is still proved rejected.
    // R deliberately has the same pointer-identity semantics as chooseDropVictim.
    if (fence_.has_value() && !EntryLess{}(*fence_, entry) && rejected_.count(entry.node) == 0) {
        if (position == heap_.ordered_.cbegin()) {
            fence_.reset();
        } else {
            fence_ = *std::prev(position);
        }
    }
}

void EvictionHeap::RejectedPrefixScope::onErase(TreeNode* node) noexcept {
    // Keep the comparator's pointer tie-breaker live. Other erasures preserve
    // the proof, but deleting the boundary rebuilds it on the next lookup.
    if (fence_.has_value() && fence_->node == node) {
        fence_.reset();
    }
}

std::optional<EvictionEntry>
EvictionHeap::RejectedPrefixScope::best(const std::function<bool(TreeNode*)>& can_evict) {
    // EntryLess uses every copied key, including the live boundary's pointer
    // tie-breaker. Erasing that boundary invalidates it before node destruction.
    auto candidate     = fence_.has_value() ? heap_.ordered_.upper_bound(*fence_) : heap_.ordered_.begin();
    auto last_rejected = heap_.ordered_.end();
    for (; candidate != heap_.ordered_.end(); ++candidate) {
        const EvictionEntry& entry = *candidate;
        if (can_evict(entry.node)) {
            if (last_rejected != heap_.ordered_.end()) {
                fence_ = *last_rejected;
            }
            return entry;
        }
        // Trusted selection predicate: false confirms/inserts the identity in
        // R. Reuse that proof without another hash lookup on every rejection.
        last_rejected = candidate;
    }
    if (last_rejected != heap_.ordered_.end()) {
        fence_ = *last_rejected;
    }
    // Both iterators die here, before any cascading drop can run.
    return std::nullopt;
}

EvictionEntry EvictionHeap::makeEntry(TreeNode* node, const CandidateMeta& meta) const {
    EvictionEntry entry;
    entry.node      = node;
    entry.cache_key = node ? node->cache_key : 0;
    switch (policy_) {
        case EvictionPolicy::LRU:
            entry.primary_key   = meta.last_access_seq;
            entry.secondary_key = meta.admission_seq;
            break;
        case EvictionPolicy::LFU:
            entry.primary_key   = meta.hit_count;
            entry.secondary_key = meta.last_access_seq;
            break;
        case EvictionPolicy::FIFO:
            entry.primary_key   = meta.admission_seq;
            entry.secondary_key = 0;
            break;
    }
    return entry;
}

void EvictionHeap::upsert(TreeNode* node, const CandidateMeta& meta) {
    if (node == nullptr)
        return;
    if (updateIfPresent(node, meta)) {
        return;
    }
    auto [ordered_it, inserted] = ordered_.insert(makeEntry(node, meta));
    RTP_LLM_CHECK_WITH_INFO(
        inserted, "EvictionHeap::upsert failed to insert a unique entry for node=%p", static_cast<void*>(node));
    try {
        auto index_result = index_.emplace(node, ordered_it);
        RTP_LLM_CHECK_WITH_INFO(
            index_result.second, "EvictionHeap::upsert failed to index node=%p", static_cast<void*>(node));
    } catch (...) {
        ordered_.erase(ordered_it);
        throw;
    }
    if (rejected_prefix_scope_ != nullptr) {
        rejected_prefix_scope_->onAdmission(ordered_it);
    }
}

bool EvictionHeap::updateIfPresent(TreeNode* node, const CandidateMeta& meta) {
    auto index_it = index_.find(node);
    if (index_it == index_.end()) {
        return false;
    }
    auto handle    = ordered_.extract(index_it->second);
    handle.value() = makeEntry(node, meta);
    auto result    = ordered_.insert(std::move(handle));
    assert(result.inserted);
    index_it->second = result.position;
    if (rejected_prefix_scope_ != nullptr) {
        rejected_prefix_scope_->onAdmission(result.position);
    }
    return true;
}

void EvictionHeap::erase(TreeNode* node) {
    auto it = index_.find(node);
    if (it == index_.end())
        return;
    if (rejected_prefix_scope_ != nullptr) {
        rejected_prefix_scope_->onErase(node);
    }
    ordered_.erase(it->second);
    index_.erase(it);
}

std::optional<EvictionEntry> EvictionHeap::best(const std::function<bool(TreeNode*)>& can_evict) const {
    for (const EvictionEntry& entry : ordered_) {
        if (can_evict(entry.node)) {
            return entry;
        }
    }
    return std::nullopt;
}

EvictionHeap::RejectedPrefixScope*
EvictionHeap::rejectedPrefixFor(const std::unordered_set<TreeNode*>& rejected, bool expect_rejected_prefix) const {
    auto* scope = rejected_prefix_scope_;
    RTP_LLM_CHECK_WITH_INFO(
        expect_rejected_prefix == (scope != nullptr) && (scope == nullptr || &scope->rejected_ == &rejected),
        "rejected-prefix scope does not match the eviction batch");
    return scope;
}

bool EvictionHeap::contains(TreeNode* node) const {
    return index_.find(node) != index_.end();
}

size_t EvictionHeap::size() const {
    return ordered_.size();
}

}  // namespace rtp_llm
