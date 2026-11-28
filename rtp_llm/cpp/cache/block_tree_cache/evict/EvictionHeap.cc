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

EvictionHeap::RejectedPrefixScope::RejectedPrefixScope(EvictionHeap&                        heap,
                                                       const std::unordered_set<TreeNode*>& rejected):
    heap_(heap), rejected_(rejected) {
    RTP_LLM_CHECK_WITH_INFO(!heap_.isSkippingRejectedPrefix(), "nested rejected-prefix selection on the same heap");
    heap_.rejected_prefix_scope_ = this;
}

EvictionHeap::RejectedPrefixScope::~RejectedPrefixScope() noexcept {
    heap_.rejected_prefix_scope_ = nullptr;
}

void EvictionHeap::updateRejectedPrefixOnAdmission(const_iterator position) {
    auto&                scope = *rejected_prefix_scope_;
    auto&                fence = scope.fence_;
    const EvictionEntry& entry = *position;
    // A non-rejected insertion or reorder inside the prefix only invalidates
    // it from that position onward. The predecessor is still proved rejected.
    if (fence.has_value() && !EntryLess{}(*fence, entry) && scope.rejected_.count(entry.node) == 0) {
        if (position == ordered_.cbegin()) {
            fence.reset();
        } else {
            fence = *std::prev(position);
        }
    }
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
        updateRejectedPrefixOnAdmission(ordered_it);
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
        updateRejectedPrefixOnAdmission(result.position);
    }
    return true;
}

void EvictionHeap::erase(TreeNode* node) {
    auto it = index_.find(node);
    if (it == index_.end())
        return;
    if (rejected_prefix_scope_ != nullptr) {
        // Invalidate the copied pointer tie-breaker before the boundary can be destroyed.
        auto& fence = rejected_prefix_scope_->fence_;
        if (fence.has_value() && fence->node == node) {
            fence.reset();
        }
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

void EvictionHeap::rememberRejectedPrefixBefore(TreeNode*                            victim,
                                                const std::unordered_set<TreeNode*>& rejected,
                                                std::optional<RejectedPrefixScope>&  scope) {
    RTP_LLM_CHECK_WITH_INFO(!scope.has_value(), "rejected-prefix storage is already active");
    const auto selected = index_.find(victim);
    RTP_LLM_CHECK_WITH_INFO(selected != index_.end(), "rejected-prefix victim must be in the heap");
    if (selected->second != ordered_.begin()) {
        const auto& boundary = *std::prev(selected->second);
        RTP_LLM_CHECK_WITH_INFO(rejected.count(boundary.node) != 0, "rejected-prefix boundary must be rejected");
        scope.emplace(*this, rejected);
        scope->fence_ = boundary;
    }
}

std::optional<EvictionEntry> EvictionHeap::bestForDrop(const std::function<bool(TreeNode*)>&     can_evict,
                                                       const std::unordered_set<TreeNode*>&      rejected,
                                                       const std::optional<RejectedPrefixScope>& batch_scope) {
    auto*       scope          = rejected_prefix_scope_;
    const auto* expected_scope = batch_scope ? &*batch_scope : nullptr;
    RTP_LLM_CHECK_WITH_INFO(scope == expected_scope && (scope == nullptr || &scope->rejected_ == &rejected),
                            "rejected-prefix scope does not match the eviction batch");
    if (scope == nullptr) {
        return best(can_evict);
    }
    auto&      fence     = scope->fence_;
    const auto start     = fence.has_value() ? ordered_.upper_bound(*fence) : ordered_.begin();
    const auto end       = ordered_.end();
    auto       candidate = start;
    for (; candidate != end; ++candidate) {
        if (can_evict(candidate->node)) {
            break;
        }
    }
    // The predicate records every rejection. This includes the last entry
    // examined when no candidate qualifies.
    if (candidate != start) {
        fence = *std::prev(candidate);
    }
    if (candidate == end) {
        return std::nullopt;
    }
    return *candidate;
}

bool EvictionHeap::contains(TreeNode* node) const {
    return index_.find(node) != index_.end();
}

size_t EvictionHeap::size() const {
    return ordered_.size();
}

}  // namespace rtp_llm
