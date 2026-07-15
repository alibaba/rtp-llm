#include "rtp_llm/cpp/cache/block_tree_cache/EvictionHeap.h"

#include <functional>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

bool EvictionHeap::EntryLess::operator()(const EvictionEntry& a, const EvictionEntry& b) const {
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
    // Replace any existing entry so at most one entry per node remains.
    erase(node);
    auto [ordered_it, inserted] = ordered_.insert(makeEntry(node, meta));
    RTP_LLM_CHECK_WITH_INFO(
        inserted, "EvictionHeap::upsert failed to insert a unique entry for node=%p", static_cast<void*>(node));
    auto index_result = index_.emplace(node, ordered_it);
    if (!index_result.second) {
        ordered_.erase(ordered_it);
        RTP_LLM_FAIL("EvictionHeap::upsert failed to index node=%p", static_cast<void*>(node));
    }
}

void EvictionHeap::erase(TreeNode* node) {
    auto it = index_.find(node);
    if (it == index_.end())
        return;
    ordered_.erase(it->second);
    index_.erase(it);
}

std::optional<EvictionEntry> EvictionHeap::takeBest() {
    if (ordered_.empty())
        return std::nullopt;
    auto          best  = ordered_.begin();
    EvictionEntry entry = *best;
    index_.erase(entry.node);
    ordered_.erase(best);
    return entry;
}

bool EvictionHeap::contains(TreeNode* node) const {
    return index_.find(node) != index_.end();
}

size_t EvictionHeap::size() const {
    return ordered_.size();
}

}  // namespace rtp_llm
