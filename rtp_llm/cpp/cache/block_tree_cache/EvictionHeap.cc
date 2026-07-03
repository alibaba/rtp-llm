#include "rtp_llm/cpp/cache/block_tree_cache/EvictionHeap.h"

#include <chrono>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
uint64_t currentTimeNs() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
}
}  // namespace

bool EvictionHeap::EntryComparator::operator()(const EvictionEntry& a, const EvictionEntry& b) const {
    // std::priority_queue is a max-heap: the element for which the comparator
    // returns false against all others is at the top.
    // We want the WORST candidate (to evict first) at the top.
    // So return true if a is "better" (should stay, evict later) than b.
    switch (policy) {
        case EvictionPolicy::LRU:
            // Higher last_access_time = more recently used = better = evict later
            return a.last_access_time > b.last_access_time;
        case EvictionPolicy::LFU:
            // Higher hit_count = more frequently used = better = evict later
            return a.hit_count > b.hit_count;
        case EvictionPolicy::FIFO:
            // Lower insert_seq = inserted earlier = worse = evict first
            // So higher insert_seq = better = evict later
            return a.insert_seq > b.insert_seq;
        case EvictionPolicy::PRIORITY:
            // Higher priority = better = evict later
            return a.priority > b.priority;
    }
    return false;
}

EvictionHeap::EvictionHeap(EvictionPolicy policy): policy_(policy), heap_(EntryComparator{policy}) {}

void EvictionHeap::push(TreeNode* node, int group_id) {
    EvictionEntry entry;
    entry.node               = node;
    entry.component_group_id = group_id;
    entry.insert_seq         = insert_seq_counter_++;
    entry.last_access_time   = currentTimeNs();
    entry.hit_count          = 0;
    entry.priority           = 0;

    entry_map_[node] = entry;
    heap_.push(entry);
}

std::optional<EvictionEntry> EvictionHeap::pop() {
    while (!heap_.empty()) {
        EvictionEntry top = heap_.top();
        heap_.pop();

        // Check if this entry is still valid (not invalidated)
        auto it = entry_map_.find(top.node);
        if (it != entry_map_.end()) {
            // Verify it's the same entry (not a stale re-push)
            if (it->second.insert_seq == top.insert_seq) {
                entry_map_.erase(it);
                return top;
            }
            // Stale entry from a re-push, skip
        }
        // Invalidated entry, skip
    }
    return std::nullopt;
}

void EvictionHeap::invalidate(TreeNode* node) {
    entry_map_.erase(node);
    // The entry remains in heap_ but will be skipped on pop()
}

void EvictionHeap::onAccess(TreeNode* node) {
    auto it = entry_map_.find(node);
    if (it == entry_map_.end()) {
        return;
    }

    switch (policy_) {
        case EvictionPolicy::LRU: {
            // Update timestamp and re-push with new insert_seq (old entry becomes stale)
            it->second.last_access_time = currentTimeNs();
            it->second.insert_seq       = insert_seq_counter_++;
            heap_.push(it->second);
            break;
        }
        case EvictionPolicy::LFU: {
            // Increment hit count and re-push with new insert_seq (old entry becomes stale)
            it->second.hit_count++;
            it->second.insert_seq = insert_seq_counter_++;
            heap_.push(it->second);
            break;
        }
        case EvictionPolicy::FIFO:
        case EvictionPolicy::PRIORITY:
            // No-op: insert_seq and priority don't change on access
            break;
    }
}

bool EvictionHeap::contains(TreeNode* node) const {
    return entry_map_.find(node) != entry_map_.end();
}

bool EvictionHeap::empty() const {
    return entry_map_.empty();
}

size_t EvictionHeap::size() const {
    return entry_map_.size();
}

}  // namespace rtp_llm
