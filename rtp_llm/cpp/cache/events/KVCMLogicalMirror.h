#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/cache/events/KVCacheEvent.h"

namespace rtp_llm::detail {

// Single-owner logical view of the keys KVCM should contain. The publisher
// worker is its only caller after startup, so synchronization stays at the
// bounded ingress queue instead of being duplicated around reconciliation.
class KVCMLogicalMirror {
public:
    explicit KVCMLogicalMirror(size_t max_keys): max_keys_(max_keys) {}

    // Replace the mirror from an authoritative cache snapshot. This is used
    // both before the worker starts and after recoverable ingress loss. The
    // replacement has a strong exception guarantee and false means the
    // supplied snapshot exceeds the configured resource ceiling.
    bool seed(const KVCacheSnapshot& snapshot);
    // Apply one drained ingress batch. Earlier transitions remain applied if
    // a later ADD crosses the ceiling; the publisher treats false as terminal
    // and never uses that partial mirror as a remote baseline.
    bool            apply(const std::vector<KVCacheEvent>& events);
    KVCacheSnapshot snapshot() const;
    void            release() noexcept;

    // Produce at most max_batch_size transitions that transform two sorted
    // snapshots. Indices are advanced in place, allowing a large diff to be
    // streamed without allocating a second source+target-sized vector.
    static std::vector<KVCacheEvent> nextMutationBatch(const std::vector<int64_t>& source_keys,
                                                       const std::vector<int64_t>& target_keys,
                                                       size_t&                     source_index,
                                                       size_t&                     target_index,
                                                       size_t                      max_batch_size);

private:
    size_t                      max_keys_;
    std::unordered_set<int64_t> keys_;
};

}  // namespace rtp_llm::detail
