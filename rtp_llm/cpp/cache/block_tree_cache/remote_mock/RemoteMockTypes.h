#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <map>
#include <vector>
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockState.h"

namespace rtp_llm {

// Immutable per-rank source identity. No local Host/Disk index or tree reference.
struct RemoteMockSourceHandle {
    RemoteMockSession session;
    uint64_t          layout_hash{0};
    uint32_t          rank{0}, group{0}, pool{0}, index{0}, checksum{0};
    uint64_t          admission_offset{0}, logical_index{0};
};
struct RemoteMockWriteTarget {
    RemoteMockSession session;
    uint64_t          record_index{0};
};
struct RemoteMockWritebackStats {
    // Per-rank physical batches, never whole-request logical counters.
    uint64_t submitted{0}, completed{0}, failed{0}, rejected{0};
    uint64_t acknowledged_payload_bytes{0};
    // Fixed backing-window reservations, including admitted queued captures.
    uint64_t retained_bytes{0}, peak_retained_bytes{0};
};
// Owner logical requests; physical rank/batch counters remain separate.
struct RemoteMockCompletionStats {
    uint64_t submitted{0}, completed{0}, failed{0}, rejected{0};
    uint64_t pending_requests{0}, pending_intents{0}, retained_descriptors{0};
};
struct RemoteMockLoadTarget {
    RemoteMockSourceHandle    source;
    std::vector<BlockIdxType> device_blocks;
};
struct RemoteMockLoadPlan {
    std::vector<RemoteMockLoadTarget>     targets;
    std::chrono::steady_clock::time_point deadline;
    // The caller freezes and pins request GPU targets before handing off the plan.
    // This owner is held through I/O, CRC, CUDA and the terminal callback.
    std::shared_ptr<void> target_owner;
};
struct RemoteMockHostSlice {
    uint32_t              group{0};
    uint64_t              record_index{0}, offset{0};
    std::shared_ptr<void> bytes;
    uint32_t              length{0};
};
struct RemoteMockHostTransfer {
    std::vector<RemoteMockHostSlice>      slices;
    std::chrono::steady_clock::time_point deadline;
    // Total backing allocation retained by the slice owners, counted once per
    // allocation, even when only a small range is sent. Must cover all slices.
    uint64_t retained_bytes{0};
};
// A transport fault invalidates this engine lifetime; there is no in-place recovery.
struct RemoteMockSafetySnapshot {
    bool   faulted{false};
    size_t quarantined_blocks{0};  // Distinct physical owner GPU sources/targets.
};
struct RemoteMockAdapterStats {
    uint64_t read_submitted{0}, read_completed{0}, read_failed{0};
    uint64_t read_payload_bytes{0};  // Existing load callback copied these application bytes.
};
struct RemoteMockRankCredits {
    uint64_t local_read_bytes{0}, local_write_bytes{0};
    uint64_t service_read_bytes{0}, service_write_bytes{0};
    uint32_t read_operations{0}, write_operations{0};
    uint32_t first_write_slot{0}, write_slots{0};
    uint32_t max_slice_bytes{0}, max_batch_records{0}, max_write_slice_bytes{0};
};
// Immutable startup capability and negotiated capacity, only when Remote exists.
// Layout records omit per-member offsets; layout_hash identifies the full manifest.
struct RemoteMockRuntimeSnapshot {
    std::map<std::string, std::string> options;
    std::string                        callback_endpoint;
    RemoteMockSession                  session;
    uint64_t                           layout_hash{0};
    uint32_t                           rank{0}, expected_ranks{0};
    RemoteMockLimits                   limits;
    RemoteMockRankCredits              credits;
    std::vector<RemoteMockLayout>      layouts;
};
}  // namespace rtp_llm
