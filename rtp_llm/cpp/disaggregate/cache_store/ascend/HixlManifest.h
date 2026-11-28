// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlProtocol.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/proto/hixl_cache_store.pb.h"

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

// A grant outlives the source request and publication callback. Only the peer
// that received it may release it, after its READ has physically completed.
// Deadlines and logical request cancellation never expire a grant.
// Each reader supplies a fresh instance ID on restart. Drain remote readers
// before destroying this registry.
class HixlSourceGrants {
public:
    size_t size() const;

    // Reader-generated monotonically increasing IDs make cancellation possible
    // even when the manifest reply is lost.
    bool reserve(const std::string& reader, const std::string& engine, uint64_t transfer, uint64_t retired_through = 0);
    // Called after reserve succeeds; cancellation can retire it before publication.
    bool acquireReserved(const std::string&                               reader,
                         uint64_t                                         transfer,
                         const std::vector<std::shared_ptr<BlockBuffer>>& blocks);
    bool retire(const std::string& reader, const std::string& engine, uint64_t transfer, uint64_t retired_through = 0);
    std::vector<std::string> readerEngines() const;

private:
    struct Session {
        std::string        engine;
        uint64_t           retired{0};
        std::set<uint64_t> completed;
        // Empty until publication; a retired sequence is never inserted again.
        std::unordered_map<uint64_t, std::vector<std::shared_ptr<BlockBuffer>>> active;
    };
    // This is a protocol window, not a time-based expiry of DMA leases.
    Session*
    sessionLocked(const std::string& reader, const std::string& engine, uint64_t transfer, uint64_t retired_through);

    std::unordered_map<std::string, Session> sessions_;
    mutable std::mutex                       mutex_;
};

// Validate descriptors from a successful manifest before submitting DMA.
// The caller retains destinations through physical completion, and releases the source
// grant even if validation rejects the response (no DMA has started in that case).
std::vector<HixlTransferSlice> makeHixlReadPlan(const RequestBlockBuffer&           destination,
                                                const hixl_proto::ManifestResponse& manifest,
                                                MemoryUtil&                         memory);

}  // namespace rtp_llm
