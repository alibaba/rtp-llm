// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0

#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlManifest.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace rtp_llm {

size_t HixlSourceGrants::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      count = 0;
    for (const auto& session : sessions_) {
        for (const auto& active : session.second.active) {
            count += !active.second.empty();
        }
    }
    return count;
}

HixlSourceGrants::Session* HixlSourceGrants::sessionLocked(const std::string& reader,
                                                           const std::string& engine,
                                                           uint64_t           transfer,
                                                           uint64_t           retired_through) {
    auto it = sessions_.find(reader);
    if (it == sessions_.end()) {
        if (transfer <= retired_through || transfer - retired_through > kHixlTransferWindow) {
            return nullptr;
        }
        // A restarted source has no grants. Resume from the reader's
        // acknowledged watermark, retaining lower in-flight IDs when
        // RPCs arrive out of order. Never advance an existing session.
        it = sessions_.emplace(reader, Session{engine, retired_through, {}, {}}).first;
    }
    auto& session = it->second;
    if (session.engine != engine || (transfer > session.retired && transfer - session.retired > kHixlTransferWindow)) {
        return nullptr;
    }
    return &session;
}

bool HixlSourceGrants::reserve(const std::string& reader,
                               const std::string& engine,
                               uint64_t           transfer,
                               uint64_t           retired_through) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto*                       session = sessionLocked(reader, engine, transfer, retired_through);
    return session && transfer > session->retired && session->completed.count(transfer) == 0
           && session->active.try_emplace(transfer).second;
}

bool HixlSourceGrants::acquireReserved(const std::string&                               reader,
                                       uint64_t                                         transfer,
                                       const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto&                       session = sessions_.at(reader);
    auto                        active  = session.active.find(transfer);
    if (active == session.active.end()) {
        return false;
    }
    active->second = blocks;
    return true;
}

bool HixlSourceGrants::retire(const std::string& reader,
                              const std::string& engine,
                              uint64_t           transfer,
                              uint64_t           retired_through) {
    // Allocator release may acquire other locks. Destroy pages outside our mutex.
    std::vector<std::shared_ptr<BlockBuffer>> retired;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto*                       session = sessionLocked(reader, engine, transfer, retired_through);
        if (!session) {
            return false;
        }
        if (transfer <= session->retired || session->completed.count(transfer)) {
            return true;
        }
        auto it = session->active.find(transfer);
        if (it != session->active.end()) {
            retired = std::move(it->second);
            session->active.erase(it);
        }
        session->completed.insert(transfer);
        while (session->completed.erase(session->retired + 1)) {
            ++session->retired;
        }
    }
    return true;
}

std::vector<std::string> HixlSourceGrants::readerEngines() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::set<std::string>       unique;
    for (const auto& item : sessions_) {
        unique.insert(item.second.engine);
    }
    return {unique.begin(), unique.end()};
}

namespace {

void validateRegion(const hixl_proto::Block& block) {
    const auto limit = std::numeric_limits<uintptr_t>::max();
    const auto base  = block.region_address();
    const auto bytes = block.region_bytes();
    if (!block.region_id() || !base || !bytes || bytes > limit - base || block.address() < base
        || block.address() - base >= bytes || !block.bytes() || block.bytes() > bytes - (block.address() - base)) {
        throw std::invalid_argument("HIXL manifest block lies outside its registered region");
    }
}

void checkRegionIdentity(const hixl_proto::Block& block, std::unordered_map<uint64_t, HixlMemoryRegion>& regions) {
    const auto region = HixlMemoryRegion{block.region_address(), block.region_bytes(), block.region_id()};
    const auto result = regions.emplace(region.id, region);
    if (!result.second
        && (result.first->second.address != region.address || result.first->second.bytes != region.bytes)) {
        throw std::invalid_argument("HIXL manifest reuses a registration ID for different regions");
    }
}

}  // namespace

std::vector<HixlTransferSlice> makeHixlReadPlan(const RequestBlockBuffer&           destination,
                                                const hixl_proto::ManifestResponse& manifest,
                                                MemoryUtil&                         memory) {
    if (manifest.engine_id().empty() || manifest.source_instance().empty()) {
        throw std::invalid_argument("HIXL manifest requires a source engine and instance");
    }
    auto blocks = destination.getBlocks();
    if (blocks.size() != static_cast<size_t>(manifest.blocks_size())) {
        throw std::invalid_argument("HIXL manifest block count does not match destination");
    }
    std::unordered_map<uint64_t, HixlMemoryRegion> regions;
    std::vector<HixlTransferSlice>                 slices;
    slices.reserve(blocks.size());
    for (const auto& remote : manifest.blocks()) {
        auto it = blocks.find(remote.key());
        if (it == blocks.end() || remote.bytes() != it->second->len) {
            throw std::invalid_argument("HIXL manifest key/length mismatch");
        }
        validateRegion(remote);
        checkRegionIdentity(remote, regions);
        const auto& local   = *it->second;
        const auto  address = reinterpret_cast<uintptr_t>(local.addr.get());
        if (!memory.isMemoryMr(local.addr.get(), local.len, local.gpu_mem, local.adopted)) {
            throw std::invalid_argument("HIXL read destination is not registered");
        }
        slices.push_back({address, remote.address(), local.len});
        blocks.erase(it);
    }
    // Core adapter preserves one descriptor per cache block.
    return slices;
}

}  // namespace rtp_llm
