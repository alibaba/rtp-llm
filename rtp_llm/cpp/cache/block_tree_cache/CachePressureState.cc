#include "rtp_llm/cpp/cache/block_tree_cache/CachePressureState.h"

namespace rtp_llm {

void CachePressureState::setHostOwner(PressureOwner owner, std::optional<PressureBlock> block) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        previous = owners_.find(owner);
    if (previous != owners_.end() && block && previous->second == *block) {
        return;
    }
    if (previous != owners_.end()) {
        const auto old   = previous->second;
        auto       count = owners_per_block_.find(old);
        if (--count->second == 0) {
            --pools_[old.first].published_host_blocks;
            owners_per_block_.erase(count);
        }
        owners_.erase(previous);
    }
    if (block) {
        if (owners_per_block_[*block]++ == 0) {
            ++pools_[block->first].published_host_blocks;
        }
        owners_.emplace(owner, *block);
    }
}

PoolEvidence CachePressureState::pool(const std::string& pool_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  found = pools_.find(pool_id);
    return found == pools_.end() ? PoolEvidence{} : found->second;
}

void CachePressureState::recordCopy(const PressureTransferKey& key, uint64_t blocks, uint64_t bytes, bool success) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto&                       evidence = transfers_[key];
    evidence.copied_blocks += blocks;
    evidence.copied_bytes += bytes;
    if (!success) {
        ++evidence.physical_failed_operations;
    }
}

void CachePressureState::recordWorkflowFailure(const PressureTransferKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++transfers_[key].failed_operations;
}

void CachePressureState::recordPublication(const PressureTransferKey& key, uint64_t blocks, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto&                       evidence = transfers_[key];
    evidence.published_blocks += blocks;
    evidence.published_bytes += bytes;
}

TransferEvidence CachePressureState::transfer(const PressureTransferKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  found = transfers_.find(key);
    return found == transfers_.end() ? TransferEvidence{} : found->second;
}

void CachePressureState::recordWatermarkTrigger(size_t group_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++triggers_[group_id];
}

uint64_t CachePressureState::watermarkTriggers(size_t group_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  found = triggers_.find(group_id);
    return found == triggers_.end() ? 0 : found->second;
}

}  // namespace rtp_llm
