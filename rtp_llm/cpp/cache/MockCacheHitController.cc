#include "rtp_llm/cpp/cache/MockCacheHitController.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rtp_llm {
namespace {
constexpr uint64_t kRatioScale = 1'000'000;
uint64_t           deficit(uint64_t eligible, uint64_t ratio, uint64_t planned) {
    const auto target = static_cast<uint64_t>((static_cast<unsigned __int128>(eligible) * ratio) / kRatioScale);
    return target > planned ? target - planned : 0;
}
}  // namespace

MockCacheHitController::MockCacheHitController(double memory_ratio, double disk_ratio, double remote_ratio) {
    if (!std::isfinite(memory_ratio) || !std::isfinite(disk_ratio) || !std::isfinite(remote_ratio) || memory_ratio < 0
        || disk_ratio < 0 || remote_ratio < 0 || memory_ratio > 1 || disk_ratio > 1 || remote_ratio > 1
        || memory_ratio + disk_ratio + remote_ratio > 1) {
        throw std::invalid_argument("mock cache active ratios must be finite in [0,1] with sum <= 1");
    }
    // Clamp in allocation-priority order so independently rounded half-ppm
    // values cannot produce an aggregate target above one.
    remote_ratio_ppm_ = static_cast<uint64_t>(std::llround(remote_ratio * kRatioScale));
    disk_ratio_ppm_ =
        std::min(static_cast<uint64_t>(std::llround(disk_ratio * kRatioScale)), kRatioScale - remote_ratio_ppm_);
    memory_ratio_ppm_ = std::min(static_cast<uint64_t>(std::llround(memory_ratio * kRatioScale)),
                                 kRatioScale - remote_ratio_ppm_ - disk_ratio_ppm_);
}

MockCacheHitDecision MockCacheHitController::plan(size_t                             eligible_blocks,
                                                  const std::function<bool(size_t)>& is_valid_gpu_prefix) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (eligible_blocks > std::numeric_limits<uint64_t>::max() - state_.cumulative_eligible_blocks) {
        throw std::overflow_error("mock cache cumulative eligible blocks overflow");
    }
    const auto eligible_next = state_.cumulative_eligible_blocks + eligible_blocks;
    size_t     remote        = std::min<uint64_t>(
        eligible_blocks, deficit(eligible_next, remote_ratio_ppm_, state_.cumulative_planned_remote_blocks));
    size_t disk = std::min<uint64_t>(eligible_blocks - remote,
                                     deficit(eligible_next, disk_ratio_ppm_, state_.cumulative_planned_disk_blocks));
    size_t memory =
        std::min<uint64_t>(eligible_blocks - remote - disk,
                           deficit(eligible_next, memory_ratio_ppm_, state_.cumulative_planned_memory_blocks));
    if (hasActiveRatios()) {
        size_t gpu_prefix = eligible_blocks - memory - disk - remote;
        while (!is_valid_gpu_prefix(gpu_prefix)) {
            if (gpu_prefix == 0) {
                throw std::invalid_argument("mock cache planning has no valid GPU prefix");
            }
            --gpu_prefix;
            if (memory_ratio_ppm_ != 0) {
                ++memory;
            } else if (disk_ratio_ppm_ != 0) {
                ++disk;
            } else {
                ++remote;
            }
        }
    }
    // Each tier's cumulative count is bounded by cumulative eligible blocks.
    state_.cumulative_eligible_blocks = eligible_next;
    state_.cumulative_planned_memory_blocks += memory;
    state_.cumulative_planned_disk_blocks += disk;
    state_.cumulative_planned_remote_blocks += remote;
    return {eligible_blocks,
            memory,
            disk,
            eligible_next,
            state_.cumulative_planned_memory_blocks,
            state_.cumulative_planned_disk_blocks,
            remote,
            state_.cumulative_planned_remote_blocks};
}

MockCacheHitSnapshot MockCacheHitController::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

}  // namespace rtp_llm
