#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>

namespace rtp_llm {

// Logical blocks in source order: GPU prefix, memory segment, disk segment, remote suffix.
// Store the complete decision on the request and reuse it on allocation retries.
struct MockCacheHitDecision {
    size_t   eligible_blocks{0};
    size_t   planned_memory_blocks{0};
    size_t   planned_disk_blocks{0};
    uint64_t cumulative_eligible_blocks{0};
    uint64_t cumulative_planned_memory_blocks{0};
    uint64_t cumulative_planned_disk_blocks{0};

    // Append fields to preserve legacy positional aggregate initialization.
    size_t   planned_remote_blocks{0};
    uint64_t cumulative_planned_remote_blocks{0};

    // Bound to the owner admission, retained on every retry.
    uint64_t remote_server_epoch{0}, remote_session_id{0}, remote_generation{0};

    size_t gpuPrefixBlocks() const {
        return eligible_blocks - planned_memory_blocks - planned_disk_blocks - planned_remote_blocks;
    }

    // Frozen logical source offset for this admission, retained with the decision
    // across retries. Later admissions cannot change it.
    uint64_t admissionOffsetBlocks() const {
        return cumulative_eligible_blocks - eligible_blocks;
    }
};

struct MockCacheHitSnapshot {
    uint64_t cumulative_eligible_blocks{0};
    uint64_t cumulative_planned_memory_blocks{0};
    uint64_t cumulative_planned_disk_blocks{0};
    uint64_t cumulative_planned_remote_blocks{0};
};

class MockCacheHitController {
public:
    MockCacheHitController(double memory_ratio, double disk_ratio, double remote_ratio = 0.0);

    // Admit exactly once per request on the mutable-tree owner. Ratios have
    // 1e-6 resolution; allocate Remote, Disk, Memory deficits in that order, carrying debt.
    // The callback receives the remaining GPU PREFIX count, NOT a load-suffix
    // count. It must validate real matched references/model/CP alignment and
    // must not reenter this controller (the controller mutex is held).
    // Infeasibility expands the suffix into effective-positive memory, otherwise
    // effective-positive disk, otherwise effective-positive remote. Activation uses quantized/clamped ppm, not raw
    // doubles. All zero bypass validation and force no reads: the caller must
    // use ordinary loader routing when hasActiveRatios() is false.
    // Overflow, callback exceptions, or no feasible prefix leave counters intact.
    MockCacheHitDecision plan(size_t eligible_blocks, const std::function<bool(size_t)>& is_valid_gpu_prefix);
    MockCacheHitSnapshot snapshot() const;

    // Immutable configuration; safe to read without the admission mutex.
    uint64_t memoryRatioPpm() const {
        return memory_ratio_ppm_;
    }
    uint64_t diskRatioPpm() const {
        return disk_ratio_ppm_;
    }

    bool hasActiveRatios() const {
        return memory_ratio_ppm_ != 0 || disk_ratio_ppm_ != 0 || remote_ratio_ppm_ != 0;
    }

private:
    uint64_t             memory_ratio_ppm_{0};
    uint64_t             disk_ratio_ppm_{0};
    uint64_t             remote_ratio_ppm_{0};
    mutable std::mutex   mutex_;
    MockCacheHitSnapshot state_;
};

}  // namespace rtp_llm
