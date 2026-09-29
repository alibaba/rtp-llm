#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

// One engine owns exactly one bounded pool. A lease owns a slot until the
// synchronous staged copy (including D2H unpack) has completed.
class StagedCopyScratchPool {
public:
    struct Limits {
        size_t max_staging_bytes_per_device;
        size_t max_tiles_per_device;
    };

    struct Stats {
        size_t capacity{0};
        size_t in_use{0};
        size_t peak_in_use{0};
        size_t acquire_misses{0};
        size_t quarantined{0};
        bool   disabled{false};
    };

    class Lease {
    public:
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        ~Lease() noexcept;

        StagedMemoryCopyScratch& scratchFor(int device_index);
        void                     quarantine() noexcept;

    private:
        friend class StagedCopyScratchPool;
        Lease(StagedCopyScratchPool& pool, size_t index) noexcept: pool_(&pool), index_(index) {}
        void reset() noexcept;

        StagedCopyScratchPool* pool_{nullptr};
        size_t                 index_{0};
        int                    active_device_{-1};
        bool                   quarantined_{false};
    };

    enum class AcquireStatus { ACQUIRED, EXHAUSTED, DISABLED };
    struct AcquireResult {
        AcquireStatus        status;
        std::optional<Lease> lease;
    };

    StagedCopyScratchPool(size_t capacity, std::vector<int> allowed_devices, Limits limits);
    StagedCopyScratchPool(const StagedCopyScratchPool&) = delete;
    StagedCopyScratchPool& operator=(const StagedCopyScratchPool&) = delete;
    ~StagedCopyScratchPool() noexcept;

    AcquireResult tryAcquire();
    bool          allowsDevice(int device_index) const noexcept;
    size_t        capacity() const noexcept { return slots_.size(); }
    Limits        limits() const noexcept { return limits_; }
    Stats         stats() const noexcept;

private:
    struct ScratchOwner {
        StagedMemoryCopyScratch scratch;
        bool                    quarantined{false};
        ScratchOwner() = default;
        ScratchOwner(const ScratchOwner&) = delete;
        ScratchOwner& operator=(const ScratchOwner&) = delete;
        ~ScratchOwner() noexcept;
    };
    enum class SlotState { FREE, LEASED, QUARANTINED };
    struct Slot {
        std::map<int, std::unique_ptr<ScratchOwner>> by_device;
        SlotState                                    state{SlotState::FREE};
    };

    void release(size_t index, bool quarantine, int active_device) noexcept;

    mutable std::mutex                 mutex_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<size_t>                free_indices_;
    std::vector<int>                   allowed_devices_;
    Limits                             limits_;
    size_t                             in_use_{0};
    size_t                             peak_in_use_{0};
    size_t                             acquire_misses_{0};
    size_t                             quarantined_{0};
    bool                               disabled_{false};
};

}  // namespace rtp_llm
