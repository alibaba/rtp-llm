#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rtp_llm::cuda_trace::detail {

// Bounded MPMC sequence queue. Producers never wait for a consumer or a lock:
// contention beyond four CAS attempts is reported as a dropped record. A
// producer publishes only after copying the complete trivially-copyable record.
template<typename T, size_t Capacity>
class RecordQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "capacity must be a power of two");
    static_assert(std::is_trivially_copyable<T>::value, "record must be POD");
    static_assert(std::atomic<size_t>::is_always_lock_free, "callbacks require lock-free atomics");
    struct Cell {
        std::atomic<size_t> sequence;
        T                   value{};
    };

public:
    RecordQueue() noexcept {
        for (size_t i = 0; i < Capacity; ++i) {
            cells_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }
    bool push(const T& value) noexcept {
        size_t pos = enqueue_.load(std::memory_order_relaxed);
        for (unsigned attempt = 0; attempt != 4; ++attempt) {
            Cell&          cell       = cells_[pos & (Capacity - 1)];
            const size_t   sequence   = cell.sequence.load(std::memory_order_acquire);
            const intptr_t difference = static_cast<intptr_t>(sequence) - static_cast<intptr_t>(pos);
            if (difference == 0) {
                if (enqueue_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    cell.value = value;
                    cell.sequence.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (difference < 0) {
                return false;
            } else {
                pos = enqueue_.load(std::memory_order_relaxed);
            }
        }
        return false;
    }
    bool pop(T& value) noexcept {
        size_t       pos      = dequeue_.load(std::memory_order_relaxed);
        Cell&        cell     = cells_[pos & (Capacity - 1)];
        const size_t sequence = cell.sequence.load(std::memory_order_acquire);
        if (static_cast<intptr_t>(sequence) - static_cast<intptr_t>(pos + 1) != 0
            || !dequeue_.compare_exchange_strong(pos, pos + 1, std::memory_order_relaxed)) {
            return false;
        }
        value = cell.value;
        cell.sequence.store(pos + Capacity, std::memory_order_release);
        return true;
    }

private:
    std::array<Cell, Capacity> cells_{};
    alignas(64) std::atomic<size_t> enqueue_{0};
    alignas(64) std::atomic<size_t> dequeue_{0};
};

}  // namespace rtp_llm::cuda_trace::detail
