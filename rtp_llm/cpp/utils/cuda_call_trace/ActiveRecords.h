#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rtp_llm::cuda_trace::detail {

// A bounded live-state ledger, independent of every lossy history queue.
// Each returned handle has exactly one owner, which calls close() once. The
// writer pins an immutable entry while copying it. Closing never waits for the
// writer: the last reader reclaims a closed pinned entry. No non-atomic value
// is read concurrently with mutation (a seqlock alone would not guarantee this).
template<typename T, size_t Capacity>
class ActiveRecords {
    static_assert(Capacity > 0 && Capacity <= 4096 && (Capacity & (Capacity - 1)) == 0);
    static_assert(std::is_trivially_copyable<T>::value);
    static_assert(std::atomic<uint64_t>::is_always_lock_free);
    static constexpr uint64_t kIndexMask = 8191;
    static constexpr uint64_t kWriting = 1, kActive = 2, kReading = 4, kClosed = 8;
    static constexpr uint64_t kFlags = 15;
    struct Slot {
        std::atomic<uint64_t> state{0};
        T                     value{};
    };
    std::array<Slot, Capacity> slots_{};

public:
    // IDs come from the process recorder; zero is reserved. The low 13 bits
    // encode slot+1, or zero for an untracked call whose history is still useful.
    static uint64_t untracked(uint64_t id) noexcept {
        return id << 13;
    }
    static uint64_t id(uint64_t handle) noexcept {
        return handle >> 13;
    }

    uint64_t begin(const T& value, uint64_t id) noexcept {
        if (id == 0 || id > (UINT64_MAX >> 13)) {
            return 0;
        }
        // The caller never waits for a contended slot; bounded probing can fail
        // before every slot is occupied, which is reported as coverage loss.
        constexpr size_t probes = Capacity < 64 ? Capacity : 64;
        for (size_t i = 0; i != probes; ++i) {
            const size_t index    = (id + i) & (Capacity - 1);
            auto&        slot     = slots_[index];
            uint64_t     expected = 0;
            if (slot.state.compare_exchange_strong(expected, (id << 4) | kWriting, std::memory_order_acquire)) {
                slot.value = value;
                slot.state.store((id << 4) | kActive, std::memory_order_release);
                return untracked(id) | (index + 1);
            }
        }
        return 0;
    }

    bool close(uint64_t handle) noexcept {
        const uint64_t encoded = handle & kIndexMask;
        if (!encoded || encoded > Capacity) {
            return false;
        }
        auto&      slot   = slots_[encoded - 1];
        const auto before = slot.state.load(std::memory_order_acquire);
        if ((before & ~kFlags) != (id(handle) << 4) || !(before & kActive) || (before & kClosed)) {
            return false;
        }
        // The unique owner is the only closer. Until it sets CLOSED, only a
        // reader can change this state, and readers cannot release ownership.
        const auto old = slot.state.fetch_or(kClosed, std::memory_order_acq_rel);
        if (!(old & kReading)) {
            slot.state.store(0, std::memory_order_release);
        }
        return true;
    }

    bool contains(uint64_t handle) const noexcept {
        const uint64_t encoded = handle & kIndexMask;
        if (!encoded || encoded > Capacity) {
            return false;
        }
        const auto state = slots_[encoded - 1].state.load(std::memory_order_acquire);
        return (state & ~kFlags) == (id(handle) << 4) && (state & kActive) && !(state & kClosed);
    }

    template<typename Visitor>
    uint64_t snapshot(Visitor visitor) {
        uint64_t missed = 0;
        for (size_t index = 0; index != Capacity; ++index) {
            auto& slot  = slots_[index];
            auto  state = slot.state.load(std::memory_order_acquire);
            if (!state) {
                continue;
            }
            if ((state & kFlags) != kActive
                || !slot.state.compare_exchange_strong(state, state | kReading, std::memory_order_acquire)) {
                ++missed;
                continue;
            }
            const T    value = slot.value;
            const auto old   = slot.state.fetch_and(~kReading, std::memory_order_acq_rel);
            if (old & kClosed) {
                slot.state.store(0, std::memory_order_release);
            } else {
                // Pass a private copy after releasing the slot, never hold a
                // read pin while allocating, formatting JSON, or doing I/O.
                visitor(value, untracked(old >> 4) | (index + 1));
            }
        }
        return missed;
    }
};
}  // namespace rtp_llm::cuda_trace::detail
