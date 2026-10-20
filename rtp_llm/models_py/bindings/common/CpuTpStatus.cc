#include "rtp_llm/models_py/bindings/common/CpuTpStatus.h"
// Local TP2 boolean consensus experiment. No CUDA work and no model mutation.
// vLLM's actual shm_broadcast.py motivates host publication/consumption rather
// than transporting a host boolean to CUDA and synchronizing it back again.
// This specialized protocol uses C++ acquire/release atomics and two generations
// per rank. Advancing to g+2 requires seeing peer g+1, which proves the peer has
// consumed g. Thus neither rank can overwrite a generation still being read.

#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <new>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

static_assert(ATOMIC_LLONG_LOCK_FREE == 2, "Shared status requires lock-free 64-bit atomics");
static_assert(ATOMIC_INT_LOCK_FREE == 2, "Shared notification requires lock-free 32-bit atomics");

namespace {
struct alignas(64) Slot {
    std::atomic<uint64_t> publication{0};
    std::atomic<uint32_t> notification{0};
};
struct alignas(64) Status {
    Slot slots[2][2];
    alignas(64) std::atomic<uint32_t> broken{0};
};

void wake(Slot& slot) {
    slot.notification.fetch_add(1, std::memory_order_release);
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(&slot.notification), FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
}

void poison(Status& state) {
    state.broken.store(1, std::memory_order_release);
    for (auto& rank : state.slots) {
        for (auto& slot : rank) {
            wake(slot);
        }
    }
}

bool aligned(const void* pointer) {
    return pointer != nullptr && reinterpret_cast<uintptr_t>(pointer) % alignof(Status) == 0;
}
}  // namespace

extern "C" size_t cpu_status_size() {
    return sizeof(Status);
}

extern "C" int cpu_status_initialize(void* pointer, size_t bytes) {
    if (!aligned(pointer) || bytes < sizeof(Status)) {
        return -EINVAL;
    }
    new (pointer) Status();
    return 0;
}

extern "C" int cpu_status_break(void* pointer) {
    if (!aligned(pointer)) {
        return -EINVAL;
    }
    poison(*static_cast<Status*>(pointer));
    return 0;
}

extern "C" int cpu_status_reduce(void* pointer, int rank, uint64_t generation, int local_ok, int timeout_ms) {
    if (!aligned(pointer) || rank < 0 || rank > 1 || generation == 0 || generation >= (uint64_t{1} << 63)
        || (local_ok != 0 && local_ok != 1) || timeout_ms <= 0) {
        return -EINVAL;
    }
    auto& state = *static_cast<Status*>(pointer);
    if (state.broken.load(std::memory_order_acquire)) {
        return -ECANCELED;
    }
    auto&          own      = state.slots[rank][generation & 1];
    auto&          peer     = state.slots[rank ^ 1][generation & 1];
    const uint64_t previous = own.publication.load(std::memory_order_acquire) >> 1;
    const uint64_t required = generation <= 2 ? 0 : generation - 2;
    if (previous != required) {
        poison(state);
        return -EPROTO;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    own.publication.store((generation << 1) | static_cast<uint64_t>(local_ok), std::memory_order_release);
    wake(own);
    unsigned spins = 0;
    while (true) {
        // Sample the wake word BEFORE checking the authoritative publication.
        // FUTEX_WAIT checks it atomically, preventing a lost publication wakeup.
        const uint32_t notification = peer.notification.load(std::memory_order_acquire);
        const uint64_t value        = peer.publication.load(std::memory_order_acquire);
        if (state.broken.load(std::memory_order_acquire)) {
            return -ECANCELED;
        }
        if ((value >> 1) == generation) {
            return local_ok && (value & 1);
        }
        if ((value >> 1) > generation) {
            poison(state);
            return -EPROTO;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            poison(state);
            return -ETIMEDOUT;
        }
        if (++spins <= 256) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#else
            std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
            continue;
        }
        const timespec timeout{0, 1000000};
        const int      rc = syscall(
            SYS_futex, reinterpret_cast<uint32_t*>(&peer.notification), FUTEX_WAIT, notification, &timeout, nullptr, 0);
        if (rc < 0 && errno != EAGAIN && errno != EINTR && errno != ETIMEDOUT) {
            const int error = errno;
            poison(state);
            return -error;
        }
    }
}
