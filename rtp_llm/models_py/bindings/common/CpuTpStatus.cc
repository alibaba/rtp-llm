#include "rtp_llm/models_py/bindings/common/CpuTpStatus.h"
// Local TP boolean consensus. No CUDA work and no model mutation.
// vLLM's actual shm_broadcast.py motivates host publication/consumption rather
// than transporting a host boolean to CUDA and synchronizing it back again.
// This specialized protocol uses C++ acquire/release atomics and two generations
// per rank. Advancing to g+2 requires seeing every peer's g+1, which proves
// every peer has consumed g. No rank can overwrite a generation still being read.

#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <time.h>
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
    std::atomic<uint32_t> broken{0};
    int                   world_size;
    size_t                broadcast_capacity;
    size_t                published_bytes = 0;
    Slot                  accept_lengths_ready;

    Slot& slot(int rank, uint64_t generation) {
        return reinterpret_cast<Slot*>(this + 1)[2 * rank + (generation & 1)];
    }

    void* broadcastBuffer() {
        return reinterpret_cast<Slot*>(this + 1) + 2 * world_size;
    }
};

constexpr int kMaxRanks = 256;

void wake(Slot& slot) {
    slot.notification.fetch_add(1, std::memory_order_release);
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(&slot.notification), FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
}

void poison(Status& state) {
    state.broken.store(1, std::memory_order_release);
    wake(state.accept_lengths_ready);
    for (int rank = 0; rank < state.world_size; ++rank) {
        for (int generation = 0; generation < 2; ++generation) {
            wake(state.slot(rank, generation));
        }
    }
}

bool aligned(const void* pointer) {
    return pointer != nullptr && reinterpret_cast<uintptr_t>(pointer) % alignof(Status) == 0;
}

int waitForPublication(Status& state, Slot& peer, uint64_t generation, std::chrono::steady_clock::time_point deadline) {
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
            return value & 1;
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
        const int      rc = syscall(SYS_futex,
                                   reinterpret_cast<uint32_t*>(&peer.notification),
                                   FUTEX_WAIT,
                                   notification,
                                   &timeout,
                                   nullptr,
                                   0);
        if (rc < 0 && errno != EAGAIN && errno != EINTR && errno != ETIMEDOUT) {
            const int error = errno;
            poison(state);
            return -error;
        }
    }
}
}  // namespace

extern "C" size_t cpu_status_size(int world_size) {
    if (world_size < 2 || world_size > kMaxRanks) {
        return 0;
    }
    return sizeof(Status) + 2 * static_cast<size_t>(world_size) * sizeof(Slot);
}

extern "C" int cpu_status_initialize(void* pointer, size_t bytes, int world_size) {
    const auto required = cpu_status_size(world_size);
    if (!aligned(pointer) || required == 0 || bytes < required) {
        return -EINVAL;
    }
    auto& state              = *new (pointer) Status();
    state.world_size         = world_size;
    state.broadcast_capacity = bytes - required;
    for (int rank = 0; rank < world_size; ++rank) {
        for (int generation = 0; generation < 2; ++generation) {
            new (&state.slot(rank, generation)) Slot();
        }
    }
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
    if (!aligned(pointer) || rank < 0 || generation == 0 || generation >= (uint64_t{1} << 63)
        || (local_ok != 0 && local_ok != 1) || timeout_ms <= 0) {
        return -EINVAL;
    }
    auto& state = *static_cast<Status*>(pointer);
    if (state.world_size < 2 || state.world_size > kMaxRanks || rank >= state.world_size) {
        return -EINVAL;
    }
    if (state.broken.load(std::memory_order_acquire)) {
        return -ECANCELED;
    }
    auto&          own      = state.slot(rank, generation);
    const uint64_t previous = own.publication.load(std::memory_order_acquire) >> 1;
    const uint64_t required = generation <= 2 ? 0 : generation - 2;
    if (previous != required) {
        poison(state);
        return -EPROTO;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    own.publication.store((generation << 1) | static_cast<uint64_t>(local_ok), std::memory_order_release);
    wake(own);
    bool global_ok = local_ok;
    for (int peer_rank = 0; peer_rank < state.world_size; ++peer_rank) {
        if (peer_rank == rank) {
            continue;
        }
        const int status = waitForPublication(state, state.slot(peer_rank, generation), generation, deadline);
        if (status < 0) {
            return status;
        }
        global_ok = global_ok && status;
    }
    return global_ok;
}

extern "C" int
cpu_status_broadcast(void* pointer, int rank, uint64_t generation, void* payload, size_t bytes, int timeout_ms) {
    if (!aligned(pointer) || rank < 0 || generation == 0 || generation >= (uint64_t{1} << 63)
        || payload == nullptr || bytes == 0 || timeout_ms <= 0) {
        return -EINVAL;
    }
    auto& state = *static_cast<Status*>(pointer);
    if (rank >= state.world_size || bytes > state.broadcast_capacity) {
        return -EINVAL;
    }
    if (state.broken.load(std::memory_order_acquire)) {
        return -ECANCELED;
    }
    auto& slot = state.accept_lengths_ready;
    if (rank == 0) {
        // The caller must have synchronized all readers of the previous chunk.
        std::memcpy(state.broadcastBuffer(), payload, bytes);
        state.published_bytes = bytes;
        slot.publication.store(generation << 1, std::memory_order_release);
        wake(slot);
        return 0;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    const int status    = waitForPublication(state, slot, generation, deadline);
    if (status < 0) {
        return status;
    }
    if (state.published_bytes != bytes) {
        poison(state);
        return -EPROTO;
    }
    std::memcpy(payload, state.broadcastBuffer(), bytes);
    return 0;
}
