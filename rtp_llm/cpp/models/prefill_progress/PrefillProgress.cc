#include "rtp_llm/cpp/models/prefill_progress/PrefillProgress.h"

#include <atomic>
#include <endian.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace rtp_llm {
namespace {

// Little-endian slot format: phase[3:0], layer+1[13:4], TID[35:14], sequence[63:36].
// An all-zero slot is idle. Zero in the layer field means unspecified.
constexpr unsigned kLayerShift    = 4;
constexpr unsigned kTidShift      = 14;
constexpr unsigned kSequenceShift = 36;
constexpr uint64_t kLayerMask     = (uint64_t{1} << 10) - 1;
constexpr uint64_t kTidMask       = (uint64_t{1} << 22) - 1;
constexpr uint64_t kSequenceMask  = (uint64_t{1} << 28) - 1;
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr));

uint64_t* mapProgressSlot(const char* path) noexcept {
    // The parent creates the file before spawning the rank. Never create or resize it here.
    const int fd = open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) {
        return nullptr;
    }
    struct stat info{};
    void*       mapping = MAP_FAILED;
    if (fstat(fd, &info) == 0 && S_ISREG(info.st_mode) && info.st_size == sizeof(uint64_t)) {
        mapping = mmap(nullptr, sizeof(uint64_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }
    close(fd);
    return mapping == MAP_FAILED ? nullptr : static_cast<uint64_t*>(mapping);
}

thread_local PrefillProgressScope* active_scope = nullptr;

}  // namespace

struct PrefillProgressSlot {
    ~PrefillProgressSlot() {
        if (auto* mapped = record.load()) {
            munmap(mapped, sizeof(*mapped));
        }
    }

    std::atomic<uint64_t*> record{nullptr};
    std::atomic_flag       occupied = ATOMIC_FLAG_INIT;
    uint64_t               sequence = 0;
};

namespace {

PrefillProgressSlot& progressSlot() noexcept {
    static PrefillProgressSlot slot;
    return slot;
}

}  // namespace

void PrefillProgressScope::initialize(const char* slot_path) noexcept {
    if (!slot_path || !*slot_path || enabled()) {
        return;
    }
    if (auto* mapped = mapProgressSlot(slot_path)) {
        uint64_t* empty = nullptr;
        if (!progressSlot().record.compare_exchange_strong(empty, mapped)) {
            munmap(mapped, sizeof(*mapped));
        }
    }
}

PrefillProgressScope::PrefillProgressScope(bool eligible) noexcept {
    if (!eligible || active_scope) {
        return;
    }
    auto& slot = progressSlot();
    if (!slot.record.load()) {
        return;
    }
    const thread_local long tid = syscall(SYS_gettid);
    if (tid <= 0 || static_cast<uint64_t>(tid) > kTidMask || slot.occupied.test_and_set(std::memory_order_acquire)) {
        return;
    }

    thread_id_bits_ = static_cast<uint64_t>(tid) << kTidShift;
    slot_           = &slot;
    active_scope    = this;
    publish(PrefillPhase::PREPARE);
}

PrefillProgressScope::~PrefillProgressScope() {
    if (!slot_) {
        return;
    }
    __atomic_store_n(slot_->record.load(std::memory_order_relaxed), uint64_t{0}, __ATOMIC_RELEASE);
    active_scope = nullptr;
    slot_->occupied.clear(std::memory_order_release);
}

bool PrefillProgressScope::enabled() noexcept {
    return progressSlot().record.load() != nullptr;
}

bool PrefillProgressScope::active() noexcept {
    return active_scope != nullptr;
}

void PrefillProgressScope::mark(PrefillPhase phase) noexcept {
    if (active_scope) {
        active_scope->publish(phase);
    }
}

void PrefillProgressScope::markLayer(int layer_index) noexcept {
    if (active_scope && layer_index >= -1 && layer_index < static_cast<int>(kLayerMask)) {
        active_scope->publish(PrefillPhase::LAYER, static_cast<uint64_t>(layer_index + 1));
    }
}

void PrefillProgressScope::publish(PrefillPhase phase, uint64_t layer) noexcept {
    slot_->sequence = (slot_->sequence + 1) & kSequenceMask;
    const uint64_t encoded =
        (slot_->sequence << kSequenceShift) | thread_id_bits_ | (layer << kLayerShift) | static_cast<uint64_t>(phase);
    __atomic_store_n(slot_->record.load(std::memory_order_relaxed), htole64(encoded), __ATOMIC_RELEASE);
}

}  // namespace rtp_llm
