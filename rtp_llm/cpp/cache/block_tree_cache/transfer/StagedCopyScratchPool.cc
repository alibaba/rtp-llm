#include "rtp_llm/cpp/cache/block_tree_cache/transfer/StagedCopyScratchPool.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

StagedCopyScratchPool::ScratchOwner::~ScratchOwner() noexcept {
    // A failed stream synchronization gives no proof that the GPU has stopped
    // using these pointers. Leave them to CUDA context/process teardown.
    if (!quarantined) {
        releaseStagedMemoryCopyScratch(scratch);
    }
}

StagedCopyScratchPool::StagedCopyScratchPool(size_t capacity, std::vector<int> allowed_devices, Limits limits):
    allowed_devices_(std::move(allowed_devices)), limits_(limits) {
    if (capacity == 0 || limits.max_staging_bytes_per_device == 0 || limits.max_tiles_per_device == 0) {
        throw std::invalid_argument("staged scratch pool capacity and limits must be positive");
    }
    std::sort(allowed_devices_.begin(), allowed_devices_.end());
    allowed_devices_.erase(std::unique(allowed_devices_.begin(), allowed_devices_.end()), allowed_devices_.end());
    if (std::any_of(allowed_devices_.begin(), allowed_devices_.end(), [](int device) { return device < 0; })) {
        throw std::invalid_argument("staged scratch pool device index must be nonnegative");
    }
    slots_.reserve(capacity);
    free_indices_.reserve(capacity);
    for (size_t index = 0; index < capacity; ++index) {
        auto slot = std::make_unique<Slot>();
        for (int device : allowed_devices_) {
            slot->by_device.emplace(device, std::make_unique<ScratchOwner>());
        }
        slots_.push_back(std::move(slot));
        free_indices_.push_back(index);
    }
}

StagedCopyScratchPool::~StagedCopyScratchPool() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(in_use_ == 0 && "staged scratch pool destroyed with outstanding leases");
    if (quarantined_ != 0) {
        RTP_LLM_LOG_WARNING("staged scratch pool closing with %zu quarantined slots", quarantined_);
    }
}

StagedCopyScratchPool::AcquireResult StagedCopyScratchPool::tryAcquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (disabled_) {
        return {AcquireStatus::DISABLED, std::nullopt};
    }
    if (free_indices_.empty()) {
        ++acquire_misses_;
        return {AcquireStatus::EXHAUSTED, std::nullopt};
    }
    const size_t index = free_indices_.back();
    free_indices_.pop_back();
    assert(slots_[index]->state == SlotState::FREE);
    slots_[index]->state = SlotState::LEASED;
    peak_in_use_ = std::max(peak_in_use_, ++in_use_);
    return {AcquireStatus::ACQUIRED, std::optional<Lease>(Lease(*this, index))};
}

bool StagedCopyScratchPool::allowsDevice(int device_index) const noexcept {
    return std::binary_search(allowed_devices_.begin(), allowed_devices_.end(), device_index);
}

StagedCopyScratchPool::Stats StagedCopyScratchPool::stats() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return {slots_.size(), in_use_, peak_in_use_, acquire_misses_, quarantined_, disabled_};
}

void StagedCopyScratchPool::release(size_t index, bool quarantine, int active_device) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(index < slots_.size() && slots_[index]->state == SlotState::LEASED && in_use_ > 0);
    --in_use_;
    if (quarantine) {
        slots_[index]->state = SlotState::QUARANTINED;
        if (active_device >= 0) {
            slots_[index]->by_device.at(active_device)->quarantined = true;
        }
        ++quarantined_;
        disabled_ = true;
        RTP_LLM_LOG_WARNING("staged scratch pool disabled after an unconfirmed CUDA completion");
    } else {
        slots_[index]->state = SlotState::FREE;
        free_indices_.push_back(index);
    }
}

StagedCopyScratchPool::Lease::Lease(Lease&& other) noexcept:
    pool_(std::exchange(other.pool_, nullptr)),
    index_(other.index_),
    active_device_(other.active_device_),
    quarantined_(other.quarantined_) {}

StagedCopyScratchPool::Lease& StagedCopyScratchPool::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        reset();
        pool_        = std::exchange(other.pool_, nullptr);
        index_       = other.index_;
        active_device_ = other.active_device_;
        quarantined_ = other.quarantined_;
    }
    return *this;
}

StagedCopyScratchPool::Lease::~Lease() noexcept {
    reset();
}

void StagedCopyScratchPool::Lease::reset() noexcept {
    if (pool_ != nullptr) {
        pool_->release(index_, quarantined_, active_device_);
        pool_ = nullptr;
    }
}

StagedMemoryCopyScratch& StagedCopyScratchPool::Lease::scratchFor(int device_index) {
    if (pool_ == nullptr) {
        throw std::logic_error("invalid staged scratch lease");
    }
    auto& scratch = pool_->slots_[index_]->by_device.at(device_index)->scratch;
    active_device_ = device_index;
    return scratch;
}

void StagedCopyScratchPool::Lease::quarantine() noexcept {
    quarantined_ = true;
}

}  // namespace rtp_llm
