#include "src/fastertransformer/core/TrackerAllocator.h"

namespace fastertransformer {

TrakcerAllocator::TrakcerAllocator(const TrackerAllocatorParams& params)
    : real_allocator_(params.real_allocator)
{
    // try reserve memory for tracker
    auto real_reserve_size = params.target_track_bytes;
    while (true) {
        void* reserved_ptr = nullptr;
        try {
            reserved_ptr = real_allocator_->malloc(real_reserve_size, false);
        } catch (std::exception& e) {
            FT_LOG_WARNING("TrackerAllocator reserve %lu bytes of memory [%d] exception: %s",
                           real_reserve_size, real_allocator_->memoryType(), e.what());
        }
        if (reserved_ptr) {
            FT_LOG_INFO("TrackerAllocator successfully reserved %lu bytes (%lu MiB) of memory [%d]",
                        real_reserve_size, real_reserve_size / 1024L / 1024L, real_allocator_->memoryType());
            memory_tracker_.reset(new MemoryTracker(reserved_ptr, real_reserve_size, params.align_size));
            break;
        }
        auto next_reserve_size = real_reserve_size - params.bytes_try_step;
        if (next_reserve_size > 0) {
            FT_LOG_WARNING("TrackerAllocator failed to reserve %lu bytes of memory [%d], "
                           "next will try %lu bytes",
                           real_reserve_size, real_allocator_->memoryType(), next_reserve_size);
        } else {
            FT_LOG_ERROR("TrackerAllocator failed to reserve %lu bytes of memory [%d]. "
                         "Give up and use real allocator directly.",
                         real_reserve_size, real_allocator_->memoryType());
            break;
        }
        real_reserve_size = next_reserve_size;
    }
}

TrakcerAllocator::~TrakcerAllocator() {
    if (memory_tracker_) {
        auto ptr = memory_tracker_->getBasePtr();
        real_allocator_->free(&ptr);
        memory_tracker_.reset();
    }
}

AllocatorType TrakcerAllocator::type() const {
    return real_allocator_->type();
}

MemoryType TrakcerAllocator::memoryType() const {
    return real_allocator_->memoryType();
}

void* TrakcerAllocator::malloc(size_t size, const bool is_set_zero) {
    if (is_set_zero) {
        throw std::runtime_error("TrakcerAllocator does not support is_set_zero = true");
    }
    void* ptr = nullptr;
    if (memory_tracker_) {
        ptr = memory_tracker_->allocate(size);
    }
    if (!ptr) {
        FT_LOG_WARNING("TrackerAllocator failed to allocate %ld bytes of memory [%d]. "
                       "Use real allocator directly as fallback.",
                       size, real_allocator_->memoryType());
        ptr = real_allocator_->malloc(size, is_set_zero);
    }
    return ptr;
}

void TrakcerAllocator::free(void** ptr) {
    if (memory_tracker_ && (memory_tracker_->isTracking(*ptr))) {
        memory_tracker_->deallocate(*ptr);
    } else {
        real_allocator_->free(ptr);
    }
}

void* TrakcerAllocator::reMalloc(void* ptr, size_t size, const bool is_set_zero) {
    free(&ptr);
    return malloc(size, is_set_zero);
}

} // namespace fastertransformer
