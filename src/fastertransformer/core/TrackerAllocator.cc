#include "src/fastertransformer/core/TrackerAllocator.h"

namespace fastertransformer {

TrackerAllocator::TrackerAllocator(const TrackerAllocatorParams& params)
    : real_allocator_(params.real_allocator)
{
    // try reserve memory for tracker
    auto real_reserve_size = params.target_track_bytes;
    if (real_reserve_size == 0) {
        FT_LOG_WARNING("TrackerAllocator target_track_bytes is 0. Use real allocator directly.");
        return;
    }
    while (true) {
        void* reserved_ptr = nullptr;
        try {
            reserved_ptr = real_allocator_->malloc(real_reserve_size);
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

TrackerAllocator::~TrackerAllocator() {
    if (memory_tracker_) {
        auto ptr = memory_tracker_->getBasePtr();
        real_allocator_->free(&ptr);
        memory_tracker_.reset();
    }
    delete real_allocator_;
}

AllocatorType TrackerAllocator::type() const {
    return real_allocator_->type();
}

MemoryType TrackerAllocator::memoryType() const {
    return real_allocator_->memoryType();
}

void* TrackerAllocator::malloc(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    if (memory_tracker_) {
        ptr = memory_tracker_->allocate(size);
    }
    if (!ptr) {
        FT_LOG_WARNING("TrackerAllocator failed to allocate %ld bytes of memory [%d]. "
                       "Use real allocator directly as fallback.",
                       size, real_allocator_->memoryType());
        ptr = real_allocator_->malloc(size);
    }
    return ptr;
}

void TrackerAllocator::free(void** ptr) {
    if (!ptr || !*ptr) {
        return;
    }
    if (memory_tracker_ && (memory_tracker_->isTracking(*ptr))) {
        memory_tracker_->deallocate(*ptr);
    } else {
        real_allocator_->free(ptr);
    }
    *ptr = nullptr;
}

void* TrackerAllocator::reMalloc(void* ptr, size_t size) {
    free(&ptr);
    return malloc(size);
}

TrackerStatus TrackerAllocator::getTrackerStatus() const {
    if (memory_tracker_) {
        return memory_tracker_->getStatus();
    }
    return TrackerStatus();
}

} // namespace fastertransformer
