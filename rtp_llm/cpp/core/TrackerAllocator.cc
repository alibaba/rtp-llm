#include "rtp_llm/cpp/core/TrackerAllocator.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

TrackerAllocator::TrackerAllocator(const TrackerAllocatorParams& params): real_allocator_(params.real_allocator) {
    // try reserve memory for tracker
    auto real_reserve_size = params.target_track_bytes;
    if (real_reserve_size < 0) {
        throw std::invalid_argument("TrackerAllocator reserve bytes num must be non-negative but got %ld"
                                    + std::to_string(real_reserve_size));
    }
    if (real_reserve_size == 0) {
        RTP_LLM_LOG_WARNING("TrackerAllocator target_track_bytes is 0. Use real allocator directly.");
        return;
    }
    while (true) {
        void* reserved_ptr = nullptr;
        try {
            reserved_ptr = real_allocator_->mallocSync(real_reserve_size);
        } catch (std::exception& e) {
            RTP_LLM_LOG_WARNING("TrackerAllocator reserve %lu bytes of memory [%d] exception: %s",
                                real_reserve_size,
                                real_allocator_->memoryType(),
                                e.what());
        }
        if (reserved_ptr) {
            RTP_LLM_LOG_INFO(
                "TrackerAllocator successfully reserved %lu bytes (%lu MiB) of memory [%d], reserved base addr [%p]",
                real_reserve_size,
                real_reserve_size / 1024L / 1024L,
                real_allocator_->memoryType(),
                reserved_ptr);
            memory_tracker_.reset(new MemoryTracker(reserved_ptr, real_reserve_size, params.align_size));
            break;
        }
        auto next_reserve_size = real_reserve_size - params.bytes_try_step;
        if (next_reserve_size > 0) {
            RTP_LLM_LOG_WARNING("TrackerAllocator failed to reserve %lu bytes of memory [%d], "
                                "next will try %lu bytes",
                                real_reserve_size,
                                real_allocator_->memoryType(),
                                next_reserve_size);
        } else {
            RTP_LLM_LOG_ERROR("TrackerAllocator failed to reserve %lu bytes of memory [%d]. "
                              "Give up and use real allocator directly.",
                              real_reserve_size,
                              real_allocator_->memoryType());
            break;
        }
        real_reserve_size = next_reserve_size;
    }
}

TrackerAllocator::~TrackerAllocator() {
    if (memory_tracker_) {
        auto chunks = memory_tracker_->getAllChunks();
        for (auto chunk : chunks) {
            if (chunk->used) {
                RTP_LLM_LOG_WARNING("TrackerAllocator is destroyed with %lu bytes of memory [%d] still in use!",
                                    chunk->size,
                                    real_allocator_->memoryType());
                real_allocator_->free(&chunk->ptr);
            }
        }
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
        if (memory_tracker_) {
            const auto tracker_status = memory_tracker_->getStatus();
            RTP_LLM_CHECK_WITH_INFO(false,
                                    "TrackerAllocator failed to allocate %ld MB of memory [%d]. "
                                    "Current memory tracker has %ld MB available, with %ld MB fragmented. "
                                    "Reserved %ld MB in total.",
                                    size / 1024 / 1024,
                                    real_allocator_->memoryType(),
                                    tracker_status.available_size / 1024 / 1024,
                                    tracker_status.fragmented_size / 1024 / 1024,
                                    (tracker_status.available_size + tracker_status.allocated_size) / 1024 / 1024);
        } else {
            RTP_LLM_CHECK_WITH_INFO(false,
                                    "TrackerAllocator failed to allocate %ld MB of memory [%d].",
                                    size / 1024 / 1024,
                                    real_allocator_->memoryType());
        }
    }
    return ptr;
}

void* TrackerAllocator::mallocSync(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    if (memory_tracker_) {
        ptr = memory_tracker_->allocate(size);
    }
    if (!ptr) {
        RTP_LLM_CHECK_WITH_INFO(false,
                                "TrackerAllocator failed to allocate %ld bytes of memory [%d].",
                                size,
                                real_allocator_->memoryType());
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

void* TrackerAllocator::mallocPrivate(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    if (memory_tracker_) {
        ptr = memory_tracker_->allocatePrivate(size);
    }
    if (!ptr) {
        RTP_LLM_LOG_ERROR("TrackerAllocator failed to allocate %ld bytes of private memory [%d]. ",
                          size,
                          real_allocator_->memoryType());
        return nullptr;
    }
    return ptr;
}

TrackerStatus TrackerAllocator::getTrackerStatus() const {
    if (memory_tracker_) {
        return memory_tracker_->getStatus();
    }
    return TrackerStatus();
}

std::vector<MemoryChunk*> TrackerAllocator::getChunks() const {
    if (memory_tracker_) {
        return memory_tracker_->getAllChunks();
    }
    return {};
}

}  // namespace rtp_llm
