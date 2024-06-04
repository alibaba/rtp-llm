#include "src/fastertransformer/devices/BufferManager.h"
#include "src/fastertransformer/core/TrackerAllocator.h"
#include "autil/StackTracer.h"

#include <numeric>
#include <mutex>
#include <unistd.h>

using namespace std;
using ReadLock = shared_lock<shared_mutex>;
using WriteLock = unique_lock<shared_mutex>;

namespace fastertransformer {

BufferManager::BufferManager(IAllocator* device_allocator, IAllocator* host_allocator)
    : device_allocator_(device_allocator)
    , host_allocator_(host_allocator)
    , device_max_allocated_bytes_(0)
    , trace_memory_(getenv("RTP_LLM_TRACE_MEMORY"))
    , trace_malloc_stack_(getenv("RTP_LLM_TRACE_MALLOC_STACK"))
{
    if (trace_memory_) {
        autil::EnvUtil::setEnv("STACK_TRACER_LOG", "true");
        DECLARE_STACK_TRACER_FILE("rtp_llm_stack.log");
    } else if (trace_malloc_stack_) {
        throw std::runtime_error("RTP_LLM_TRACE_MALLOC_STACK must be used with RTP_LLM_TRACE_MALLOC_STACK");
    }
}

BufferManager::~BufferManager() {}

BufferPtr BufferManager::allocate(const BufferParams& params, const BufferHints& hints) {
    auto buffer = doAllocate(params, hints);
    recordAllcation(params, hints, buffer);
    return move(buffer);
}

void BufferManager::recycle(Buffer* buffer, IAllocator* allocator) {
    recordRecycle(buffer);
    doRecycle(buffer, allocator);
}

BufferPtr BufferManager::doAllocate(const BufferParams& params, const BufferHints& hints) {
    const auto allocator = (params.allocation == AllocationType::DEVICE) ? device_allocator_ : host_allocator_;
    const auto shape = params.dims;
    const auto alloc_bytes = accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>())
                           * getTypeSize(params.type);
    const auto data = allocator->malloc(alloc_bytes);
    const auto deleter = [this, allocator](Buffer* buffer) { this->recycle(buffer, allocator); };
    const auto buffer = new Buffer(allocator->memoryType(), params.type, shape, data, deleter);
    return BufferPtr(buffer);
}

void BufferManager::doRecycle(Buffer* buffer, IAllocator* allocator) {
    void* data = buffer->data();
    allocator->free(&data);
}

void BufferManager::recordAllcation(const BufferParams& params, const BufferHints& hints, const BufferPtr& buffer) {
    if (trace_memory_) {
        auto stack_trace_id = autil::StackTracer::getInstance()->getTraceId();
        if (trace_malloc_stack_) {
            FT_LOG_INFO("record allocation: %p, size: %zu, tag: [%s], trace id [%lu]",
                        buffer->data(), buffer->sizeBytes(), hints.tag.c_str(), stack_trace_id);
        }
        if (auto tracker_allocator_ = dynamic_cast<TrackerAllocator*>(device_allocator_)) {
            auto tracker_status = tracker_allocator_->getTrackerStatus();
            if (tracker_status.allocated_size > device_max_allocated_bytes_) {
                FT_LOG_INFO("Device allocated size or fragmented size reached new maximum, \n"
                            "previous is %zu bytes, stack trace id[%lu]\n  %s",
                            device_max_allocated_bytes_, stack_trace_id,
                            tracker_status.toString().c_str());
                device_max_allocated_bytes_ = tracker_status.allocated_size;
            }
        }
    }
    {
        WriteLock lock(mutex_);
        AllocationRecord record = {params.allocation, true, buffer->sizeBytes(), hints};
        allocation_records_[buffer->data()] = record;
    }
}

void BufferManager::recordRecycle(Buffer* buffer) {
    if (trace_memory_) {
        FT_LOG_DEBUG("record recycle: %p [%s]",
                     buffer->data(), allocation_records_[buffer->data()].hints.tag.c_str());
    }
    {
        WriteLock lock(mutex_);
        allocation_records_.erase(buffer->data());
    }
}

BufferStatus BufferManager::queryStatus() {
    auto status = BufferStatus();
    ReadLock lock(mutex_);
    for (const auto& [_, record] : allocation_records_) {
        if (record.allocation_type == AllocationType::HOST) {
            status.host_allocated_bytes += record.bytes;
        } else {
            status.device_allocated_bytes += record.bytes;
        }
    }
    if (auto tracker_allocator_ = dynamic_cast<TrackerAllocator*>(device_allocator_)) {
        const auto tracker_status = tracker_allocator_->getTrackerStatus();
        status.device_preserved_bytes = tracker_status.available_size;
        status.device_fragmented_bytes = tracker_status.fragmented_size;
    }
    return move(status);
}

} // namespace fastertransformer

