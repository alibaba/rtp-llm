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
    auto stack_trace_id = trace_malloc_stack_ ? autil::StackTracer::getInstance()->getTraceId() : 0;
    {
        WriteLock lock(mutex_);
        AllocationRecord record = {params.allocation, buffer->sizeBytes(), hints, stack_trace_id};
        allocation_records_[buffer->data()] = record;
    }
    if (trace_memory_) {
        FT_LOG_DEBUG("record allocation: %p, size: %zu, tag: [%s], trace id [%lu]",
                     buffer->data(), buffer->sizeBytes(), hints.tag.c_str(), stack_trace_id);
        auto status = queryStatus();
        if (status.device_allocated_bytes > device_max_allocated_bytes_) {
            FT_LOG_INFO("Device allocated size or fragmented size reached new maximum %zu, \n"
                        "previous is %zu bytes, current stack trace id[%lu]\n  %s",
                        status.device_allocated_bytes,
                        device_max_allocated_bytes_, stack_trace_id,
                        printAllocationRecords(device_allocator_).c_str());
            device_max_allocated_bytes_ = status.device_allocated_bytes;
        }
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

string BufferManager::printAllocationRecords(IAllocator* allocator) {
    if (auto tracker_allocator = dynamic_cast<TrackerAllocator*>(allocator)) {
        auto tracker_status = tracker_allocator->getTrackerStatus();
        std::ostringstream info;
        info << "Memory Tracker [" << (int32_t)tracker_allocator->type() << "] Status:\n";
        info << "allocated " << tracker_status.allocated_chunk_count
             << " chunks, size: " << tracker_status.allocated_size << "\n"
             << "available " << tracker_status.available_size
             << " bytes, with " << tracker_status.fragment_chunk_count
             << " fragments of size: " << tracker_status.fragmented_size << "\n";
        info << "--------------------------------------------------------------------------\n";
        info << "|        ADDR |         size (     hex) | AVAIL| TRACE|              TAG |\n";
        info << "--------------------------------------------------------------------------\n";
        {
            ReadLock lock(mutex_);
            for (const auto chunk: tracker_status.chunks) {
                info << "| " << chunk.ptr
                     << " | " << setw(12) << chunk.size
                     << " (" << std::setw(8) << std::hex << chunk.size << std::dec << ")"
                     << " | " << (chunk.used ? "USED" : "FREE");
                const auto alloc_record = allocation_records_.find(chunk.ptr);
                if (alloc_record != allocation_records_.end()) {
                    info << " | " << setw(4) << alloc_record->second.trace_id
                         << " | " << setw(16) << alloc_record->second.hints.tag.c_str()
                         << " |\n";
                } else {
                    info << " |      |                  |\n";
                }
            }
        }
        info << "--------------------------------------------------------------------------\n";
        return info.str();
    } else {
        FT_LOG_WARNING("BufferManager::printAllocationRecords is only effective when using TrackerAllocator!");
        return "";
    }
}

} // namespace fastertransformer

