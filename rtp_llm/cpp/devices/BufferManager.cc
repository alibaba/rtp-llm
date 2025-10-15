#include "rtp_llm/cpp/devices/BufferManager.h"
#include "rtp_llm/cpp/core/TrackerAllocator.h"
#include "rtp_llm/cpp/utils/StackTrace.h"
#include "autil/StackTracer.h"

#include <numeric>
#include <mutex>
#include <unistd.h>
#include "rtp_llm/cpp/config/ConfigModules.h"
using namespace std;
using ReadLock  = shared_lock<shared_mutex>;
using WriteLock = unique_lock<shared_mutex>;

namespace rtp_llm {

BufferManager::BufferManager(IAllocator*                        device_allocator,
                             IAllocator*                        host_allocator,
                             const ProfilingDebugLoggingConfig& config):
    device_allocator_(device_allocator),
    host_allocator_(host_allocator),
    trace_memory_(config.trace_memory),
    trace_malloc_stack_(config.trace_malloc_stack),
    profiling_debug_logging_config_(config) {
    if (trace_memory_) {
        autil::EnvUtil::setEnv("STACK_TRACER_LOG", "true");
        DECLARE_STACK_TRACER_FILE("rtp_llm_stack.log");
    } else if (trace_malloc_stack_) {
        throw std::runtime_error("RTP_LLM_TRACE_MALLOC_STACK must be used with RTP_LLM_TRACE_MALLOC_STACK");
    }
}

BufferManager::~BufferManager() {}

BufferPtr BufferManager::allocate(const BufferParams& params, const BufferHints& hints) {
    try {
        auto buffer = doAllocate(params, hints);
        recordAllcation(params, hints, buffer);
        return buffer;
    } catch (std::exception& e) {
        RTP_LLM_STACKTRACE_LOG_INFO(
            "allocate buffer failed: size %lu, exception: %s, current allocation records:\n%s \n stack traces: ",
            params.sizeInBytes(),
            e.what(),
            printAllocationRecords(device_allocator_).c_str());
        printStackTrace();
        throw;
    }
}

void BufferManager::recycle(Buffer* buffer, IAllocator* allocator) {
    auto data = buffer->data();

    if (recycle_held_) {
        RTP_LLM_LOG_DEBUG("hold recycle buffer: %p [%s][%s]",
                          data,
                          allocation_records_[data].hints.tag.c_str(),
                          buffer->debugString().c_str());
        held_data_.push_back(std::make_pair(data, allocator));
        return;
    }

    recordRecycle(data);
    doRecycle(data, allocator);
}

BufferPtr BufferManager::doAllocate(const BufferParams& params, const BufferHints& hints) {
    const auto allocator = (params.allocation == AllocationType::DEVICE) ? device_allocator_ : host_allocator_;
    const auto shape     = params.dims;
    const auto alloc_bytes =
        accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>()) * getTypeSize(params.type);

    if (params.vmem_ctl == VmemCtl::ForcePhysical) {
        if (auto vmem_allocator = dynamic_cast<IVirtualMemAllocator*>(allocator)) {
            const auto data    = vmem_allocator->mallocPhysical(alloc_bytes);
            const auto deleter = [this, vmem_allocator](Buffer* buffer) { this->recycle(buffer, vmem_allocator); };
            const auto buffer  = new Buffer(vmem_allocator->memoryType(), params.type, shape, data, deleter);
            return BufferPtr(buffer);
        }
        // if allocator is not virtual memory allocator, fallback to default logic.
    }

    const auto data    = params.private_alloc ? allocator->mallocPrivate(alloc_bytes) : allocator->malloc(alloc_bytes);
    const auto deleter = [this, allocator](Buffer* buffer) { this->recycle(buffer, allocator); };
    const auto buffer  = new Buffer(allocator->memoryType(), params.type, shape, data, deleter);
    return BufferPtr(buffer);
}

void BufferManager::doRecycle(void* data, IAllocator* allocator) {
    allocator->free(&data);
}

void BufferManager::setTraceMemory(bool trace_memory) {
    trace_memory_ = trace_memory;
}

void BufferManager::recordAllcation(const BufferParams& params, const BufferHints& hints, const BufferPtr& buffer) {
    if (trace_memory_) {
        auto stack_trace_id = trace_malloc_stack_ ? autil::StackTracer::getInstance()->getTraceId() : 0;
        {
            WriteLock        lock(mutex_);
            AllocationRecord record             = {params.allocation, buffer->sizeBytes(), hints, stack_trace_id};
            allocation_records_[buffer->data()] = record;
        }
        RTP_LLM_LOG_INFO("record allocation: %p, size: %zu, tag: [%s], trace id [%lu]",
                         buffer->data(),
                         buffer->sizeBytes(),
                         hints.tag.c_str(),
                         stack_trace_id);
        auto       status                = queryStatus();
        const auto device_consumed_bytes = status.device_allocated_bytes + status.device_fragmented_bytes;
        if (device_consumed_bytes > device_max_consumed_bytes_) {
            RTP_LLM_LOG_INFO("Device allocated size + fragmented size reached new maximum %zu, \n"
                             "previous is %zu bytes, current stack trace id[%lu]\n  %s",
                             device_consumed_bytes,
                             device_max_allocated_bytes_,
                             stack_trace_id,
                             printAllocationRecords(device_allocator_).c_str());
            device_max_consumed_bytes_ = device_consumed_bytes;
        }
        if (status.device_allocated_bytes > device_max_allocated_bytes_) {
            device_max_allocated_bytes_ = status.device_allocated_bytes;
        }
    }
}

void BufferManager::recordRecycle(void* data) {
    if (trace_memory_) {
        {
            WriteLock lock(mutex_);
            allocation_records_.erase(data);
        }
        RTP_LLM_LOG_DEBUG("record recycle: %p [%s]", data, allocation_records_[data].hints.tag.c_str());
    }
}

BufferStatus BufferManager::queryStatus() {
    auto     status = BufferStatus();
    ReadLock lock(mutex_);
    for (const auto& [_, record] : allocation_records_) {
        if (record.allocation_type == AllocationType::HOST) {
            status.host_allocated_bytes += record.bytes;
        } else {
            status.device_allocated_bytes += record.bytes;
        }
    }
    if (auto tracker_allocator_ = dynamic_cast<TrackerAllocator*>(device_allocator_)) {
        const auto tracker_status        = tracker_allocator_->getTrackerStatus();
        status.device_preserved_bytes    = tracker_status.available_size;
        status.device_fragmented_bytes   = tracker_status.fragmented_size;
        status.device_freezed_bytes      = tracker_status.freezed_bytes;
        status.device_max_consumed_bytes = device_max_consumed_bytes_;
    }
    return status;
}

void BufferManager::holdRecycle() {
    if (recycle_held_) {
        throw std::runtime_error("last buffer manager recycle hold is not released");
    }
    recycle_held_ = true;
}

void BufferManager::releaseRecycleHold() {
    if (!recycle_held_) {
        throw std::runtime_error("buffer manager recycle is not held");
    }
    for (const auto& data_pair : held_data_) {
        auto data      = data_pair.first;
        auto allocator = data_pair.second;
        RTP_LLM_LOG_DEBUG("release held buffer data %p [%s]", data, allocation_records_[data].hints.tag.c_str());
        recordRecycle(data);
        doRecycle(data, allocator);
    }
    held_data_.clear();
    recycle_held_ = false;
}

string BufferManager::printAllocationRecords(IAllocator* allocator) {
    if (auto tracker_allocator = dynamic_cast<TrackerAllocator*>(allocator)) {
        auto               tracker_status = tracker_allocator->getTrackerStatus();
        std::ostringstream info;
        std::set<void*>    allocated_ptrs;
        info << "Memory Tracker [" << (int32_t)tracker_allocator->type() << "] Status:\n";
        info << "allocated " << tracker_status.allocated_chunk_count
             << " chunks, size: " << tracker_status.allocated_size << "\n"
             << "available " << tracker_status.available_size << " bytes, with " << tracker_status.fragment_chunk_count
             << " fragments of size: " << tracker_status.fragmented_size << "\n";
        info << "--------------------------------------------------------------------------\n";
        info << "|        ADDR |         size (     hex) | AVAIL| TRACE|              TAG |\n";
        info << "--------------------------------------------------------------------------\n";
        {
            ReadLock lock(mutex_);
            for (const auto& [ptr, record] : allocation_records_) {
                allocated_ptrs.insert(ptr);
            }
            for (const auto chunk : tracker_status.chunks) {
                info << "| " << chunk.ptr << " | " << setw(12) << chunk.size << " (" << std::setw(8) << std::hex
                     << chunk.size << std::dec << ")"
                     << " | " << (chunk.used ? "USED" : "FREE");
                const auto alloc_record = allocation_records_.find(chunk.ptr);
                if (alloc_record != allocation_records_.end()) {
                    allocated_ptrs.erase(chunk.ptr);
                    info << " | " << setw(4) << alloc_record->second.trace_id << " | " << setw(16)
                         << alloc_record->second.hints.tag.c_str() << " |\n";
                } else {
                    info << " |      |                  |\n";
                }
            }
            info << "--------------------------------------------------------------------------\n";
            if (allocated_ptrs.size()) {
                info << "There are also " << allocated_ptrs.size() << " buffers allocated but not tracked, "
                     << "they are not shown in the list: \n";
                info << "--------------------------------------------------------------------------\n";
                for (const auto ptr : allocated_ptrs) {
                    const auto alloc_record = allocation_records_.find(ptr);
                    info << "| " << ptr << " | " << setw(12) << alloc_record->second.bytes << " (" << std::setw(8)
                         << std::hex << alloc_record->second.bytes << std::dec << ")"
                         << " |     "
                         << " | " << setw(4) << alloc_record->second.trace_id << " | " << setw(16)
                         << alloc_record->second.hints.tag.c_str() << " |\n";
                }
                info << "--------------------------------------------------------------------------\n";
            }
        }
        return info.str();
    } else {
        RTP_LLM_LOG_WARNING("BufferManager::printAllocationRecords is only effective when using TrackerAllocator!");
        return "";
    }
}

}  // namespace rtp_llm
