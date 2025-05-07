#pragma once
#include "maga_transformer/cpp/core/allocator.h"
#include "maga_transformer/cpp/core/Buffer.h"

#include <string>
#include <numeric>
#include <functional>
#include <shared_mutex>

namespace rtp_llm {

struct BufferParams {
    BufferParams(DataType type, const std::vector<size_t>& dims,
                 AllocationType allocation = AllocationType::DEVICE)
    : type(type), dims(dims), allocation(allocation) {}

    // for allocating pure buffer space
    BufferParams(const std::vector<size_t>& dims,
                 AllocationType allocation = AllocationType::DEVICE)
    : type(DataType::TYPE_UINT8), dims(dims), allocation(allocation) {}
    BufferParams(const size_t size_bytes,
                 AllocationType allocation = AllocationType::DEVICE)
    : type(DataType::TYPE_UINT8), dims({size_bytes}), allocation(allocation) {}

    size_t sizeInBytes() const {
        if (dims.empty()) {
            return 0;
        }
        auto size = std::accumulate(dims.begin(), dims.end(), (size_t)1, std::multiplies<size_t>());
        return size * getTypeSize(type);
    }

    DataType type;
    std::vector<size_t> dims;
    AllocationType allocation;
};

struct BufferStatus {
    size_t device_free_bytes          = 0;
    size_t host_allocated_bytes       = 0;
    size_t device_allocated_bytes     = 0;
    size_t device_preserved_bytes     = 0;
    size_t device_fragmented_bytes    = 0;
    size_t device_max_consumed_bytes  = 0;
};

struct AllocationRecord {
    AllocationType allocation_type;
    size_t bytes;
    BufferHints hints;
    size_t trace_id;
};

class BufferManager {
public:
    BufferManager(IAllocator* device_allocator, IAllocator* host_allocator);
    virtual ~BufferManager();

public:
    BufferPtr allocate(const BufferParams& params, const BufferHints& hints);
    void recycle(Buffer* buffer, IAllocator* allocator);
    void setTraceMemory(bool trace_memory);
    virtual BufferStatus queryStatus();
    std::string printAllocationRecords(IAllocator* allocator);

private:
    virtual BufferPtr doAllocate(const BufferParams& params, const BufferHints& hints);
    virtual void doRecycle(Buffer* buffer, IAllocator* allocator);
    void recordAllcation(const BufferParams& params, const BufferHints& hints, const BufferPtr& buffer);
    void recordRecycle(Buffer* buffer);

private:
    IAllocator* device_allocator_;
    IAllocator* host_allocator_;

    std::unordered_map<void*, AllocationRecord> allocation_records_;
    std::shared_mutex mutex_;

    // these two values are only recorded when trace_memory_ is true
    size_t device_max_allocated_bytes_     = 0;
    size_t device_max_consumed_bytes_      = 0; // this includes allocaetd bytes + fragmented bytes
    bool trace_memory_;
    const bool trace_malloc_stack_;
};

} // namespace rtp_llm
