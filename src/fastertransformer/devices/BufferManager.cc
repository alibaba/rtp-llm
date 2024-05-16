#include "src/fastertransformer/devices/BufferManager.h"

#include <numeric>
#include <mutex>

using namespace std;
using ReadLock = shared_lock<shared_mutex>;
using WriteLock = unique_lock<shared_mutex>;

namespace fastertransformer {

BufferManager::BufferManager(IAllocator* device_allocator, IAllocator* host_allocator)
    : device_allocator_(device_allocator)
    , host_allocator_(host_allocator)
{}

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
    WriteLock lock(mutex_);
    AllocationRecord record = {params.allocation, true, buffer->sizeBytes(), hints};
    FT_LOG_DEBUG("record allocation: %p, size: %zu, tag: [%s]",
                 buffer->data(), buffer->sizeBytes(), hints.tag.c_str());
    allocation_records_[buffer->data()] = record;
}

void BufferManager::recordRecycle(Buffer* buffer) {
    WriteLock lock(mutex_);
    FT_LOG_DEBUG("record recycle: %p [%s]",
        buffer->data(), allocation_records_[buffer->data()].hints.tag.c_str());
    allocation_records_.erase(buffer->data());
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
    return move(status);
}

} // namespace fastertransformer

