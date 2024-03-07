#include "src/fastertransformer/devices/BufferManager.h"

#include <numeric>

using namespace std;

namespace fastertransformer {

BufferManager::BufferManager(IAllocator* device_allocator, IAllocator* host_allocator)
    : device_allocator_(device_allocator)
    , host_allocator_(host_allocator)
{}

BufferManager::~BufferManager() {}

unique_ptr<Buffer> BufferManager::allocate(const BufferParams& params, const BufferHints& hints) {
    const auto allocator = (params.allocation == AllocationType::DEVICE) ? device_allocator_ : host_allocator_;
    const auto shape = params.dims;
    const auto alloc_bytes = accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>())
                           * getTypeSize(params.type);
    const auto data = allocator->malloc(alloc_bytes);
    const auto deleter = [this, allocator](Buffer* buffer) { this->recycle(buffer, allocator); };
    const auto buffer = new Buffer(allocator->memoryType(), params.type, shape, data, deleter);
    return unique_ptr<Buffer>(buffer);
}

void BufferManager::recycle(Buffer* buffer, IAllocator* allocator) {
    void* data = buffer->data();
    allocator->free(&data);
}

} // namespace fastertransformer

