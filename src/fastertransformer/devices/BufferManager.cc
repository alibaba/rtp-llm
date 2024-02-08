#include "src/fastertransformer/devices/BufferManager.h"

namespace fastertransformer {

BufferManager::BufferManager(IAllocator* device_allocator, IAllocator* host_allocator)
    : device_allocator_(device_allocator)
    , host_allocator_(host_allocator)
{}

BufferManager::~BufferManager() {}

Tensor BufferManager::allocate(const BufferParams& params, const BufferHints& hints) {
    const auto allocator = (params.allocation == AllocationType::DEVICE) ? device_allocator_ : host_allocator_;
    const auto shape = params.dims;
    const auto alloc_bytes = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>())
                            * Tensor::getTypeSize(params.type);
    const auto data = allocator->malloc(alloc_bytes);
    const auto deleter = [this, allocator](Tensor& tensor) { this->recycle(tensor, allocator); };
    return Tensor(allocator->memoryType(), params.type, params.dims, data, deleter);
}

void BufferManager::recycle(Tensor& tensor, IAllocator* allocator) {
    allocator->free(tensor.dataPtr());
}

} // namespace fastertransformer

