#include "src/fastertransformer/devices/BufferManager.h"

namespace fastertransformer {

BufferManager::BufferManager(IAllocator* deviceAllocator, IAllocator* hostAllocator)
    : deviceAllocator_(deviceAllocator)
    , hostAllocator_(hostAllocator)
{}

BufferManager::~BufferManager() {}

Tensor BufferManager::allocate(const BufferParams& params, const BufferHints& hints) {
    auto allocator = (params.allocation == AllocationType::DEVICE) ? deviceAllocator_ : hostAllocator_;
    return Tensor(allocator, params.type, params.dims);
}

} // namespace fastertransformer

