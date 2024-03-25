#pragma once
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/Buffer.h"

#include <string>

namespace fastertransformer {

enum class AllocationType {
    HOST   = 0,
    DEVICE = 1,
};

enum class BufferLifecycleType {
    SHORT,
    LONG
};

enum class SpaceComplexityType {
    UNKNOWN,
    CONSTANT,
    LINEAR,
    QUADRATIC
};

struct BufferHints {
    std::string tag;
    BufferLifecycleType lifecycle;
    SpaceComplexityType space_complexity;
};

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

    DataType type;
    std::vector<size_t> dims;
    AllocationType allocation;
};

class BufferManager {
public:
    BufferManager(IAllocator* device_allocator, IAllocator* host_allocator);
    ~BufferManager();

    std::unique_ptr<Buffer> allocate(const BufferParams& params, const BufferHints& hints);
    void recycle(Buffer* buffer, IAllocator* allocator);

private:
    IAllocator* device_allocator_;
    IAllocator* host_allocator_;
};

} // namespace fastertransformer

