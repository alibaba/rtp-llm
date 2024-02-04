#pragma once
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/Tensor.h"

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
    DataType type;
    std::vector<size_t> dims;
    AllocationType allocation;
};

class BufferManager {
public:
    BufferManager(IAllocator* deviceAllocator, IAllocator* hostAllocator);
    ~BufferManager();

    Tensor allocate(const BufferParams& params, const BufferHints& hints);

private:
    IAllocator* deviceAllocator_;
    IAllocator* hostAllocator_;
};

} // namespace fastertransformer

