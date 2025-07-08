#include <unordered_map>
#include <vector>
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void* IAllocator::mallocPrivate(size_t size) {
    throw std::runtime_error("mallocPrivate is not implemented in base IAllocator class. "
                             "CudaGraph should be used with TrackerAllocator.");
}

}  // namespace rtp_llm
