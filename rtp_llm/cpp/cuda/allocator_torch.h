#pragma once
#include "rtp_llm/cpp/cuda/allocator_cuda.h"
#include "torch/extension.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <unordered_map>
#include <vector>

namespace rtp_llm {

template<>
class Allocator<AllocatorType::TH>: public ICudaAllocator, public TypedAllocator<AllocatorType::TH> {
    bool isExist(void* address) const {
        return pointer_mapping_->count(address) > 0;
    }
    ReallocType isReMalloc(void* address, size_t size) const {
        RTP_LLM_CHECK(isExist(address));
        size_t current_buffer_size = 1;
        for (int i = 0; i < pointer_mapping_->at(address).dim(); i++) {
            current_buffer_size *= pointer_mapping_->at(address).size(i);
        }
        RTP_LLM_LOG_DEBUG(
            "current_buffer_size: %d, original buffer: %p, new buffer: %d", current_buffer_size, address, size);
        if (current_buffer_size < size) {
            return ReallocType::INCREASE;
        } else if (current_buffer_size == size) {
            return ReallocType::REUSE;
        } else {
            return ReallocType::DECREASE;
        }
    }

public:
    Allocator();

    void* malloc(size_t size);
    void* mallocSync(size_t size);
    void  free(void** ptr);

    virtual ~Allocator();

private:
    std::unique_ptr<std::unordered_map<void*, torch::Tensor>> pointer_mapping_;
};

}  // namespace rtp_llm
