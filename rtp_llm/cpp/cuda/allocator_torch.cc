#include "allocator_torch.h"

#include <cuda_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

Allocator<AllocatorType::TH>::Allocator():
    ICudaAllocator(0), pointer_mapping_(new std::unordered_map<void*, torch::Tensor>) {}

Allocator<AllocatorType::TH>::~Allocator() {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    while (!pointer_mapping_->empty()) {
        void* ptr = pointer_mapping_->begin()->second.data_ptr();
        free((void**)(&ptr));
    }
    pointer_mapping_->clear();
}

void Allocator<AllocatorType::TH>::free(void** ptr) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    void* address = *ptr;
    pointer_mapping_->erase(address);
    *ptr = nullptr;
    return;
}

void* Allocator<AllocatorType::TH>::malloc(size_t size) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    int64_t       buf_size = static_cast<int64_t>(ceil(size / 32.)) * 32;
    torch::Tensor buf;
    buf       = torch::empty({buf_size}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    void* ptr = buf.data_ptr();
    RTP_LLM_LOG_DEBUG("malloc buffer %p with size %ld", ptr, buf_size);
    pointer_mapping_->insert({ptr, buf});
    return ptr;
}

void* Allocator<AllocatorType::TH>::mallocSync(size_t size) {
    return malloc(size);
}

}  // namespace rtp_llm
