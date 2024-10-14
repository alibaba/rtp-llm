#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"

#include <cstring>
#include <cstdlib>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, MemoryUtil);

MemoryUtil::MemoryUtil(std::unique_ptr<MemoryUtilBase> impl): instance_(std::move(impl)) {}

bool MemoryUtil::rdmaMode() {
    return getInstance().isRdmaMode();
}

MemoryUtilBase& MemoryUtil::getInstance() {
    return *instance_;
}

void* MemoryUtil::mallocCPU(size_t size) {
    return malloc(size);
}

void MemoryUtil::freeCPU(void* ptr) {
    free(ptr);
}

void* MemoryUtil::mallocGPU(size_t size) {
    void* ptr = nullptr;
    if (cudaMallocHost(&ptr, size) != cudaSuccess) {
        AUTIL_LOG(WARN, "cuda malloc host failed, size %lu", size);
        return nullptr;
    }
    return ptr;
}

void MemoryUtil::freeGPU(void* ptr) {
    if (cudaFreeHost(ptr) != cudaSuccess) {
        AUTIL_LOG(WARN, "cuda free host failed");
    }
}

bool MemoryUtil::regUserMr(void* buf, uint64_t size, bool gpu) {
    return getInstance().regUserMr(buf, size, gpu);
}

bool MemoryUtil::deregUserMr(void* buf, bool gpu) {
    return getInstance().deregUserMr(buf, gpu);
}

bool MemoryUtil::isMemoryMr(void* ptr, uint64_t size, bool gpu, bool adopted) {
    return getInstance().isMemoryMr(ptr, size, gpu, adopted);
}

bool MemoryUtil::findMemoryMr(void* mem_info, void* buf, uint64_t size, bool gpu, bool adopted) {
    return getInstance().findMemoryMr(mem_info, buf, size, gpu, adopted);
}

void MemoryUtil::memsetCPU(void* ptr, int value, size_t len) {
    memset(ptr, value, len);
}

bool MemoryUtil::memsetGPU(void* ptr, int value, size_t len) {
    return cudaMemset(ptr, value, len) == cudaSuccess;
}

bool MemoryUtil::memcopy(void* dst, bool dst_gpu, const void* src, bool src_gpu, size_t size) {
    if (src == nullptr || dst == nullptr || size == 0) {
        return false;
    }

    if (dst_gpu) {
        if (src_gpu) {
            return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) == cudaSuccess;
        }
        return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (src_gpu) {
        return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    memcpy(dst, src, size);
    return true;
}

bool MemoryUtil::gpuEventBarrier(void* event) {
    if (event == nullptr) {  // wait event only when pass event
        return true;
    }
    return cudaEventSynchronize(*(cudaEvent_t*)event) == cudaSuccess;
}

}  // namespace rtp_llm
