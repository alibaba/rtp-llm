#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "maga_transformer/cpp/utils/Logger.h"

#include <cstring>
#include <cstdlib>
#include <iostream>

namespace rtp_llm {

MemoryUtil::MemoryUtil(std::unique_ptr<MemoryUtilBase> impl): instance_(std::move(impl)) {
    auto err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (err != cudaError_t::cudaSuccess) {
        throw std::runtime_error("failed to create stream, error is " + std::string(cudaGetErrorString(err)));
    }
}

bool MemoryUtil::rdmaMode() {
    return getInstance().isRdmaMode();
}

void MemoryUtil::setRdmaMode(bool rdma_mode) {
    rdma_mode_ = rdma_mode;
}

MemoryUtilBase& MemoryUtil::getInstance() {
    return *instance_;
}

void* MemoryUtil::mallocCPU(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        FT_LOG_WARNING("cuda malloc host failed, size %lu, error is %s", size, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void MemoryUtil::freeCPU(void* ptr) {
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
        FT_LOG_WARNING("cuda free host failed, error is %s", cudaGetErrorString(err));
    }
}

void* MemoryUtil::mallocGPU(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        FT_LOG_WARNING("cuda malloc host failed, size %lu, error is %s", size, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void MemoryUtil::freeGPU(void* ptr) {
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
        FT_LOG_WARNING("cuda free host failed, error is %s", cudaGetErrorString(err));
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

bool MemoryUtil::memcpyImpl(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    auto succ = cudaMemcpyAsync(dst, src, count, kind, stream_);
    if (succ != cudaSuccess) {
        return false;
    }
    return cudaStreamSynchronize(stream_);
}

bool MemoryUtil::memcopy(void* dst, bool dst_gpu, const void* src, bool src_gpu, size_t size) {
    if (src == nullptr || dst == nullptr || size == 0) {
        return false;
    }

    if (dst_gpu) {
        if (src_gpu) {
            return memcpyImpl(dst, src, size, cudaMemcpyDeviceToDevice) == cudaSuccess;
        }
        return memcpyImpl(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (src_gpu) {
        return memcpyImpl(dst, src, size, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    memcpy(dst, src, size);
    return true;
}

bool MemoryUtil::gpuEventBarrier(void* event) {
    if (event == nullptr) {  // wait event only when pass event
        return true;
    }
    cudaError_t err = cudaEventSynchronize(*(cudaEvent_t*)event);
    if (err != cudaSuccess) {
        FT_LOG_WARNING("cuda event sync failed, error is %s", cudaGetErrorString(err));
    }
    return err == cudaSuccess;
}

}  // namespace rtp_llm
