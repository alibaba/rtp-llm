#pragma once

#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <cuda.h>
#include <cuda_runtime.h>

namespace rtp_llm {

class MemoryUtilBase {
public:
    virtual ~MemoryUtilBase() = default;

    virtual bool regUserMr(void* buf, uint64_t size, bool gpu)                                  = 0;
    virtual bool deregUserMr(void* buf, bool gpu)                                               = 0;
    virtual bool isMemoryMr(void* ptr, uint64_t size, bool gpu, bool adopted)                   = 0;
    virtual bool findMemoryMr(void* mem_info, void* buf, uint64_t size, bool gpu, bool adopted) = 0;
    virtual bool isRdmaMode()                                                                   = 0;
};

// 用于封装内存/显存的分配和释放
class MemoryUtil {
public:
    MemoryUtil(std::unique_ptr<MemoryUtilBase> impl);

public:
    void* mallocCPU(size_t size);
    void  freeCPU(void* ptr);
    void* mallocGPU(size_t size);
    void  freeGPU(void* ptr);

    void         memsetCPU(void*, int value, size_t len);
    bool         memsetGPU(void*, int value, size_t len);
    bool         memcopy(void* dst, bool dst_gpu, const void* src, bool src_gpu, size_t size);
    virtual bool gpuEventBarrier(void* event);

    bool rdmaMode();

    bool         regUserMr(void* buf, uint64_t size, bool gpu);
    bool         deregUserMr(void* buf, bool gpu);
    bool         isMemoryMr(void* ptr, uint64_t size, bool gpu, bool adopted);
    virtual bool findMemoryMr(void* mem_info, void* buf, uint64_t size, bool gpu, bool adopted);

private:
    MemoryUtilBase& getInstance();

    // methods for unittest
    void setRdmaMode(bool rdma_mode);

    bool                            rdma_mode_{false};
    bool                            memcpyImpl(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
    std::unique_ptr<MemoryUtilBase> instance_;
    cudaStream_t                    stream_;
};

}  // namespace rtp_llm
