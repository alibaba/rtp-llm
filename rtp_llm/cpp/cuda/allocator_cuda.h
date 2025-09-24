#pragma once
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include <mutex>
#include <unordered_set>

namespace rtp_llm {

class ICudaAllocator: virtual public IAllocator {
public:
    ICudaAllocator(int device_id): device_id_(device_id) {}
    virtual ~ICudaAllocator() {};

    MemoryType memoryType() const override {
        return MEMORY_GPU;
    }

    void setStream(cudaStream_t stream) {
        stream_ = stream;
    }

    cudaStream_t returnStream() {
        return stream_;
    };

    void* reMalloc(void* ptr, size_t size) override;

protected:
    virtual bool        isExist(void* address) const                 = 0;
    virtual ReallocType isReMalloc(void* address, size_t size) const = 0;

protected:
    cudaStream_t stream_ = 0;  // initialize as default stream
    const int    device_id_;
};

class PurePointerCudaAllocator: public ICudaAllocator {
public:
    PurePointerCudaAllocator(int device_id);
    ~PurePointerCudaAllocator();

public:
    void* malloc(size_t size) override;
    void* mallocSync(size_t size) override;
    void  free(void** ptr) override;

protected:
    virtual bool        isExist(void* address) const;
    virtual ReallocType isReMalloc(void* address, size_t size) const;

    virtual void* doMalloc(size_t size)     = 0;
    virtual void* doMallocSync(size_t size) = 0;
    virtual void  doFree(void* ptr)         = 0;

    void destroy();

private:
    std::unique_ptr<std::unordered_map<void*, size_t>> pointer_mapping_;
    std::mutex                                         lock_;
};

template<>
class Allocator<AllocatorType::CUDA>: public PurePointerCudaAllocator, public TypedAllocator<AllocatorType::CUDA> {
public:
    Allocator(int device_id);
    ~Allocator();

    void* doMalloc(size_t size) override;
    void* doMallocSync(size_t size) override;
    void  doFree(void* ptr) override;
};

template<>
class Allocator<AllocatorType::CUDA_HOST>:
    public PurePointerCudaAllocator,
    public TypedAllocator<AllocatorType::CUDA_HOST> {
public:
    Allocator(int device_id);
    ~Allocator();

    MemoryType memoryType() const override {
        return MEMORY_CPU_PINNED;
    }

    void* doMalloc(size_t size) override;
    void* doMallocSync(size_t size) override;
    void  doFree(void* ptr) override;
};

}  // namespace rtp_llm
