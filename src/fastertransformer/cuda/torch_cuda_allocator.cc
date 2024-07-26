#include "src/fastertransformer/cuda/torch_cuda_allocator.h"
#include <iostream>

namespace fastertransformer {

TorchCudaAllocator torch_cuda_allocator;

void local_raw_delete(void* ptr) {
    torch_cuda_allocator.free(&ptr);
}

void TorchCudaAllocator::malloc(void** devPtr, int device, size_t size, cudaStream_t stream) {
    *devPtr = allocator_->malloc(size);
}

void TorchCudaAllocator::free(void** ptr) {
    if (*ptr) {
        allocator_->free(ptr);
    }
}

at::DataPtr TorchCudaAllocator::allocate(size_t size) const {
    auto  device = c10::Device(at::DeviceType::CUDA, 0);
    void* ptr    = allocator_->malloc(size);
    return {ptr, ptr, &local_raw_delete, device};
}

at::DeleterFnPtr TorchCudaAllocator::raw_deleter() const {
    return &local_raw_delete;
}

void* TorchCudaAllocator::raw_alloc(size_t nbytes) {
    return raw_alloc_with_stream(nbytes, at::cuda::getCurrentCUDAStream());
}

void* TorchCudaAllocator::raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
    if (nbytes == 0) {
        return nullptr;
    }
    int   device = 0;
    void* r      = nullptr;
    malloc(&r, device, nbytes, stream);
    return r;
}

cudaError_t TorchCudaAllocator::memcpyAsync(
    void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream, bool p2p_enabled) {
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

void TorchCudaAllocator::raw_delete(void* ptr) {
    free(&ptr);
}

c10::cuda::CUDACachingAllocator::CUDAAllocator* getTorchCUDAAllocator() {
    return &torch_cuda_allocator;
}

void initTorchCUDAAllocator(IAllocator* allocator) {
    torch_cuda_allocator.init(allocator);
}

}  // namespace fastertransformer
