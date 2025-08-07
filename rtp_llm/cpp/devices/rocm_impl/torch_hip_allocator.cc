#include "torch_hip_allocator.h"
#include <iostream>

namespace rtp_llm {

TorchHipAllocator torch_hip_allocator;
#if 1
void local_raw_delete(void* ptr) {
    torch_hip_allocator.free(&ptr);
}

void TorchHipAllocator::malloc(void** devPtr, int device, size_t size, hipStream_t stream) {
    *devPtr = device_->nativeGraphCapturing() ? allocator_->mallocPrivate(size) : allocator_->malloc(size);
}

void TorchHipAllocator::free(void** ptr) {
    if (*ptr) {
        allocator_->free(ptr);
    }
}

at::DataPtr TorchHipAllocator::allocate(size_t size) {
    auto  device = c10::Device(at::DeviceType::HIP, device_id_);
    void* ptr    = device_->nativeGraphCapturing() ? allocator_->mallocPrivate(size) : allocator_->malloc(size);
    return {ptr, ptr, &local_raw_delete, device};
}

at::DeleterFnPtr TorchHipAllocator::raw_deleter() const {
    return &local_raw_delete;
}

void* TorchHipAllocator::raw_alloc(size_t nbytes) {
    return raw_alloc_with_stream(nbytes, at::hip::getCurrentHIPStream());
}

void* TorchHipAllocator::raw_alloc_with_stream(size_t nbytes, hipStream_t stream) {
    if (nbytes == 0) {
        return nullptr;
    }
    void* r = nullptr;
    malloc(&r, device_id_, nbytes, stream);
    return r;
}

hipError_t TorchHipAllocator::memcpyAsync(
    void* dst, int dstDevice, const void* src, int srcDevice, size_t count, hipStream_t stream, bool p2p_enabled) {
    return hipMemcpyAsync(dst, src, count, hipMemcpyDefault, stream);
}

void TorchHipAllocator::raw_delete(void* ptr) {
    free(&ptr);
}
#endif

c10::hip::HIPCachingAllocator::HIPAllocator* getTorchHIPAllocator() {
    return &torch_hip_allocator;
}
void initTorchHIPAllocator(IAllocator* allocator, int device_id, DeviceBase* device) {
    torch_hip_allocator.init(allocator, device_id, device);
}

}  // namespace rtp_llm
