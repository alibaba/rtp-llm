#include "src/fastertransformer/core/torch_utils/torch_cuda_allocator.h"
#include <iostream>

namespace fastertransformer {

TorchCudaAllocator::TorchCudaAllocator(DeviceBase* device)
    : device_(device)
    , torch_device_(c10::DeviceType::CUDA, device->getDeviceProperties().id)
    {}

void TorchCudaAllocator::init(int device_count) {}

bool TorchCudaAllocator::initialized() {
    return device_;
}

at::DataPtr TorchCudaAllocator::allocate(size_t size) const {
    auto buffer = device_->allocateBuffer({size}, {"torch_allocated"});
    auto buffer_ctx = new BufferPtr(buffer);
    const auto ptr = buffer->data();
    const auto ctx_deleter = [](void* ctx_ptr) {
        auto ptr = (BufferPtr *)ctx_ptr;
        delete ptr;
    };
    return at::DataPtr(ptr, buffer_ctx, ctx_deleter, torch_device_);
}

void TorchCudaAllocator::malloc(void** devPtr, int device, size_t size, cudaStream_t stream) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::free(void** ptr) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::setMemoryFraction(double fraction, int device) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::recordHistory(bool                                            enabled,
                                       at::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
                                       size_t                                          alloc_trace_max_entries,
                                       at::cuda::CUDACachingAllocator::RecordContext   when) {
    throw std::runtime_error("not implemented.");
}

bool TorchCudaAllocator::isHistoryEnabled() {
    return false;
}

bool TorchCudaAllocator::checkPoolLiveAllocations(int                              device,
                                                  at::cuda::MempoolId_t            mempool_id,
                                                  const std::unordered_set<void*>& expected_live_allocations) {
    return true;
}

void TorchCudaAllocator::attachOutOfMemoryObserver(at::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::emptyCache() {}

void* TorchCudaAllocator::getBaseAllocation(void* ptr, size_t* outSize) {
    return ptr;
}

void TorchCudaAllocator::recordStream(const at::DataPtr& ptr, at::cuda::CUDAStream stream) {
    throw std::runtime_error("not implemented.");
}

at::cuda::CUDACachingAllocator::SnapshotInfo TorchCudaAllocator::snapshot() {
    throw std::runtime_error("not implemented.");
}

std::shared_ptr<at::cuda::CUDACachingAllocator::AllocatorState>
TorchCudaAllocator::getCheckpointState(int device, at::cuda::MempoolId_t id) {
    throw std::runtime_error("not implemented.");
}

at::cuda::CUDACachingAllocator::CheckpointDelta
TorchCudaAllocator::setCheckpointPoolState(int device, std::shared_ptr<at::cuda::CUDACachingAllocator::AllocatorState> as) {
    at::cuda::CUDACachingAllocator::CheckpointDelta cpd;
    return cpd;
}

at::DeleterFnPtr TorchCudaAllocator::raw_deleter() const {
    throw std::runtime_error("not implemented.");
}

void* TorchCudaAllocator::raw_alloc(size_t nbytes) {
    throw std::runtime_error("not implemented.");
}

void* TorchCudaAllocator::raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
    throw std::runtime_error("not implemented.");
}

cudaError_t TorchCudaAllocator::memcpyAsync(
    void* dst, int dstDevice, const void* src, int srcDevice,
    size_t count, cudaStream_t stream, bool p2p_enabled)
{
    FT_CHECK_WITH_INFO(((srcDevice == dstDevice) || (p2p_enabled)),
                       "p2p is required to copy across device.");
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream);
}

void TorchCudaAllocator::raw_delete(void* ptr) {
    throw std::runtime_error("not implemented.");
}

std::shared_ptr<void> TorchCudaAllocator::getIpcDevPtr(std::string handle) {
    return nullptr;
}

std::string TorchCudaAllocator::name() {
    return "torch_cuda_allocator";
}

void TorchCudaAllocator::beginAllocateStreamToPool(int device, cudaStream_t stream, at::cuda::MempoolId_t mempool_id) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::endAllocateStreamToPool(int device, cudaStream_t stream) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::releasePool(int device, at::cuda::MempoolId_t mempool_id) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::enablePeerAccess(int dev, int dev_to_access) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::cacheInfo(int dev_id, size_t* largestBlock) {}

void TorchCudaAllocator::assertValidDevice(int device) {}

at::cuda::CUDACachingAllocator::DeviceStats TorchCudaAllocator::getDeviceStats(int device) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::resetAccumulatedStats(int device) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::resetPeakStats(int device) {
    throw std::runtime_error("not implemented.");
}

}  // namespace fastertransformer
