#pragma once

#include "c10/cuda/CUDACachingAllocator.h"
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include <torch/version.h>

#if defined(TORCH_VERSION_MAJOR) && ((TORCH_VERSION_MAJOR == 2) & (TORCH_VERSION_MINOR >= 6))
#define UNDER_TORCH_2_6
#define TORCH_CUDA_ALLOCATOR_INDEX_DTYPE c10::DeviceIndex
#else
#define TORCH_CUDA_ALLOCATOR_INDEX_DTYPE int
#endif

#if defined(TORCH_VERSION_MAJOR) && ((TORCH_VERSION_MAJOR == 2) & (TORCH_VERSION_MINOR >= 8))
#define UNDER_TORCH_2_8
#endif

namespace rtp_llm {

class TorchCudaAllocator: public c10::cuda::CUDACachingAllocator::CUDAAllocator {
private:
    DeviceBase*  device_;
    c10::Device  torch_device_;
    bool         allocate_private_          = false;
    mutable bool enable_python_stack_trace_ = false;

public:
    TorchCudaAllocator(DeviceBase* device);

    void init(int device_count) override;

    bool initialized() override;

    void malloc(void** devPtr, TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device, size_t size, cudaStream_t stream);

    void free(void** ptr);

#ifdef UNDER_TORCH_2_6
    void copy_data(void* dest, const void* src, size_t count) const override;

    double getMemoryFraction(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device) override;

    void enable(bool value) override;

    bool isEnabled() const override;

    void beginAllocateToPool(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE  device,
                             at::cuda::MempoolId_t             mempool_id,
                             std::function<bool(cudaStream_t)> filter) override;

    void endAllocateToPool(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device, at::cuda::MempoolId_t mempool_id) override;

    void attachAllocatorTraceTracker(c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) override;

    c10::cuda::CUDACachingAllocator::ShareableHandle shareIpcHandle(void* ptr);

    at::DataPtr allocate(size_t size) override;
#else
    void beginAllocateStreamToPool(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device,
                                   cudaStream_t                     stream,
                                   at::cuda::MempoolId_t            mempool_id) override;

    void endAllocateStreamToPool(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device, cudaStream_t stream) override;

    at::DataPtr allocate(size_t size) const override;
#endif

    void setMemoryFraction(double fraction, TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device) override;

#ifdef UNDER_TORCH_2_8
    void recordHistory(bool                                            enabled,
                       at::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
                       size_t                                          alloc_trace_max_entries,
                       at::cuda::CUDACachingAllocator::RecordContext   when,
                       bool                                            clearHistory) override;
#else
    void recordHistory(bool                                            enabled,
                       at::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
                       size_t                                          alloc_trace_max_entries,
                       at::cuda::CUDACachingAllocator::RecordContext   when) override;
#endif

    bool isHistoryEnabled() override;

    bool checkPoolLiveAllocations(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device,
                                  at::cuda::MempoolId_t            mempool_id,
                                  const std::unordered_set<void*>& expected_live_allocations) override;

    void attachOutOfMemoryObserver(at::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override;

#ifdef UNDER_TORCH_2_8
    void emptyCache(at::cuda::MempoolId_t mempool_id = {0, 0}) override;
#else
    void emptyCache() override;
#endif

    void* getBaseAllocation(void* ptr, size_t* outSize) override;

    void recordStream(const at::DataPtr& ptr, at::cuda::CUDAStream stream) override;

#ifdef UNDER_TORCH_2_8
    at::cuda::CUDACachingAllocator::SnapshotInfo snapshot(at::cuda::MempoolId_t mempool_id = {0, 0}) override;
#else
    at::cuda::CUDACachingAllocator::SnapshotInfo snapshot() override;
#endif

    std::shared_ptr<at::cuda::CUDACachingAllocator::AllocatorState>
    getCheckpointState(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device, at::cuda::MempoolId_t id) override;

    at::cuda::CUDACachingAllocator::CheckpointDelta
    setCheckpointPoolState(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE                                device,
                           std::shared_ptr<at::cuda::CUDACachingAllocator::AllocatorState> as) override;

    at::DeleterFnPtr raw_deleter() const override;

    void cacheInfo(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device, size_t* largestBlock) override;

    void assertValidDevice(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device);

    at::cuda::CUDACachingAllocator::DeviceStats getDeviceStats(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device) override;

    void resetAccumulatedStats(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device) override;

    void resetPeakStats(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device) override;

    void releasePool(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device, at::cuda::MempoolId_t mempool_id) override;

    void* raw_alloc(size_t nbytes) override;

    void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override;

    void enablePeerAccess(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE dev,
                          TORCH_CUDA_ALLOCATOR_INDEX_DTYPE dev_to_access) override;

    cudaError_t memcpyAsync(void*        dst,
                            int          dstDevice,
                            const void*  src,
                            int          srcDevice,
                            size_t       count,
                            cudaStream_t stream,
                            bool         p2p_enabled) override;

    void raw_delete(void* ptr) override;

    std::shared_ptr<void> getIpcDevPtr(std::string handle) override;

    std::string name() override;
};

}  // namespace rtp_llm
