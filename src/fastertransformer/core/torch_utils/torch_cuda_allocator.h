#pragma once

#include "c10/cuda/CUDACachingAllocator.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/devices/DeviceBase.h"

namespace fastertransformer {

class TorchCudaAllocator: public c10::cuda::CUDACachingAllocator::CUDAAllocator {
private:
    DeviceBase*                                 device_;
    c10::Device                                 torch_device_;

public:
    TorchCudaAllocator(DeviceBase* device);

    void init(int device_count) override;

    bool initialized() override;

    void malloc(void** devPtr, int device, size_t size, cudaStream_t stream);

    void free(void** ptr);

    void setMemoryFraction(double fraction, int device) override;

    void recordHistory(bool                                            enabled,
                       at::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
                       size_t                                          alloc_trace_max_entries,
                       at::cuda::CUDACachingAllocator::RecordContext   when) override;

    bool isHistoryEnabled() override;

    bool checkPoolLiveAllocations(int                              device,
                                  at::cuda::MempoolId_t            mempool_id,
                                  const std::unordered_set<void*>& expected_live_allocations) override;

    void attachOutOfMemoryObserver(at::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override;

    void emptyCache() override;

    void* getBaseAllocation(void* ptr, size_t* outSize) override;

    void recordStream(const at::DataPtr& ptr, at::cuda::CUDAStream stream) override;

    at::cuda::CUDACachingAllocator::SnapshotInfo snapshot() override;

    std::shared_ptr<at::cuda::CUDACachingAllocator::AllocatorState>
    getCheckpointState(int device, at::cuda::MempoolId_t id) override;

    at::cuda::CUDACachingAllocator::CheckpointDelta
    setCheckpointPoolState(int device, std::shared_ptr<at::cuda::CUDACachingAllocator::AllocatorState> as) override;

    at::DataPtr allocate(size_t size) const override;

    at::DeleterFnPtr raw_deleter() const override;

    void cacheInfo(int dev_id, size_t* largestBlock) override;

    void assertValidDevice(int device);

    at::cuda::CUDACachingAllocator::DeviceStats getDeviceStats(int device) override;

    void resetAccumulatedStats(int device) override;

    void resetPeakStats(int device) override;

    void beginAllocateStreamToPool(int device, cudaStream_t stream, at::cuda::MempoolId_t mempool_id) override;

    void endAllocateStreamToPool(int device, cudaStream_t stream) override;

    void releasePool(int device, at::cuda::MempoolId_t mempool_id) override;

    void* raw_alloc(size_t nbytes) override;

    void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override;

    void enablePeerAccess(int dev, int dev_to_access) override;

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

}  // namespace fastertransformer
