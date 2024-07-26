#pragma once

#include "c10/cuda/CUDACachingAllocator.h"
#include "src/fastertransformer/core/allocator.h"

namespace fastertransformer {

class TorchCudaAllocator: public c10::cuda::CUDACachingAllocator::CUDAAllocator {
private:
    IAllocator*                                 allocator_;
    at::cuda::CUDACachingAllocator::DeviceStats stats;

    std::mutex mutex;

public:
    void init(IAllocator* allocator) {
        allocator_ = allocator;
    }

    void init(int device_count) override {}

    bool initialized() override {
        return allocator_;
    }

    void malloc(void** devPtr, int device, size_t size, cudaStream_t stream);

    void free(void** ptr);

    void setMemoryFraction(double fraction, int device) override {}

    void recordHistory(bool                                            enabled,
                       at::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
                       size_t                                          alloc_trace_max_entries,
                       at::cuda::CUDACachingAllocator::RecordContext   when) override {}

    bool isHistoryEnabled() override {
        return false;
    }

    bool checkPoolLiveAllocations(int                              device,
                                  at::cuda::MempoolId_t            mempool_id,
                                  const std::unordered_set<void*>& expected_live_allocations) override {
        return true;
    }

    void attachOutOfMemoryObserver(at::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override {}

    void emptyCache() override {}

    void* getBaseAllocation(void* ptr, size_t* outSize) override {
        return ptr;
    }

    void recordStream(const at::DataPtr& ptr, at::cuda::CUDAStream stream) override {}

    at::cuda::CUDACachingAllocator::SnapshotInfo snapshot() override {
        at::cuda::CUDACachingAllocator::SnapshotInfo result;
        return result;
    }

    std::shared_ptr<at::cuda::CUDACachingAllocator::AllocatorState>
    getCheckpointState(int device, at::cuda::MempoolId_t id) override {
        return nullptr;
    }

    /**
     * @brief Checkpoint the private pool state identified in `as` to its prior
     * state
     *
     * @param device - device of the pool to manipulate
     * @param as - allocator state
     * @param stale_live_storages - storages of tensors which are currently
     * allocated but which will be not be allocated after the checkpoint is set.
     * For these storages we will remove their deleter function.
     * @return CheckpointDelta - Freed Pointers and DataPtrs that contain deleter
     * functions for all allocated blocks in the new checkpoint state.
     */
    at::cuda::CUDACachingAllocator::CheckpointDelta
    setCheckpointPoolState(int device, std::shared_ptr<at::cuda::CUDACachingAllocator::AllocatorState> as) override {
        at::cuda::CUDACachingAllocator::CheckpointDelta cpd;
        return cpd;
    }

    at::DataPtr allocate(size_t size) const override;

    at::DeleterFnPtr raw_deleter() const override;

    void cacheInfo(int dev_id, size_t* largestBlock) override {}

    void assertValidDevice(int device) {}

    at::cuda::CUDACachingAllocator::DeviceStats getDeviceStats(int device) override {
        return stats;
    }

    void resetAccumulatedStats(int device) override {}

    void resetPeakStats(int device) override {}
    // CUDAGraph interactions
    void beginAllocateStreamToPool(int device, cudaStream_t stream, at::cuda::MempoolId_t mempool_id) override {}

    void endAllocateStreamToPool(int device, cudaStream_t stream) override {}

    void releasePool(int device, at::cuda::MempoolId_t mempool_id) override {}

    void* raw_alloc(size_t nbytes) override;

    void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override;

    void enablePeerAccess(int dev, int dev_to_access) override {}

    cudaError_t memcpyAsync(void*        dst,
                            int          dstDevice,
                            const void*  src,
                            int          srcDevice,
                            size_t       count,
                            cudaStream_t stream,
                            bool         p2p_enabled) override;

    void raw_delete(void* ptr) override;

    // In CUDA IPC, sender sends a tensor to receiver, getIpcDevPtr
    // is called by the receiving process to map the CUDA memory from the sending
    // process into its own address space.
    //
    // CUDA IPC only allows sharing a big memory block associated with a
    // cudaIpcMemHandle_t and it can be opened only **once** per context per
    // process. There can be multiple types of storage in the same IPC mem block,
    // so we must cache the device ptr to construct typed storage as it comes.
    //
    // ipcMemHandle_to_devptr maps a cudaIpcMemHandle_t to a device pointer in the
    // process that can be used to access the memory block in the sender process.
    // It only saves a weak_ptr of the device pointer in the map, the shared_ptr
    // will be used to reconstruct all storages in this CudaMalloc allocation. And
    // it will deleted in cudaIpcCloseMemHandle when its reference count is 0.
    //
    std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
        return nullptr;
    }
    std::string name() override {
        return "torch_cuda_allocator";
    }
};

c10::cuda::CUDACachingAllocator::CUDAAllocator* getTorchCUDAAllocator();

void initTorchCUDAAllocator(IAllocator* allocator);

}  // namespace fastertransformer
