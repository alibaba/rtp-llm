#pragma once

#include "c10/hip/HIPCachingAllocator.h"
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
class TorchHipAllocator: public c10::hip::HIPCachingAllocator::HIPAllocator {
#if 1
private:
    IAllocator*                               allocator_;
    at::hip::HIPCachingAllocator::DeviceStats stats;

    int         device_id_;
    std::mutex  mutex;
    DeviceBase* device_;

public:
    void init(IAllocator* allocator, int device_id, DeviceBase* device) {
        allocator_ = allocator;
        device_id_ = device_id;
        device_    = device;
    }

    void init(int device_count) override {}

    bool initialized() override {
        return allocator_;
    }

    double getMemoryFraction(c10::DeviceIndex device) override {
        return 0.0;
    }

    void enable(bool value) override {}

    bool isEnabled() const override {
        return false;
    }

    c10::hip::HIPCachingAllocator::ShareableHandle shareIpcHandle(void* ptr) override {
        return {};
    }

    void malloc(void** devPtr, int device, size_t size, hipStream_t stream);

    void free(void** ptr);

    void setMemoryFraction(double fraction, c10::DeviceIndex device) override {}

    void recordHistory(bool                                          enabled,
                       at::hip::HIPCachingAllocator::CreateContextFn context_recorder,
                       size_t                                        alloc_trace_max_entries,
                       at::hip::HIPCachingAllocator::RecordContext   when) override {}

    bool isHistoryEnabled() override {
        return false;
    }

    bool checkPoolLiveAllocations(int                              device,
                                  at::hip::MempoolId_t             mempool_id,
                                  const std::unordered_set<void*>& expected_live_allocations) {
        return true;
    }

    void attachOutOfMemoryObserver(at::hip::HIPCachingAllocator::OutOfMemoryObserver observer) override {}

    void emptyCache() override {}

    void* getBaseAllocation(void* ptr, size_t* outSize) override {
        return ptr;
    }

    void recordStream(const at::DataPtr& ptr, at::hip::HIPStream stream) override {}

    at::hip::HIPCachingAllocator::SnapshotInfo snapshot() override {
        at::hip::HIPCachingAllocator::SnapshotInfo result;
        return result;
    }

    std::shared_ptr<at::hip::HIPCachingAllocator::AllocatorState> getCheckpointState(c10::DeviceIndex     device,
                                                                                     at::hip::MempoolId_t id) override {
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
    at::hip::HIPCachingAllocator::CheckpointDelta
    setCheckpointPoolState(c10::DeviceIndex                                              device,
                           std::shared_ptr<at::hip::HIPCachingAllocator::AllocatorState> pps) override {
        at::hip::HIPCachingAllocator::CheckpointDelta cpd;
        return cpd;
    }

    at::DataPtr allocate(size_t size) override;

    at::DeleterFnPtr raw_deleter() const override;

    void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override {}

    void assertValidDevice(int device) {}

    at::hip::HIPCachingAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override {
        return stats;
    }

    void resetAccumulatedStats(c10::DeviceIndex device) override {}

    void resetPeakStats(c10::DeviceIndex device) override {}
    // HIPGraph interactions
    void beginAllocateStreamToPool(int device, hipStream_t stream, at::hip::MempoolId_t mempool_id) {}

    void endAllocateStreamToPool(int device, hipStream_t stream) {}

    void releasePool(c10::DeviceIndex device, c10::hip::MempoolId_t mempool_id) override {}

    void* raw_alloc(size_t nbytes) override;

    void* raw_alloc_with_stream(size_t nbytes, hipStream_t stream) override;

    void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) override {}

    hipError_t memcpyAsync(void*       dst,
                           int         dstDevice,
                           const void* src,
                           int         srcDevice,
                           size_t      count,
                           hipStream_t stream,
                           bool        p2p_enabled) override;

    void raw_delete(void* ptr) override;

    // In HIP IPC, sender sends a tensor to receiver, getIpcDevPtr
    // is called by the receiving process to map the HIP memory from the sending
    // process into its own address space.
    //
    // HIP IPC only allows sharing a big memory block associated with a
    // hipIpcMemHandle_t and it can be opened only **once** per context per
    // process. There can be multiple types of storage in the same IPC mem block,
    // so we must cache the device ptr to construct typed storage as it comes.
    //
    // ipcMemHandle_to_devptr maps a hipIpcMemHandle_t to a device pointer in the
    // process that can be used to access the memory block in the sender process.
    // It only saves a weak_ptr of the device pointer in the map, the shared_ptr
    // will be used to reconstruct all storages in this HipMalloc allocation. And
    // it will deleted in hipIpcCloseMemHandle when its reference count is 0.
    //
    std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
        return nullptr;
    }
    std::string name() override {
        return "torch_hip_allocator";
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {}

    void beginAllocateToPool(c10::DeviceIndex                 device,
                             c10::hip::MempoolId_t            mempool_id,
                             std::function<bool(hipStream_t)> filter) override {
        RTP_LLM_LOG_INFO("TorchHipAllocator::beginAllocateToPool called, setting nativeGraphCapturing = true");
        device_->nativeGraphBeginCapture();
    }

    void endAllocateToPool(c10::DeviceIndex device, c10::hip::MempoolId_t mempool_id) override {
        RTP_LLM_LOG_INFO("TorchHipAllocator::endAllocateToPool called, setting nativeGraphCapturing = false");
        device_->nativeGraphEndCapture();
    }

    void attachAllocatorTraceTracker(c10::hip::HIPCachingAllocator::AllocatorTraceTracker tracker) override {};

#endif
};

c10::hip::HIPCachingAllocator::HIPAllocator* getTorchHIPAllocator();

void initTorchHIPAllocator(IAllocator* allocator, int device_id, DeviceBase* device);

}  // namespace rtp_llm
