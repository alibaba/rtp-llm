#include "rtp_llm/cpp/core/torch_utils/torch_cuda_allocator.h"
#include "rtp_llm/cpp/utils/StackTrace.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

TorchCudaAllocator::TorchCudaAllocator(DeviceBase* device):
    device_(device), torch_device_(c10::DeviceType::CUDA, device->getDeviceProperties().id) {
    const char* env_value      = std::getenv("RTP_LLM_PRINT_TORCH_BUF");
    enable_python_stack_trace_ = (env_value && std::string(env_value) == "1");
}

void TorchCudaAllocator::init(int device_count) {}

bool TorchCudaAllocator::initialized() {
    return device_;
}

#ifdef UNDER_TORCH_2_6
at::DataPtr TorchCudaAllocator::allocate(size_t size) {
#else
at::DataPtr TorchCudaAllocator::allocate(size_t size) const {
#endif
    std::string tag = "torch_allocated";
    if (enable_python_stack_trace_) {
        tag = rtp_llm::getPythonStackTrace();
    }

    auto       buffer      = device_->allocateBuffer({size, AllocationType::DEVICE, allocate_private_}, {tag});
    auto       buffer_ctx  = new BufferPtr(buffer);
    const auto ptr         = buffer->data();
    const auto ctx_deleter = [](void* ctx_ptr) {
        auto ptr = (BufferPtr*)ctx_ptr;
        delete ptr;
    };
    return at::DataPtr(ptr, buffer_ctx, ctx_deleter, torch_device_);
}

void TorchCudaAllocator::malloc(void**                           devPtr,
                                TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device,
                                size_t                           size,
                                cudaStream_t                     stream) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::free(void** ptr) {
    throw std::runtime_error("not implemented.");
}

#ifdef UNDER_TORCH_2_6
void TorchCudaAllocator::copy_data(void* dest, const void* src, size_t count) const {
    throw std::runtime_error("not implemented.");
}

double TorchCudaAllocator::getMemoryFraction(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::enable(bool value) {
    throw std::runtime_error("not implemented.");
}

bool TorchCudaAllocator::isEnabled() const {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::beginAllocateToPool(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE  device,
                                             at::cuda::MempoolId_t             mempool_id,
                                             std::function<bool(cudaStream_t)> filter) {
    // printStackTrace();
    RTP_LLM_LOG_DEBUG("beginAllocateToPool: device %d, mempool_id %d", device, mempool_id);
    allocate_private_ = true;
};

void TorchCudaAllocator::endAllocateToPool(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device, at::cuda::MempoolId_t mempool_id) {
    // printStackTrace();
    RTP_LLM_LOG_DEBUG("endAllocateToPool: device %d, mempool_id %d", device, mempool_id);
    allocate_private_ = false;
}

void TorchCudaAllocator::attachAllocatorTraceTracker(c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) {
    throw std::runtime_error("not implemented.");
}

c10::cuda::CUDACachingAllocator::ShareableHandle TorchCudaAllocator::shareIpcHandle(void* ptr) {
    throw std::runtime_error("not implemented.");
}
#else
void TorchCudaAllocator::beginAllocateStreamToPool(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device,
                                                   cudaStream_t                     stream,
                                                   at::cuda::MempoolId_t            mempool_id) {
    throw std::runtime_error("not implemented.");
};

void TorchCudaAllocator::endAllocateStreamToPool(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device, cudaStream_t stream) {
    throw std::runtime_error("not implemented.");
}
#endif

void TorchCudaAllocator::setMemoryFraction(double fraction, TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device) {
    throw std::runtime_error("not implemented.");
}

#ifdef UNDER_TORCH_2_8
void TorchCudaAllocator::recordHistory(bool                                            enabled,
                                       at::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
                                       size_t                                          alloc_trace_max_entries,
                                       at::cuda::CUDACachingAllocator::RecordContext   when,
                                       bool                                            clearHistory) {
    throw std::runtime_error("not implemented.");
}
#else
void TorchCudaAllocator::recordHistory(bool                                            enabled,
                                       at::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
                                       size_t                                          alloc_trace_max_entries,
                                       at::cuda::CUDACachingAllocator::RecordContext   when) {
    throw std::runtime_error("not implemented.");
}
#endif

bool TorchCudaAllocator::isHistoryEnabled() {
    return false;
}

bool TorchCudaAllocator::checkPoolLiveAllocations(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device,
                                                  at::cuda::MempoolId_t            mempool_id,
                                                  const std::unordered_set<void*>& expected_live_allocations) {
    return true;
}

void TorchCudaAllocator::attachOutOfMemoryObserver(at::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
    throw std::runtime_error("not implemented.");
}

#ifdef UNDER_TORCH_2_8
void TorchCudaAllocator::emptyCache(at::cuda::MempoolId_t mempool_id) {}
#else
void TorchCudaAllocator::emptyCache() {}
#endif

void* TorchCudaAllocator::getBaseAllocation(void* ptr, size_t* outSize) {
    return ptr;
}

void TorchCudaAllocator::recordStream(const at::DataPtr& ptr, at::cuda::CUDAStream stream) {}

#ifdef UNDER_TORCH_2_8
at::cuda::CUDACachingAllocator::SnapshotInfo TorchCudaAllocator::snapshot(at::cuda::MempoolId_t mempool_id) {
    throw std::runtime_error("not implemented.");
}
#else
at::cuda::CUDACachingAllocator::SnapshotInfo TorchCudaAllocator::snapshot() {
    throw std::runtime_error("not implemented.");
}
#endif

std::shared_ptr<at::cuda::CUDACachingAllocator::AllocatorState>
TorchCudaAllocator::getCheckpointState(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device, at::cuda::MempoolId_t id) {
    throw std::runtime_error("not implemented.");
}

at::cuda::CUDACachingAllocator::CheckpointDelta
TorchCudaAllocator::setCheckpointPoolState(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE                                device,
                                           std::shared_ptr<at::cuda::CUDACachingAllocator::AllocatorState> as) {
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
    void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream, bool p2p_enabled) {
    RTP_LLM_CHECK_WITH_INFO(((srcDevice == dstDevice) || (p2p_enabled)), "p2p is required to copy across device.");
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

void TorchCudaAllocator::releasePool(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device, at::cuda::MempoolId_t mempool_id) {}

void TorchCudaAllocator::enablePeerAccess(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE dev,
                                          TORCH_CUDA_ALLOCATOR_INDEX_DTYPE dev_to_access) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::cacheInfo(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device, size_t* largestBlock) {}

void TorchCudaAllocator::assertValidDevice(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device) {}

at::cuda::CUDACachingAllocator::DeviceStats
TorchCudaAllocator::getDeviceStats(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::resetAccumulatedStats(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device) {
    throw std::runtime_error("not implemented.");
}

void TorchCudaAllocator::resetPeakStats(TORCH_CUDA_ALLOCATOR_INDEX_DTYPE device) {
    throw std::runtime_error("not implemented.");
}
}  // namespace rtp_llm
