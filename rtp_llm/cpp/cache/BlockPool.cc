#include <ATen/cuda/CachingHostAllocator.h>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/KVCachePhysicalMemoryController.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

#include <cstdlib>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <utility>

#include <sys/mman.h>
#include <unistd.h>

#if USING_CUDA
#include <cuda_runtime.h>
#include <ATen/cuda/MemPool.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif

namespace rtp_llm {

namespace {

bool shouldPinHostBlockPool();

class AllocationRegionGuard {
public:
    AllocationRegionGuard(VmmBackend& backend, const std::string& tag):
        backend_(backend), active_(backend.isAvailable() && backend.beginAllocationRegion(tag)) {}

    ~AllocationRegionGuard() {
        if (active_) {
            backend_.endAllocationRegion();
        }
    }

    bool active() const {
        return active_;
    }

private:
    VmmBackend& backend_;
    bool        active_;
};

const char* allocationTypeName(AllocationType allocation_type) {
    switch (allocation_type) {
        case AllocationType::HOST:
            return "HOST";
        case AllocationType::DEVICE:
            return "DEVICE";
    }
    return "UNKNOWN";
}

const char* memoryTypeName(MemoryType memory_type) {
    switch (memory_type) {
        case MemoryType::MEMORY_CPU:
            return "CPU";
        case MemoryType::MEMORY_CPU_PINNED:
            return "CPU_PINNED";
        case MemoryType::MEMORY_GPU:
            return "GPU";
    }
    return "UNKNOWN";
}

const char*
requestedBackingName(AllocationType allocation_type, bool use_pinned_cpu_backing, bool use_cuda_malloc_backing) {
    if (allocation_type == AllocationType::HOST) {
        return shouldPinHostBlockPool() ? "CPU_PINNED_OR_CPU_FALLBACK" : "CPU";
    }
    if (use_cuda_malloc_backing) {
        return "GPU_CUDA_MALLOC";
    }
    return use_pinned_cpu_backing ? "CPU_PINNED" : "GPU";
}

bool shouldPinHostBlockPool() {
    const char* value = std::getenv("RTP_LLM_PIN_HOST_BLOCK_POOL");
    if (value == nullptr) {
        return true;
    }
    const std::string flag(value);
    return flag != "0" && flag != "false" && flag != "FALSE" && flag != "off" && flag != "OFF";
}

void markHostBlockPoolDontDump(void* ptr, size_t size) {
#ifdef MADV_DONTDUMP
    if (ptr == nullptr || size == 0) {
        return;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        page_size = 4096;
    }

    const auto begin         = reinterpret_cast<uintptr_t>(ptr);
    const auto page_mask     = static_cast<uintptr_t>(page_size - 1);
    const auto aligned_begin = begin & ~page_mask;
    const auto aligned_end   = (begin + size + page_mask) & ~page_mask;
    const auto aligned_size  = static_cast<size_t>(aligned_end - aligned_begin);

    if (madvise(reinterpret_cast<void*>(aligned_begin), aligned_size, MADV_DONTDUMP) != 0) {
        RTP_LLM_LOG_WARNING("madvise MADV_DONTDUMP failed for host block pool, ptr=%p, size=%zu, error=%s",
                            ptr,
                            size,
                            std::strerror(errno));
    } else {
        RTP_LLM_LOG_INFO("madvise MADV_DONTDUMP success for host block pool, ptr=%p, size=%zu, aligned_ptr=%p, "
                         "aligned_size=%zu",
                         ptr,
                         size,
                         reinterpret_cast<void*>(aligned_begin),
                         aligned_size);
    }
#else
    RTP_LLM_LOG_WARNING(
        "MADV_DONTDUMP is not defined, host block pool may be included in coredump, ptr=%p, size=%zu", ptr, size);
#endif
}

}  // namespace

BlockPool::BlockPool(const BlockPoolConfig& config,
                     AllocationType         allocation_type,
                     bool                   use_pinned_cpu_backing,
                     bool                   use_cuda_malloc_backing):
    config_(config),
    allocation_type_(allocation_type),
    use_pinned_cpu_backing_(use_pinned_cpu_backing),
    use_cuda_malloc_backing_(use_cuda_malloc_backing) {}

BlockPool::BlockPool(const BlockPoolConfig& config, torch::Tensor device_backing):
    BlockPool(config, AllocationType::DEVICE, false, false) {
    external_device_backing_ = std::move(device_backing);
}

BlockPool::~BlockPool() {
    cache_aligned_buffer_ = torch::Tensor();
}

torch::Tensor BlockPool::allocatePausableDeviceBacking(size_t size_bytes) {
    const auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    VmmBackend vmm_backend;

#if USING_CUDA
    if (vmm_backend.isAvailable()) {
        // Route the KV big-buffer through the torch_memory_saver preload shim so its physical
        // pages become pausable under the "kv_cache" tag. Two ingredients are required; setting
        // the shim's interesting-region flag alone is NOT enough (that was the sleep-mode bug):
        //
        //   1) The allocation must trigger a *fresh* cudaMalloc. The shim intercepts cudaMalloc
        //      (armed by the flag) and backs it with VMM pages; if torch serves the request from a
        //      pre-existing cached block (allocated before the flag was set, e.g. a model-load
        //      transient), no cudaMalloc happens, the pages are plain non-VMM memory, and a later
        //      tms_pause("kv_cache") frees nothing. A brand-new private pool starts empty, so the
        //      first allocation into it always misses the cache and issues a real cudaMalloc.
        //   2) The VMM segment must stay isolated from the default pool. torch's default-pool
        //      emptyCache()/cudaFree() paths cannot handle an unmapped/paused VMM range and raise
        //      cudaErrorInvalidValue; keeping it in a dedicated pool prevents those paths from ever
        //      touching it. This mirrors the proven weights path (torch_memory_saver.region(),
        //      which enters use_mem_pool(primary_pool) before tagging).
        //
        // The pool is created and never released: the arena lives for the whole process, and there
        // is exactly one per KV cache, so a permanent private-pool refcount is intentional.
        const auto device  = c10::cuda::current_device();
        const auto pool_id = at::cuda::MemPool::graph_pool_handle(/*is_user_created=*/true);
        c10::cuda::CUDACachingAllocator::createOrIncrefPool(device, pool_id);
        // Return unused default-pool reservations to the driver first (belt-and-suspenders; the
        // fresh private pool already guarantees a cache miss).
        c10::cuda::CUDACachingAllocator::emptyCache();
        c10::cuda::CUDACachingAllocator::beginAllocateToPool(device, pool_id, [](cudaStream_t) { return true; });

        torch::Tensor buffer;
        {
            AllocationRegionGuard region_guard(vmm_backend, KVCachePhysicalMemoryController::kDefaultTag);
            try {
                buffer = torch::empty({static_cast<int64_t>(size_bytes)}, options);
            } catch (...) {
                c10::cuda::CUDACachingAllocator::endAllocateToPool(device, pool_id);
                throw;
            }
        }
        c10::cuda::CUDACachingAllocator::endAllocateToPool(device, pool_id);
        RTP_LLM_LOG_INFO("device backing (%zu bytes) allocated under VMM tag '%s' in isolated pool (%llu,%llu)",
                         size_bytes,
                         KVCachePhysicalMemoryController::kDefaultTag,
                         static_cast<unsigned long long>(pool_id.first),
                         static_cast<unsigned long long>(pool_id.second));
        return buffer;
    }
#endif

    // Shim unavailable: plain device allocation (not pausable; sleep mode inactive).
    return torch::empty({static_cast<int64_t>(size_bytes)}, options);
}

void BlockPool::validateConfig() const {
    RTP_LLM_CHECK_WITH_INFO(!config_.memory_layouts.empty(), "BlockPoolConfig.memory_layouts must not be empty");
    RTP_LLM_CHECK_WITH_INFO(config_.block_num > 0, "BlockPoolConfig.block_num must be > 0");

    for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
        const auto& layout_cfg = config_.memory_layouts[layout_idx];

        RTP_LLM_CHECK_WITH_INFO(layout_cfg.block_num == config_.block_num,
                                "MemoryLayoutConfig.block_num mismatch: layout[%zu].block_num=%u, pool.block_num=%u",
                                layout_idx,
                                layout_cfg.block_num,
                                config_.block_num);
        RTP_LLM_CHECK_WITH_INFO(
            layout_cfg.layer_num > 0, "MemoryLayoutConfig.layer_num must be > 0 (layout=%zu)", layout_idx);
        RTP_LLM_CHECK_WITH_INFO(layout_cfg.kv_block_pool_size_bytes > 0,
                                "MemoryLayoutConfig.kv_block_pool_size_bytes must be > 0 (layout=%zu)",
                                layout_idx);
    }
}

void BlockPool::initializeCacheBuffer() {
    if (external_device_backing_.defined()) {
        RTP_LLM_CHECK_WITH_INFO(external_device_backing_.is_cuda() && external_device_backing_.is_contiguous(),
                                "external backing must be a contiguous CUDA tensor, pool_name=%s",
                                config_.pool_name.c_str());
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(external_device_backing_.numel()) == config_.total_size_bytes,
                                "external backing size mismatch, pool_name=%s, expected=%zu, actual=%ld",
                                config_.pool_name.c_str(),
                                config_.total_size_bytes,
                                external_device_backing_.numel());
        cache_aligned_buffer_ = external_device_backing_;
    } else if (allocation_type_ == AllocationType::HOST) {
        auto cpu_buffer = torch::empty({static_cast<int64_t>(config_.total_size_bytes)},
                                       torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        if (shouldPinHostBlockPool()) {
            try {
                cache_aligned_buffer_ = cpu_buffer.pin_memory();
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING(
                    "pin host block pool failed, fallback to pageable CPU memory, total_size=%zu bytes, error=%s",
                    config_.total_size_bytes,
                    e.what());
                cache_aligned_buffer_ = std::move(cpu_buffer);
            }
        } else {
            RTP_LLM_LOG_INFO("host block pool uses pageable CPU memory, total_size=%zu bytes",
                             config_.total_size_bytes);
            cache_aligned_buffer_ = std::move(cpu_buffer);
        }
        RTP_LLM_LOG_INFO("mark host block pool dont dump, ptr=%p, size=%zu",
                         cache_aligned_buffer_.data_ptr(),
                         config_.total_size_bytes);
        markHostBlockPoolDontDump(cache_aligned_buffer_.data_ptr(), config_.total_size_bytes);
    } else if (use_pinned_cpu_backing_) {
        initializePinnedCpuBuffer("device block pool pinned CPU backing");
    } else if (use_cuda_malloc_backing_) {
        initializeCudaMallocBuffer();
    } else {
        cache_aligned_buffer_ = allocatePausableDeviceBacking(config_.total_size_bytes);
    }
    cache_base_ptr_ = cache_aligned_buffer_.data_ptr();
    RTP_LLM_CHECK_WITH_INFO(cache_base_ptr_ != nullptr, "block pool allocate cache aligned buffer is null");
    const bool              is_cuda     = cache_aligned_buffer_.is_cuda();
    const bool              is_pinned   = !is_cuda && cache_aligned_buffer_.is_pinned();
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    RTP_LLM_LOG_INFO("BlockPool backing selected: pool_name=%s allocation_type=%s requested_backing=%s "
                     "actual_backing=%s is_cuda=%d is_pinned=%d ptr=%p total_size=%zu bytes total_size_mb=%.2f "
                     "block_num=%u memory_layouts=%zu",
                     config_.pool_name.c_str(),
                     allocationTypeName(allocation_type_),
                     requestedBackingName(allocation_type_, use_pinned_cpu_backing_, use_cuda_malloc_backing_),
                     memoryTypeName(where()),
                     is_cuda,
                     is_pinned,
                     cache_base_ptr_,
                     config_.total_size_bytes,
                     static_cast<double>(config_.total_size_bytes) / kBytesPerMB,
                     config_.block_num,
                     config_.memory_layouts.size());
}

void BlockPool::initializePinnedCpuBuffer(const char* log_context) {
    RTP_LLM_LOG_WARNING(
        "%s, pool_name=%s, total_size=%zu bytes", log_context, config_.pool_name.c_str(), config_.total_size_bytes);
    auto cpu_buffer = torch::empty({static_cast<int64_t>(config_.total_size_bytes)},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    try {
        cache_aligned_buffer_ = cpu_buffer.pin_memory();
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("%s pin failed, total_size=%zu bytes, error=%s", log_context, config_.total_size_bytes, e.what());
    }
}

void BlockPool::initializeCudaMallocBuffer() {
#if USING_CUDA
    RTP_LLM_CHECK_WITH_INFO(allocation_type_ == AllocationType::DEVICE,
                            "cudaMalloc block pool backing requires DEVICE allocation");
    RTP_LLM_CHECK_WITH_INFO(config_.total_size_bytes > 0, "cudaMalloc block pool total_size_bytes must be > 0");

    int  device_id  = -1;
    auto device_err = cudaGetDevice(&device_id);
    RTP_LLM_CHECK_WITH_INFO(device_err == cudaSuccess,
                            "cudaGetDevice failed before cudaMalloc block pool allocation, error=%s",
                            cudaGetErrorString(device_err));

    void*      ptr = nullptr;
    const auto err = cudaMalloc(&ptr, config_.total_size_bytes);
    RTP_LLM_CHECK_WITH_INFO(err == cudaSuccess,
                            "cudaMalloc block pool failed, pool_name=%s, total_size=%zu bytes, error=%s",
                            config_.pool_name.c_str(),
                            config_.total_size_bytes,
                            cudaGetErrorString(err));

    auto deleter = [device_id](void* p) {
        if (p == nullptr) {
            return;
        }
        int current_device = -1;
        if (cudaGetDevice(&current_device) == cudaSuccess && current_device != device_id) {
            (void)cudaSetDevice(device_id);
            (void)cudaFree(p);
            (void)cudaSetDevice(current_device);
            return;
        }
        (void)cudaFree(p);
    };
    cache_aligned_buffer_ =
        torch::from_blob(ptr,
                         {static_cast<int64_t>(config_.total_size_bytes)},
                         std::move(deleter),
                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device_id)));
    RTP_LLM_LOG_INFO("cudaMalloc block pool backing allocated, pool_name=%s, ptr=%p, total_size=%zu bytes, device=%d",
                     config_.pool_name.c_str(),
                     ptr,
                     config_.total_size_bytes,
                     device_id);
#else
    RTP_LLM_FAIL("cudaMalloc block pool backing requested but this binary was not built with CUDA");
#endif
}

void BlockPool::initializeLayerMappings() {
    torch::Tensor full_tensor = cache_aligned_buffer_;

    size_t total_layers = 0;
    for (const auto& layout_cfg : config_.memory_layouts) {
        total_layers += static_cast<size_t>(layout_cfg.layer_num);
    }
    global_layer_to_local_.assign(total_layers, {-1, -1});
    global_layer_kv_tensors_.assign(total_layers, torch::Tensor());
    global_layer_kv_scale_tensors_.assign(total_layers, torch::Tensor());
}

void BlockPool::initializeLayoutStrategies() {
    layout_strategies_.resize(config_.memory_layouts.size());
    torch::Tensor full_tensor = cache_aligned_buffer_;

    size_t global_layer_begin = 0;
    for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
        processMemoryLayout(layout_idx, full_tensor, global_layer_begin);
        global_layer_begin += static_cast<size_t>(config_.memory_layouts[layout_idx].layer_num);
    }
}

void BlockPool::processMemoryLayout(size_t layout_idx, const torch::Tensor& full_tensor, size_t& global_layer_begin) {
    const auto& layout_cfg = config_.memory_layouts[layout_idx];

    // 创建 KV 缓存张量
    torch::Tensor kv_cache_tensor = createTensor(full_tensor,
                                                 static_cast<int64_t>(layout_cfg.kv_cache_offset_bytes),
                                                 static_cast<int64_t>(layout_cfg.kv_block_pool_size_bytes),
                                                 layout_idx,
                                                 "kv");
    // 创建缩放张量（如果需要）
    torch::Tensor kv_scale_tensor;
    if (layout_cfg.hasScale()) {
        kv_scale_tensor = createTensor(full_tensor,
                                       static_cast<int64_t>(layout_cfg.kv_scale_offset_bytes),
                                       static_cast<int64_t>(layout_cfg.kv_scale_pool_size_bytes),
                                       layout_idx,
                                       "kv_scale");
    }

    // 初始化内存布局策略
    initializeLayoutStrategy(layout_idx, layout_cfg, kv_cache_tensor, kv_scale_tensor);

    // 处理层张量映射
    processLayerTensors(layout_idx, layout_cfg, global_layer_begin);

    // 记录初始化信息
    RTP_LLM_LOG_INFO(
        "MemoryLayout[%zu] initialized: layer_num=%u block_num=%u kv_off=%zu kv_bytes=%zu scale_off=%zu scale_bytes=%zu",
        layout_idx,
        layout_cfg.layer_num,
        layout_cfg.block_num,
        layout_cfg.kv_cache_offset_bytes,
        layout_cfg.kv_block_pool_size_bytes,
        layout_cfg.kv_scale_offset_bytes,
        layout_cfg.kv_scale_pool_size_bytes);
}

torch::Tensor BlockPool::createTensor(
    const torch::Tensor& full_tensor, int64_t offset, int64_t size, size_t layout_idx, const std::string& tensor_type) {
    RTP_LLM_CHECK_WITH_INFO(offset >= 0 && size >= 0 && offset + size <= full_tensor.numel(),
                            "layout[%zu] %s tensor out of range: off=%ld bytes=%ld full=%ld",
                            layout_idx,
                            tensor_type.c_str(),
                            offset,
                            size,
                            full_tensor.numel());
    return full_tensor.narrow(0, offset, size);
}

void BlockPool::initializeLayoutStrategy(size_t                    layout_idx,
                                         const MemoryLayoutConfig& layout_cfg,
                                         torch::Tensor&            kv_cache_tensor,
                                         torch::Tensor&            kv_scale_tensor) {
    void* layout_cache_base_ptr =
        static_cast<void*>(static_cast<char*>(cache_base_ptr_) + layout_cfg.kv_cache_offset_bytes);

    layout_strategies_[layout_idx] = std::make_unique<MemoryLayoutStrategy>();
    RTP_LLM_CHECK_WITH_INFO(layout_strategies_[layout_idx] != nullptr,
                            "Failed to create memory layout strategy for layout[%zu]",
                            layout_idx);

    RTP_LLM_CHECK_WITH_INFO(
        layout_strategies_[layout_idx]->init(layout_cfg, kv_cache_tensor, kv_scale_tensor, layout_cache_base_ptr),
        "Failed to initialize memory layout strategy for layout[%zu]",
        layout_idx);
}

void BlockPool::processLayerTensors(size_t                    layout_idx,
                                    const MemoryLayoutConfig& layout_cfg,
                                    size_t&                   global_layer_begin) {
    // 获取层张量
    auto layer_tensors = layout_strategies_[layout_idx]->getLayerCacheTensors();
    RTP_LLM_CHECK_WITH_INFO(layer_tensors.size() == static_cast<size_t>(layout_cfg.layer_num),
                            "layout[%zu] layer tensors size mismatch: got=%zu expect=%u",
                            layout_idx,
                            layer_tensors.size(),
                            layout_cfg.layer_num);

    // 映射全局层到局部层，并设置KV张量
    for (size_t local_layer = 0; local_layer < static_cast<size_t>(layout_cfg.layer_num); ++local_layer) {
        const size_t global_layer = global_layer_begin + local_layer;
        RTP_LLM_CHECK_WITH_INFO(global_layer < global_layer_to_local_.size(), "global layer index out of range");
        global_layer_to_local_[global_layer]   = {static_cast<int>(layout_idx), static_cast<int>(local_layer)};
        global_layer_kv_tensors_[global_layer] = layer_tensors[local_layer];
    }

    // 处理缩放张量（如果存在）
    auto scale_tensors = layout_strategies_[layout_idx]->getLayerScaleCacheTensors();
    if (!scale_tensors.empty()) {
        RTP_LLM_CHECK_WITH_INFO(scale_tensors.size() == static_cast<size_t>(layout_cfg.layer_num),
                                "layout[%zu] scale tensors size mismatch: got=%zu expect=%u",
                                layout_idx,
                                scale_tensors.size(),
                                layout_cfg.layer_num);
        for (size_t local_layer = 0; local_layer < static_cast<size_t>(layout_cfg.layer_num); ++local_layer) {
            const size_t global_layer                    = global_layer_begin + local_layer;
            global_layer_kv_scale_tensors_[global_layer] = scale_tensors[local_layer];
        }
    }
}

bool BlockPool::init() {
    validateConfig();
    initializeCacheBuffer();
    initializeLayerMappings();
    initializeLayoutStrategies();
    initFreeBlocks();

    RTP_LLM_LOG_INFO("BlockPool init success: memory_layouts=%zu, total_layers=%zu, total_size=%zu bytes",
                     config_.memory_layouts.size(),
                     global_layer_to_local_.size(),
                     config_.total_size_bytes);
    return true;
}

void BlockPool::initFreeBlocks() {
    // block 0 is reserved
    for (BlockIdxType i = 1; i < static_cast<BlockIdxType>(config_.block_num); ++i) {
        free_block_ids_.insert(i);
    }
    request_ref_counter_.init(config_.block_num);
    connector_ref_counter_.init(config_.block_num);
    req_con_ref_counter_.init(config_.block_num);
    block_cache_ref_counter_.init(config_.block_num);
    req_cache_ref_counter_.init(config_.block_num);
    block_cache_ = std::make_shared<BlockCache>();
}

BlockCachePtr BlockPool::blockCache() {
    return block_cache_;
}

void BlockPool::resetMetadata() {
    std::scoped_lock lock(ref_mu_, free_mu_);
    free_block_ids_.clear();
    // block 0 is reserved, same as initFreeBlocks()
    for (BlockIdxType i = 1; i < static_cast<BlockIdxType>(config_.block_num); ++i) {
        free_block_ids_.insert(i);
    }
    request_ref_counter_.init(config_.block_num);
    connector_ref_counter_.init(config_.block_num);
    req_con_ref_counter_.init(config_.block_num);
    block_cache_ref_counter_.init(config_.block_num);
    req_cache_ref_counter_.init(config_.block_num);
    RTP_LLM_LOG_INFO("BlockPool metadata reset to fresh state: free_blocks=%zu, total_blocks=%u",
                     free_block_ids_.size(),
                     config_.block_num);
}

void BlockPool::releaseHostBuffer() {
    RTP_LLM_CHECK_WITH_INFO(allocation_type_ == AllocationType::HOST,
                            "releaseHostBuffer is only valid for HOST block pool");
    if (host_released_) {
        return;
    }
    {
        // Prevent malloc() from handing out blocks that point into the freed buffer.
        std::scoped_lock lock(ref_mu_, free_mu_);
        free_block_ids_.clear();
    }
    // Drop everything that views into cache_aligned_buffer_ so the tensor's refcount
    // reaches zero and the pinned host memory is actually returned to the OS.
    layout_strategies_.clear();
    global_layer_kv_tensors_.clear();
    global_layer_kv_scale_tensors_.clear();
    global_layer_to_local_.clear();
    block_cache_          = std::make_shared<BlockCache>();
    cache_base_ptr_       = nullptr;
    cache_aligned_buffer_ = torch::Tensor();
    // The buffer was allocated via torch's CachingHostAllocator (pin_memory()); dropping
    // the tensor only returns the block to torch's pinned-memory *cache*, NOT to the OS.
    // Flush that cache so the pinned pages are actually cudaHostFree'd and RAM is reclaimed
    // (the whole point of discarding the memory cache on sleep). Only frees unused blocks.
    at::getHostAllocator(at::kCUDA)->empty_cache();
    host_released_ = true;
    RTP_LLM_LOG_INFO("BlockPool host buffer released for sleep (%zu bytes freed)", config_.total_size_bytes);
}

void BlockPool::reallocateHostBuffer() {
    RTP_LLM_CHECK_WITH_INFO(allocation_type_ == AllocationType::HOST,
                            "reallocateHostBuffer is only valid for HOST block pool");
    if (!host_released_) {
        return;
    }
    // Mirror init(): re-create the pinned buffer and every derived layer view/layout.
    initializeCacheBuffer();
    initializeLayerMappings();
    initializeLayoutStrategies();
    {
        // Locked reset of block metadata to a fresh pool (same as initFreeBlocks()),
        // safe against the connector's metrics-reporter thread reading counts.
        std::scoped_lock lock(ref_mu_, free_mu_);
        free_block_ids_.clear();
        for (BlockIdxType i = 1; i < static_cast<BlockIdxType>(config_.block_num); ++i) {
            free_block_ids_.insert(i);
        }
        request_ref_counter_.init(config_.block_num);
        connector_ref_counter_.init(config_.block_num);
        req_con_ref_counter_.init(config_.block_num);
        block_cache_ref_counter_.init(config_.block_num);
        req_cache_ref_counter_.init(config_.block_num);
        block_cache_ = std::make_shared<BlockCache>();
    }
    host_released_ = false;
    RTP_LLM_LOG_INFO("BlockPool host buffer reallocated on wake (%zu bytes)", config_.total_size_bytes);
}

std::vector<torch::Tensor> BlockPool::allLayerCacheBase() const {
    return global_layer_kv_tensors_;
}

std::vector<torch::Tensor> BlockPool::allLayerScaleCacheBase() const {
    return global_layer_kv_scale_tensors_;
}

BlockIndicesType BlockPool::malloc(int num_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    if (num_blocks <= 0) {
        return {};
    }
    BlockIndicesType block_ids;
    block_ids.reserve(num_blocks);

    {
        std::scoped_lock lock(ref_mu_, free_mu_);
        if (free_block_ids_.size() < static_cast<size_t>(num_blocks)) {
            RTP_LLM_LOG_WARNING(
                "Block pool only has %zu free blocks, cannot allocate %d blocks", free_block_ids_.size(), num_blocks);
            return {};
        }
        auto first = free_block_ids_.begin();
        auto last  = std::next(first, num_blocks);
        block_ids.assign(first, last);
        free_block_ids_.erase(first, last);
        request_ref_counter_.incrementRefCounter(block_ids);
        req_con_ref_counter_.incrementRefCounter(block_ids);
        req_cache_ref_counter_.incrementRefCounter(block_ids);
    }

    return block_ids;
}

void BlockPool::requestFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    requestFree(block_ids);
}

void BlockPool::requestFree(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    request_ref_counter_.decrementRefCounter(block_ids);
    req_con_ref_counter_.decrementRefCounter(block_ids);
    req_cache_ref_counter_.decrementRefCounter(block_ids);
    tryFreeBlocks(block_ids);
}

void BlockPool::connectorFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    connectorFree(block_ids);
}

void BlockPool::connectorFree(const BlockIndicesType& block_indices) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    connector_ref_counter_.decrementRefCounter(block_indices);
    req_con_ref_counter_.decrementRefCounter(block_indices);
    tryFreeBlocks(block_indices);
}

void BlockPool::blockCacheFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    blockCacheFree(block_ids);
}

void BlockPool::blockCacheFree(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    block_cache_ref_counter_.decrementRefCounter(block_ids);
    req_cache_ref_counter_.decrementRefCounter(block_ids);
    tryFreeBlocks(block_ids);
}

// Must be called with ref_mu_ and free_mu_ held.
void BlockPool::tryFreeBlocks(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    for (const auto& block_id : block_ids) {
        if (req_con_ref_counter_.getRefCounter(block_id) == 0
            && block_cache_ref_counter_.getRefCounter(block_id) == 0) {
            free_block_ids_.insert(block_id);
        }
    }
}

void BlockPool::requestReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    requestReference(block_ids);
}

void BlockPool::requestReference(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    request_ref_counter_.incrementRefCounter(block_ids);
    req_con_ref_counter_.incrementRefCounter(block_ids);
    req_cache_ref_counter_.incrementRefCounter(block_ids);
    for (const auto& block_id : block_ids) {
        free_block_ids_.erase(block_id);
    }
}

void BlockPool::connectorReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    connectorReference(block_ids);
}

void BlockPool::connectorReference(const BlockIndicesType& block_indices) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    connector_ref_counter_.incrementRefCounter(block_indices);
    req_con_ref_counter_.incrementRefCounter(block_indices);
    for (const auto& block_id : block_indices) {
        free_block_ids_.erase(block_id);
    }
}

void BlockPool::blockCacheReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    blockCacheReference(block_ids);
}

void BlockPool::blockCacheReference(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    block_cache_ref_counter_.incrementRefCounter(block_ids);
    req_cache_ref_counter_.incrementRefCounter(block_ids);
    for (const auto& block_id : block_ids) {
        free_block_ids_.erase(block_id);
    }
}

void BlockPool::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    if (cache_store) {
        cache_store_ = std::move(cache_store);
    }
    if (cache_store_ && !kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to register user mr");
        auto       memory_util = cache_store_->getMemoryUtil();
        const bool gpu         = where() == MemoryType::MEMORY_GPU;

        // Track buffers registered in THIS call so a mid-loop failure rolls them back instead of
        // leaking already-registered MRs (a leak also makes a wake retry double-register). On the
        // wake path this runs inside the registerMr hook, whose invokeHookNoThrow catches the throw
        // below and drives the controller to ERROR -- a clean recoverable failure, not an abort.
        struct Registered {
            size_t      layout_idx;
            size_t      offset_bytes;
            std::string type;
        };
        std::vector<Registered> registered;
        auto                    rollback = [&]() {
            for (auto it = registered.rbegin(); it != registered.rend(); ++it) {
                deregisterUserMrForBuffer(memory_util, it->layout_idx, it->offset_bytes, gpu, it->type);
            }
        };

        for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
            const auto& layout_cfg = config_.memory_layouts[layout_idx];

            // Register KV buffer
            if (!registerUserMrForBuffer(memory_util,
                                         layout_idx,
                                         layout_cfg.kv_cache_offset_bytes,
                                         layout_cfg.kv_block_pool_size_bytes,
                                         layout_cfg.kv_block_stride_bytes,
                                         gpu,
                                         "kv")) {
                rollback();
                throw RTP_EXCEPTION("register user mr for block pool layout[%zu] kv buffer failed (rolled back %zu)",
                                    layout_idx,
                                    registered.size());
            }
            registered.push_back({layout_idx, layout_cfg.kv_cache_offset_bytes, "kv"});

            // Register scale buffer if present
            if (layout_cfg.hasScale()) {
                if (!registerUserMrForBuffer(memory_util,
                                             layout_idx,
                                             layout_cfg.kv_scale_offset_bytes,
                                             layout_cfg.kv_scale_pool_size_bytes,
                                             layout_cfg.kv_scale_stride_bytes,
                                             gpu,
                                             "scale")) {
                    rollback();
                    throw RTP_EXCEPTION(
                        "register user mr for block pool layout[%zu] scale buffer failed (rolled back %zu)",
                        layout_idx,
                        registered.size());
                }
                registered.push_back({layout_idx, layout_cfg.kv_scale_offset_bytes, "scale"});
            }
        }

        kvcache_reg_mr_ = true;
    }
}

void BlockPool::deregUserMr() {
    if (kvcache_reg_mr_ && cache_store_) {
        RTP_LLM_LOG_INFO("start to deregister user mr");
        auto       memory_util = cache_store_->getMemoryUtil();
        const bool gpu         = where() == MemoryType::MEMORY_GPU;

        // Attempt ALL deregs even if one fails, so we never leave a subset registered.
        bool all_ok = true;
        for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
            const auto& layout_cfg = config_.memory_layouts[layout_idx];

            // Deregister KV buffer
            all_ok &= deregisterUserMrForBuffer(memory_util, layout_idx, layout_cfg.kv_cache_offset_bytes, gpu, "kv");

            // Deregister scale buffer if present
            if (layout_cfg.hasScale()) {
                all_ok &=
                    deregisterUserMrForBuffer(memory_util, layout_idx, layout_cfg.kv_scale_offset_bytes, gpu, "scale");
            }
        }

        if (!all_ok) {
            // A dereg failed: the MR may still pin KV pages the sleep path is about to VMM-unmap.
            // Leave kvcache_reg_mr_ set (MRs believed live) and throw so the synchronizeAndDeregisterMr
            // hook fails into ERROR rather than releasing physical memory under a live MR -- a dangling
            // MR is exactly what historically forced a GPU reset. Caught by invokeHookNoThrow.
            throw RTP_EXCEPTION("deregister user mr for block pool failed for one or more buffers");
        }

        RTP_LLM_LOG_INFO("deregister user mr for block pool success");
        kvcache_reg_mr_ = false;
    }
}

bool BlockPool::registerUserMrForBuffer(std::shared_ptr<rtp_llm::MemoryUtil> memory_util,
                                        size_t                               layout_idx,
                                        size_t                               offset_bytes,
                                        size_t                               bytes,
                                        size_t                               stride_bytes,
                                        bool                                 gpu,
                                        const std::string&                   buffer_type) {
    void* base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_) + static_cast<ptrdiff_t>(offset_bytes));
    auto  start_us = currentTimeUs();

    if (!memory_util->regUserMr(base_ptr, bytes, gpu, stride_bytes)) {
        RTP_LLM_LOG_ERROR(
            "register user mr for block pool layout[%zu] %s buffer failed", layout_idx, buffer_type.c_str());
        return false;
    }

    auto cost_ms = (currentTimeUs() - start_us) / 1000;
    mr_cost_time_ms_ += cost_ms;

    RTP_LLM_LOG_INFO("register user mr success: layout[%zu] %s base=%p len=%zu aligned=%zu cost=%ld ms",
                     layout_idx,
                     buffer_type.c_str(),
                     base_ptr,
                     bytes,
                     stride_bytes,
                     cost_ms);
    return true;
}

bool BlockPool::deregisterUserMrForBuffer(std::shared_ptr<rtp_llm::MemoryUtil> memory_util,
                                          size_t                               layout_idx,
                                          size_t                               offset_bytes,
                                          bool                                 gpu,
                                          const std::string&                   buffer_type) {
    void* base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_) + static_cast<ptrdiff_t>(offset_bytes));

    if (!memory_util->deregUserMr(base_ptr, gpu)) {
        RTP_LLM_LOG_ERROR(
            "deregister user mr for block pool layout[%zu] %s buffer failed", layout_idx, buffer_type.c_str());
        return false;
    }
    return true;
}

size_t BlockPool::freeBlocksNum() const {
    std::lock_guard<std::mutex> free_lock(free_mu_);
    return free_block_ids_.size();
}

size_t BlockPool::totalBlocksNum() const {
    // reserve block 0 for internal use
    return config_.block_num - 1;
}

// Available blocks need to satisfy two conditions:
// 1. not referenced by a request
// 2. not referenced by connector(read or write)
size_t BlockPool::availableBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return req_con_ref_counter_.freeBlockNum();
}

size_t BlockPool::requestRefBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return request_ref_counter_.busyBlockNum();
}

size_t BlockPool::connectorRefBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return connector_ref_counter_.busyBlockNum();
}

size_t BlockPool::blockCacheRefBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return block_cache_ref_counter_.busyBlockNum();
}

size_t BlockPool::notInUseBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return req_cache_ref_counter_.freeBlockNum();
}

// MTP support: Map global_layer_id to (model_index, local_layer_id).
// Returns {layout_index, local_layer_id}. layout_index is the index in BlockPoolConfig.memory_layouts.
std::pair<int, int> BlockPool::mapGlobalLayerIdToLocal(int global_layer_id) const {
    if (global_layer_id < 0 || static_cast<size_t>(global_layer_id) >= global_layer_to_local_.size()) {
        RTP_LLM_LOG_ERROR(
            "Global layer_id %d out of range (total layers: %zu)", global_layer_id, global_layer_to_local_.size());
        return {-1, -1};
    }

    return global_layer_to_local_[static_cast<size_t>(global_layer_id)];
}

BlockAddrInfo BlockPool::convertIndexToAddr(int layer_id, int block_id) const {
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    return layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToAddr(local_layer_id, block_id);
}

std::vector<BlockInfo> BlockPool::convertIndexToBuffer(int layer_id, int block_id) const {
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    return layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToBuffer(local_layer_id, block_id);
}

std::vector<BlockInfo>
BlockPool::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);

    return layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToBuffer(
        local_layer_id, block_id, partition_count, partition_id);
}

MemoryType BlockPool::where() const {
    if (cache_aligned_buffer_.is_cuda()) {
        return MemoryType::MEMORY_GPU;
    }
    return cache_aligned_buffer_.is_pinned() ? MemoryType::MEMORY_CPU_PINNED : MemoryType::MEMORY_CPU;
}

void BlockPool::checkLayoutValidity(int layout_id) const {
    RTP_LLM_CHECK_WITH_INFO(layout_id >= 0 && static_cast<size_t>(layout_id) < layout_strategies_.size(),
                            "Memory layout ID %d out of range (max: %zu)",
                            layout_id,
                            layout_strategies_.size());
}

}  // namespace rtp_llm
