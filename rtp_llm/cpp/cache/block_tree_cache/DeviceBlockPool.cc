#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#if USING_CUDA
#include <cuda_runtime.h>
#endif

namespace rtp_llm {

namespace {

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

}  // namespace

std::shared_ptr<const DeviceBlockPoolConfig> DeviceBlockPool::normalizeConfig(const std::shared_ptr<const DeviceBlockPoolConfig>& config) {
    RTP_LLM_CHECK(config != nullptr);
    RTP_LLM_CHECK(config->pool_type == BlockPoolType::DEVICE);
    RTP_LLM_CHECK(config->free_block_order_policy == FreeBlockOrderPolicy::ANY_ORDER);
    RTP_LLM_CHECK_WITH_INFO(
        !config->memory_layouts.empty(), "device block pool [%s] memory_layouts must not be empty",
        config->pool_name.c_str());

    // Every memory layout shares the same block id space (see rtp_llm::BlockPool /
    // rtp_llm::BlockPoolConfig), so physical_block_count is reconciled against
    // memory_layouts[*].block_num - the first layout's block_num is authoritative and
    // every other layout must agree with it.
    const size_t computed_physical_block_count = static_cast<size_t>(config->memory_layouts.front().block_num);
    for (size_t layout_idx = 0; layout_idx < config->memory_layouts.size(); ++layout_idx) {
        const auto& layout_cfg = config->memory_layouts[layout_idx];
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(layout_cfg.block_num) == computed_physical_block_count,
                                "device block pool [%s] memory_layouts[%zu].block_num=%u mismatches "
                                "memory_layouts[0].block_num=%zu",
                                config->pool_name.c_str(),
                                layout_idx,
                                layout_cfg.block_num,
                                computed_physical_block_count);
        RTP_LLM_CHECK_WITH_INFO(
            layout_cfg.layer_num > 0, "device block pool [%s] memory_layouts[%zu].layer_num must be > 0",
            config->pool_name.c_str(), layout_idx);
        RTP_LLM_CHECK_WITH_INFO(layout_cfg.kv_block_pool_size_bytes > 0,
                                "device block pool [%s] memory_layouts[%zu].kv_block_pool_size_bytes must be > 0",
                                config->pool_name.c_str(),
                                layout_idx);
    }

    if (config->physical_block_count != 0) {
        RTP_LLM_CHECK_WITH_INFO(config->physical_block_count == computed_physical_block_count,
                                "device block pool [%s] physical_block_count [%zu] does not match "
                                "memory_layouts[*].block_num [%zu]",
                                config->pool_name.c_str(),
                                config->physical_block_count,
                                computed_physical_block_count);
    }
    RTP_LLM_CHECK_WITH_INFO(computed_physical_block_count > 1,
                            "device block pool [%s] physical_block_count [%zu] (from memory_layouts[*].block_num) "
                            "must be > 1",
                            config->pool_name.c_str(),
                            computed_physical_block_count);

    auto normalized                  = std::make_shared<DeviceBlockPoolConfig>(*config);
    normalized->physical_block_count = computed_physical_block_count;
    return normalized;
}

DeviceBlockPool::DeviceBlockPool(std::shared_ptr<const DeviceBlockPoolConfig> config): IBlockPool(normalizeConfig(config)) {}

DeviceBlockPool::~DeviceBlockPool() {
    cache_aligned_buffer_ = torch::Tensor();
}

const DeviceBlockPoolConfig& DeviceBlockPool::config() const {
    return configAs<DeviceBlockPoolConfig>(BlockPoolType::DEVICE);
}

bool DeviceBlockPool::init() {
    RTP_LLM_CHECK(!initialized());
    const auto& cfg = config();

    initializeCacheBuffer();
    initializeLayerMappings();
    initializeLayoutStrategies();

    markInitialized();

    RTP_LLM_LOG_INFO(
        "block_tree_cache::DeviceBlockPool init success: pool_name=%s memory_layouts=%zu total_layers=%zu "
        "total_size=%zu bytes physical_block_count=%zu",
        cfg.pool_name.c_str(),
        cfg.memory_layouts.size(),
        global_layer_to_local_.size(),
        cfg.total_size_bytes,
        cfg.physical_block_count);
    return true;
}

// Adapted from rtp_llm::BlockPool::initializeCacheBuffer (rtp_llm/cpp/cache/BlockPool.cc).
// The host-backed (pageable CPU with pin fallback + MADV_DONTDUMP) branch of the old
// pool is intentionally dropped here: this class is the DEVICE pool of the v4 family
// (rtp_llm::block_tree_cache::HostBlockPool already covers host backing), so only the
// DEVICE allocation_type path (plain CUDA tensor, pinned-CPU-backed, or cudaMalloc-backed)
// is kept.
void DeviceBlockPool::initializeCacheBuffer() {
    const auto& cfg = config();
    RTP_LLM_CHECK_WITH_INFO(cfg.allocation_type == AllocationType::DEVICE,
                            "block_tree_cache::DeviceBlockPool [%s] only supports AllocationType::DEVICE",
                            cfg.pool_name.c_str());
    RTP_LLM_CHECK_WITH_INFO(
        cfg.total_size_bytes > 0, "device block pool [%s] total_size_bytes must be > 0", cfg.pool_name.c_str());

    if (cfg.use_pinned_cpu_backing) {
        initializePinnedCpuBuffer("device block pool pinned CPU backing");
    } else if (cfg.use_cuda_malloc_backing) {
        initializeCudaMallocBuffer();
    } else {
        cache_aligned_buffer_ = torch::empty({static_cast<int64_t>(cfg.total_size_bytes)},
                                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    }
    cache_base_ptr_ = cache_aligned_buffer_.data_ptr();
    RTP_LLM_CHECK_WITH_INFO(
        cache_base_ptr_ != nullptr, "device block pool [%s] allocate cache aligned buffer is null",
        cfg.pool_name.c_str());

    const bool           is_cuda   = cache_aligned_buffer_.is_cuda();
    const bool           is_pinned = !is_cuda && cache_aligned_buffer_.is_pinned();
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    RTP_LLM_LOG_INFO("block_tree_cache::DeviceBlockPool backing selected: pool_name=%s allocation_type=%s "
                     "actual_backing=%s is_cuda=%d is_pinned=%d ptr=%p total_size=%zu bytes total_size_mb=%.2f "
                     "memory_layouts=%zu",
                     cfg.pool_name.c_str(),
                     allocationTypeName(cfg.allocation_type),
                     memoryTypeName(where()),
                     is_cuda,
                     is_pinned,
                     cache_base_ptr_,
                     cfg.total_size_bytes,
                     static_cast<double>(cfg.total_size_bytes) / kBytesPerMB,
                     cfg.memory_layouts.size());
}

void DeviceBlockPool::initializePinnedCpuBuffer(const char* log_context) {
    const auto& cfg = config();
    RTP_LLM_LOG_WARNING(
        "%s, pool_name=%s, total_size=%zu bytes", log_context, cfg.pool_name.c_str(), cfg.total_size_bytes);
    auto cpu_buffer = torch::empty({static_cast<int64_t>(cfg.total_size_bytes)},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    try {
        cache_aligned_buffer_ = cpu_buffer.pin_memory();
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("%s pin failed, pool_name=%s total_size=%zu bytes, error=%s",
                     log_context,
                     cfg.pool_name.c_str(),
                     cfg.total_size_bytes,
                     e.what());
    }
}

void DeviceBlockPool::initializeCudaMallocBuffer() {
#if USING_CUDA
    const auto& cfg = config();
    RTP_LLM_CHECK_WITH_INFO(
        cfg.total_size_bytes > 0, "cudaMalloc device block pool total_size_bytes must be > 0");

    int  device_id  = -1;
    auto device_err = cudaGetDevice(&device_id);
    RTP_LLM_CHECK_WITH_INFO(device_err == cudaSuccess,
                            "cudaGetDevice failed before cudaMalloc device block pool allocation, error=%s",
                            cudaGetErrorString(device_err));

    void*      ptr = nullptr;
    const auto err = cudaMalloc(&ptr, cfg.total_size_bytes);
    RTP_LLM_CHECK_WITH_INFO(err == cudaSuccess,
                            "cudaMalloc device block pool failed, pool_name=%s, total_size=%zu bytes, error=%s",
                            cfg.pool_name.c_str(),
                            cfg.total_size_bytes,
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
                         {static_cast<int64_t>(cfg.total_size_bytes)},
                         std::move(deleter),
                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device_id)));
    RTP_LLM_LOG_INFO(
        "cudaMalloc device block pool backing allocated, pool_name=%s, ptr=%p, total_size=%zu bytes, device=%d",
        cfg.pool_name.c_str(),
        ptr,
        cfg.total_size_bytes,
        device_id);
#else
    RTP_LLM_FAIL("cudaMalloc device block pool backing requested but this binary was not built with CUDA, "
                 "pool_name=%s",
                 config().pool_name.c_str());
#endif
}

void DeviceBlockPool::initializeLayerMappings() {
    const auto& cfg = config();
    size_t      total_layers = 0;
    for (const auto& layout_cfg : cfg.memory_layouts) {
        total_layers += static_cast<size_t>(layout_cfg.layer_num);
    }
    global_layer_to_local_.assign(total_layers, {-1, -1});
}

void DeviceBlockPool::initializeLayoutStrategies() {
    const auto&   cfg         = config();
    torch::Tensor full_tensor = cache_aligned_buffer_;

    layout_strategies_.resize(cfg.memory_layouts.size());

    size_t global_layer_begin = 0;
    for (size_t layout_idx = 0; layout_idx < cfg.memory_layouts.size(); ++layout_idx) {
        processMemoryLayout(layout_idx, full_tensor, global_layer_begin);
        global_layer_begin += static_cast<size_t>(cfg.memory_layouts[layout_idx].layer_num);
    }
}

void DeviceBlockPool::processMemoryLayout(size_t layout_idx, const torch::Tensor& full_tensor, size_t& global_layer_begin) {
    const auto& layout_cfg = config().memory_layouts[layout_idx];

    torch::Tensor kv_cache_tensor = createTensor(full_tensor,
                                                 static_cast<int64_t>(layout_cfg.kv_cache_offset_bytes),
                                                 static_cast<int64_t>(layout_cfg.kv_block_pool_size_bytes),
                                                 layout_idx,
                                                 "kv");
    torch::Tensor kv_scale_tensor;
    if (layout_cfg.hasScale()) {
        kv_scale_tensor = createTensor(full_tensor,
                                       static_cast<int64_t>(layout_cfg.kv_scale_offset_bytes),
                                       static_cast<int64_t>(layout_cfg.kv_scale_pool_size_bytes),
                                       layout_idx,
                                       "kv_scale");
    }

    initializeLayoutStrategy(layout_idx, layout_cfg, kv_cache_tensor, kv_scale_tensor);
    processLayerTensors(layout_idx, layout_cfg, global_layer_begin);

    RTP_LLM_LOG_INFO("block_tree_cache::DeviceBlockPool MemoryLayout[%zu] initialized: pool_name=%s layer_num=%u "
                     "block_num=%u kv_off=%zu kv_bytes=%zu scale_off=%zu scale_bytes=%zu",
                     layout_idx,
                     config().pool_name.c_str(),
                     layout_cfg.layer_num,
                     layout_cfg.block_num,
                     layout_cfg.kv_cache_offset_bytes,
                     layout_cfg.kv_block_pool_size_bytes,
                     layout_cfg.kv_scale_offset_bytes,
                     layout_cfg.kv_scale_pool_size_bytes);
}

torch::Tensor DeviceBlockPool::createTensor(
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

void DeviceBlockPool::initializeLayoutStrategy(size_t                    layout_idx,
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

void DeviceBlockPool::processLayerTensors(size_t                    layout_idx,
                                    const MemoryLayoutConfig& layout_cfg,
                                    size_t&                   global_layer_begin) {
    // Unlike rtp_llm::BlockPool, this v4 pool does not keep the global layer -> KV
    // tensor mapping around (allLayerCacheBase()/allLayerScaleCacheBase() are not part
    // of this class's interface); it only needs the global layer -> (layout_index,
    // local_layer_id) mapping used by blockBuffers().
    for (size_t local_layer = 0; local_layer < static_cast<size_t>(layout_cfg.layer_num); ++local_layer) {
        const size_t global_layer = global_layer_begin + local_layer;
        RTP_LLM_CHECK_WITH_INFO(global_layer < global_layer_to_local_.size(), "global layer index out of range");
        global_layer_to_local_[global_layer] = {static_cast<int>(layout_idx), static_cast<int>(local_layer)};
    }
}

std::pair<int, int> DeviceBlockPool::mapGlobalLayerIdToLocal(int global_layer_id) const {
    if (global_layer_id < 0 || static_cast<size_t>(global_layer_id) >= global_layer_to_local_.size()) {
        RTP_LLM_LOG_ERROR("Global layer_id %d out of range (total layers: %zu), pool_name=%s",
                          global_layer_id,
                          global_layer_to_local_.size(),
                          config().pool_name.c_str());
        return {-1, -1};
    }
    return global_layer_to_local_[static_cast<size_t>(global_layer_id)];
}

void DeviceBlockPool::checkLayoutValidity(int layout_id) const {
    RTP_LLM_CHECK_WITH_INFO(layout_id >= 0 && static_cast<size_t>(layout_id) < layout_strategies_.size(),
                            "Memory layout ID %d out of range (max: %zu)",
                            layout_id,
                            layout_strategies_.size());
}

std::vector<DeviceBlockBuffer>
DeviceBlockPool::toDeviceBlockBuffers(const std::vector<BlockInfo>& infos, BlockIdxType block) const {
    std::vector<DeviceBlockBuffer> buffers;
    buffers.reserve(infos.size());
    for (const auto& info : infos) {
        buffers.push_back(DeviceBlockBuffer{block, info.addr, info.size_bytes});
    }
    return buffers;
}

std::vector<DeviceBlockBuffer> DeviceBlockPool::blockBuffers(int layer_id, BlockIdxType block) const {
    RTP_LLM_CHECK(initialized());
    RTP_LLM_CHECK(isAllocated(block));

    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    auto infos = layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToBuffer(local_layer_id, block);
    return toDeviceBlockBuffers(infos, block);
}

std::vector<DeviceBlockBuffer>
DeviceBlockPool::blockBuffers(int layer_id, BlockIdxType block, int partition_count, int partition_id) const {
    RTP_LLM_CHECK(initialized());
    RTP_LLM_CHECK(isAllocated(block));

    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    auto infos = layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToBuffer(
        local_layer_id, block, partition_count, partition_id);
    return toDeviceBlockBuffers(infos, block);
}

MemoryType DeviceBlockPool::where() const {
    if (cache_aligned_buffer_.is_cuda()) {
        return MemoryType::MEMORY_GPU;
    }
    return cache_aligned_buffer_.is_pinned() ? MemoryType::MEMORY_CPU_PINNED : MemoryType::MEMORY_CPU;
}

std::string DeviceBlockPool::debugString() const {
    std::ostringstream oss;
    oss << "DeviceBlockPool{" << IBlockPool::debugString() << ", pool_name=" << config().pool_name
        << ", memory_layouts=" << config().memory_layouts.size() << ", total_size_bytes=" << config().total_size_bytes
        << ", where=" << memoryTypeName(where()) << "}";
    return oss.str();
}

}  // namespace rtp_llm
