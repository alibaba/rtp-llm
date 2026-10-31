#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"

#include <algorithm>
#include <cctype>
#include <cassert>
#include <limits>
#include <unordered_map>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#include <torch/cuda.h>
#endif

namespace rtp_llm {

namespace {

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

DeviceBlockPool::DeviceBlockPool(std::shared_ptr<const DeviceBlockPoolConfig> config):
    IBlockPool(normalizeConfig(config)) {
    refcounts_.assign(this->config().physical_block_count, 0);
    if (isOffload() && this->config().offload->hbm_blocks < this->config().offload->total_blocks) {
        const char*       value       = std::getenv("RTP_LLM_SPARSE_KV_OFFLOAD_HBM_SHARE_DENOMINATOR");
        const std::string denominator = value ? value : "3";
        RTP_LLM_CHECK_WITH_INFO(
            !denominator.empty()
                && std::all_of(denominator.begin(), denominator.end(), [](unsigned char c) { return std::isdigit(c); }),
            "RTP_LLM_SPARSE_KV_OFFLOAD_HBM_SHARE_DENOMINATOR must be 0 or an integer >= 2");
        offload_hbm_share_denominator_ = std::stoull(denominator);
        RTP_LLM_CHECK_WITH_INFO(offload_hbm_share_denominator_ != 1,
                                "RTP_LLM_SPARSE_KV_OFFLOAD_HBM_SHARE_DENOMINATOR must be 0 or >= 2");
        if (offload_hbm_share_denominator_ > 0) {
            splitFreeBlocksNoLock(this->config().offload->hbm_blocks);
        }
    }
}

std::optional<BlockIdList> DeviceBlockPool::mallocForRequest(size_t n, size_t request_blocks, BlockIdxType last_block) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool                        prefer_host = false;
    if (offload_hbm_share_denominator_ > 0 && request_blocks > 0) {
        prefer_host = last_block > 0 ? static_cast<uint32_t>(last_block) >= config().offload->hbm_blocks :
                                       request_blocks > freeLowerRangeBlocksNoLock() / offload_hbm_share_denominator_;
    }
    return mallocNoLock(n, prefer_host);
}

void DeviceBlockPool::incRef(BlockIdxType block) {
    incRef(BlockIdList{block});
}

void DeviceBlockPool::incRef(const BlockIdList& blocks) {
    mutateAllocatedBlocks(
        blocks,
        [](BlockIdxType) {},
        [this](BlockIdxType block) {
            const bool was_active     = isActiveNoLock(block);
            const bool was_available  = isAvailableNoLock(block);
            const bool was_referenced = hasRequestRefNoLock(block);
            ++refcounts_[block];
            if (!was_referenced) {
                ++request_referenced_blocks_num_;
            }
            updateActiveBlocksNumNoLock(block, was_active);
            updateAvailableBlocksNumNoLock(block, was_available);
        });
}

void DeviceBlockPool::decRef(BlockIdxType block) {
    decRef(BlockIdList{block});
}

void DeviceBlockPool::decRef(const BlockIdList& blocks) {
    mutateAllocatedBlocks(
        blocks,
        [this](BlockIdxType block) {
            RTP_LLM_CHECK_WITH_INFO(hasRequestRefNoLock(block),
                                    "cannot decRef block [%d] of pool [%s] without a request reference",
                                    block,
                                    poolName().c_str());
        },
        [this](BlockIdxType block) {
            const bool was_active    = isActiveNoLock(block);
            const bool was_available = isAvailableNoLock(block);
            --refcounts_[block];
            if (!hasRequestRefNoLock(block)) {
                --request_referenced_blocks_num_;
            }
            if (refcounts_[block] == 0) {
                freeAllocatedBlockNoLock(block);
            }
            updateActiveBlocksNumNoLock(block, was_active);
            updateAvailableBlocksNumNoLock(block, was_available);
        });
}

bool DeviceBlockPool::tryReplaceRequestReferences(const std::vector<RequestReferenceUpdate>& reference_updates,
                                                  int                                        replacement_count,
                                                  BlockIndicesType&                          replacements,
                                                  int&                                       required_free_blocks) {
    replacements.clear();
    required_free_blocks = 0;
    RTP_LLM_CHECK(replacement_count >= 0);

    std::vector<RequestReferenceUpdate>      changes;
    std::unordered_map<BlockIdxType, size_t> positions;
    changes.reserve(reference_updates.size());
    positions.reserve(reference_updates.size());
    for (const auto& update : reference_updates) {
        RTP_LLM_CHECK(update.old_count > 0 && update.new_count >= 0);
        const auto [it, inserted] = positions.emplace(update.block, changes.size());
        if (inserted) {
            changes.push_back(update);
        } else {
            auto& change = changes[it->second];
            RTP_LLM_CHECK(change.old_count <= std::numeric_limits<int>::max() - update.old_count);
            RTP_LLM_CHECK(change.new_count <= std::numeric_limits<int>::max() - update.new_count);
            change.old_count += update.old_count;
            change.new_count += update.new_count;
        }
    }

    std::vector<size_t> reclaimable;
    reclaimable.reserve(changes.size());
    replacements.reserve(replacement_count);
    std::function<void()> notify;
    bool                  capacity_increased = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        checkInitializedNoLock();
        for (size_t i = 0; i < changes.size(); ++i) {
            const auto& change = changes[i];
            checkAllocatedNoLock(change.block);
            const uint32_t tree_hold    = treeRefCountNoLock(change.block) > 0 ? 1 : 0;
            const uint32_t request_refs = refcounts_[change.block] - tree_hold;
            RTP_LLM_CHECK_WITH_INFO(request_refs >= static_cast<uint32_t>(change.old_count),
                                    "replacing more request references than owned for block %d",
                                    change.block);
            RTP_LLM_CHECK(static_cast<uint64_t>(refcounts_[change.block]) - change.old_count + change.new_count
                          <= std::numeric_limits<uint32_t>::max());
            // CACHE, LOAD, EVICTION and STORE holds all prevent donor reuse.
            if (request_refs == static_cast<uint32_t>(change.old_count) && change.new_count == 0 && tree_hold == 0) {
                reclaimable.push_back(i);
            }
        }

        const size_t transfer_count = std::min(reclaimable.size(), static_cast<size_t>(replacement_count));
        required_free_blocks        = replacement_count - static_cast<int>(transfer_count);
        const size_t free_before    = availableFreeBlocksNoLock();
        if (free_before < static_cast<size_t>(required_free_blocks)) {
            return false;
        }

        // Prepare all potentially allocating work before committing ownership.
        notify = capacityChangeCallbackNoLock();
        reserveReleasedBlocksNoLock(reclaimable.size() - transfer_count);
        auto fresh_blocks = mallocNoLock(required_free_blocks);
        RTP_LLM_CHECK(fresh_blocks.has_value());
        for (size_t i = 0; i < transfer_count; ++i) {
            auto& change     = changes[reclaimable[i]];
            change.new_count = 1;
            onAllocatedNoLock(change.block);
            replacements.push_back(change.block);
        }
        replacements.insert(replacements.end(), fresh_blocks->begin(), fresh_blocks->end());

        for (const auto& change : changes) {
            const auto block          = change.block;
            const bool was_active     = isActiveNoLock(block);
            const bool was_available  = isAvailableNoLock(block);
            const bool was_referenced = hasRequestRefNoLock(block);
            refcounts_[block]         = refcounts_[block] - change.old_count + change.new_count;
            if (was_referenced && !hasRequestRefNoLock(block)) {
                --request_referenced_blocks_num_;
            }
            if (refcounts_[block] == 0) {
                freeAllocatedBlockNoLock(block);
            }
            updateActiveBlocksNumNoLock(block, was_active);
            updateAvailableBlocksNumNoLock(block, was_available);
            capacity_increased |= !was_available && isAvailableNoLock(block);
        }
        for (const auto block : *fresh_blocks) {
            const bool was_active = isActiveNoLock(block);
            refcounts_[block]     = 1;
            ++request_referenced_blocks_num_;
            updateActiveBlocksNumNoLock(block, was_active);
        }
        capacity_increased |= availableFreeBlocksNoLock() > free_before;
    }
    // Tree callbacks acquire their own locks and must run outside the pool lock.
    if (capacity_increased && notify) {
        notify();
    }
    return true;
}

bool DeviceBlockPool::isExclusiveRequestBlock(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkAllocatedNoLock(block);
    return refcounts_[block] == 1 && treeRefCountNoLock(block) == 0;
}

uint32_t DeviceBlockPool::refCount(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkAllocatedNoLock(block);
    return refcounts_[block];
}

size_t DeviceBlockPool::referencedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    return request_referenced_blocks_num_;
}

void DeviceBlockPool::onFirstTreeRefNoLock(BlockIdxType block) {
    ++refcounts_[block];
}

bool DeviceBlockPool::isOffload() const {
    return config().offload.has_value();
}

const std::vector<OffloadStoragePtr>& DeviceBlockPool::allLayerOffloadStorage() const {
    return offload_storages_;
}

void DeviceBlockPool::onAllocatedNoLock(BlockIdxType block) {
    if (!isOffload()) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(config().offload_version_owner, "only the allocator rank may change offload generations");
    auto& version = offload_cpu_generations_[block];
    RTP_LLM_CHECK_WITH_INFO(version < std::numeric_limits<int64_t>::max(), "offload generation overflow");
    ++version;
    if (!offload_dirty_[block]) {
        offload_dirty_[block] = 1;
        offload_dirty_blocks_.push_back(block);
    }
}

void DeviceBlockPool::markOffloadOverwrite(const BlockIdList& blocks) {
    if (!isOffload()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    for (const auto block : blocks) {
        checkAllocatedNoLock(block);
    }
    for (const auto block : blocks) {
        onAllocatedNoLock(block);
    }
}

std::vector<std::pair<BlockIdxType, int64_t>> DeviceBlockPool::takeOffloadVersionUpdates() {
    std::lock_guard<std::mutex>                   lock(mutex_);
    std::vector<std::pair<BlockIdxType, int64_t>> updates;
    updates.reserve(offload_dirty_blocks_.size());
    for (const auto block : offload_dirty_blocks_) {
        updates.emplace_back(block, offload_cpu_generations_[block]);
        offload_dirty_[block] = 0;
    }
    offload_dirty_blocks_.clear();
    return updates;
}

bool DeviceBlockPool::onLastTreeRefNoLock(BlockIdxType block) {
    assert(refcounts_[block] > 0);
    --refcounts_[block];
    return refcounts_[block] == 0;
}

bool DeviceBlockPool::hasRequestRefNoLock(BlockIdxType block) const {
    return refcounts_[block] > static_cast<uint32_t>(treeRefCountNoLock(block) > 0);
}

bool DeviceBlockPool::hasExternalRefNoLock(BlockIdxType block) const {
    return hasRequestRefNoLock(block);
}

DeviceBlockPool::~DeviceBlockPool() {
    cache_aligned_buffer_ = torch::Tensor();
}

std::shared_ptr<const DeviceBlockPoolConfig>
DeviceBlockPool::normalizeConfig(const std::shared_ptr<const DeviceBlockPoolConfig>& config) {
    RTP_LLM_CHECK(config != nullptr);
    RTP_LLM_CHECK(config->pool_type == BlockPoolType::DEVICE);
    RTP_LLM_CHECK_WITH_INFO(!config->memory_layouts.empty(),
                            "device block pool [%s] memory_layouts must not be empty",
                            config->pool_name.c_str());

    // Every memory layout shares the same block id space, so physical_block_count is
    // reconciled against memory_layouts[*].block_num before IBlockPool construction.
    // The first layout's block_num is authoritative and every other layout must agree with it.
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
        RTP_LLM_CHECK_WITH_INFO(layout_cfg.layer_num > 0,
                                "device block pool [%s] memory_layouts[%zu].layer_num must be > 0",
                                config->pool_name.c_str(),
                                layout_idx);
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

const DeviceBlockPoolConfig& DeviceBlockPool::config() const {
    return configAs<DeviceBlockPoolConfig>(BlockPoolType::DEVICE);
}

void DeviceBlockPool::initializeCacheBuffer() {
    const auto& cfg = config();
    RTP_LLM_CHECK_WITH_INFO(
        cfg.total_size_bytes > 0, "device block pool [%s] total_size_bytes must be > 0", cfg.pool_name.c_str());
    RTP_LLM_CHECK_WITH_INFO(!(cfg.use_pinned_cpu_backing && cfg.use_device_malloc_backing),
                            "device block pool [%s] cannot use pinned CPU and cudaMalloc backing together",
                            cfg.pool_name.c_str());

    if (cfg.use_pinned_cpu_backing) {
        initializePinnedCpuBuffer();
    } else if (cfg.use_device_malloc_backing) {
        initializeDeviceMallocBuffer(cfg.total_size_bytes);
    } else {
        cache_aligned_buffer_ = torch::empty({static_cast<int64_t>(cfg.total_size_bytes)},
                                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    }
#if USING_ROCM
    // Gluon PA can load logically unused V-cache lanes: NaN * 0 is still NaN.
    // Initialize every backing mode once, before publishing the pool. Finite
    // writes/reuse preserve this invariant; imported non-finite data does not.
    cache_aligned_buffer_.zero_();
    if (cache_aligned_buffer_.is_cuda()) {
        torch::cuda::synchronize(cache_aligned_buffer_.get_device());
    }
#endif
    cache_base_ptr_ = cache_aligned_buffer_.data_ptr();
    RTP_LLM_CHECK_WITH_INFO(cache_base_ptr_ != nullptr,
                            "device block pool [%s] allocate cache aligned buffer is null",
                            cfg.pool_name.c_str());

    const bool              is_cuda     = cache_aligned_buffer_.is_cuda();
    const bool              is_pinned   = !is_cuda && cache_aligned_buffer_.is_pinned();
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    RTP_LLM_LOG_INFO("backing selected: pool_name=%s "
                     "actual_backing=%s is_cuda=%d is_pinned=%d ptr=%p total_size=%zu bytes total_size_mb=%.2f "
                     "memory_layouts=%zu",
                     cfg.pool_name.c_str(),
                     memoryTypeName(where()),
                     is_cuda,
                     is_pinned,
                     cache_base_ptr_,
                     cfg.total_size_bytes,
                     static_cast<double>(cfg.total_size_bytes) / kBytesPerMB,
                     cfg.memory_layouts.size());
}

void DeviceBlockPool::initializePinnedCpuBuffer() {
    const auto& cfg        = config();
    auto        cpu_buffer = torch::empty({static_cast<int64_t>(cfg.total_size_bytes)},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    try {
        cache_aligned_buffer_ = cpu_buffer.pin_memory();
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("pin device block pool backing failed, pool_name=%s total_size=%zu bytes, error=%s",
                     cfg.pool_name.c_str(),
                     cfg.total_size_bytes,
                     e.what());
    }
    RTP_LLM_CHECK_WITH_INFO(cache_aligned_buffer_.is_pinned(),
                            "device block pool [%s] requested pinned CPU backing but allocation is not pinned",
                            cfg.pool_name.c_str());
}

void DeviceBlockPool::initializeDeviceMallocBuffer(size_t bytes) {
    const auto& cfg = config();
#if USING_CUDA
    RTP_LLM_CHECK_WITH_INFO(!cfg.use_pinned_cpu_backing,
                            "raw device malloc block pool backing requires DEVICE allocation");
    RTP_LLM_CHECK_WITH_INFO(bytes > 0, "raw device malloc block pool total_size_bytes must be > 0");

    int  device_id  = -1;
    auto device_err = cudaGetDevice(&device_id);
    RTP_LLM_CHECK_WITH_INFO(device_err == cudaSuccess,
                            "cudaGetDevice failed before cudaMalloc block pool allocation, error=%s",
                            cudaGetErrorString(device_err));

    void*      ptr = nullptr;
    const auto err = cudaMalloc(&ptr, bytes);
    RTP_LLM_CHECK_WITH_INFO(err == cudaSuccess,
                            "cudaMalloc block pool failed, pool_name=%s, total_size=%zu bytes, error=%s",
                            cfg.pool_name.c_str(),
                            bytes,
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
                         {static_cast<int64_t>(bytes)},
                         std::move(deleter),
                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device_id)));
    RTP_LLM_LOG_INFO("cudaMalloc block pool backing allocated, pool_name=%s, ptr=%p, total_size=%zu bytes, device=%d",
                     cfg.pool_name.c_str(),
                     ptr,
                     bytes,
                     device_id);
#elif USING_ROCM
    RTP_LLM_CHECK_WITH_INFO(!cfg.use_pinned_cpu_backing,
                            "raw device malloc block pool backing requires DEVICE allocation");
    RTP_LLM_CHECK_WITH_INFO(bytes > 0, "raw device malloc block pool total_size_bytes must be > 0");

    int  device_id  = -1;
    auto device_err = hipGetDevice(&device_id);
    RTP_LLM_CHECK_WITH_INFO(device_err == hipSuccess,
                            "hipGetDevice failed before hipMalloc block pool allocation, error=%s",
                            hipGetErrorString(device_err));

    void*      ptr = nullptr;
    const auto err = hipMalloc(&ptr, bytes);
    RTP_LLM_CHECK_WITH_INFO(err == hipSuccess,
                            "hipMalloc block pool failed, pool_name=%s, total_size=%zu bytes, error=%s",
                            cfg.pool_name.c_str(),
                            bytes,
                            hipGetErrorString(err));

    auto deleter = [device_id](void* p) {
        if (p == nullptr) {
            return;
        }
        int current_device = -1;
        if (hipGetDevice(&current_device) == hipSuccess && current_device != device_id) {
            (void)hipSetDevice(device_id);
            (void)hipFree(p);
            (void)hipSetDevice(current_device);
            return;
        }
        (void)hipFree(p);
    };
    cache_aligned_buffer_ =
        torch::from_blob(ptr,
                         {static_cast<int64_t>(bytes)},
                         std::move(deleter),
                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device_id)));
    RTP_LLM_LOG_INFO("hipMalloc block pool backing allocated, pool_name=%s, ptr=%p, total_size=%zu bytes, device=%d",
                     cfg.pool_name.c_str(),
                     ptr,
                     bytes,
                     device_id);
#else
    (void)bytes;
    RTP_LLM_FAIL("raw device malloc block pool backing requires a CUDA or ROCm build, pool_name=%s",
                 cfg.pool_name.c_str());
#endif
}

void DeviceBlockPool::initializeOffloadStorage() {
#if !USING_CUDA
    RTP_LLM_FAIL("sparse KV offload requires CUDA");
#else
    const auto& cfg    = config();
    const auto& tiered = *cfg.offload;
    RTP_LLM_CHECK_WITH_INFO(!cfg.memory_layouts.empty() && !cfg.use_pinned_cpu_backing,
                            "offload requires native layouts with DEVICE placement");
    const auto&  layout    = cfg.memory_layouts.front();
    size_t       total_layers = 0;
    for (const auto& module_layout : cfg.memory_layouts) {
        RTP_LLM_CHECK_WITH_INFO(module_layout.kv_block_stride_bytes == layout.kv_block_stride_bytes
                                    && module_layout.kv_scale_stride_bytes == layout.kv_scale_stride_bytes
                                    && module_layout.kernel_blocks_per_kv_block == layout.kernel_blocks_per_kv_block
                                    && module_layout.seq_size_per_block == layout.seq_size_per_block
                                    && module_layout.dtype == layout.dtype,
                                "offload main and draft layouts must have matching native page geometry");
        total_layers += module_layout.layer_num;
    }
    const size_t pages     = layout.kernel_blocks_per_kv_block;
    const size_t page_rows = cfg.offload_layout.rows_per_page;
    RTP_LLM_CHECK_WITH_INFO(page_rows > 0 && pages > 0 && pages <= INT32_MAX / page_rows,
                            "offload requires complete native pages");
    const size_t rows     = page_rows * pages;
    const size_t resident = tiered.residentRows(page_rows);
    RTP_LLM_CHECK_WITH_INFO(tiered.total_blocks == cfg.physical_block_count && tiered.hbm_blocks >= 2
                                && tiered.hbm_blocks <= tiered.total_blocks && tiered.max_selected_rows > 0
                                && static_cast<uint64_t>(tiered.total_blocks) * rows <= INT32_MAX
                                && static_cast<uint64_t>(tiered.hbm_blocks) * rows + resident <= INT32_MAX,
                            "invalid offload block counts or row capacity");
    RTP_LLM_CHECK_WITH_INFO(cfg.offload_owner_layers.size() == total_layers, "offload owner layer count mismatch");
    const auto checked_product = [](size_t a, size_t b) {
        RTP_LLM_CHECK_WITH_INFO(a == 0 || b <= static_cast<size_t>(INT64_MAX) / a, "offload allocation overflow");
        return a * b;
    };
    RTP_LLM_CHECK_WITH_INFO(layout.kv_block_stride_bytes % pages == 0 && layout.kv_scale_stride_bytes % pages == 0
                                && layout.kv_scale_stride_bytes <= SIZE_MAX - layout.kv_block_stride_bytes,
                            "invalid offload native page stride");
    const size_t gpu_pages      = static_cast<size_t>(tiered.hbm_blocks) * pages + resident / page_rows;
    const size_t host_pages     = static_cast<size_t>(tiered.total_blocks - tiered.hbm_blocks) * pages;
    const size_t bytes_per_page = (layout.kv_block_stride_bytes + layout.kv_scale_stride_bytes) / pages;
    cfg.offload_layout.validate(layout.kv_block_stride_bytes / pages, layout.kv_scale_stride_bytes / pages);
    const size_t gpu_bytes  = checked_product(checked_product(gpu_pages, bytes_per_page), total_layers);
    const size_t host_bytes = checked_product(checked_product(host_pages, bytes_per_page), total_layers);
    if (cfg.use_device_malloc_backing) {
        initializeDeviceMallocBuffer(gpu_bytes);
    } else {
        cache_aligned_buffer_ = torch::empty({static_cast<int64_t>(gpu_bytes)},
                                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    }
    cache_base_ptr_ = cache_aligned_buffer_.data_ptr();
    // Retain a pinned allocation for empty host views (P=0) as well.
    offload_host_buffer_ =
        torch::empty({static_cast<int64_t>(std::max<size_t>(host_bytes, 8))},
                     torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    void*      mapped = nullptr;
    const auto err    = cudaHostGetDevicePointer(&mapped, offload_host_buffer_.data_ptr(), 0);
    RTP_LLM_CHECK_WITH_INFO(err == cudaSuccess && mapped == offload_host_buffer_.data_ptr(),
                            "offload pinned storage requires identical host and CUDA UVA addresses");
    auto generations =
        torch::zeros({tiered.total_blocks}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    if (cfg.offload_version_owner) {
        offload_cpu_generations_.assign(tiered.total_blocks, 0);
        offload_dirty_.assign(tiered.total_blocks, 0);
        offload_dirty_blocks_.reserve(tiered.total_blocks);
    }
    size_t     gpu_offset  = 0;
    size_t     host_offset = 0;
    const auto plane = [&](const torch::Tensor& buffer, size_t offset, size_t count, const CacheEntryPlane& view) {
        const size_t kv_page_bytes = layout.kv_block_stride_bytes / pages;
        const size_t page_bytes    = view.scale ? layout.kv_scale_stride_bytes / pages : kv_page_bytes;
        if (view.scale) {
            offset += checked_product(count, kv_page_bytes);
        }
        const auto   dtype   = dataTypeToTorchType(view.dtype);
        const size_t element = c10::elementSize(dtype);
        auto         typed   = buffer.narrow(0, offset, checked_product(count, page_bytes)).view(dtype);
        auto         shape   = view.shape;
        auto         strides = view.strides;
        shape.insert(shape.begin(), static_cast<int64_t>(count));
        strides.insert(strides.begin(), static_cast<int64_t>(page_bytes / element));
        return typed.as_strided(shape, strides, typed.storage_offset() + view.offset_bytes / element);
    };
    for (size_t layer = 0; layer < total_layers; ++layer) {
        auto storage               = std::make_shared<OffloadStorage>();
        storage->tag               = cfg.pool_name;
        storage->owner_layer       = cfg.offload_owner_layers[layer];
        storage->total_blocks      = tiered.total_blocks;
        storage->hbm_blocks        = tiered.hbm_blocks;
        storage->rows_per_block    = rows;
        storage->rows_per_page     = page_rows;
        storage->max_selected_rows = tiered.max_selected_rows;
        storage->generations       = generations;
        for (const auto& view : cfg.offload_layout.planes) {
            storage->resident_planes.push_back(plane(cache_aligned_buffer_, gpu_offset, gpu_pages, view));
            storage->host_planes.push_back(plane(offload_host_buffer_, host_offset, host_pages, view));
            storage->resident_planes.back().narrow(0, 0, pages).zero_();
        }
        gpu_offset += checked_product(gpu_pages, bytes_per_page);
        host_offset += checked_product(host_pages, bytes_per_page);
        offload_storages_.push_back(std::move(storage));
    }
    RTP_LLM_CHECK(gpu_offset == gpu_bytes && host_offset == host_bytes);
    RTP_LLM_LOG_INFO("offload backing pool=%s H=%u P=%u device_bytes=%zu host_bytes=%zu logical_block_bytes=%zu",
                     cfg.pool_name.c_str(),
                     tiered.hbm_blocks,
                     tiered.total_blocks - tiered.hbm_blocks,
                     offloadDeviceSizeBytes(),
                     offloadHostSizeBytes(),
                     blockSizeBytes());
#endif
}

void DeviceBlockPool::initializeLayerMappings() {
    const auto& cfg          = config();
    size_t      total_layers = 0;
    for (const auto& layout_cfg : cfg.memory_layouts) {
        total_layers += static_cast<size_t>(layout_cfg.layer_num);
    }
    global_layer_to_local_.assign(total_layers, {-1, -1});
    // 待删除: aligned/padded model-facing scale exposure, for legacy-L1 consumers.
    global_layer_kv_tensors_.assign(total_layers, torch::Tensor());
    global_layer_kv_scale_tensors_.assign(total_layers, torch::Tensor());
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

void DeviceBlockPool::processMemoryLayout(size_t               layout_idx,
                                          const torch::Tensor& full_tensor,
                                          size_t&              global_layer_begin) {
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

    RTP_LLM_LOG_INFO("MemoryLayout[%zu] initialized: pool_name=%s layer_num=%u "
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
    auto&      strategy      = layout_strategies_[layout_idx];
    const auto layer_tensors = strategy->getLayerCacheTensors();
    const auto scale_tensors = strategy->getLayerScaleCacheTensors();
    for (size_t local_layer = 0; local_layer < static_cast<size_t>(layout_cfg.layer_num); ++local_layer) {
        const size_t global_layer = global_layer_begin + local_layer;
        RTP_LLM_CHECK_WITH_INFO(global_layer < global_layer_to_local_.size(), "global layer index out of range");
        global_layer_to_local_[global_layer] = {static_cast<int>(layout_idx), static_cast<int>(local_layer)};

        global_layer_kv_tensors_[global_layer] = layer_tensors[local_layer];
        if (layout_cfg.hasScale()) {
            global_layer_kv_scale_tensors_[global_layer] = scale_tensors[local_layer];
        }
    }
}

bool DeviceBlockPool::init() {
    RTP_LLM_CHECK(!initialized());
    const auto& cfg = config();

    if (isOffload()) {
        initializeOffloadStorage();
    } else {
        initializeCacheBuffer();
        initializeLayerMappings();
        initializeLayoutStrategies();
    }

    markInitialized();

    RTP_LLM_LOG_INFO("init success: pool_name=%s memory_layouts=%zu total_layers=%zu "
                     "total_size=%zu bytes physical_block_count=%zu",
                     cfg.pool_name.c_str(),
                     cfg.memory_layouts.size(),
                     global_layer_to_local_.size(),
                     cfg.total_size_bytes,
                     cfg.physical_block_count);
    return true;
}

void DeviceBlockPool::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    if (cache_store) {
        cache_store_ = std::move(cache_store);
    }
    if (cache_store_ && !kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to register user mr, pool_name=%s", config().pool_name.c_str());
        auto memory_util = cache_store_->getMemoryUtil();
        if (isOffload()) {
            changeOffloadMr(memory_util, true);
            kvcache_reg_mr_ = true;
            return;
        }
        const bool gpu = where() == MemoryType::MEMORY_GPU;

        for (size_t layout_idx = 0; layout_idx < config().memory_layouts.size(); ++layout_idx) {
            const auto& layout_cfg = config().memory_layouts[layout_idx];

            registerUserMrForBuffer(memory_util,
                                    layout_idx,
                                    layout_cfg.kv_cache_offset_bytes,
                                    layout_cfg.kv_block_pool_size_bytes,
                                    layout_cfg.kv_block_stride_bytes,
                                    gpu,
                                    "kv");

            if (layout_cfg.hasScale()) {
                registerUserMrForBuffer(memory_util,
                                        layout_idx,
                                        layout_cfg.kv_scale_offset_bytes,
                                        layout_cfg.kv_scale_pool_size_bytes,
                                        layout_cfg.kv_scale_stride_bytes,
                                        gpu,
                                        "scale");
            }
        }
        kvcache_reg_mr_ = true;
    }
}

void DeviceBlockPool::deregUserMr() {
    if (kvcache_reg_mr_ && cache_store_) {
        RTP_LLM_LOG_INFO("start to deregister user mr, pool_name=%s", config().pool_name.c_str());
        auto memory_util = cache_store_->getMemoryUtil();
        if (isOffload()) {
            changeOffloadMr(memory_util, false);
            kvcache_reg_mr_ = false;
            return;
        }
        const bool gpu = where() == MemoryType::MEMORY_GPU;

        for (size_t layout_idx = 0; layout_idx < config().memory_layouts.size(); ++layout_idx) {
            const auto& layout_cfg = config().memory_layouts[layout_idx];
            deregisterUserMrForBuffer(memory_util, layout_idx, layout_cfg.kv_cache_offset_bytes, gpu, "kv");
            if (layout_cfg.hasScale()) {
                deregisterUserMrForBuffer(memory_util, layout_idx, layout_cfg.kv_scale_offset_bytes, gpu, "scale");
            }
        }
        RTP_LLM_LOG_INFO("deregister user mr for block pool success, pool_name=%s", config().pool_name.c_str());
        kvcache_reg_mr_ = false;
    }
}

void DeviceBlockPool::changeOffloadMr(const std::shared_ptr<MemoryUtil>& memory_util, bool registering) {
    for (size_t layer = 0; layer < offload_storages_.size(); ++layer) {
        const auto& storage = offload_storages_[layer];
        for (const bool gpu : {true, false}) {
            const size_t blocks = gpu ? storage->hbm_blocks : storage->total_blocks - storage->hbm_blocks;
            if (blocks == 0) {
                continue;
            }
            // Register native buffers once; entry planes may share the same buffer.
            for (const auto& buffer : convertIndexToBuffer(layer, gpu ? 0 : storage->hbm_blocks)) {
                const auto start_us = currentTimeUs();
                const bool ok =
                    registering ?
                        memory_util->regUserMr(buffer.addr, blocks * buffer.size_bytes, gpu, buffer.size_bytes) :
                        memory_util->deregUserMr(buffer.addr, gpu);
                RTP_LLM_CHECK_WITH_INFO(ok,
                                        "offload MR %s failed: tag=%s owner=%d gpu=%d",
                                        registering ? "registration" : "deregistration",
                                        storage->tag.c_str(),
                                        storage->owner_layer,
                                        gpu);
                if (registering) {
                    mr_cost_time_ms_ += (currentTimeUs() - start_us) / 1000;
                }
            }
        }
    }
}

void DeviceBlockPool::registerUserMrForBuffer(std::shared_ptr<MemoryUtil> memory_util,
                                              size_t                      layout_idx,
                                              size_t                      offset_bytes,
                                              size_t                      bytes,
                                              size_t                      stride_bytes,
                                              bool                        gpu,
                                              const std::string&          buffer_type) {
    void* base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_) + static_cast<ptrdiff_t>(offset_bytes));
    auto  start_us = currentTimeUs();

    if (!memory_util->regUserMr(base_ptr, bytes, gpu, stride_bytes)) {
        RTP_LLM_FAIL("register user mr for block pool layout[%zu] %s buffer failed, pool_name=%s",
                     layout_idx,
                     buffer_type.c_str(),
                     config().pool_name.c_str());
    }

    auto cost_ms = (currentTimeUs() - start_us) / 1000;
    mr_cost_time_ms_ += cost_ms;

    RTP_LLM_LOG_INFO("register user mr success: pool_name=%s layout[%zu] %s base=%p len=%zu aligned=%zu cost=%ld ms",
                     config().pool_name.c_str(),
                     layout_idx,
                     buffer_type.c_str(),
                     base_ptr,
                     bytes,
                     stride_bytes,
                     cost_ms);
}

void DeviceBlockPool::deregisterUserMrForBuffer(std::shared_ptr<MemoryUtil> memory_util,
                                                size_t                      layout_idx,
                                                size_t                      offset_bytes,
                                                bool                        gpu,
                                                const std::string&          buffer_type) {
    void* base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_) + static_cast<ptrdiff_t>(offset_bytes));
    if (!memory_util->deregUserMr(base_ptr, gpu)) {
        RTP_LLM_FAIL("deregister user mr for block pool layout[%zu] %s buffer failed, pool_name=%s",
                     layout_idx,
                     buffer_type.c_str(),
                     config().pool_name.c_str());
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

std::vector<torch::Tensor> DeviceBlockPool::allLayerCacheBase() const {
    RTP_LLM_CHECK_WITH_INFO(!isOffload(), "offload pool has no legacy full-GPU cache view");
    return global_layer_kv_tensors_;
}

std::vector<torch::Tensor> DeviceBlockPool::allLayerScaleCacheBase() const {
    RTP_LLM_CHECK_WITH_INFO(!isOffload(), "offload pool has no legacy full-GPU scale view");
    return global_layer_kv_scale_tensors_;
}

BlockAddrInfo DeviceBlockPool::convertIndexToAddr(int layer_id, BlockIdxType block) const {
    RTP_LLM_CHECK(initialized());
    if (isOffload()) {
        const auto buffers = convertIndexToBuffer(layer_id, block);
        return {buffers[0].addr, buffers.size() > 1 ? buffers[1].addr : nullptr};
    }
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    return layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToAddr(local_layer_id, block);
}

std::vector<BlockInfo> DeviceBlockPool::convertIndexToBuffer(int layer_id, BlockIdxType block) const {
    RTP_LLM_CHECK(initialized());
    if (isOffload()) {
        RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < offload_storages_.size(),
                                "invalid offload local layer id=%d",
                                layer_id);
        const auto& storage = *offload_storages_[layer_id];
        RTP_LLM_CHECK_WITH_INFO(
            block >= 0 && static_cast<uint32_t>(block) < storage.total_blocks, "invalid offload block id=%d", block);
        const bool             gpu         = static_cast<uint32_t>(block) < storage.hbm_blocks;
        const size_t           local_block = gpu ? block : block - storage.hbm_blocks;
        const auto&            backing     = gpu ? cache_aligned_buffer_ : offload_host_buffer_;
        const auto&            layout      = config().memory_layouts.front();
        const size_t           pages       = layout.kernel_blocks_per_kv_block;
        const size_t           page_count  = (gpu ? storage.resident_planes : storage.host_planes).front().size(0);
        const size_t           kv_bytes    = page_count * (layout.kv_block_stride_bytes / pages);
        const size_t           layer_bytes = kv_bytes + page_count * (layout.kv_scale_stride_bytes / pages);
        char*                  base        = static_cast<char*>(backing.data_ptr()) + layer_id * layer_bytes;
        std::vector<BlockInfo> result;
        const auto             append = [&](size_t offset, size_t stride, DataType dtype) {
            result.push_back({gpu,
                              gpu ? backing.get_device() : 0,
                              static_cast<int32_t>(dataTypeToTorchType(dtype)),
                              base + offset + local_block * stride,
                              stride});
        };
        append(0, layout.kv_block_stride_bytes, layout.dtype);
        if (layout.kv_scale_stride_bytes > 0) {
            const auto& views = config().offload_layout.planes;
            const auto  scale = std::find_if(views.begin(), views.end(), [](const auto& view) { return view.scale; });
            RTP_LLM_CHECK_WITH_INFO(scale != views.end(), "offload scale buffer has no entry view");
            append(kv_bytes, layout.kv_scale_stride_bytes, scale->dtype);
        }
        return result;
    }
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    return layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToBuffer(local_layer_id, block);
}

std::vector<BlockInfo>
DeviceBlockPool::convertIndexToBuffer(int layer_id, BlockIdxType block, int partition_count, int partition_id) const {
    RTP_LLM_CHECK(initialized());
    if (isOffload()) {
        const auto  buffers = convertIndexToBuffer(layer_id, block);
        const auto& layout  = config().memory_layouts.front();
        if (!config().offload_layout.partition_heads || layout.enable_hybrid_attention) {
            return buffers;
        }
        RTP_LLM_CHECK_WITH_INFO(partition_count == 1 || layout.kernel_blocks_per_kv_block == 1,
                                "offload MHA asymmetric TP requires one kernel page per physical block");
        std::vector<BlockInfo> result;
        for (const auto& buffer : buffers) {
            const auto parts = splitKVPartitionBytes(buffer.size_bytes,
                                                     buffer.size_bytes / 2,
                                                     buffer.size_bytes / 2,
                                                     layout.local_head_num_kv,
                                                     partition_count,
                                                     partition_id,
                                                     "offload KV");
            auto       key   = buffer;
            key.addr         = static_cast<char*>(buffer.addr) + parts.k_off;
            key.size_bytes   = parts.k_sz;
            auto value       = buffer;
            value.addr       = static_cast<char*>(buffer.addr) + parts.v_off;
            value.size_bytes = parts.v_sz;
            result.push_back(key);
            result.push_back(value);
        }
        return result;
    }
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    return layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToBuffer(
        local_layer_id, block, partition_count, partition_id);
}

MemoryType DeviceBlockPool::where() const {
    if (cache_aligned_buffer_.is_cuda()) {
        return MemoryType::MEMORY_GPU;
    }
    return cache_aligned_buffer_.is_pinned() ? MemoryType::MEMORY_CPU_PINNED : MemoryType::MEMORY_CPU;
}

void DeviceBlockPool::checkLayoutValidity(int layout_id) const {
    RTP_LLM_CHECK_WITH_INFO(layout_id >= 0 && static_cast<size_t>(layout_id) < layout_strategies_.size(),
                            "Memory layout ID %d out of range (max: %zu)",
                            layout_id,
                            layout_strategies_.size());
}

int DeviceBlockPool::deviceIndex() const {
    if (cache_aligned_buffer_.is_cuda()) {
        return static_cast<int>(cache_aligned_buffer_.get_device());
    }
    return -1;
}

size_t DeviceBlockPool::getTotalSizeBytes() const {
    if (isOffload()) {
        return offloadDeviceSizeBytes() + offloadHostSizeBytes();
    }
    return config().total_size_bytes;
}

size_t DeviceBlockPool::offloadHostSizeBytes() const {
    return offload_host_buffer_.defined() ? offload_host_buffer_.nbytes() : 0;
}

size_t DeviceBlockPool::offloadDeviceSizeBytes() const {
    return cache_aligned_buffer_.defined() ?
               cache_aligned_buffer_.nbytes()
                   + (offload_storages_.empty() ? 0 : offload_storages_.front()->generations.nbytes()) :
               0;
}

size_t DeviceBlockPool::blockSizeBytes() const {
    RTP_LLM_CHECK_WITH_INFO(config().physical_block_count > 0,
                            "device block pool [%s] physical_block_count must be > 0",
                            config().pool_name.c_str());
    RTP_LLM_CHECK_WITH_INFO(
        config().total_size_bytes % config().physical_block_count == 0,
        "device block pool [%s] total_size_bytes [%zu] is not divisible by physical_block_count [%zu]",
        config().pool_name.c_str(),
        config().total_size_bytes,
        config().physical_block_count);
    return config().total_size_bytes / config().physical_block_count;
}

std::string DeviceBlockPool::debugString() const {
    size_t request_ref_blocks;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        request_ref_blocks = request_referenced_blocks_num_;
    }

    std::ostringstream oss;
    oss << "DeviceBlockPool{" << IBlockPool::debugString() << ", pool_name=" << config().pool_name
        << ", memory_layouts=" << config().memory_layouts.size() << ", total_size_bytes=" << config().total_size_bytes
        << ", request_ref_blocks=" << request_ref_blocks << ", where=" << memoryTypeName(where()) << "}";
    return oss.str();
}

}  // namespace rtp_llm
