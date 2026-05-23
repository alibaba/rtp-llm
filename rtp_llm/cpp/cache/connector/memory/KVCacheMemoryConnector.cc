#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

// When set on MultiCopyParams, execNoBlockCopy may try CUDA split scatter/gather (SplitKvCacheCopy; not on PPU).
// This legacy SM-copy path is only used for non typed layer-region layouts.
static void applySplitKvMultiCopyFieldsIfEligible(bool enable_sm_copy, const CacheConfig& cfg, MultiCopyParams& out) {
    if (!enable_sm_copy) {
        return;
    }
    out.split_kv_layer_num          = static_cast<int>(cfg.layer_all_num);
    out.split_kv_cache_stride_bytes = cfg.kv_block_stride_bytes;
    out.split_kv_scale_stride_bytes = cfg.kv_scale_stride_bytes;
}

static void
appendBatchedMemoryCopyTile(void* dst, const void* src, size_t bytes, std::vector<BatchedMemoryCopyTile>& tiles) {
    if (bytes > 0) {
        tiles.push_back(BatchedMemoryCopyTile{dst, src, bytes});
    }
}

static void
appendStagedMemoryCopyTile(void* gpu, size_t host_offset, size_t bytes, std::vector<StagedMemoryCopyTile>& tiles) {
    if (gpu != nullptr && bytes > 0) {
        tiles.push_back(StagedMemoryCopyTile{gpu, host_offset, bytes});
    }
}

static void appendStagedMemoryCopyHostSegment(void*                                     host,
                                              size_t                                    host_offset,
                                              size_t                                    bytes,
                                              std::vector<StagedMemoryCopyHostSegment>& segments) {
    if (host == nullptr || bytes == 0) {
        return;
    }
    if (!segments.empty()) {
        auto& prev = segments.back();
        if (static_cast<char*>(prev.host) + prev.bytes == host && prev.host_offset + prev.bytes == host_offset) {
            prev.bytes += bytes;
            return;
        }
    }
    segments.push_back(StagedMemoryCopyHostSegment{host, host_offset, bytes});
}

static size_t alignUp(size_t value, size_t alignment) {
    RTP_LLM_CHECK_WITH_INFO(alignment != 0, "alignment must not be zero");
    return ((value + alignment - 1) / alignment) * alignment;
}

static size_t regionIndex(KVCacheRegionName region_name) {
    return static_cast<size_t>(region_name);
}

KVCacheMemoryConnector::KVCacheMemoryConnector(const CacheConfig&                       cache_config,
                                               const KVCacheConfig&                     kv_cache_config,
                                               const ParallelismConfig&                 parallelism_config,
                                               const std::shared_ptr<KVCacheAllocator>& allocator,
                                               const std::vector<std::string>&          tp_addrs,
                                               const kmonitor::MetricsReporterPtr&      metrics_reporter):
    cache_config_(cache_config),
    kv_cache_config_(kv_cache_config),
    parallelism_config_(parallelism_config),
    allocator_(allocator),
    tp_addrs_(tp_addrs),
    metrics_reporter_(metrics_reporter) {}

KVCacheMemoryConnector::KVCacheMemoryConnector(const CacheConfig&                       cache_config,
                                               const KVCacheConfig&                     kv_cache_config,
                                               const std::shared_ptr<KVCacheAllocator>& allocator,
                                               const std::vector<std::string>&          tp_addrs,
                                               const kmonitor::MetricsReporterPtr&      metrics_reporter):
    KVCacheMemoryConnector(cache_config, kv_cache_config, ParallelismConfig{}, allocator, tp_addrs, metrics_reporter) {}

KVCacheMemoryConnector::~KVCacheMemoryConnector() {
    RTP_LLM_LOG_INFO("KVCacheMemoryConnector destructor");
    stop_.store(true);
    if (metrics_reporter_thread_) {
        metrics_reporter_thread_->join();
        metrics_reporter_thread_.reset();
    }
    if (wait_done_thread_pool_) {
        wait_done_thread_pool_->stop();
        wait_done_thread_pool_.reset();
    }
    broadcast_manager_.reset();
    block_pool_.reset();
    block_cache_.reset();
    {
        std::lock_guard<std::mutex> lock(staged_copy_scratch_mutex_);
        for (auto& [_, scratch] : staged_copy_scratch_by_device_) {
            if (scratch) {
                releaseStagedMemoryCopyScratch(*scratch);
            }
        }
        staged_copy_scratch_by_device_.clear();
    }
}

bool KVCacheMemoryConnector::init() {
    const auto memory_cache_sync_timeout_ms = kv_cache_config_.memory_cache_sync_timeout_ms;
    RTP_LLM_CHECK_WITH_INFO(memory_cache_sync_timeout_ms > 0,
                            "init failed, sync timeout is invalid, sync timeout: %ld ms",
                            memory_cache_sync_timeout_ms);
    RTP_LLM_CHECK_WITH_INFO(!kv_cache_config_.enable_memory_cache_disk || kv_cache_config_.enable_memory_cache,
                            "init failed, disk memory cache requires enable_memory_cache");
    RTP_LLM_CHECK_WITH_INFO(!(kv_cache_config_.enable_memory_cache_disk && kv_cache_config_.enable_tiered_memory_cache),
                            "init failed, enable_memory_cache_disk cannot be used with enable_tiered_memory_cache");
    RTP_LLM_CHECK_WITH_INFO(!kv_cache_config_.enable_memory_cache_disk
                                || kv_cache_config_.memory_cache_disk_sync_timeout_ms > 0,
                            "init failed, disk sync timeout is invalid, sync timeout: %ld ms",
                            kv_cache_config_.memory_cache_disk_sync_timeout_ms);

    checkLayerBlockStrideBytes();

    initBlockPool();
    if (diskCacheEnabled()) {
        initDiskBlockPool(memoryCacheBlockSizeBytes());
    }
    block_cache_ = std::make_shared<MemoryDiskBlockCache>();

    broadcast_manager_ = std::make_shared<BroadcastManager>(tp_addrs_);
    RTP_LLM_CHECK_WITH_INFO(broadcast_manager_->init(), "init failed, broadcast manager init failed");

    wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(8, 1000, nullptr, "WaitDoneThreadPool");
    RTP_LLM_CHECK_WITH_INFO(wait_done_thread_pool_->start(), "init failed, wait done thread pool start failed");

    if (metrics_reporter_) {
        metrics_reporter_thread_ = std::make_shared<std::thread>([this]() { reportMetricsLoop(); });
    }
    return true;
}

void KVCacheMemoryConnector::checkLayerBlockStrideBytes() const {
    const auto slots = layerRegionSlots();
    RTP_LLM_CHECK_WITH_INFO(!slots.empty(), "layer-attn slots must not be empty");
    for (const auto& slot : slots) {
        RTP_LLM_CHECK_WITH_INFO(slot.stride_bytes > 0,
                                "invalid block stride bytes at layer=%d region_name=%d group=%d: %zu",
                                slot.layer_id,
                                static_cast<int>(slot.region_name),
                                slot.group_id,
                                slot.stride_bytes);
    }
}

void KVCacheMemoryConnector::initBlockPool() {
    const auto memory_cache_size_mb = kv_cache_config_.memory_cache_size_mb;
    RTP_LLM_CHECK_WITH_INFO(memory_cache_size_mb > 0,
                            "init block pool failed, memory size is invalid, memory size: %ld MB",
                            memory_cache_size_mb);

    const auto block_size = memoryCacheBlockSizeBytes();
    RTP_LLM_CHECK_WITH_INFO(block_size > 0, "block size is invalid: %zu", block_size);

    block_pool_ = createBlockPool(block_size, memory_cache_size_mb);
    RTP_LLM_CHECK_WITH_INFO(block_pool_ != nullptr, "init block pool failed, create block pool failed");
}

size_t KVCacheMemoryConnector::memoryCacheBlockSizeBytes() const {
    const auto slots      = layerRegionSlots();
    size_t     block_size = 0;
    for (const auto& slot : slots) {
        block_size += slot.stride_bytes;
    }
    return block_size;
}

bool KVCacheMemoryConnector::diskCacheEnabled() const {
    return kv_cache_config_.enable_memory_cache_disk;
}

void KVCacheMemoryConnector::initDiskBlockPool(size_t block_size_bytes) {
    const auto paths = split(kv_cache_config_.memory_cache_disk_paths, ',');
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config_.memory_cache_disk_size_mb > 0,
                            "init disk block pool failed, disk size must be positive, size=%ld MB",
                            kv_cache_config_.memory_cache_disk_size_mb);
    RTP_LLM_CHECK_WITH_INFO(
        paths.size() == static_cast<size_t>(parallelism_config_.local_world_size),
        "init disk block pool failed, disk path count must equal local_world_size, paths=%zu, local_world_size=%ld",
        paths.size(),
        parallelism_config_.local_world_size);
    RTP_LLM_CHECK_WITH_INFO(parallelism_config_.local_rank >= 0
                                && parallelism_config_.local_rank < parallelism_config_.local_world_size,
                            "init disk block pool failed, invalid local rank, local_rank=%ld, local_world_size=%ld",
                            parallelism_config_.local_rank,
                            parallelism_config_.local_world_size);

    DiskBlockPoolConfig config;
    config.mount_path       = paths.at(static_cast<size_t>(parallelism_config_.local_rank));
    config.local_rank       = parallelism_config_.local_rank;
    config.world_rank       = parallelism_config_.world_rank;
    config.disk_size_bytes  = static_cast<size_t>(kv_cache_config_.memory_cache_disk_size_mb) * 1024UL * 1024UL;
    config.block_size_bytes = block_size_bytes;
    config.buffered_io      = kv_cache_config_.memory_cache_disk_buffered_io;
    disk_block_pool_        = std::make_shared<DiskBlockPool>(std::move(config));
    RTP_LLM_CHECK_WITH_INFO(disk_block_pool_->init(), "init disk block pool failed");
}

std::vector<KVCacheMemoryConnector::LayerRegionSlot> KVCacheMemoryConnector::layerRegionSlots() const {
    std::vector<LayerRegionSlot> slots;
    const size_t                 layer_num         = cache_config_.layer_all_num;
    const size_t                 region_name_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);

    auto group_stride = [this](int gid, int layer_id) -> size_t {
        if (gid >= 0 && static_cast<size_t>(gid) < cache_config_.group_kv_block_stride_bytes.size()) {
            const size_t kv_stride    = cache_config_.group_kv_block_stride_bytes[static_cast<size_t>(gid)];
            const size_t scale_stride = static_cast<size_t>(gid) < cache_config_.group_kv_scale_stride_bytes.size() ?
                                            cache_config_.group_kv_scale_stride_bytes[static_cast<size_t>(gid)] :
                                            0;
            if (kv_stride + scale_stride > 0) {
                return kv_stride + scale_stride;
            }
        }
        if (layer_id >= 0 && static_cast<size_t>(layer_id) < cache_config_.layer_to_block_stride_bytes.size()) {
            return static_cast<size_t>(cache_config_.layer_to_block_stride_bytes[static_cast<size_t>(layer_id)]);
        }
        return cache_config_.kv_block_stride_bytes + cache_config_.kv_scale_stride_bytes;
    };

    for (size_t layer = 0; layer < layer_num; ++layer) {
        bool has_typed_slot = false;
        if (layer < cache_config_.layer_region_to_group_id.size()) {
            const auto&  dense = cache_config_.layer_region_to_group_id[layer];
            const size_t n     = std::min(region_name_count, dense.size());
            for (size_t attn = 0; attn < n; ++attn) {
                const int gid = dense[attn];
                if (gid < 0) {
                    continue;
                }
                slots.push_back(LayerRegionSlot{static_cast<int>(layer),
                                                static_cast<KVCacheRegionName>(attn),
                                                gid,
                                                group_stride(gid, static_cast<int>(layer))});
                has_typed_slot = true;
            }
        }
        if (!has_typed_slot) {
            int gid = 0;
            if (layer < cache_config_.layer_to_group_id.size() && cache_config_.layer_to_group_id[layer] >= 0) {
                gid = cache_config_.layer_to_group_id[layer];
            }
            slots.push_back(LayerRegionSlot{
                static_cast<int>(layer), KVCacheRegionName::DEFAULT, gid, group_stride(gid, static_cast<int>(layer))});
        }
    }
    return slots;
}

bool KVCacheMemoryConnector::hasTypedLayerRegionSlots(const std::vector<LayerRegionSlot>& slots) const {
    if (slots.size() != cache_config_.layer_all_num) {
        return true;
    }
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].layer_id != static_cast<int>(i) || slots[i].region_name != KVCacheRegionName::DEFAULT) {
            return true;
        }
    }
    return false;
}

bool KVCacheMemoryConnector::isDsv4TypedCacheLayout(const std::vector<LayerRegionSlot>& slots) const {
    // Keep this gate deliberately strict: this staged SM path is enabled by the current DSV4 typed
    // cache schema, not by model name. DSV4 Flash/Pro currently use seven typed pools in the fixed
    // order below, 256 tokens/block, sparse opaque KV cache store, and SWA_KV in group 6. If that
    // schema changes, update this matcher together with HybridPoolConfigCreator coverage.
    constexpr size_t            kDsv4PoolNum                      = 7;
    constexpr size_t            kDsv4TokensPerBlock               = 256;
    constexpr KVCacheRegionName kExpectedRegions[kDsv4PoolNum]    = {KVCacheRegionName::CSA_KV,
                                                                  KVCacheRegionName::HCA_KV,
                                                                  KVCacheRegionName::INDEXER_KV,
                                                                  KVCacheRegionName::INDEXER_STATE,
                                                                  KVCacheRegionName::CSA_STATE,
                                                                  KVCacheRegionName::HCA_STATE,
                                                                  KVCacheRegionName::SWA_KV};
    constexpr CacheGroupType    kExpectedGroupTypes[kDsv4PoolNum] = {CacheGroupType::FULL,
                                                                  CacheGroupType::FULL,
                                                                  CacheGroupType::FULL,
                                                                  CacheGroupType::SWA,
                                                                  CacheGroupType::SWA,
                                                                  CacheGroupType::SWA,
                                                                  CacheGroupType::SWA};

    if (slots.empty() || !hasTypedLayerRegionSlots(slots)) {
        return false;
    }
    if (!cache_config_.use_typed_cache_regions || !cache_config_.use_opaque_kv_cache_store
        || !cache_config_.use_independent_block_pools || !cache_config_.is_sparse) {
        return false;
    }
    if (cache_config_.seq_size_per_block != kDsv4TokensPerBlock
        || cache_config_.kernel_seq_size_per_block != kDsv4TokensPerBlock) {
        return false;
    }
    if (cache_config_.cache_specs.size() != kDsv4PoolNum || cache_config_.group_region_names.size() != kDsv4PoolNum
        || cache_config_.group_types.size() != kDsv4PoolNum
        || cache_config_.group_kv_block_stride_bytes.size() < kDsv4PoolNum
        || cache_config_.group_kv_scale_stride_bytes.size() < kDsv4PoolNum
        || cache_config_.layer_region_to_group_id.size() < cache_config_.layer_all_num) {
        return false;
    }

    for (size_t gid = 0; gid < kDsv4PoolNum; ++gid) {
        if (cache_config_.group_region_names[gid] != kExpectedRegions[gid]
            || cache_config_.group_types[gid] != kExpectedGroupTypes[gid]) {
            return false;
        }
        if (cache_config_.group_kv_block_stride_bytes[gid] + cache_config_.group_kv_scale_stride_bytes[gid] == 0) {
            return false;
        }
    }

    bool saw_csa_layer = false;
    bool saw_hca_layer = false;
    for (size_t layer = 0; layer < cache_config_.layer_all_num; ++layer) {
        const auto& row = cache_config_.layer_region_to_group_id[layer];
        if (row.size() < regionIndex(KVCacheRegionName::REGION_COUNT)) {
            return false;
        }
        if (row[regionIndex(KVCacheRegionName::DEFAULT)] >= 0 || row[regionIndex(KVCacheRegionName::SWA_KV)] != 6) {
            return false;
        }

        const bool has_csa_region = row[regionIndex(KVCacheRegionName::CSA_KV)] >= 0
                                    || row[regionIndex(KVCacheRegionName::INDEXER_KV)] >= 0
                                    || row[regionIndex(KVCacheRegionName::INDEXER_STATE)] >= 0
                                    || row[regionIndex(KVCacheRegionName::CSA_STATE)] >= 0;
        const bool has_hca_region =
            row[regionIndex(KVCacheRegionName::HCA_KV)] >= 0 || row[regionIndex(KVCacheRegionName::HCA_STATE)] >= 0;
        const bool complete_csa_region = row[regionIndex(KVCacheRegionName::CSA_KV)] == 0
                                         && row[regionIndex(KVCacheRegionName::INDEXER_KV)] == 2
                                         && row[regionIndex(KVCacheRegionName::INDEXER_STATE)] == 3
                                         && row[regionIndex(KVCacheRegionName::CSA_STATE)] == 4;
        const bool complete_hca_region =
            row[regionIndex(KVCacheRegionName::HCA_KV)] == 1 && row[regionIndex(KVCacheRegionName::HCA_STATE)] == 5;
        if (has_csa_region != complete_csa_region || has_hca_region != complete_hca_region
            || (complete_csa_region && complete_hca_region)) {
            return false;
        }
        saw_csa_layer = saw_csa_layer || complete_csa_region;
        saw_hca_layer = saw_hca_layer || complete_hca_region;
    }
    if (!saw_csa_layer || !saw_hca_layer) {
        return false;
    }

    for (const auto& slot : slots) {
        const auto layer = static_cast<size_t>(slot.layer_id);
        const auto attn  = regionIndex(slot.region_name);
        if (slot.group_id < 0 || static_cast<size_t>(slot.group_id) >= kDsv4PoolNum
            || layer >= cache_config_.layer_region_to_group_id.size()
            || attn >= cache_config_.layer_region_to_group_id[layer].size()
            || cache_config_.layer_region_to_group_id[layer][attn] != slot.group_id) {
            return false;
        }
        const auto group_id = static_cast<size_t>(slot.group_id);
        const auto group_stride =
            cache_config_.group_kv_block_stride_bytes[group_id] + cache_config_.group_kv_scale_stride_bytes[group_id];
        if (slot.stride_bytes != group_stride) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<AsyncMatchContext> KVCacheMemoryConnector::asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                                      const std::shared_ptr<Meta>&            meta) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK_WITH_INFO(meta != nullptr, "async match failed, meta is null");
    RTP_LLM_CHECK_WITH_INFO(resource != nullptr, "async match failed, resource is null");
    if (!meta->enableMemoryCache()) {
        return nullptr;
    }

    const auto& cache_keys = resource->cacheKeys();
    // do not match last block, whether it is aligned or not, otherwise may cause core dump in computing ops.
    const auto cache_keys_size = cache_keys.empty() ? 0 : cache_keys.size() - 1;
    if (cache_keys_size == 0) {
        RTP_LLM_LOG_DEBUG("async match skip, cache keys is empty");
        return nullptr;
    }

    const auto slots                = layerRegionSlots();
    const auto layer_attn_block_ids = resourceLayerRegionBlocks(*resource, slots);
    if (!checkLayerRegionBlocks(layer_attn_block_ids, slots, cache_keys_size)) {
        RTP_LLM_LOG_WARNING("async match failed, invalid layer_attn_block_ids, cache_keys_size=%zu", cache_keys_size);
        return nullptr;
    }

    const size_t already_reuse_num = resource->reuseBlockNum();
    if (already_reuse_num >= cache_keys_size) {
        // gpu has already matched all cache keys, no need to match in memory
        RTP_LLM_LOG_DEBUG(
            "async match skip, already reuse num is greater than cache keys size, cache_keys size: %zu, already_reuse_num: %zu",
            cache_keys_size,
            already_reuse_num);
        return nullptr;
    }

    autil::ScopedTime2 timer;

    // matched_num must end at a key that satisfies BOTH:
    // - memory cache key is complete
    // - all gpu blocks for this key are valid (non-null)
    //
    // Notes:
    // - If a key is complete, we allow gpu blocks to be partially invalid and keep matching further.
    // - If all gpu blocks are valid, the final matched key must be complete.
    size_t matched_num  = already_reuse_num;
    bool   matched_disk = false;
    for (size_t i = already_reuse_num; i < cache_keys_size; ++i) {
        const auto cache_key    = cache_keys.at(i);
        const auto match_result = block_cache_->match(static_cast<CacheKeyType>(cache_key));
        if (match_result.backing_type == CacheBackingType::MEMORY && isNullBlockIdx(match_result.matched_index)) {
            break;  // only continuous prefix
        }
        if (match_result.backing_type == CacheBackingType::DISK && match_result.disk_slot < 0) {
            break;
        }
        matched_disk = matched_disk || match_result.backing_type == CacheBackingType::DISK;
        if (match_result.is_complete && gpuBlocksAllValid(layer_attn_block_ids, slots, i)) {
            matched_num = i + 1;
        }
    }

    if (matched_num <= already_reuse_num) {
        RTP_LLM_LOG_DEBUG("not matched cache in memory, cache keys size: %zu, already_reuse_num: %zu",
                          cache_keys_size,
                          already_reuse_num);
        reportMatchMetrics(/*success=*/false, timer.done_us(), cache_keys_size, matched_num);
        reportDiskMatchMetrics(/*success=*/false, timer.done_us(), cache_keys_size, 0);
        return nullptr;
    }
    RTP_LLM_LOG_INFO("memory cache matched blocks: already_reuse=%zu matched=%zu cache_keys=%zu",
                     already_reuse_num,
                     matched_num,
                     cache_keys_size);
    reportMatchMetrics(/*success=*/true, timer.done_us(), cache_keys_size, matched_num);
    reportDiskMatchMetrics(/*success=*/true, timer.done_us(), cache_keys_size, matched_disk ? matched_num : 0);
    return std::make_shared<MemoryAsyncMatchContext>(matched_num);
}

bool KVCacheMemoryConnector::gpuBlocksAllValid(const LayerBlockIds& layer_block_ids, size_t key_index) const {
    for (size_t layer = 0; layer < cache_config_.layer_all_num; ++layer) {
        const auto& blocks = layer_block_ids.at(layer)->blocks();
        if (isNullBlockIdx(blocks.at(key_index))) {
            return false;
        }
    }
    return true;
}

bool KVCacheMemoryConnector::gpuBlocksAllValid(const LayerAttnBlockIds&            layer_attn_block_ids,
                                               const std::vector<LayerRegionSlot>& slots,
                                               size_t                              key_index) const {
    for (const auto& slot : slots) {
        const auto layer = static_cast<size_t>(slot.layer_id);
        const auto attn  = static_cast<size_t>(slot.region_name);
        if (layer >= layer_attn_block_ids.size() || attn >= layer_attn_block_ids[layer].size()
            || layer_attn_block_ids[layer][attn] == nullptr) {
            return false;
        }
        const auto& blocks = layer_attn_block_ids[layer][attn]->blocks();
        if (key_index >= blocks.size() || isNullBlockIdx(blocks[key_index])) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<AsyncContext> KVCacheMemoryConnector::asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                                const std::shared_ptr<Meta>&              meta,
                                                                const std::shared_ptr<AsyncMatchContext>& match_context,
                                                                int start_read_block_index,
                                                                int read_block_num) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK_WITH_INFO(resource != nullptr, "async read failed, resource is null");
    const auto& cache_keys      = resource->cacheKeys();
    const auto  cache_keys_size = cache_keys.empty() ? 0 : cache_keys.size() - 1;
    if (cache_keys_size == 0) {
        RTP_LLM_LOG_DEBUG("async read skip, cache keys is empty");
        return nullptr;
    }

    autil::ScopedTime2 timer;

    const auto slots                = layerRegionSlots();
    const auto layer_attn_block_ids = resourceLayerRegionBlocks(*resource, slots);
    if (!checkLayerRegionBlocks(layer_attn_block_ids, slots, cache_keys_size)) {
        reportReadMetrics(false, timer.done_us(), cache_keys_size, 0);
        return nullptr;
    }

    if (start_read_block_index < 0 || read_block_num <= 0
        || start_read_block_index + read_block_num > cache_keys_size) {
        RTP_LLM_LOG_WARNING(
            "async read failed, invalid block range, start_read_block_index: %d, read_block_num: %d, cache_keys size: %zu",
            start_read_block_index,
            read_block_num,
            cache_keys_size);
        reportReadMetrics(false, timer.done_us(), cache_keys_size, 0);
        return nullptr;
    }

    auto copy_plan =
        buildCopyPlanForRead(cache_keys, layer_attn_block_ids, slots, start_read_block_index, read_block_num);
    if (!copy_plan || copy_plan->copy_infos.empty()) {
        reportReadMetrics(false, timer.done_us(), cache_keys_size, 0);
        return nullptr;
    }

    const auto total_block_num = cache_keys_size;
    auto       read_done = [resource, copy_plan, total_block_num, read_block_num, timer, this](bool success) mutable {
        RTP_LLM_LOG_DEBUG("async read done, success: %d", success);
        int64_t disk_read_block_num = 0;
        for (const auto& copy_info : copy_plan->copy_infos) {
            if (copy_info.backing_type == CacheBackingType::DISK) {
                ++disk_read_block_num;
            }
        }
        if (success) {
            resource->setMemoryReuseBlockNum(read_block_num);
            for (const auto& copy_info : copy_plan->copy_infos) {
                const auto removed_item = block_cache_->removeIfMatch(
                    copy_info.cache_key, copy_info.backing_type, copy_info.mem_block, copy_info.disk_slot);
                if (!removed_item.has_value()) {
                    continue;
                }
                releaseCacheBacking(*removed_item);
            }
            RTP_LLM_LOG_INFO("memory cache read success: read_blocks=%d released_blocks=%zu total_blocks=%zu",
                             read_block_num,
                             copy_plan->copy_infos.size(),
                             total_block_num);
        }
        // reset ptr to release memory block refs
        copy_plan.reset();
        reportReadMetrics(success, timer.done_us(), total_block_num, read_block_num);
        if (disk_read_block_num > 0) {
            reportDiskReadMetrics(success, timer.done_us(), total_block_num, success ? disk_read_block_num : 0);
        }
    };

    auto context = std::make_shared<MemoryAsyncContext>(read_done);
    if (!startCopyAsync(context, copy_plan)) {
        RTP_LLM_LOG_WARNING("async read failed, start copy plan async failed");
        read_done(false);
        return nullptr;
    }
    return context;
}

std::shared_ptr<KVCacheMemoryConnector::CopyPlan>
KVCacheMemoryConnector::buildCopyPlanForRead(const CacheKeysType&                cache_keys,
                                             const LayerAttnBlockIds&            layer_attn_block_ids,
                                             const std::vector<LayerRegionSlot>& slots,
                                             int                                 start_index,
                                             int                                 read_num) {
    std::vector<CopyInfoPerKey> copy_infos;
    bool                        success = true;

    for (int i = start_index; i < start_index + read_num; ++i) {
        const auto cache_key    = cache_keys.at(i);
        const auto match_result = block_cache_->matchAndMarkInFlight(static_cast<CacheKeyType>(cache_key));
        if (match_result.backing_type == CacheBackingType::MEMORY && isNullBlockIdx(match_result.matched_index)) {
            RTP_LLM_LOG_WARNING("build copy plan for read failed, cache key not found, cache key: %ld", cache_key);
            success = false;
            break;
        }
        if (match_result.backing_type == CacheBackingType::DISK && match_result.disk_slot < 0) {
            RTP_LLM_LOG_WARNING("build copy plan for read failed, invalid disk slot, cache key: %ld", cache_key);
            success = false;
            break;
        }
        // 每次都加引用的原因是为了确保match到的block不会被释放(避免在写时malloc如果cache满弹出该block)
        if (match_result.backing_type == CacheBackingType::MEMORY) {
            referenceBlocks({match_result.matched_index}, /*cache_ref=*/false);
        } else {
            disk_block_pool_->requestReference(match_result.disk_slot);
        }

        CopyInfoPerKey copy_info;
        copy_info.cache_key    = cache_key;
        copy_info.backing_type = match_result.backing_type;
        copy_info.mem_block    = match_result.matched_index;
        copy_info.disk_slot    = match_result.disk_slot;
        copy_info.gpu_blocks.reserve(slots.size());
        for (const auto& slot : slots) {
            // Do NOT skip NULL_BLOCK_IDX here. The merged memory block layout requires reserving
            // per-layer+attn stride even when this slot has no gpu block (-1).
            const auto layer = static_cast<size_t>(slot.layer_id);
            const auto attn  = static_cast<size_t>(slot.region_name);
            copy_info.gpu_blocks.push_back(layer_attn_block_ids.at(layer).at(attn)->blocks().at(i));
        }
        copy_info.is_complete = match_result.is_complete;
        copy_infos.emplace_back(std::move(copy_info));
    }

    // 在match时已经保证了最后一个key是complete, 这里再校验下
    if (success && !copy_infos.empty() && !copy_infos.back().is_complete) {
        RTP_LLM_LOG_WARNING("build copy plan for read failed, last key is not complete, cache key: %ld",
                            copy_infos.back().cache_key);
        success = false;
    }

    // free blocks in destructor
    auto plan = createCopyPlan(copy_infos, CopyDirection::H2D);
    return success ? plan : nullptr;
}

std::shared_ptr<AsyncContext> KVCacheMemoryConnector::asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                                 const std::shared_ptr<Meta>&            meta) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK_WITH_INFO(meta != nullptr, "async write failed, meta is null");
    RTP_LLM_CHECK_WITH_INFO(resource != nullptr, "async write failed, resource is null");
    if (!meta->enableMemoryCache()) {
        return nullptr;
    }

    const auto& cache_keys = resource->cacheKeys();
    const auto  cache_keys_size =
        cache_keys.empty() ? 0 : (resource->lastBlockAligned() ? cache_keys.size() : cache_keys.size() - 1);
    if (cache_keys_size == 0) {
        RTP_LLM_LOG_DEBUG("async write skip, cache keys is empty");
        return nullptr;
    }

    autil::ScopedTime2 timer;

    const auto slots                = layerRegionSlots();
    const auto layer_attn_block_ids = resourceLayerRegionBlocks(*resource, slots);
    if (!checkLayerRegionBlocks(layer_attn_block_ids, slots, cache_keys_size)) {
        reportWriteMetrics(false, timer.done_us(), cache_keys_size, 0);
        return nullptr;
    }

    // 计算内存中已存在的前缀长度
    size_t mem_matched_num = 0;
    for (; mem_matched_num < cache_keys_size; ++mem_matched_num) {
        if (!block_cache_->contains(static_cast<CacheKeyType>(cache_keys[mem_matched_num]))) {
            break;
        }
    }
    if (mem_matched_num == cache_keys_size) {
        RTP_LLM_LOG_DEBUG(
            "async write skip, all cache keys already in memory cache, matched num: %zu, cache keys size: %zu",
            mem_matched_num,
            cache_keys_size);
        reportWriteMetrics(true, timer.done_us(), static_cast<int64_t>(cache_keys_size), 0);
        return nullptr;
    }

    bool no_need_write = false;
    auto copy_plan     = buildCopyPlanForWrite(
        cache_keys, layer_attn_block_ids, slots, mem_matched_num, cache_keys_size - mem_matched_num, no_need_write);
    if (!copy_plan || copy_plan->copy_infos.empty()) {
        reportWriteMetrics(no_need_write, timer.done_us(), static_cast<int64_t>(cache_keys_size), 0);
        return nullptr;
    }

    auto write_done =
        [copy_plan, resource_copy = resource, timer, total_block_num = cache_keys_size, this](bool success) mutable {
            RTP_LLM_LOG_DEBUG("async write done, success: %d", success);
            int64_t disk_write_block_num = 0;
            for (const auto& copy_info : copy_plan->copy_infos) {
                if (copy_info.backing_type == CacheBackingType::DISK) {
                    ++disk_write_block_num;
                }
            }

            if (success) {
                for (const auto& copy_info : copy_plan->copy_infos) {
                    MemoryDiskBlockCache::CacheItem item;
                    item.cache_key    = copy_info.cache_key;
                    item.backing_type = copy_info.backing_type;
                    item.block_index  = copy_info.mem_block;
                    item.disk_slot    = copy_info.disk_slot;
                    item.is_resident  = false;
                    item.is_complete  = copy_info.is_complete;
                    putToCache(item);
                }
            }
            // reset resource to decrease block ref count in destructor
            resource_copy.reset();
            const int64_t write_block_num = success ? static_cast<int64_t>(copy_plan->copy_infos.size()) : 0;
            // reset copy plan to release memory block refs
            copy_plan.reset();
            reportWriteMetrics(success, timer.done_us(), total_block_num, write_block_num);
            if (disk_write_block_num > 0) {
                reportDiskWriteMetrics(success, timer.done_us(), total_block_num, success ? disk_write_block_num : 0);
            }
        };

    auto context = std::make_shared<MemoryAsyncContext>(write_done);
    if (!startCopyAsync(context, copy_plan)) {
        RTP_LLM_LOG_WARNING("async write failed, start copy plan async failed");
        write_done(false);
        return nullptr;
    }
    return context;
}

std::shared_ptr<KVCacheMemoryConnector::CopyPlan>
KVCacheMemoryConnector::buildCopyPlanForWrite(const CacheKeysType&                cache_keys,
                                              const LayerAttnBlockIds&            layer_attn_block_ids,
                                              const std::vector<LayerRegionSlot>& slots,
                                              int                                 start_index,
                                              int                                 write_num,
                                              bool&                               no_need_write) {
    std::vector<CopyInfoPerKey> copy_infos;
    copy_infos.reserve(write_num);

    // Hybrid-attn support:
    // We allow writing "partial" keys (incomplete KV) to keep prefix continuity,
    // BUT the final written key MUST be "complete" (complete KV on all layers),
    // otherwise the written tail cannot be reused by asyncMatch.
    int last_complete_index = -1;  // cache_key index in [start_index, start_index + write_num)

    for (int i = start_index; i < start_index + write_num; ++i) {
        const auto                cache_key = cache_keys.at(i);
        std::vector<BlockIdxType> gpu_blocks;
        gpu_blocks.reserve(slots.size());
        size_t null_block_num = 0;
        for (const auto& slot : slots) {
            const auto layer         = static_cast<size_t>(slot.layer_id);
            const auto attn          = static_cast<size_t>(slot.region_name);
            const int  gpu_block_idx = layer_attn_block_ids.at(layer).at(attn)->blocks().at(i);
            // Do NOT skip NULL_BLOCK_IDX here. We must keep per-layer+attn stride slots in the merged big block.
            if (isNullBlockIdx(gpu_block_idx)) {
                ++null_block_num;
            }
            gpu_blocks.push_back(gpu_block_idx);
        }

        const bool is_complete = null_block_num == 0;
        if (is_complete) {
            last_complete_index = i;
        }

        CopyInfoPerKey copy_info;
        copy_info.cache_key   = cache_key;
        copy_info.mem_block   = NULL_BLOCK_IDX;
        copy_info.gpu_blocks  = std::move(gpu_blocks);
        copy_info.is_complete = is_complete;
        copy_infos.emplace_back(std::move(copy_info));
    }

    // ensure the final written key is complete
    no_need_write = last_complete_index < start_index;
    if (no_need_write) {
        return nullptr;
    }

    // drop keys behind the last complete key
    const size_t keep_cnt = static_cast<size_t>(last_complete_index - start_index + 1);
    copy_infos.resize(keep_cnt);

    if (!allocateBackingsForWrite(copy_infos)) {
        RTP_LLM_LOG_WARNING("build copy plan for write failed, allocate backing failed, need blocks: %zu",
                            copy_infos.size());
        return nullptr;
    }

    // free blocks in destructor
    auto plan = createCopyPlan(copy_infos, CopyDirection::D2H);
    return plan;
}

std::shared_ptr<KVCacheMemoryConnector::CopyPlan>
KVCacheMemoryConnector::createCopyPlan(const std::vector<CopyInfoPerKey>& copy_infos, const CopyDirection& direction) {
    auto plan        = new CopyPlan();
    plan->copy_infos = copy_infos;
    plan->direction  = direction;
    auto deleter     = [this](CopyPlan* plan) {
        for (const auto& copy_info : plan->copy_infos) {
            if (plan->direction == CopyDirection::H2D) {
                block_cache_->releaseInFlight(
                    copy_info.cache_key, copy_info.backing_type, copy_info.mem_block, copy_info.disk_slot);
            }
            releaseRequestBacking(copy_info);
        }
        delete plan;
    };
    return std::shared_ptr<CopyPlan>(plan, deleter);
}

bool KVCacheMemoryConnector::startCopyAsync(const std::shared_ptr<MemoryAsyncContext>& context,
                                            const std::shared_ptr<CopyPlan>&           copy_plan) {
    if (stop_.load()) {
        return false;
    }
    auto task_copy_plan = copy_plan;
    auto code           = wait_done_thread_pool_->pushTask([this, context, task_copy_plan]() mutable {
        auto send_result = sendCopyPlan(task_copy_plan);
        context->setBroadcastResult(send_result);
        task_copy_plan.reset();
        context->waitDone();
    });
    if (code != autil::ThreadPoolBase::ERROR_NONE) {
        RTP_LLM_LOG_WARNING("start copy plan async failed, push send+wait task failed, code=%d", code);
        return false;
    }
    return true;
}

std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>
KVCacheMemoryConnector::sendCopyPlan(const std::shared_ptr<CopyPlan>& copy_plan) const {
    MemoryOperationRequestPB mem_req;
    mem_req.set_copy_direction(copy_plan->direction == CopyDirection::H2D ? MemoryOperationRequestPB::H2D :
                                                                            MemoryOperationRequestPB::D2H);
    for (const auto& copy_info : copy_plan->copy_infos) {
        auto* item = mem_req.add_copy_items();
        item->set_mem_block(copy_info.mem_block);
        item->set_backing_type(copy_info.backing_type == CacheBackingType::MEMORY ? MemoryOperationRequestPB::MEMORY :
                                                                                    MemoryOperationRequestPB::DISK);
        item->set_disk_slot(copy_info.disk_slot);
        for (const auto& block : copy_info.gpu_blocks) {
            item->add_gpu_blocks(block);
        }
    }

    std::vector<FunctionRequestPB> requests;
    requests.reserve(broadcast_manager_->workerNum());
    for (size_t i = 0; i < broadcast_manager_->workerNum(); ++i) {
        FunctionRequestPB req;
        req.mutable_mem_request()->CopyFrom(mem_req);
        requests.emplace_back(std::move(req));
    }

    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const FunctionRequestPB&                    request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->AsyncExecuteFunction(context.get(), request, completion_queue);
    };
    return broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        requests, copyPlanTimeoutMs(copy_plan), rpc_call);
}

void KVCacheMemoryConnector::printCopyPlan(const std::shared_ptr<CopyPlan>& copy_plan) const {
    std::ostringstream oss;
    oss << "copy plan direction: " << (copy_plan->direction == CopyDirection::H2D ? "H2D" : "D2H")
        << ", copy infos size: " << copy_plan->copy_infos.size() << "\n";
    for (int i = 0; i < copy_plan->copy_infos.size(); ++i) {
        const auto& copy_info = copy_plan->copy_infos.at(i);
        oss << "copy info " << i << ": cache key: " << copy_info.cache_key
            << ", backing: " << (copy_info.backing_type == CacheBackingType::MEMORY ? "MEMORY" : "DISK")
            << ", mem block: " << copy_info.mem_block << ", disk slot: " << copy_info.disk_slot
            << ", gpu layer blocks: [";
        for (const auto& gpu_block : copy_info.gpu_blocks) {
            oss << gpu_block << ", ";
        }
        oss << "]\n";
    }
    RTP_LLM_LOG_INFO("%s", oss.str().c_str());
}

bool KVCacheMemoryConnector::copyCache(const MemoryOperationRequestPB& request, MemoryOperationResponsePB& response) {
    RTP_LLM_PROFILE_FUNCTION();
    autil::ScopedTime2 timer;
    const auto         copy_direction =
        (request.copy_direction() == MemoryOperationRequestPB::H2D) ? CopyDirection::H2D : CopyDirection::D2H;
    const auto slots            = layerRegionSlots();
    const bool has_typed_slots  = hasTypedLayerRegionSlots(slots);
    bool       has_disk_items   = false;
    bool       has_memory_items = false;

    for (int i = 0; i < request.copy_items_size(); ++i) {
        const auto& item = request.copy_items(i);
        if (!validateCopyItemBacking(item)) {
            response.set_success(false);
            reportCopyMetrics(false, timer.done_us(), copy_direction);
            return false;
        }
        if (item.backing_type() == MemoryOperationRequestPB::DISK) {
            has_disk_items = true;
        } else {
            has_memory_items = true;
        }
    }

    if (has_disk_items) {
        bool success = true;
        if (has_memory_items) {
            success = copyMemoryItemsGeneric(request, copy_direction, slots);
        }
        if (success) {
            success = copyDiskItems(request, copy_direction, slots);
        }
        response.set_success(success);
        reportCopyMetrics(success, timer.done_us(), copy_direction);
        reportDiskCopyMetrics(success, timer.done_us(), copy_direction);
        return success;
    }

    if (tryCopyCacheWithStagedMemoryCopy(request, copy_direction, slots)) {
        response.set_success(true);
        reportCopyMetrics(true, timer.done_us(), copy_direction);
        return true;
    }
    if (has_typed_slots && tryCopyCacheWithBatchedMemoryCopy(request, copy_direction, slots)) {
        response.set_success(true);
        reportCopyMetrics(true, timer.done_us(), copy_direction);
        return true;
    }

    if (!copyMemoryItemsGeneric(request, copy_direction, slots)) {
        response.set_success(false);
        reportCopyMetrics(false, timer.done_us(), copy_direction);
        return false;
    }

    response.set_success(true);
    reportCopyMetrics(true, timer.done_us(), copy_direction);
    return true;
}

bool KVCacheMemoryConnector::validateCopyItemBacking(const MemoryOperationRequestPB::CopyItem& item) const {
    const bool has_disk_slot = item.disk_slot_presence_case() == MemoryOperationRequestPB::CopyItem::kDiskSlot;
    if (item.backing_type() == MemoryOperationRequestPB::MEMORY) {
        if (has_disk_slot && item.disk_slot() != -1) {
            RTP_LLM_LOG_WARNING("memory copy item has invalid disk_slot=%d", item.disk_slot());
            return false;
        }
        return true;
    }
    if (item.backing_type() == MemoryOperationRequestPB::DISK) {
        if (!isNullBlockIdx(static_cast<BlockIdxType>(item.mem_block()))) {
            RTP_LLM_LOG_WARNING("disk copy item has non-null mem_block=%d", item.mem_block());
            return false;
        }
        if (!has_disk_slot || item.disk_slot() < 0) {
            RTP_LLM_LOG_WARNING("disk copy item has invalid disk_slot");
            return false;
        }
        return true;
    }
    RTP_LLM_LOG_WARNING("copy item has unknown backing_type=%d", static_cast<int>(item.backing_type()));
    return false;
}

bool KVCacheMemoryConnector::copyMemoryItemsGeneric(const MemoryOperationRequestPB&     request,
                                                    CopyDirection                       direction,
                                                    const std::vector<LayerRegionSlot>& slots) {
    std::vector<torch::Tensor> dst_buffers;
    std::vector<torch::Tensor> src_buffers;
    for (int i = 0; i < request.copy_items_size(); ++i) {
        const auto& item = request.copy_items(i);
        if (item.backing_type() != MemoryOperationRequestPB::MEMORY) {
            continue;
        }
        const auto                      mem_block = static_cast<BlockIdxType>(item.mem_block());
        const std::vector<BlockIdxType> gpu_blocks(item.gpu_blocks().begin(), item.gpu_blocks().end());

        if (!prepareCopyBuffers(mem_block, gpu_blocks, direction, dst_buffers, src_buffers)) {
            RTP_LLM_LOG_WARNING("copy cache failed, prepare memory copy buffers failed, mem_block=%d, direction=%s",
                                mem_block,
                                direction == CopyDirection::H2D ? "H2D" : "D2H");
            return false;
        }
    }

    if (!dst_buffers.empty()) {
        MultiCopyParams mc{dst_buffers, src_buffers};
        const bool      can_use_split_kv_copy = !hasTypedLayerRegionSlots(slots);
        applySplitKvMultiCopyFieldsIfEligible(
            kv_cache_config_.enable_memory_cache_sm_copy && can_use_split_kv_copy, cache_config_, mc);
        execNoBlockCopy(mc);
    }
    return true;
}

bool KVCacheMemoryConnector::copyDiskItems(const MemoryOperationRequestPB&     request,
                                           CopyDirection                       direction,
                                           const std::vector<LayerRegionSlot>& slots) {
    for (int i = 0; i < request.copy_items_size(); ++i) {
        const auto& item = request.copy_items(i);
        if (item.backing_type() != MemoryOperationRequestPB::DISK) {
            continue;
        }
        if (!copyDiskItem(item, direction, slots)) {
            return false;
        }
    }
    return true;
}

bool KVCacheMemoryConnector::copyDiskItem(const MemoryOperationRequestPB::CopyItem& item,
                                          CopyDirection                             direction,
                                          const std::vector<LayerRegionSlot>&       slots) {
    if (!disk_block_pool_ || item.gpu_blocks_size() != static_cast<int>(slots.size())) {
        return false;
    }

    void*        raw_buffer   = nullptr;
    const size_t stride_bytes = disk_block_pool_->slotStrideBytes();
    if (::posix_memalign(&raw_buffer, 4096, stride_bytes) != 0 || raw_buffer == nullptr) {
        RTP_LLM_LOG_WARNING("allocate disk staging buffer failed, bytes=%zu", stride_bytes);
        return false;
    }
    std::unique_ptr<void, decltype(&std::free)> staging(raw_buffer, &std::free);
    std::memset(raw_buffer, 0, stride_bytes);

    const auto disk_slot = item.disk_slot();
    if (direction == CopyDirection::H2D && !disk_block_pool_->read(disk_slot, raw_buffer, stride_bytes)) {
        RTP_LLM_LOG_WARNING("disk cache read failed, slot=%d, bytes=%zu", disk_slot, stride_bytes);
        return false;
    }

    BlockInfo disk_block;
    disk_block.is_cuda    = false;
    disk_block.addr       = raw_buffer;
    disk_block.size_bytes = disk_block_pool_->blockSizeBytes();

    std::vector<torch::Tensor> dst_buffers;
    std::vector<torch::Tensor> src_buffers;
    size_t                     byte_off = 0;
    for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
        const auto& slot      = slots[slot_idx];
        const auto  gpu_block = static_cast<BlockIdxType>(item.gpu_blocks(static_cast<int>(slot_idx)));
        if (isNullBlockIdx(gpu_block)) {
            byte_off += slot.stride_bytes;
            continue;
        }
        const auto gpu_buffers      = allocator_->convertIndexToBuffer(slot.layer_id, slot.region_name, gpu_block);
        size_t     within_layer_off = 0;
        for (const auto& gpu_buffer : gpu_buffers) {
            const auto off = byte_off + within_layer_off;
            if (!appendCopyBytesToBuffers(disk_block, gpu_buffer, off, direction, dst_buffers, src_buffers)) {
                return false;
            }
            within_layer_off += gpu_buffer.size_bytes;
        }
        byte_off += slot.stride_bytes;
    }

    if (!dst_buffers.empty()) {
        MultiCopyParams mc{dst_buffers, src_buffers};
        execNoBlockCopy(mc);
    }

    if (direction == CopyDirection::D2H && !disk_block_pool_->write(disk_slot, raw_buffer, stride_bytes)) {
        RTP_LLM_LOG_WARNING("disk cache write failed, slot=%d, bytes=%zu", disk_slot, stride_bytes);
        return false;
    }
    return true;
}

bool KVCacheMemoryConnector::tryCopyCacheWithStagedMemoryCopy(const MemoryOperationRequestPB&     request,
                                                              CopyDirection                       direction,
                                                              const std::vector<LayerRegionSlot>& slots) {
    RTP_LLM_PROFILE_SCOPE("reuse_cache.memory.copy.plan_staged");
    if (!isDsv4TypedCacheLayout(slots)) {
        return false;
    }
    if (block_pool_ == nullptr || allocator_ == nullptr) {
        return false;
    }

    StagedMemoryCopyParams params;
    params.direction =
        direction == CopyDirection::H2D ? StagedMemoryCopyDirection::H2D : StagedMemoryCopyDirection::D2H;

    size_t logical_rows  = 0;
    size_t payload_bytes = 0;

    for (int i = 0; i < request.copy_items_size(); ++i) {
        const auto&                     item      = request.copy_items(i);
        const auto                      mem_block = static_cast<BlockIdxType>(item.mem_block());
        const std::vector<BlockIdxType> gpu_blocks(item.gpu_blocks().begin(), item.gpu_blocks().end());

        if (isNullBlockIdx(mem_block) || gpu_blocks.size() != slots.size()) {
            return false;
        }

        auto mem_buffers = block_pool_->convertIndexToBuffer(/*layer_id=*/0, mem_block);
        if (mem_buffers.size() != 1u || mem_buffers[0].addr == nullptr || mem_buffers[0].size_bytes == 0
            || mem_buffers[0].is_cuda) {
            return false;
        }
        const auto& mem_buffer = mem_buffers[0];
        auto*       mem_addr   = static_cast<char*>(mem_buffer.addr);

        size_t byte_off = 0;
        for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
            const auto& slot         = slots[slot_idx];
            const auto  gpu_block    = gpu_blocks.at(slot_idx);
            const auto  layer_stride = slot.stride_bytes;

            if (isNullBlockIdx(gpu_block)) {
                byte_off += layer_stride;
                continue;
            }

            const auto gpu_buffers      = allocator_->convertIndexToBuffer(slot.layer_id, slot.region_name, gpu_block);
            size_t     within_layer_off = 0;
            for (const auto& gpu_buffer : gpu_buffers) {
                if (gpu_buffer.addr == nullptr || gpu_buffer.size_bytes == 0) {
                    within_layer_off += gpu_buffer.size_bytes;
                    continue;
                }
                if (!gpu_buffer.is_cuda) {
                    return false;
                }
                if (within_layer_off + gpu_buffer.size_bytes > layer_stride
                    || byte_off + within_layer_off + gpu_buffer.size_bytes > mem_buffer.size_bytes) {
                    return false;
                }
                if (params.device_index < 0) {
                    params.device_index = gpu_buffer.device_index;
                } else if (params.device_index != gpu_buffer.device_index) {
                    return false;
                }

                // The SM copy kernels vectorize with int4/int2. Keep every staged tile aligned so compact
                // staging does not trade fewer memcpy calls for misaligned vector accesses.
                constexpr size_t kStagedTileAlignment = 16;
                const size_t     staging_offset       = alignUp(params.host_bytes, kStagedTileAlignment);
                params.host_bytes                     = staging_offset;
                auto* host_addr                       = mem_addr + byte_off + within_layer_off;
                appendStagedMemoryCopyHostSegment(
                    host_addr, staging_offset, gpu_buffer.size_bytes, params.host_segments);
                appendStagedMemoryCopyTile(gpu_buffer.addr, staging_offset, gpu_buffer.size_bytes, params.tiles);
                params.host_bytes += gpu_buffer.size_bytes;
                ++logical_rows;
                payload_bytes += gpu_buffer.size_bytes;
                within_layer_off += gpu_buffer.size_bytes;
            }
            byte_off += layer_stride;
        }
    }

    if (params.tiles.empty()) {
        return true;
    }

    RTP_LLM_LOG_DEBUG("cuda staged memory copy, direction=%s, rows=%zu, tiles=%zu, bytes=%zu, span=%zu, device=%d",
                      direction == CopyDirection::H2D ? "H2D" : "D2H",
                      logical_rows,
                      params.tiles.size(),
                      payload_bytes,
                      params.host_bytes,
                      params.device_index);
    RTP_LLM_PROFILE_SCOPE("reuse_cache.memory.copy.exec_staged");
    std::lock_guard<std::mutex> scratch_lock(staged_copy_scratch_mutex_);
    return execStagedMemoryCopy(params, &stagedCopyScratchForDevice(params.device_index));
}

StagedMemoryCopyScratch& KVCacheMemoryConnector::stagedCopyScratchForDevice(int device_index) {
    auto& scratch = staged_copy_scratch_by_device_[device_index];
    if (!scratch) {
        scratch = std::make_unique<StagedMemoryCopyScratch>();
    }
    return *scratch;
}

bool KVCacheMemoryConnector::tryCopyCacheWithBatchedMemoryCopy(const MemoryOperationRequestPB&     request,
                                                               CopyDirection                       direction,
                                                               const std::vector<LayerRegionSlot>& slots) {
    RTP_LLM_PROFILE_SCOPE("reuse_cache.memory.copy.plan_batch");
    if (block_pool_ == nullptr || allocator_ == nullptr) {
        return false;
    }

    BatchedMemoryCopyParams params;
    size_t                  logical_rows  = 0;
    size_t                  payload_bytes = 0;

    for (int i = 0; i < request.copy_items_size(); ++i) {
        const auto&                     item      = request.copy_items(i);
        const auto                      mem_block = static_cast<BlockIdxType>(item.mem_block());
        const std::vector<BlockIdxType> gpu_blocks(item.gpu_blocks().begin(), item.gpu_blocks().end());

        if (isNullBlockIdx(mem_block) || gpu_blocks.size() != slots.size()) {
            return false;
        }

        auto mem_buffers = block_pool_->convertIndexToBuffer(/*layer_id=*/0, mem_block);
        if (mem_buffers.size() != 1u || mem_buffers[0].addr == nullptr || mem_buffers[0].size_bytes == 0) {
            return false;
        }
        const auto& mem_buffer = mem_buffers[0];

        size_t byte_off = 0;
        for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
            const auto& slot         = slots[slot_idx];
            const auto  gpu_block    = gpu_blocks.at(slot_idx);
            const auto  layer_stride = slot.stride_bytes;

            if (isNullBlockIdx(gpu_block)) {
                byte_off += layer_stride;
                continue;
            }

            const auto gpu_buffers      = allocator_->convertIndexToBuffer(slot.layer_id, slot.region_name, gpu_block);
            size_t     within_layer_off = 0;
            for (const auto& gpu_buffer : gpu_buffers) {
                if (gpu_buffer.addr == nullptr || gpu_buffer.size_bytes == 0) {
                    within_layer_off += gpu_buffer.size_bytes;
                    continue;
                }
                if (!gpu_buffer.is_cuda) {
                    return false;
                }
                if (within_layer_off + gpu_buffer.size_bytes > layer_stride
                    || byte_off + within_layer_off + gpu_buffer.size_bytes > mem_buffer.size_bytes) {
                    return false;
                }
                if (params.device_index < 0) {
                    params.device_index = gpu_buffer.device_index;
                } else if (params.device_index != gpu_buffer.device_index) {
                    return false;
                }

                auto* mem_addr = static_cast<void*>(static_cast<char*>(mem_buffer.addr) + byte_off + within_layer_off);
                if (direction == CopyDirection::H2D) {
                    appendBatchedMemoryCopyTile(gpu_buffer.addr, mem_addr, gpu_buffer.size_bytes, params.tiles);
                } else {
                    appendBatchedMemoryCopyTile(mem_addr, gpu_buffer.addr, gpu_buffer.size_bytes, params.tiles);
                }
                ++logical_rows;
                payload_bytes += gpu_buffer.size_bytes;
                within_layer_off += gpu_buffer.size_bytes;
            }
            byte_off += layer_stride;
        }
    }

    if (params.tiles.empty()) {
        return true;
    }

    RTP_LLM_LOG_DEBUG("cuda memcpy batch, direction=%s, rows=%zu, tiles=%zu, bytes=%zu, device=%d",
                      direction == CopyDirection::H2D ? "H2D" : "D2H",
                      logical_rows,
                      params.tiles.size(),
                      payload_bytes,
                      params.device_index);
    RTP_LLM_PROFILE_SCOPE("reuse_cache.memory.copy.exec_batch");
    return execBatchedMemoryCopy(params);
}

bool KVCacheMemoryConnector::prepareCopyBuffers(BlockIdxType                     mem_block,
                                                const std::vector<BlockIdxType>& gpu_blocks,
                                                CopyDirection                    direction,
                                                std::vector<torch::Tensor>&      dst,
                                                std::vector<torch::Tensor>&      src) {
    RTP_LLM_CHECK_WITH_INFO(mem_block != NULL_BLOCK_IDX, "mem block is null");
    RTP_LLM_CHECK_WITH_INFO(block_pool_ != nullptr, "block pool is null");
    auto mem_buffers = block_pool_->convertIndexToBuffer(/*layer_id=*/0, mem_block);
    if (mem_buffers.empty()) {
        RTP_LLM_LOG_WARNING("prepare copy buffers failed, mem buffers are empty, block=%d, direction=%s",
                            mem_block,
                            direction == CopyDirection::H2D ? "H2D" : "D2H");
        return false;
    }

    // memory has only one buffer
    const auto& mem_buffer = mem_buffers[0];
    RTP_LLM_CHECK_WITH_INFO(mem_buffer.addr != nullptr && mem_buffer.size_bytes > 0,
                            "mem buffer address is null or size is 0, addr=%p, size=%zu, block=%d, direction=%s",
                            mem_buffer.addr,
                            mem_buffer.size_bytes,
                            mem_block,
                            direction == CopyDirection::H2D ? "H2D" : "D2H");

    const auto slots = layerRegionSlots();
    RTP_LLM_CHECK_WITH_INFO(gpu_blocks.size() == slots.size(),
                            "gpu_blocks must contain all layer-attn slots, got=%zu need=%zu",
                            gpu_blocks.size(),
                            slots.size());

    size_t byte_off = 0;
    for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
        const auto& slot         = slots[slot_idx];
        const auto  gpu_block    = gpu_blocks.at(slot_idx);
        const auto  layer_stride = slot.stride_bytes;

        if (isNullBlockIdx(gpu_block)) {
            byte_off += layer_stride;
            continue;
        }

        const auto gpu_buffers      = allocator_->convertIndexToBuffer(slot.layer_id, slot.region_name, gpu_block);
        size_t     within_layer_off = 0;
        for (const auto& gpu_buffer : gpu_buffers) {
            if (within_layer_off + gpu_buffer.size_bytes > layer_stride) {
                RTP_LLM_LOG_WARNING("prepare copy buffers failed, gpu buffer overflow: "
                                    "layer=%d region_name=%d byte_off=%zu within_layer_off=%zu gpu_buffer_size=%zu",
                                    slot.layer_id,
                                    static_cast<int>(slot.region_name),
                                    byte_off,
                                    within_layer_off,
                                    gpu_buffer.size_bytes);
                return false;
            }
            const size_t off = byte_off + within_layer_off;
            if (!appendCopyBytesToBuffers(mem_buffer, gpu_buffer, off, direction, dst, src)) {
                return false;
            }
            within_layer_off += gpu_buffer.size_bytes;
        }
        byte_off += layer_stride;
    }
    return true;
}

bool KVCacheMemoryConnector::appendCopyBytesToBuffers(const BlockInfo&            mem_block,
                                                      const BlockInfo&            gpu_block,
                                                      size_t                      byte_off,
                                                      CopyDirection               direction,
                                                      std::vector<torch::Tensor>& dst,
                                                      std::vector<torch::Tensor>& src) {
    if (!gpu_block.addr || gpu_block.size_bytes == 0) {
        return true;
    }
    if (byte_off + gpu_block.size_bytes > mem_block.size_bytes) {
        RTP_LLM_LOG_WARNING(
            "append copy bytes to buffers failed, mem block overflow: offset=%zu bytes=%zu mem_size=%zu",
            byte_off,
            gpu_block.size_bytes,
            mem_block.size_bytes);
        return false;
    }

    auto mem_device = mem_block.is_cuda ? torch::kCUDA : torch::kCPU;
    auto gpu_device = gpu_block.is_cuda ? torch::kCUDA : torch::kCPU;
    auto mem_tensor = torch::from_blob(static_cast<void*>(static_cast<char*>(mem_block.addr) + byte_off),
                                       {(int64_t)gpu_block.size_bytes},
                                       torch::TensorOptions().dtype(torch::kUInt8).device(mem_device));
    auto gpu_tensor = torch::from_blob(gpu_block.addr,
                                       {(int64_t)gpu_block.size_bytes},
                                       torch::TensorOptions().dtype(torch::kUInt8).device(gpu_device));
    if (direction == CopyDirection::H2D) {
        src.push_back(mem_tensor);
        dst.push_back(gpu_tensor);
    } else {
        src.push_back(gpu_tensor);
        dst.push_back(mem_tensor);
    }
    return true;
}

bool KVCacheMemoryConnector::checkLayerBlocks(const LayerBlockIds& layer_block_ids, size_t required_len) const {
    if (layer_block_ids.empty()) {
        RTP_LLM_LOG_WARNING(
            "check layer blocks failed, layer_block_ids is empty (required_len=%zu, layer_block_ids.size=%zu)",
            required_len,
            layer_block_ids.size());
        return false;
    }

    const auto layer_num = cache_config_.layer_all_num;
    if (layer_block_ids.size() != layer_num) {
        RTP_LLM_LOG_WARNING(
            "check layer blocks failed, layer block ids size is not equal to layer num, layer block ids size: %zu, layer num: %zu",
            layer_block_ids.size(),
            layer_num);
        return false;
    }
    for (const auto& blocks : layer_block_ids) {
        if (blocks->blocksNum() < required_len) {
            RTP_LLM_LOG_WARNING(
                "check layer blocks failed, layer blocksNum is less than required_len, blocksNum: %zu, required_len: %zu",
                blocks->blocksNum(),
                required_len);
            return false;
        }
    }
    return true;
}

LayerAttnBlockIds
KVCacheMemoryConnector::resourceLayerRegionBlocks(const KVCacheResource&                resource,
                                                  const std::vector<LayerRegionSlot>& slots) const {
    if (!resource.layerAttnBlocks().empty()) {
        return resource.layerAttnBlocks();
    }

    const auto& legacy_layer_blocks = resource.layerBlocks();
    if (legacy_layer_blocks.empty()) {
        return {};
    }

    LayerAttnBlockIds layer_region_blocks;
    layer_region_blocks.resize(static_cast<size_t>(cache_config_.layer_all_num));
    for (auto& regions : layer_region_blocks) {
        regions.resize(static_cast<size_t>(KVCacheRegionName::REGION_COUNT));
    }

    for (const auto& slot : slots) {
        const auto layer = static_cast<size_t>(slot.layer_id);
        const auto attn  = static_cast<size_t>(slot.region_name);
        if (layer >= legacy_layer_blocks.size() || legacy_layer_blocks[layer] == nullptr) {
            return {};
        }
        layer_region_blocks[layer][attn] = legacy_layer_blocks[layer];
    }
    return layer_region_blocks;
}

bool KVCacheMemoryConnector::checkLayerRegionBlocks(const LayerAttnBlockIds&            layer_attn_block_ids,
                                                    const std::vector<LayerRegionSlot>& slots,
                                                    size_t                              required_len) const {
    if (layer_attn_block_ids.empty()) {
        RTP_LLM_LOG_WARNING("check layer-attn blocks failed, layer_attn_block_ids is empty (required_len=%zu)",
                            required_len);
        return false;
    }
    for (const auto& slot : slots) {
        const auto layer = static_cast<size_t>(slot.layer_id);
        const auto attn  = static_cast<size_t>(slot.region_name);
        if (layer >= layer_attn_block_ids.size() || attn >= layer_attn_block_ids[layer].size()
            || layer_attn_block_ids[layer][attn] == nullptr) {
            RTP_LLM_LOG_WARNING("check layer-attn blocks failed, missing slot layer=%d region_name=%d",
                                slot.layer_id,
                                static_cast<int>(slot.region_name));
            return false;
        }
        if (layer_attn_block_ids[layer][attn]->blocksNum() < required_len) {
            RTP_LLM_LOG_WARNING(
                "check layer-attn blocks failed, blocksNum is less than required_len, layer=%d region_name=%d blocksNum=%zu required_len=%zu",
                slot.layer_id,
                static_cast<int>(slot.region_name),
                layer_attn_block_ids[layer][attn]->blocksNum(),
                required_len);
            return false;
        }
    }
    return true;
}

bool KVCacheMemoryConnector::freeBlocks(const std::vector<BlockIdxType>& blocks, bool cache_free) {
    RTP_LLM_PROFILE_FUNCTION();
    std::vector<int> need_free_blocks;
    need_free_blocks.reserve(blocks.size());
    for (const auto& block : blocks) {
        if (isNullBlockIdx(block)) {
            continue;
        }
        need_free_blocks.push_back(static_cast<int>(block));
    }
    if (need_free_blocks.empty()) {
        return true;
    }

    RTP_LLM_CHECK_WITH_INFO(block_pool_ != nullptr, "block pool is null");
    if (cache_free) {
        // cache中的block需要blockCacheFree
        block_pool_->blockCacheFree(need_free_blocks);
    } else {
        // malloc的block需要requestFree
        block_pool_->requestFree(need_free_blocks);
    }
    return true;
}

void KVCacheMemoryConnector::referenceBlocks(const std::vector<BlockIdxType>& blocks, bool cache_ref) {
    RTP_LLM_CHECK_WITH_INFO(block_pool_ != nullptr, "block pool is null");
    if (cache_ref) {
        block_pool_->blockCacheReference(blocks);
    } else {
        block_pool_->requestReference(blocks);
    }
}

bool KVCacheMemoryConnector::allocateBackingsForWrite(std::vector<CopyInfoPerKey>& copy_infos) {
    std::unique_lock<std::mutex> lock(malloc_mutex_);
    std::vector<size_t>          allocated_indices;
    allocated_indices.reserve(copy_infos.size());
    for (size_t i = 0; i < copy_infos.size(); ++i) {
        if (!allocateOneBacking(copy_infos[i])) {
            for (const auto idx : allocated_indices) {
                releaseRequestBacking(copy_infos[idx]);
            }
            return false;
        }
        allocated_indices.push_back(i);
    }
    return true;
}

bool KVCacheMemoryConnector::allocateOneBacking(CopyInfoPerKey& copy_info) {
    BlockIdxType mem_block = NULL_BLOCK_IDX;
    if (tryMallocMemoryBlock(mem_block)) {
        copy_info.backing_type = CacheBackingType::MEMORY;
        copy_info.mem_block    = mem_block;
        copy_info.disk_slot    = -1;
        return true;
    }

    int32_t disk_slot = -1;
    if (diskCacheEnabled() && tryMallocDiskSlot(disk_slot)) {
        copy_info.backing_type = CacheBackingType::DISK;
        copy_info.mem_block    = NULL_BLOCK_IDX;
        copy_info.disk_slot    = disk_slot;
        return true;
    }

    while (true) {
        auto evicted = block_cache_->popOldestEvictable();
        if (!evicted.has_value()) {
            return false;
        }
        const auto target_backing = evicted->backing_type;
        releaseCacheBacking(*evicted);
        if (target_backing == CacheBackingType::MEMORY) {
            if (tryMallocMemoryBlock(mem_block)) {
                copy_info.backing_type = CacheBackingType::MEMORY;
                copy_info.mem_block    = mem_block;
                copy_info.disk_slot    = -1;
                return true;
            }
            continue;
        }
        if (target_backing == CacheBackingType::DISK && tryMallocDiskSlot(disk_slot)) {
            copy_info.backing_type = CacheBackingType::DISK;
            copy_info.mem_block    = NULL_BLOCK_IDX;
            copy_info.disk_slot    = disk_slot;
            return true;
        }
        RTP_LLM_LOG_WARNING("allocate backing failed after evicting backing=%d, retrying",
                            static_cast<int>(target_backing));
    }
}

bool KVCacheMemoryConnector::tryMallocMemoryBlock(BlockIdxType& block) {
    block = NULL_BLOCK_IDX;
    if (block_pool_ == nullptr || block_pool_->freeBlocksNum() == 0) {
        return false;
    }
    auto blocks = block_pool_->malloc(1);
    if (blocks.size() != 1) {
        return false;
    }
    block = blocks[0];
    return true;
}

bool KVCacheMemoryConnector::tryMallocDiskSlot(int32_t& slot) {
    slot = -1;
    if (!disk_block_pool_) {
        return false;
    }
    auto allocated_slot = disk_block_pool_->malloc();
    if (!allocated_slot.has_value()) {
        return false;
    }
    slot = *allocated_slot;
    return true;
}

void KVCacheMemoryConnector::releaseRequestBacking(const CopyInfoPerKey& copy_info) {
    if (copy_info.backing_type == CacheBackingType::MEMORY) {
        freeBlocks({copy_info.mem_block}, /*cache_free=*/false);
    } else if (copy_info.backing_type == CacheBackingType::DISK && disk_block_pool_) {
        disk_block_pool_->requestFree(copy_info.disk_slot);
    }
}

void KVCacheMemoryConnector::releaseCacheBacking(const MemoryDiskBlockCache::CacheItem& item) {
    if (item.backing_type == CacheBackingType::MEMORY) {
        freeBlocks({item.block_index}, /*cache_free=*/true);
    } else if (item.backing_type == CacheBackingType::DISK && disk_block_pool_) {
        disk_block_pool_->blockCacheFree(item.disk_slot);
    }
}

void KVCacheMemoryConnector::referenceCacheBacking(const MemoryDiskBlockCache::CacheItem& item) {
    if (item.backing_type == CacheBackingType::MEMORY) {
        referenceBlocks({item.block_index}, /*cache_ref=*/true);
    } else if (item.backing_type == CacheBackingType::DISK && disk_block_pool_) {
        disk_block_pool_->blockCacheReference(item.disk_slot);
    }
}

std::shared_ptr<BlockPool> KVCacheMemoryConnector::createBlockPool(size_t block_size, size_t pool_size_mb) const {
    RTP_LLM_CHECK_WITH_INFO(pool_size_mb > 0, "pool size must be > 0");
    const int64_t block_num = pool_size_mb * 1024 * 1024 / static_cast<int64_t>(block_size);
    RTP_LLM_CHECK_WITH_INFO(
        block_num > 0, "pool_size_mb=%ld is too small for block_size=%zu (block_num=0)", pool_size_mb, block_size);
    RTP_LLM_LOG_INFO("create memory block pool, pool size: %ld MB, block num: %ld, block size: %zu",
                     pool_size_mb,
                     block_num,
                     block_size);
    const auto pool_config = BlockPoolConfigHelper::createConfig(
        /*layer_num=*/1, static_cast<uint32_t>(block_num), static_cast<uint32_t>(block_size), rtp_llm::TYPE_INT8);
    auto pool = std::make_shared<BlockPool>(pool_config, AllocationType::HOST);
    RTP_LLM_CHECK_WITH_INFO(pool->init(), "memory block pool init failed, block size: %zu", block_size);
    return pool;
}

std::string KVCacheMemoryConnector::blockPoolDebugString() const {
    std::stringstream oss;
    oss << "total blocks num: " << block_pool_->totalBlocksNum()
        << ", free blocks num: " << block_pool_->freeBlocksNum()
        << ", available blocks num: " << block_pool_->availableBlocksNum();
    return oss.str();
}

void KVCacheMemoryConnector::putToCache(const MemoryBlockCache::CacheItem& item) {
    RTP_LLM_PROFILE_FUNCTION();
    MemoryDiskBlockCache::CacheItem new_item;
    new_item.cache_key    = item.cache_key;
    new_item.backing_type = CacheBackingType::MEMORY;
    new_item.block_index  = item.block_index;
    new_item.disk_slot    = -1;
    new_item.block_size   = item.block_size;
    new_item.is_resident  = item.is_resident;
    new_item.is_complete  = item.is_complete;
    putToCache(new_item);
}

void KVCacheMemoryConnector::putToCache(const MemoryDiskBlockCache::CacheItem& item) {
    RTP_LLM_PROFILE_FUNCTION();
    if (auto [success, popped_item_opt] = block_cache_->putCommitted(item); success) {
        RTP_LLM_LOG_DEBUG("write cache, cache key: %ld, backing: %d, block index: %d, disk slot: %d, block size: %zu",
                          item.cache_key,
                          static_cast<int>(item.backing_type),
                          item.block_index,
                          item.disk_slot,
                          item.block_size);
        referenceCacheBacking(item);
        if (popped_item_opt.has_value()) {
            const auto popped_item = popped_item_opt.value();
            releaseCacheBacking(popped_item);
        }
    }
}

int64_t KVCacheMemoryConnector::copyPlanTimeoutMs(const std::shared_ptr<CopyPlan>& copy_plan) const {
    bool has_disk_item = false;
    for (const auto& copy_info : copy_plan->copy_infos) {
        if (copy_info.backing_type == CacheBackingType::DISK) {
            has_disk_item = true;
            break;
        }
    }
    if (!has_disk_item) {
        return kv_cache_config_.memory_cache_sync_timeout_ms;
    }
    return std::max(kv_cache_config_.memory_cache_sync_timeout_ms, kv_cache_config_.memory_cache_disk_sync_timeout_ms);
}

std::vector<CacheKeyType> KVCacheMemoryConnector::cacheKeys() const {
    RTP_LLM_CHECK_WITH_INFO(block_cache_ != nullptr, "block cache should not be null");
    return block_cache_->cacheKeys();
}

void KVCacheMemoryConnector::reportMatchMetrics(bool    success,
                                                int64_t latency_us,
                                                int64_t input_block_num,
                                                int64_t matched_block_num) {
    if (!metrics_reporter_) {
        return;
    }

    RtpLLMMemoryCacheMatchMetricsCollector collector;
    collector.failed        = !success;
    collector.latency_us    = latency_us;
    collector.input_token   = input_block_num * cache_config_.seq_size_per_block;
    collector.matched_token = matched_block_num * cache_config_.seq_size_per_block;

    metrics_reporter_->report<RtpLLMMemoryCacheMetrics, RtpLLMMemoryCacheMatchMetricsCollector>(nullptr, &collector);
}

void KVCacheMemoryConnector::reportReadMetrics(bool    success,
                                               int64_t latency_us,
                                               int64_t input_block_num,
                                               int64_t read_block_num) {
    if (!metrics_reporter_) {
        return;
    }

    RtpLLMMemoryCacheReadMetricsCollector collector;
    collector.failed      = !success;
    collector.latency_us  = latency_us;
    collector.input_token = input_block_num * cache_config_.seq_size_per_block;
    collector.read_token  = read_block_num * cache_config_.seq_size_per_block;

    metrics_reporter_->report<RtpLLMMemoryCacheMetrics, RtpLLMMemoryCacheReadMetricsCollector>(nullptr, &collector);
}

void KVCacheMemoryConnector::reportWriteMetrics(bool    success,
                                                int64_t latency_us,
                                                int64_t input_block_num,
                                                int64_t write_block_num) {
    if (!metrics_reporter_) {
        return;
    }

    RtpLLMMemoryCacheWriteMetricsCollector collector;
    collector.failed      = !success;
    collector.latency_us  = latency_us;
    collector.input_token = input_block_num * cache_config_.seq_size_per_block;
    collector.write_token = write_block_num * cache_config_.seq_size_per_block;

    metrics_reporter_->report<RtpLLMMemoryCacheMetrics, RtpLLMMemoryCacheWriteMetricsCollector>(nullptr, &collector);
}

void KVCacheMemoryConnector::reportCopyMetrics(bool success, int64_t latency_us, CopyDirection direction) {
    if (!metrics_reporter_) {
        return;
    }

    RtpLLMMemoryCacheCopyMetricsCollector collector;
    collector.failed     = !success;
    collector.latency_us = latency_us;
    collector.from_gpu   = direction == CopyDirection::D2H;

    metrics_reporter_->report<RtpLLMMemoryCacheMetrics, RtpLLMMemoryCacheCopyMetricsCollector>(nullptr, &collector);
}

void KVCacheMemoryConnector::reportDiskMatchMetrics(bool    success,
                                                    int64_t latency_us,
                                                    int64_t input_block_num,
                                                    int64_t matched_block_num) {
    if (!metrics_reporter_ || !diskCacheEnabled()) {
        return;
    }
    RtpLLMDiskCacheMatchMetricsCollector collector;
    collector.failed        = !success;
    collector.latency_us    = latency_us;
    collector.input_token   = input_block_num * cache_config_.seq_size_per_block;
    collector.matched_token = matched_block_num * cache_config_.seq_size_per_block;
    metrics_reporter_->report<RtpLLMDiskCacheMetrics, RtpLLMDiskCacheMatchMetricsCollector>(nullptr, &collector);
}

void KVCacheMemoryConnector::reportDiskReadMetrics(bool    success,
                                                   int64_t latency_us,
                                                   int64_t input_block_num,
                                                   int64_t read_block_num) {
    if (!metrics_reporter_ || !diskCacheEnabled()) {
        return;
    }
    RtpLLMDiskCacheReadMetricsCollector collector;
    collector.failed      = !success;
    collector.latency_us  = latency_us;
    collector.input_token = input_block_num * cache_config_.seq_size_per_block;
    collector.read_token  = read_block_num * cache_config_.seq_size_per_block;
    metrics_reporter_->report<RtpLLMDiskCacheMetrics, RtpLLMDiskCacheReadMetricsCollector>(nullptr, &collector);
}

void KVCacheMemoryConnector::reportDiskWriteMetrics(bool    success,
                                                    int64_t latency_us,
                                                    int64_t input_block_num,
                                                    int64_t write_block_num) {
    if (!metrics_reporter_ || !diskCacheEnabled()) {
        return;
    }
    RtpLLMDiskCacheWriteMetricsCollector collector;
    collector.failed      = !success;
    collector.latency_us  = latency_us;
    collector.input_token = input_block_num * cache_config_.seq_size_per_block;
    collector.write_token = write_block_num * cache_config_.seq_size_per_block;
    metrics_reporter_->report<RtpLLMDiskCacheMetrics, RtpLLMDiskCacheWriteMetricsCollector>(nullptr, &collector);
}

void KVCacheMemoryConnector::reportDiskCopyMetrics(bool success, int64_t latency_us, CopyDirection direction) {
    if (!metrics_reporter_ || !diskCacheEnabled()) {
        return;
    }
    RtpLLMDiskCacheCopyMetricsCollector collector;
    collector.failed     = !success;
    collector.latency_us = latency_us;
    collector.from_gpu   = direction == CopyDirection::D2H;
    metrics_reporter_->report<RtpLLMDiskCacheMetrics, RtpLLMDiskCacheCopyMetricsCollector>(nullptr, &collector);
}

void KVCacheMemoryConnector::reportMetricsLoop() {
    size_t last_disk_read_bytes  = 0;
    size_t last_disk_write_bytes = 0;
    while (!stop_.load()) {
        if (metrics_reporter_) {
            if (!block_pool_) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            const auto total_blocks     = block_pool_->totalBlocksNum();
            const auto free_blocks      = block_pool_->freeBlocksNum();
            const auto available_blocks = block_pool_->availableBlocksNum();

            RtpLLMMemoryCacheStatusMetricsCollector collector;
            collector.total_block_num     = total_blocks;
            collector.allocated_block_num = total_blocks - free_blocks;
            collector.available_block_num = available_blocks;

            metrics_reporter_->report<RtpLLMMemoryCacheMetrics, RtpLLMMemoryCacheStatusMetricsCollector>(nullptr,
                                                                                                         &collector);
            if (disk_block_pool_) {
                const auto total_disk_slots     = disk_block_pool_->totalSlots();
                const auto free_disk_slots      = disk_block_pool_->freeSlots();
                const auto available_disk_slots = disk_block_pool_->availableSlots();
                const auto read_bytes           = disk_block_pool_->readBytes();
                const auto write_bytes          = disk_block_pool_->writeBytes();

                RtpLLMDiskCacheStatusMetricsCollector disk_collector;
                disk_collector.total_block_num     = total_disk_slots;
                disk_collector.allocated_block_num = total_disk_slots - free_disk_slots;
                disk_collector.available_block_num = available_disk_slots;
                disk_collector.in_flight_block_num = static_cast<int64_t>(total_disk_slots - available_disk_slots);
                disk_collector.read_bytes          = read_bytes;
                disk_collector.write_bytes         = write_bytes;
                disk_collector.read_bandwidth =
                    read_bytes >= last_disk_read_bytes ? read_bytes - last_disk_read_bytes : 0;
                disk_collector.write_bandwidth =
                    write_bytes >= last_disk_write_bytes ? write_bytes - last_disk_write_bytes : 0;
                last_disk_read_bytes  = read_bytes;
                last_disk_write_bytes = write_bytes;

                metrics_reporter_->report<RtpLLMDiskCacheMetrics, RtpLLMDiskCacheStatusMetricsCollector>(
                    nullptr, &disk_collector);
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace rtp_llm
