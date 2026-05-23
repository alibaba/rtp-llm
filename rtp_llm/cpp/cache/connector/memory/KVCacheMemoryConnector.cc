#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"

#include <algorithm>
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
                                               const std::shared_ptr<KVCacheAllocator>& allocator,
                                               const std::vector<std::string>&          tp_addrs,
                                               const kmonitor::MetricsReporterPtr&      metrics_reporter,
                                               int                                      tp_rank,
                                               int                                      tp_size):
    cache_config_(cache_config),
    kv_cache_config_(kv_cache_config),
    allocator_(allocator),
    tp_addrs_(tp_addrs),
    tp_rank_(tp_rank),
    tp_size_(tp_size),
    metrics_reporter_(metrics_reporter) {}

KVCacheMemoryConnector::~KVCacheMemoryConnector() {
    RTP_LLM_LOG_INFO("KVCacheMemoryConnector destructor");
    stop_.store(true);
    if (metrics_reporter_thread_) {
        metrics_reporter_thread_->join();
        metrics_reporter_thread_.reset();
    }
    if (commit_coordinator_) {
        commit_coordinator_->stop();
        commit_coordinator_.reset();
    }
    if (wait_done_thread_pool_) {
        wait_done_thread_pool_->stop();
        wait_done_thread_pool_.reset();
    }
    broadcast_manager_.reset();
    if (disk_spill_cache_) {
        disk_spill_cache_->shutdown();
        disk_spill_cache_.reset();
    }
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

    checkLayerBlockStrideBytes();

    initBlockPool();
    block_cache_ = std::make_shared<MemoryBlockCache>();

    broadcast_manager_ = std::make_shared<BroadcastManager>(tp_addrs_);
    RTP_LLM_CHECK_WITH_INFO(broadcast_manager_->init(), "init failed, broadcast manager init failed");

    initDiskSpillCache();

    wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(8, 1000, nullptr, "WaitDoneThreadPool");
    RTP_LLM_CHECK_WITH_INFO(wait_done_thread_pool_->start(), "init failed, wait done thread pool start failed");

    if (metrics_reporter_) {
        metrics_reporter_thread_ = std::make_shared<std::thread>([this]() { reportMetricsLoop(); });
    }
    // NOTE: disk spill capability handshake is deferred to postInit() which the
    // coordinator MUST call after memory_connector_ assignment completes. We
    // can't broadcast here because executeFunction's RTP_LLM_CHECK on the
    // receiving rank dereferences memory_connector_ — which is still null on
    // every rank until each rank's coordinator finishes initMemoryConnector().
    return true;
}

bool KVCacheMemoryConnector::postInit() {
    if (!kv_cache_config_.enable_memory_cache_disk_spill || !isMaster()
        || !broadcast_manager_ || broadcast_manager_->workerNum() == 0) {
        return true;
    }
    return runDiskSpillHandshake();
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

    const auto slots = layerRegionSlots();

    // block_size here means "one cache-key across all layer+attn slots" total bytes (kv + scale).
    // Use per-slot strides so NULL_BLOCK_IDX entries still occupy stable offsets in the merged layout.
    size_t block_size = 0;
    for (const auto& slot : slots) {
        block_size += slot.stride_bytes;
    }
    RTP_LLM_CHECK_WITH_INFO(block_size > 0, "block size is invalid: %zu", block_size);

    block_pool_ = createBlockPool(block_size, memory_cache_size_mb);
    RTP_LLM_CHECK_WITH_INFO(block_pool_ != nullptr, "init block pool failed, create block pool failed");
}

void KVCacheMemoryConnector::initDiskSpillCache() {
    if (!kv_cache_config_.enable_memory_cache_disk_spill) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(!kv_cache_config_.memory_cache_disk_paths.empty(),
                            "init disk spill failed, MEMORY_CACHE_DISK_PATHS is empty");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config_.memory_cache_disk_init_timeout_ms > 0,
                            "init disk spill failed, init timeout is invalid: %ld",
                            kv_cache_config_.memory_cache_disk_init_timeout_ms);
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config_.memory_cache_disk_stage_ack_timeout_ms > 0,
                            "init disk spill failed, stage ack timeout is invalid: %ld",
                            kv_cache_config_.memory_cache_disk_stage_ack_timeout_ms);

    schema_hash_ = computeSchemaHash();

    DiskSpillBlockCache::InitConfig config;
    config.disks = DiskSpillBlockCache::parseDiskConfigs(kv_cache_config_.memory_cache_disk_paths);
    config.block_size                    = memoryBlockSizeBytes();
    config.align_bytes                   = static_cast<size_t>(kv_cache_config_.memory_cache_disk_align_bytes);
    if (config.align_bytes == 0) {
        config.align_bytes = 4096;
    }
    config.segment_bytes                 = static_cast<size_t>(kv_cache_config_.memory_cache_disk_segment_mb)
                                           * 1024UL * 1024UL;
    config.direct_io                     = kv_cache_config_.memory_cache_disk_direct_io;
    config.direct_io_required            = kv_cache_config_.memory_cache_disk_direct_io_required;
    config.io_threads_per_disk           = kv_cache_config_.memory_cache_disk_io_threads_per_disk;
    config.io_queue_size                 = kv_cache_config_.memory_cache_disk_queue_size_per_disk;
    config.max_staging_buffers_per_disk  = kv_cache_config_.memory_cache_disk_max_staging_buffers_per_disk;
    config.cleanup_on_destroy            = true;
    config.cleanup_old_startup_dirs      = true;
    config.schema_hash                   = schema_hash_;
    config.world_rank                    = tp_rank_;
    disk_spill_cache_                    = DiskSpillBlockCache::create(std::move(config));
    disk_spill_cache_->setMasterMode(isMaster());
    RTP_LLM_CHECK_WITH_INFO(disk_spill_cache_->init(), "init disk spill failed, disk cache init failed");

    if (isMaster()) {
        DiskSpillCommitCoordinator::Config cccfg;
        cccfg.stage_ack_timeout_ms   = kv_cache_config_.memory_cache_disk_stage_ack_timeout_ms;
        cccfg.commit_timeout_ms      = kv_cache_config_.memory_cache_disk_spill_commit_timeout_ms;
        cccfg.poll_interval_ms       = 50;
        const int worker_count = static_cast<int>(broadcast_manager_ ? broadcast_manager_->workerNum() : 0);
        commit_coordinator_          = std::make_shared<DiskSpillCommitCoordinator>(
            disk_spill_cache_,
            cccfg,
            worker_count,
            [this](SpillJobId id, const DiskSpillBlockCache::DiskItem& slot) {
                return broadcastSpillToWorkers(id, slot);
            },
            [this](const DiskSpillBlockCache::DiskItem& slot) { return broadcastDeleteToWorkers(slot); },
            [this](int worker_idx, SpillJobId id) { return pollWorkerSpillStatus(worker_idx, id); });
        RTP_LLM_CHECK_WITH_INFO(commit_coordinator_->start(), "init disk spill failed, commit coordinator start failed");
    }
}

size_t KVCacheMemoryConnector::memoryBlockSizeBytes() const {
    const auto slots      = layerRegionSlots();
    size_t     block_size = 0;
    for (const auto& slot : slots) {
        block_size += slot.stride_bytes;
    }
    return block_size;
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

    const auto  slots                = layerRegionSlots();
    const auto& layer_attn_block_ids = resource->layerAttnBlocks();
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
    size_t       matched_num                    = already_reuse_num;
    size_t       inspected_num                  = already_reuse_num;
    CacheKeyType first_unmatched_key            = 0;
    const char*  first_unmatched_reason         = "none";
    bool         first_block_recorded           = false;
    size_t       matched_entries                = 0;
    size_t       complete_entries               = 0;
    size_t       gpu_valid_entries              = 0;
    size_t       complete_gpu_valid_entries     = 0;
    int          first_complete_index           = -1;
    int          first_complete_gpu_valid_index = -1;
    int          first_not_found_index          = -1;
    for (size_t i = already_reuse_num; i < cache_keys_size; ++i) {
        const auto cache_key           = static_cast<CacheKeyType>(cache_keys.at(i));
        bool       matched             = false;
        bool       matched_is_complete = false;
        bool       matched_in_memory   = false;
        bool       matched_on_disk     = false;
        const auto memory_match_result = block_cache_->match(cache_key);
        if (!isNullBlockIdx(memory_match_result.matched_index)) {
            matched             = true;
            matched_is_complete = memory_match_result.is_complete;
            matched_in_memory   = true;
        } else if (disk_spill_cache_) {
            const auto disk_match_result = disk_spill_cache_->match(cache_key);
            if (disk_match_result.matched) {
                matched             = true;
                matched_is_complete = disk_match_result.is_complete;
                matched_on_disk     = true;
            }
        }
        if (!matched) {
            inspected_num          = i;
            first_unmatched_key    = cache_key;
            first_unmatched_reason = "not_found";
            first_not_found_index  = static_cast<int>(i);
            break;
        }
        const bool gpu_blocks_valid = gpuBlocksAllValid(layer_attn_block_ids, slots, i);
        ++matched_entries;
        if (matched_is_complete) {
            ++complete_entries;
            if (first_complete_index < 0) {
                first_complete_index = static_cast<int>(i);
            }
        }
        if (gpu_blocks_valid) {
            ++gpu_valid_entries;
        }
        if (matched_is_complete && gpu_blocks_valid) {
            ++complete_gpu_valid_entries;
            if (first_complete_gpu_valid_index < 0) {
                first_complete_gpu_valid_index = static_cast<int>(i);
            }
        }
        if (matched_is_complete && gpu_blocks_valid) {
            matched_num = i + 1;
        } else if (matched_num <= already_reuse_num && !first_block_recorded) {
            inspected_num          = i;
            first_unmatched_key    = cache_key;
            first_unmatched_reason = matched_is_complete ? "gpu_blocks_invalid" : "incomplete";
            first_block_recorded   = true;
            if (disk_spill_cache_) {
                RTP_LLM_LOG_INFO("memory disk cache match blocked, index=%zu cache_key=%ld source=%s complete=%d "
                                 "gpu_blocks_valid=%d already_reuse=%zu cache_keys=%zu",
                                 i,
                                 cache_key,
                                 matched_in_memory ? "memory" : (matched_on_disk ? "disk" : "unknown"),
                                 matched_is_complete,
                                 gpu_blocks_valid,
                                 already_reuse_num,
                                 cache_keys_size);
            }
        }
    }

    if (matched_num <= already_reuse_num) {
        if (disk_spill_cache_) {
            RTP_LLM_LOG_INFO("not matched cache in memory/disk, cache_keys=%zu already_reuse=%zu inspected=%zu "
                             "cache_key=%ld reason=%s matched_entries=%zu complete=%zu gpu_valid=%zu "
                             "complete_gpu_valid=%zu first_complete=%d first_complete_gpu_valid=%d first_not_found=%d",
                             cache_keys_size,
                             already_reuse_num,
                             inspected_num,
                             first_unmatched_key,
                             first_unmatched_reason,
                             matched_entries,
                             complete_entries,
                             gpu_valid_entries,
                             complete_gpu_valid_entries,
                             first_complete_index,
                             first_complete_gpu_valid_index,
                             first_not_found_index);
        } else {
            RTP_LLM_LOG_DEBUG("not matched cache in memory, cache keys size: %zu, already_reuse_num: %zu",
                              cache_keys_size,
                              already_reuse_num);
        }
        reportMatchMetrics(/*success=*/false, timer.done_us(), cache_keys_size, matched_num);
        return nullptr;
    }
    RTP_LLM_LOG_INFO("memory cache matched blocks: already_reuse=%zu matched=%zu cache_keys=%zu "
                     "matched_entries=%zu complete=%zu gpu_valid=%zu complete_gpu_valid=%zu "
                     "first_complete=%d first_complete_gpu_valid=%d first_not_found=%d",
                     already_reuse_num,
                     matched_num,
                     cache_keys_size,
                     matched_entries,
                     complete_entries,
                     gpu_valid_entries,
                     complete_gpu_valid_entries,
                     first_complete_index,
                     first_complete_gpu_valid_index,
                     first_not_found_index);
    reportMatchMetrics(/*success=*/true, timer.done_us(), cache_keys_size, matched_num);
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

    const auto  slots                = layerRegionSlots();
    const auto& layer_attn_block_ids = resource->layerAttnBlocks();
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
        if (success) {
            resource->setMemoryReuseBlockNum(read_block_num);
            for (const auto& copy_info : copy_plan->copy_infos) {
                if (copy_info.source_type == CopyInfoPerKey::SourceType::DISK_SLOT) {
                    if (disk_spill_cache_) {
                        disk_spill_cache_->releaseTakenSlot(copy_info.disk_slot);
                    }
                    continue;
                }
                const auto removed_item = block_cache_->removeIfMatch(copy_info.cache_key, copy_info.mem_block);
                if (!removed_item.has_value()) {
                    continue;
                }
                freeBlocks({removed_item->block_index}, /*cache_free=*/true);
            }
            RTP_LLM_LOG_INFO("memory cache read success: read_blocks=%d released_blocks=%zu total_blocks=%zu",
                             read_block_num,
                             copy_plan->copy_infos.size(),
                             total_block_num);
        } else {
            for (const auto& copy_info : copy_plan->copy_infos) {
                if (copy_info.source_type == CopyInfoPerKey::SourceType::DISK_SLOT && disk_spill_cache_) {
                    disk_spill_cache_->releaseTakenSlot(copy_info.disk_slot);
                }
            }
        }
        // reset ptr to release memory block refs
        copy_plan.reset();
        reportReadMetrics(success, timer.done_us(), total_block_num, read_block_num);
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
        const auto                    cache_key           = static_cast<CacheKeyType>(cache_keys.at(i));
        const auto                    memory_match_result = block_cache_->match(cache_key);
        bool                          source_found        = false;
        bool                          source_is_complete  = false;
        BlockIdxType                  source_memory_block = NULL_BLOCK_IDX;
        DiskSpillBlockCache::DiskSlot source_disk_slot;
        CopyInfoPerKey::SourceType    source_type = CopyInfoPerKey::SourceType::MEMORY_BLOCK;

        if (!isNullBlockIdx(memory_match_result.matched_index)) {
            source_found        = true;
            source_is_complete  = memory_match_result.is_complete;
            source_memory_block = memory_match_result.matched_index;
            source_type         = CopyInfoPerKey::SourceType::MEMORY_BLOCK;
            // 每次都加引用的原因是为了确保match到的block不会被释放(避免在写时malloc如果cache满弹出该block)
            referenceBlocks({source_memory_block}, /*cache_ref=*/false);
        } else if (disk_spill_cache_) {
            auto disk_slot = disk_spill_cache_->takeForReadRaw(cache_key);
            if (disk_slot.has_value()) {
                source_found       = true;
                source_is_complete = disk_slot->is_complete;
                source_disk_slot   = *disk_slot;
                source_type        = CopyInfoPerKey::SourceType::DISK_SLOT;
            }
        }

        if (!source_found) {
            RTP_LLM_LOG_WARNING("build copy plan for read failed, cache key not found, cache key: %ld", cache_key);
            success = false;
            break;
        }

        CopyInfoPerKey copy_info;
        copy_info.cache_key   = cache_key;
        copy_info.mem_block   = source_memory_block;
        copy_info.source_type = source_type;
        copy_info.disk_slot   = source_disk_slot;
        copy_info.gpu_blocks.reserve(slots.size());
        for (const auto& slot : slots) {
            // Do NOT skip NULL_BLOCK_IDX here. The merged memory block layout requires reserving
            // per-layer+attn stride even when this slot has no gpu block (-1).
            const auto layer = static_cast<size_t>(slot.layer_id);
            const auto attn  = static_cast<size_t>(slot.region_name);
            copy_info.gpu_blocks.push_back(layer_attn_block_ids.at(layer).at(attn)->blocks().at(i));
        }
        copy_info.is_complete = source_is_complete;
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
    if (!success) {
        for (const auto& copy_info : copy_infos) {
            if (copy_info.source_type == CopyInfoPerKey::SourceType::DISK_SLOT && disk_spill_cache_) {
                disk_spill_cache_->releaseTakenSlot(copy_info.disk_slot);
            }
        }
        return nullptr;
    }
    return plan;
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

    const auto  slots                = layerRegionSlots();
    const auto& layer_attn_block_ids = resource->layerAttnBlocks();
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
    if (disk_spill_cache_) {
        RTP_LLM_LOG_INFO("memory disk cache write plan, mem_matched=%zu cache_keys=%zu write_blocks=%zu "
                         "first_cache_key=%ld last_cache_key=%ld",
                         mem_matched_num,
                         cache_keys_size,
                         copy_plan->copy_infos.size(),
                         copy_plan->copy_infos.front().cache_key,
                         copy_plan->copy_infos.back().cache_key);
    }

    auto write_done =
        [copy_plan, resource_copy = resource, timer, total_block_num = cache_keys_size, this](bool success) mutable {
            RTP_LLM_LOG_DEBUG("async write done, success: %d", success);

            if (success) {
                for (const auto& copy_info : copy_plan->copy_infos) {
                    MemoryBlockCache::CacheItem item;
                    item.cache_key   = copy_info.cache_key;
                    item.block_index = copy_info.mem_block;
                    item.is_resident = false;
                    item.is_complete = copy_info.is_complete;
                    putToCache(item);
                }
            }
            // reset resource to decrease block ref count in destructor
            resource_copy.reset();
            const int64_t write_block_num = success ? static_cast<int64_t>(copy_plan->copy_infos.size()) : 0;
            // reset copy plan to release memory block refs
            copy_plan.reset();
            reportWriteMetrics(success, timer.done_us(), total_block_num, write_block_num);
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
    int    last_complete_index  = -1;  // cache_key index in [start_index, start_index + write_num)
    int    first_complete_index = -1;
    size_t complete_count       = 0;

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
            if (first_complete_index < 0) {
                first_complete_index = i;
            }
            last_complete_index = i;
            ++complete_count;
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
    if (disk_spill_cache_) {
        RTP_LLM_LOG_INFO("memory disk cache write completeness, start=%d write_num=%d complete=%zu "
                         "first_complete=%d last_complete=%d",
                         start_index,
                         write_num,
                         complete_count,
                         first_complete_index,
                         last_complete_index);
    }
    if (no_need_write) {
        return nullptr;
    }

    // drop keys behind the last complete key
    const size_t keep_cnt = static_cast<size_t>(last_complete_index - start_index + 1);
    if (disk_spill_cache_) {
        RTP_LLM_LOG_INFO("memory disk cache write keep, start=%d keep=%zu first_key=%ld last_key=%ld",
                         start_index,
                         keep_cnt,
                         static_cast<CacheKeyType>(cache_keys.at(start_index)),
                         static_cast<CacheKeyType>(cache_keys.at(last_complete_index)));
    }
    copy_infos.resize(keep_cnt);

    std::vector<BlockIdxType> mem_blocks;
    if (!mallocBlocks(copy_infos.size(), mem_blocks)) {
        RTP_LLM_LOG_WARNING("build copy plan for write failed, malloc blocks failed, need blocks: %zu",
                            copy_infos.size());
        return nullptr;
    }
    for (size_t i = 0; i < copy_infos.size(); ++i) {
        copy_infos[i].mem_block = mem_blocks[i];
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
        std::vector<BlockIdxType> blocks;
        blocks.reserve(plan->copy_infos.size());
        for (const auto& copy_info : plan->copy_infos) {
            blocks.push_back(copy_info.mem_block);
        }
        freeBlocks(blocks, /*cache_free=*/false);
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
    bool                     has_disk_source = false;
    for (const auto& copy_info : copy_plan->copy_infos) {
        has_disk_source = has_disk_source || copy_info.source_type == CopyInfoPerKey::SourceType::DISK_SLOT;
    }
    if (has_disk_source) {
        mem_req.set_memory_op_protocol_version(1);
        mem_req.set_op_type(MemoryOperationRequestPB::H2D_MIXED_MEMORY_DISK);
    }
    mem_req.set_copy_direction(copy_plan->direction == CopyDirection::H2D ? MemoryOperationRequestPB::H2D :
                                                                            MemoryOperationRequestPB::D2H);
    for (const auto& copy_info : copy_plan->copy_infos) {
        auto* item = mem_req.add_copy_items();
        item->set_cache_key(copy_info.cache_key);
        item->set_mem_block(copy_info.mem_block);
        item->set_source_type(copy_info.source_type == CopyInfoPerKey::SourceType::DISK_SLOT ?
                                  MemoryOperationRequestPB::DISK_SLOT :
                                  MemoryOperationRequestPB::MEMORY_BLOCK);
        if (copy_info.source_type == CopyInfoPerKey::SourceType::DISK_SLOT) {
            item->set_disk_id(copy_info.disk_slot.disk_id);
            item->set_disk_slot_id(copy_info.disk_slot.slot_id);
            item->set_generation(copy_info.disk_slot.gen.slot_gen);
            item->set_key_generation(copy_info.disk_slot.gen.key_gen);
            item->set_logical_bytes(copy_info.disk_slot.block_size);
            item->set_is_complete(copy_info.disk_slot.is_complete);
        }
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
    const auto timeout_ms = has_disk_source ? std::max(kv_cache_config_.memory_cache_sync_timeout_ms,
                                                       kv_cache_config_.memory_cache_disk_stage_ack_timeout_ms) :
                                              kv_cache_config_.memory_cache_sync_timeout_ms;
    return broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, timeout_ms, rpc_call);
}

void KVCacheMemoryConnector::printCopyPlan(const std::shared_ptr<CopyPlan>& copy_plan) const {
    std::ostringstream oss;
    oss << "copy plan direction: " << (copy_plan->direction == CopyDirection::H2D ? "H2D" : "D2H")
        << ", copy infos size: " << copy_plan->copy_infos.size() << "\n";
    for (int i = 0; i < copy_plan->copy_infos.size(); ++i) {
        const auto& copy_info = copy_plan->copy_infos.at(i);
        oss << "copy info " << i << ": cache key: " << copy_info.cache_key << ", mem block: " << copy_info.mem_block
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
    if (request.memory_op_protocol_version() != 0) {
        if (request.op_type() == MemoryOperationRequestPB::UNSPECIFIED) {
            response.set_success(false);
            response.set_error_type(disk_error::kProtocolViolation);
            reportCopyMetrics(false, timer.done_us(), CopyDirection::H2D);
            reportDiskError(disk_error::kProtocolViolation, "copyCache");
            return false;
        }
        // op_sequence FIFO check (per peer rank). We use schema_hash bucket as a
        // simple peer-id proxy when source rank id isn't available.
        if (request.op_sequence() > 0) {
            const int  peer_id = static_cast<int>(std::hash<std::string>{}(request.schema_hash()) & 0x7FFF);
            auto&      tracker = trackerForIncomingRank(peer_id);
            const auto chk     = tracker.checkReceived(request.op_sequence());
            if (chk != OpSequenceTracker::CheckResult::OK) {
                response.set_success(false);
                response.set_error_type(disk_error::kProtocolViolation);
                reportDiskError(disk_error::kProtocolViolation, "op_sequence");
                return false;
            }
        }
        const auto slots = layerRegionSlots();
        switch (request.op_type()) {
            case MemoryOperationRequestPB::DISK_SPILL_HELLO: {
                handleDiskSpillHello(request, response);
                return response.success();
            }
            case MemoryOperationRequestPB::SPILL_WRITE_STATUS: {
                handleSpillWriteStatus(request, response);
                return response.success();
            }
            case MemoryOperationRequestPB::H2D_MIXED_MEMORY_DISK: {
                const bool success = copyMixedMemoryDiskToDevice(request, slots);
                response.set_success(success);
                if (!success) {
                    response.set_error_type(disk_error::kH2d);
                    reportDiskError(disk_error::kH2d, "mixed_read");
                }
                reportCopyMetrics(success, timer.done_us(), CopyDirection::H2D);
                reportDiskCopyMetrics(success, timer.done_us(), "DISK_TO_GPU", -1);
                return success;
            }
            case MemoryOperationRequestPB::SPILL_MEMORY_TO_DISK: {
                const bool success = spillMemoryToDisk(request);
                response.set_success(success);
                if (!success) {
                    response.set_error_type(disk_error::kRankWrite);
                    reportDiskError(disk_error::kRankWrite, "spill_worker");
                }
                reportCopyMetrics(success, timer.done_us(), CopyDirection::D2H);
                reportDiskCopyMetrics(success, timer.done_us(), "MEMORY_TO_DISK", -1);
                return success;
            }
            case MemoryOperationRequestPB::DELETE_DISK_SLOT: {
                const bool success = deleteDiskSlots(request);
                response.set_success(success);
                if (!success) {
                    response.set_error_type(disk_error::kDeleteSlot);
                    reportDiskError(disk_error::kDeleteSlot, "delete");
                }
                reportCopyMetrics(success, timer.done_us(), CopyDirection::H2D);
                reportDiskCopyMetrics(success, timer.done_us(), "DELETE_DISK_SLOT", -1);
                return success;
            }
            case MemoryOperationRequestPB::LEGACY_MEMORY_COPY:
            case MemoryOperationRequestPB::H2D_MEMORY:
            case MemoryOperationRequestPB::D2H_MEMORY:
                break;
            default:
                response.set_success(false);
                response.set_error_type("unsupported_op_type");
                reportCopyMetrics(false, timer.done_us(), CopyDirection::H2D);
                reportDiskError("unsupported_op_type", "copyCache");
                return false;
        }
    }
    const auto copy_direction =
        (request.copy_direction() == MemoryOperationRequestPB::H2D) ? CopyDirection::H2D : CopyDirection::D2H;
    const auto slots           = layerRegionSlots();
    const bool has_typed_slots = hasTypedLayerRegionSlots(slots);

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

    std::vector<torch::Tensor> dst_buffers;
    std::vector<torch::Tensor> src_buffers;
    for (int i = 0; i < request.copy_items_size(); ++i) {
        const auto&                     item      = request.copy_items(i);
        const auto                      mem_block = static_cast<BlockIdxType>(item.mem_block());
        const std::vector<BlockIdxType> gpu_blocks(item.gpu_blocks().begin(), item.gpu_blocks().end());

        if (!prepareCopyBuffers(mem_block, gpu_blocks, copy_direction, dst_buffers, src_buffers)) {
            RTP_LLM_LOG_WARNING("copy cache failed, prepare copy buffers failed, mem_block=%d, direction=%s",
                                mem_block,
                                copy_direction == CopyDirection::H2D ? "H2D" : "D2H");
            response.set_success(false);
            reportCopyMetrics(false, timer.done_us(), copy_direction);
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

    response.set_success(true);
    reportCopyMetrics(true, timer.done_us(), copy_direction);
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

bool KVCacheMemoryConnector::copyMixedMemoryDiskToDevice(const MemoryOperationRequestPB&     request,
                                                         const std::vector<LayerRegionSlot>& slots) {
    if (request.copy_direction() != MemoryOperationRequestPB::H2D) {
        return false;
    }
    if (block_pool_ == nullptr || allocator_ == nullptr) {
        return false;
    }

    std::vector<std::vector<char>> disk_buffers;
    disk_buffers.reserve(request.copy_items_size());
    std::vector<torch::Tensor> dst_buffers;
    std::vector<torch::Tensor> src_buffers;

    for (int i = 0; i < request.copy_items_size(); ++i) {
        const auto&                     item = request.copy_items(i);
        const std::vector<BlockIdxType> gpu_blocks(item.gpu_blocks().begin(), item.gpu_blocks().end());
        if (gpu_blocks.size() != slots.size()) {
            return false;
        }

        BlockInfo source_block;
        if (item.source_type() == MemoryOperationRequestPB::DISK_SLOT) {
            if (!disk_spill_cache_) {
                return false;
            }
            const size_t logical_bytes =
                item.logical_bytes() == 0 ? memoryBlockSizeBytes() : static_cast<size_t>(item.logical_bytes());
            disk_buffers.emplace_back(logical_bytes, 0);
            DiskSpillBlockCache::DiskSlot disk_slot;
            disk_slot.cache_key   = static_cast<CacheKeyType>(item.cache_key());
            disk_slot.disk_id     = item.disk_id();
            disk_slot.slot_id     = item.disk_slot_id();
            disk_slot.gen.slot_gen = item.generation();
            disk_slot.gen.key_gen  = item.key_generation();
            disk_slot.block_size  = logical_bytes;
            disk_slot.is_complete = item.is_complete();
            if (!disk_spill_cache_->readTaken(disk_slot, disk_buffers.back().data(), logical_bytes)) {
                return false;
            }
            source_block.addr       = disk_buffers.back().data();
            source_block.size_bytes = logical_bytes;
            source_block.is_cuda    = false;
        } else {
            const auto mem_block = static_cast<BlockIdxType>(item.mem_block());
            if (isNullBlockIdx(mem_block)) {
                return false;
            }
            auto mem_buffers = block_pool_->convertIndexToBuffer(/*layer_id=*/0, mem_block);
            if (mem_buffers.size() != 1u || mem_buffers[0].addr == nullptr || mem_buffers[0].size_bytes == 0) {
                return false;
            }
            source_block = mem_buffers[0];
        }

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
                    return false;
                }
                if (!appendCopyBytesToBuffers(source_block,
                                              gpu_buffer,
                                              byte_off + within_layer_off,
                                              CopyDirection::H2D,
                                              dst_buffers,
                                              src_buffers)) {
                    return false;
                }
                within_layer_off += gpu_buffer.size_bytes;
            }
            byte_off += layer_stride;
        }
    }

    if (!dst_buffers.empty()) {
        MultiCopyParams mc{dst_buffers, src_buffers};
        execNoBlockCopy(mc);
    }
    size_t disk_source_count = 0;
    for (int i = 0; i < request.copy_items_size(); ++i) {
        if (request.copy_items(i).source_type() == MemoryOperationRequestPB::DISK_SLOT) {
            ++disk_source_count;
        }
    }
    if (disk_source_count > 0) {
        RTP_LLM_LOG_INFO("memory disk cache mixed read success, copy_items=%d disk_sources=%zu",
                         request.copy_items_size(),
                         disk_source_count);
    }
    return true;
}

bool KVCacheMemoryConnector::spillMemoryToDisk(const MemoryOperationRequestPB& request) {
    if (!disk_spill_cache_ || !block_pool_) {
        return false;
    }
    for (int i = 0; i < request.copy_items_size(); ++i) {
        const auto& item      = request.copy_items(i);
        const auto  mem_block = static_cast<BlockIdxType>(item.mem_block());
        if (isNullBlockIdx(mem_block)) {
            return false;
        }
        auto mem_buffers = block_pool_->convertIndexToBuffer(/*layer_id=*/0, mem_block);
        if (mem_buffers.size() != 1u || mem_buffers[0].addr == nullptr || mem_buffers[0].size_bytes == 0) {
            return false;
        }
        DiskSpillBlockCache::DiskSlot disk_slot;
        disk_slot.cache_key    = static_cast<CacheKeyType>(item.cache_key());
        disk_slot.disk_id      = item.disk_id();
        disk_slot.slot_id      = item.disk_slot_id();
        disk_slot.gen.slot_gen = item.generation();
        disk_slot.gen.key_gen  = item.key_generation();
        disk_slot.block_size   = item.logical_bytes() == 0 ? mem_buffers[0].size_bytes : item.logical_bytes();
        disk_slot.is_complete  = item.is_complete();
        if (!disk_spill_cache_->putExternalSlot(disk_slot, mem_buffers[0].addr, disk_slot.block_size)) {
            return false;
        }
        const auto removed_item = block_cache_->removeIfMatch(disk_slot.cache_key, mem_block);
        if (removed_item.has_value()) {
            freeBlocks({removed_item->block_index}, /*cache_free=*/true);
        }
    }
    return true;
}

bool KVCacheMemoryConnector::deleteDiskSlots(const MemoryOperationRequestPB& request) {
    if (!disk_spill_cache_) {
        return true;
    }
    for (int i = 0; i < request.copy_items_size(); ++i) {
        const auto&                   item = request.copy_items(i);
        DiskSpillBlockCache::DiskSlot disk_slot;
        disk_slot.cache_key    = static_cast<CacheKeyType>(item.cache_key());
        disk_slot.disk_id      = item.disk_id();
        disk_slot.slot_id      = item.disk_slot_id();
        disk_slot.gen.slot_gen = item.generation();
        disk_slot.gen.key_gen  = item.key_generation();
        disk_slot.block_size   = item.logical_bytes();
        disk_slot.is_complete  = item.is_complete();
        if (!disk_spill_cache_->deleteSlot(disk_slot)) {
            RTP_LLM_LOG_DEBUG("ignore stale disk slot delete, cache_key=%ld disk_id=%d slot_id=%d slot_gen=%lu "
                              "key_gen=%lu",
                              disk_slot.cache_key,
                              disk_slot.disk_id,
                              disk_slot.slot_id,
                              disk_slot.gen.slot_gen,
                              disk_slot.gen.key_gen);
        }
    }
    return true;
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

bool KVCacheMemoryConnector::mallocBlocks(size_t need_blocks, std::vector<BlockIdxType>& malloced_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    if (need_blocks == 0) {
        RTP_LLM_LOG_WARNING("malloc memory blocks failed, need blocks cannot be 0");
        return false;
    }

    std::vector<PendingSpill> pendings;

    {
        // make sure `ensure + malloc` is atomic with respect to other allocators.
        std::unique_lock<std::mutex> lock(malloc_mutex_);

        if (!ensureEnoughFreeBlocks(need_blocks)) {
            RTP_LLM_LOG_WARNING(
                "malloc memory blocks failed, ensure enough free blocks failed, need blocks: %zu, free blocks: %zu",
                need_blocks,
                block_pool_->freeBlocksNum());
            return false;
        }

        auto blocks = block_pool_->malloc(need_blocks);
        if (blocks.size() != need_blocks) {
            RTP_LLM_LOG_WARNING("malloc memory blocks failed, malloc failed, need blocks: %zu, allocated blocks: %zu",
                                need_blocks,
                                blocks.size());
            freeBlocks(blocks, /*cache_free=*/false);
            return false;
        }
        malloced_blocks = std::move(blocks);
        // Steal pending spills (set by ensureEnoughFreeBlocks via member) so we
        // can dispatch them OUTSIDE the malloc_mutex_ — broadcast + pwrite must
        // never block other allocators on the disk RPC RTT.
        // (pendings is local; flushPendingSpills runs lockless below.)
    }
    // No spills pending? Fast exit.
    // The two-phase contract: ensureEnoughFreeBlocks ALWAYS performs staging
    // (memcpy + slot reserve) under the lock and immediately calls
    // submitSpillsForAsyncCommit which forwards to the coordinator OUTSIDE the
    // critical section. We don't need to do anything here.
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
    if (disk_spill_cache_) {
        const auto invalidated_slot = disk_spill_cache_->invalidate(item.cache_key);
        if (invalidated_slot.has_value()) {
            sendDeleteDiskSlot(*invalidated_slot);
        }
    }
    if (auto [success, popped_item_opt] = block_cache_->put(item); success) {
        RTP_LLM_LOG_DEBUG("write cache, cache key: %ld, block index: %d, block size: %zu",
                          item.cache_key,
                          item.block_index,
                          item.block_size);
        referenceBlocks({item.block_index}, /*cache_ref=*/true);
        if (popped_item_opt.has_value()) {
            const auto popped_item = popped_item_opt.value();
            freeBlocks({popped_item.block_index}, /*cache_free=*/true);
        }
    }
}

bool KVCacheMemoryConnector::spillMemoryItemToDisk(const MemoryBlockCache::CacheItem& item) {
    if (!disk_spill_cache_ || !block_pool_) {
        return false;
    }
    auto mem_buffers = block_pool_->convertIndexToBuffer(/*layer_id=*/0, item.block_index);
    if (mem_buffers.size() != 1u || mem_buffers[0].addr == nullptr || mem_buffers[0].size_bytes == 0) {
        RTP_LLM_LOG_WARNING(
            "disk spill failed, invalid memory block: cache_key=%ld block=%d", item.cache_key, item.block_index);
        return false;
    }
    auto disk_slot = disk_spill_cache_->reserve(item.cache_key, mem_buffers[0].size_bytes, item.is_complete);
    if (!disk_slot.has_value()) {
        RTP_LLM_LOG_WARNING("disk spill failed, reserve slot failed, cache_key=%ld", item.cache_key);
        return false;
    }
    if (!disk_spill_cache_->writeReserved(*disk_slot, mem_buffers[0].addr, mem_buffers[0].size_bytes)) {
        disk_spill_cache_->abort(*disk_slot);
        RTP_LLM_LOG_WARNING("disk spill failed, local write failed, cache_key=%ld", item.cache_key);
        return false;
    }

    MemoryOperationRequestPB mem_req;
    mem_req.set_memory_op_protocol_version(1);
    mem_req.set_op_type(MemoryOperationRequestPB::SPILL_MEMORY_TO_DISK);
    mem_req.set_copy_direction(MemoryOperationRequestPB::D2H);
    auto* pb_item = mem_req.add_copy_items();
    pb_item->set_cache_key(item.cache_key);
    pb_item->set_mem_block(item.block_index);
    pb_item->set_source_type(MemoryOperationRequestPB::MEMORY_BLOCK);
    pb_item->set_disk_id(disk_slot->disk_id);
    pb_item->set_disk_slot_id(disk_slot->slot_id);
    pb_item->set_generation(disk_slot->gen.slot_gen);
    pb_item->set_key_generation(disk_slot->gen.key_gen);
    pb_item->set_logical_bytes(mem_buffers[0].size_bytes);
    pb_item->set_is_complete(disk_slot->is_complete);

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
    auto result = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        requests, kv_cache_config_.memory_cache_disk_stage_ack_timeout_ms, rpc_call);
    bool success = result != nullptr && result->waitDone(kv_cache_config_.memory_cache_disk_stage_ack_timeout_ms)
                   && result->success();
    if (success) {
        for (const auto& response : result->responses()) {
            if (!response.has_mem_response() || !response.mem_response().success()) {
                success = false;
                break;
            }
        }
    }
    if (!success) {
        disk_spill_cache_->abort(*disk_slot);
        sendDeleteDiskSlot(*disk_slot);
        RTP_LLM_LOG_WARNING("disk spill failed, worker stage ack failed, cache_key=%ld", item.cache_key);
        return false;
    }
    if (!disk_spill_cache_->commit(*disk_slot)) {
        disk_spill_cache_->abort(*disk_slot);
        sendDeleteDiskSlot(*disk_slot);
        RTP_LLM_LOG_WARNING("disk spill failed, commit failed, cache_key=%ld", item.cache_key);
        return false;
    }
    RTP_LLM_LOG_INFO("disk spill success, cache_key=%ld disk_id=%d slot_id=%d slot_gen=%lu",
                     item.cache_key,
                     disk_slot->disk_id,
                     disk_slot->slot_id,
                     disk_slot->gen.slot_gen);
    return true;
}

bool KVCacheMemoryConnector::sendDeleteDiskSlot(const DiskSpillBlockCache::DiskSlot& slot) const {
    if (!broadcast_manager_) {
        return false;
    }
    MemoryOperationRequestPB mem_req;
    mem_req.set_memory_op_protocol_version(1);
    mem_req.set_op_type(MemoryOperationRequestPB::DELETE_DISK_SLOT);
    auto* item = mem_req.add_copy_items();
    item->set_cache_key(slot.cache_key);
    item->set_source_type(MemoryOperationRequestPB::DISK_SLOT);
    item->set_disk_id(slot.disk_id);
    item->set_disk_slot_id(slot.slot_id);
    item->set_generation(slot.gen.slot_gen);
    item->set_key_generation(slot.gen.key_gen);
    item->set_logical_bytes(slot.block_size);
    item->set_is_complete(slot.is_complete);

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
    auto result = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        requests, kv_cache_config_.memory_cache_sync_timeout_ms, rpc_call);
    if (!result) {
        return false;
    }
    return result->waitDone(kv_cache_config_.memory_cache_sync_timeout_ms);
}

// Called under malloc_mutex_. Performs the SYNCHRONOUS phase of spill:
//   1. takeLRUItems (only_complete=true) from MemoryBlockCache
//   2. for each: reserve disk slot, memcpy mem block to heap staging buffer
//      (NULL slot ranges zeroed via valid_slots), then freeBlocks(cache_free=true)
//   3. ALSO drop partial items via a second takeLRUItems(only_complete=false) call
// The ASYNC phase (broadcast + pwrite + commit + abort) runs OUTSIDE the lock
// in the commit coordinator's worker thread.
bool KVCacheMemoryConnector::ensureEnoughFreeBlocks(size_t need_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    auto free_blocks = block_pool_->freeBlocksNum();
    if (free_blocks >= need_blocks) {
        return true;
    }
    const auto need_evict_blocks = need_blocks - free_blocks;
    if (disk_spill_cache_ && isMaster() && commit_coordinator_) {
        std::vector<PendingSpill> pendings;
        const auto                evict_items = block_cache_->takeLRUItems(need_evict_blocks, /*only_complete=*/true);
        for (const auto& item : evict_items) {
            if (item.is_resident) {
                freeBlocks({item.block_index}, /*cache_free=*/true);
                continue;
            }
            // Stage memory bytes into heap buffer under lock; release mem block.
            const auto mem_buffers = block_pool_->convertIndexToBuffer(/*layer_id=*/0, item.block_index);
            if (mem_buffers.size() != 1u || mem_buffers[0].addr == nullptr || mem_buffers[0].size_bytes == 0) {
                reportDiskError(disk_error::kInvalidMemoryBlock, "spill");
                freeBlocks({item.block_index}, /*cache_free=*/true);
                continue;
            }
            const size_t bytes = mem_buffers[0].size_bytes;
            PendingSpill pending;
            pending.item = item;
            pending.staging.resize(bytes);
            std::memcpy(pending.staging.data(), mem_buffers[0].addr, bytes);
            zeroNullSlots(item, pending.staging);
            // Reserve slot under malloc_mutex_; if reserve fails (disk full and
            // no in-place reuse candidate), drop the spill but still free memory.
            auto slot = disk_spill_cache_->reserve(item.cache_key, bytes, item.is_complete);
            if (!slot.has_value()) {
                reportDiskError(disk_error::kNoSlot, "spill");
                freeBlocks({item.block_index}, /*cache_free=*/true);
                continue;
            }
            pending.reserved_slot = *slot;
            pending.slot_reserved = true;
            pendings.push_back(std::move(pending));
            freeBlocks({item.block_index}, /*cache_free=*/true);
        }
        // Drop any remaining partial items so they don't keep memory blocks reserved.
        if (block_pool_->freeBlocksNum() < need_blocks) {
            const auto remaining   = need_blocks - block_pool_->freeBlocksNum();
            const auto partial_items = block_cache_->takeLRUItems(remaining, /*only_complete=*/false);
            for (const auto& item : partial_items) {
                freeBlocks({item.block_index}, /*cache_free=*/true);
            }
        }
        RTP_LLM_LOG_INFO("memory disk cache eviction, need_blocks=%zu free_before=%zu spilled=%zu free_after=%zu",
                         need_blocks,
                         free_blocks,
                         pendings.size(),
                         block_pool_->freeBlocksNum());
        // Dispatch pending spills to async coordinator outside this function
        // call site — we call directly here, but flushPendingSpills only does
        // SubmitSpill which is lock-free into the coordinator's queue.
        flushPendingSpills(std::move(pendings));
    } else {
        // Legacy path: no disk spill, or worker rank — just evict.
        const auto evict_blocks = block_cache_->pop(need_evict_blocks);
        if (!evict_blocks.empty()) {
            freeBlocks(evict_blocks, /*cache_free=*/true);
        }
    }
    return block_pool_->freeBlocksNum() >= need_blocks;
}

void KVCacheMemoryConnector::flushPendingSpills(std::vector<PendingSpill>&& pendings) {
    if (!commit_coordinator_ || pendings.empty()) {
        // Abort any reserved slots if we have no coordinator (shouldn't happen on master)
        for (auto& p : pendings) {
            if (p.slot_reserved) {
                disk_spill_cache_->abort(p.reserved_slot);
            }
        }
        return;
    }
    for (auto& p : pendings) {
        if (!p.slot_reserved) {
            continue;
        }
        auto reserved = p.reserved_slot;
        commit_coordinator_->submitSpill(
            reserved,
            std::move(p.staging),
            [this, reserved](SpillJobId id, SpillStageState state) {
                if (state == SpillStageState::COMMITTED) {
                    reportDiskWriteMetrics(/*success=*/true, 0, 1, 1);
                    RTP_LLM_LOG_DEBUG("disk spill committed job=%lu key=%ld disk=%d slot=%d",
                                      id,
                                      reserved.cache_key,
                                      reserved.disk_id,
                                      reserved.slot_id);
                } else {
                    reportDiskWriteMetrics(/*success=*/false, 0, 1, 0);
                    RTP_LLM_LOG_WARNING("disk spill ended without commit, job=%lu key=%ld state=%d",
                                        id,
                                        reserved.cache_key,
                                        static_cast<int>(state));
                }
            });
    }
}

void KVCacheMemoryConnector::zeroNullSlots(const MemoryBlockCache::CacheItem& item,
                                           std::vector<char>&                 staging) const {
    if (item.valid_slots.empty()) {
        return;  // legacy non-DSV4 layout: assume all slots valid
    }
    const auto slots = layerRegionSlots();
    if (slots.size() != item.valid_slots.size()) {
        return;
    }
    size_t off = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        const size_t bytes = slots[i].stride_bytes;
        if (!item.valid_slots[i]) {
            if (off + bytes <= staging.size()) {
                std::memset(staging.data() + off, 0, bytes);
            }
        }
        off += bytes;
    }
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

void KVCacheMemoryConnector::reportMetricsLoop() {
    int64_t periodic_log_acc_ms = 0;
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

            if (disk_spill_cache_) {
                const auto disk_status = disk_spill_cache_->status();
                RtpLLMMemoryDiskCacheStatusMetricsCollector disk_collector;
                disk_collector.total_block_num          = static_cast<int64_t>(disk_status.total_slot_num);
                disk_collector.allocated_block_num =
                    static_cast<int64_t>(disk_status.total_slot_num - disk_status.free_slot_num);
                disk_collector.available_block_num      = static_cast<int64_t>(disk_status.free_slot_num);
                disk_collector.committed_block_num      = static_cast<int64_t>(disk_status.committed_slot_num);
                disk_collector.inflight_write_block_num = static_cast<int64_t>(disk_status.inflight_write_slot_num);
                disk_collector.inflight_read_block_num  = static_cast<int64_t>(disk_status.inflight_read_slot_num);
                disk_collector.used_bytes               = static_cast<int64_t>(disk_status.used_bytes);
                disk_collector.free_bytes               = static_cast<int64_t>(disk_status.free_bytes);
                disk_collector.staging_used_bytes       = static_cast<int64_t>(disk_status.staging_used);
                disk_collector.unhealthy_disk_num       = static_cast<int64_t>(disk_status.unhealthy_disk_num);
                disk_collector.leaked_block_num         = static_cast<int64_t>(disk_status.leaked_slot_num);
                metrics_reporter_->report<RtpLLMMemoryDiskCacheMetrics,
                                          RtpLLMMemoryDiskCacheStatusMetricsCollector>(nullptr, &disk_collector);
            }
        }
        // Periodic textual summary for SRE / log analysis.
        periodic_log_acc_ms += 1000;
        if (disk_spill_cache_ && periodic_log_acc_ms
            >= kv_cache_config_.memory_cache_disk_metrics_report_interval_ms) {
            periodic_log_acc_ms       = 0;
            const auto disk_status    = disk_spill_cache_->status();
            const auto fms            = disk_spill_cache_->fileManagers();
            RTP_LLM_LOG_INFO("memory-disk raw: enabled=1 disks=%zu total_slots=%zu committed=%zu inflight_w=%zu "
                             "inflight_r=%zu free=%zu staging_used=%zu unhealthy=%zu leaked=%zu",
                             fms.size(),
                             disk_status.total_slot_num,
                             disk_status.committed_slot_num,
                             disk_status.inflight_write_slot_num,
                             disk_status.inflight_read_slot_num,
                             disk_status.free_slot_num,
                             disk_status.staging_used,
                             disk_status.unhealthy_disk_num,
                             disk_status.leaked_slot_num);
            for (size_t i = 0; i < fms.size(); ++i) {
                if (!fms[i]) {
                    continue;
                }
                const auto fs = fms[i]->getStats();
                RTP_LLM_LOG_INFO("memory-disk raw disk[%zu]: path_hash=%s io_mode=%s slots=%zu staging=%zu/%zu "
                                 "unhealthy=%d",
                                 i,
                                 fms[i]->pathHash().c_str(),
                                 fs.io_mode == DiskSpillFileManager::IoMode::DIRECT ? "direct" : "buffered",
                                 fs.slot_count,
                                 fs.staging_used,
                                 fs.staging_total,
                                 fs.unhealthy ? 1 : 0);
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// =================== Disk spill handshake + broadcast helpers ===================

std::string KVCacheMemoryConnector::computeSchemaHash() const {
    std::ostringstream oss;
    oss << "layer_all_num=" << cache_config_.layer_all_num;
    oss << ",block_size=" << memoryBlockSizeBytes();
    oss << ",seq=" << cache_config_.seq_size_per_block;
    oss << ",kernel_seq=" << cache_config_.kernel_seq_size_per_block;
    oss << ",sparse=" << cache_config_.is_sparse;
    oss << ",typed=" << cache_config_.use_typed_cache_regions;
    oss << ",opaque=" << cache_config_.use_opaque_kv_cache_store;
    oss << ",indep=" << cache_config_.use_independent_block_pools;
    oss << ",world=" << tp_size_;
    const auto slots = layerRegionSlots();
    for (const auto& s : slots) {
        oss << ",[" << s.layer_id << "," << static_cast<int>(s.region_name) << "," << s.group_id << ","
            << s.stride_bytes << "]";
    }
    const auto raw = oss.str();
    std::hash<std::string> h;
    std::ostringstream hex;
    hex << std::hex << h(raw);
    return hex.str();
}

bool KVCacheMemoryConnector::runDiskSpillHandshake() {
    if (!broadcast_manager_) {
        return false;
    }
    if (broadcast_manager_->workerNum() == 0) {
        return true;  // TP=1 (no remote workers) — vacuously ok
    }
    // Worker ranks may finish initDiskSpillCache() slightly after master.
    // Retry the HELLO on transient "capability_mismatch" responses (which
    // a worker returns when its disk_spill_cache_ isn't ready yet) until
    // memory_cache_disk_init_timeout_ms elapses. Real schema mismatches will
    // surface as the same error_type but the cache will be initialised on
    // every retry, so any persistent mismatch eventually still fails out
    // after timeout.
    const auto deadline_ms      = kv_cache_config_.memory_cache_disk_init_timeout_ms;
    const auto deadline         = std::chrono::steady_clock::now() + std::chrono::milliseconds(deadline_ms);
    const auto per_call_timeout = std::min<int64_t>(2000, deadline_ms);
    int        attempts         = 0;
    while (true) {
        ++attempts;
        MemoryOperationRequestPB hello;
        hello.set_memory_op_protocol_version(kDiskSpillProtocolVersion);
        hello.set_op_type(MemoryOperationRequestPB::DISK_SPILL_HELLO);
        hello.set_op_sequence(outgoing_op_sequence_.next());
        hello.set_disk_spill_capability_mask(static_cast<uint64_t>(disk_spill_cache_->blockSize()));
        hello.set_schema_hash(schema_hash_);

        std::vector<FunctionRequestPB> requests;
        requests.reserve(broadcast_manager_->workerNum());
        for (size_t i = 0; i < broadcast_manager_->workerNum(); ++i) {
            FunctionRequestPB req;
            req.mutable_mem_request()->CopyFrom(hello);
            requests.emplace_back(std::move(req));
        }
        auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                           const std::shared_ptr<grpc::ClientContext>& context,
                           const FunctionRequestPB&                    request,
                           grpc::CompletionQueue*                      completion_queue) {
            return stub->AsyncExecuteFunction(context.get(), request, completion_queue);
        };
        auto result = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
            requests, static_cast<int>(per_call_timeout), rpc_call);

        bool transient_failure = false;
        bool permanent_failure = false;
        if (!result || !result->waitDone(static_cast<int>(per_call_timeout))) {
            transient_failure = true;
        } else if (!result->success()) {
            transient_failure = true;
        } else {
            for (const auto& response : result->responses()) {
                if (!response.has_mem_response()) {
                    transient_failure = true;
                    break;
                }
                const auto& mr = response.mem_response();
                if (!mr.success()) {
                    // capability_mismatch and protocol_violation can be transient if
                    // worker isn't ready or our op_sequence is ahead of theirs after
                    // a partial rollout.
                    if (mr.error_type() == disk_error::kCapabilityMismatch
                        || mr.error_type() == disk_error::kProtocolViolation) {
                        transient_failure = true;
                        if (attempts % 25 == 1) {
                            RTP_LLM_LOG_WARNING("disk spill handshake transient reject error=%s attempt=%d",
                                                mr.error_type().c_str(),
                                                attempts);
                        }
                    } else {
                        permanent_failure = true;
                        RTP_LLM_LOG_ERROR("disk spill handshake rejected, error=%s", mr.error_type().c_str());
                    }
                    break;
                }
                if (mr.schema_hash() != schema_hash_) {
                    permanent_failure = true;
                    RTP_LLM_LOG_ERROR("disk spill schema_hash mismatch: master=%s worker=%s",
                                      schema_hash_.c_str(),
                                      mr.schema_hash().c_str());
                    break;
                }
                if (mr.disk_spill_capability_mask() != static_cast<uint64_t>(disk_spill_cache_->blockSize())) {
                    permanent_failure = true;
                    RTP_LLM_LOG_ERROR("disk spill capability_mask mismatch: master=%zu worker=%lu",
                                      disk_spill_cache_->blockSize(),
                                      mr.disk_spill_capability_mask());
                    break;
                }
            }
        }
        if (!transient_failure && !permanent_failure) {
            RTP_LLM_LOG_INFO("disk spill capability handshake ok, peers=%zu schema_hash=%s attempts=%d",
                             broadcast_manager_->workerNum(),
                             schema_hash_.c_str(),
                             attempts);
            return true;
        }
        if (permanent_failure) {
            reportDiskError(disk_error::kCapabilityMismatch, "hello");
            return false;
        }
        // transient — back off and retry
        if (std::chrono::steady_clock::now() >= deadline) {
            reportDiskError(disk_error::kTpBroadcastTimeout, "hello");
            RTP_LLM_LOG_ERROR("disk spill handshake exhausted retries after %d attempts", attempts);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

bool KVCacheMemoryConnector::handleDiskSpillHello(const MemoryOperationRequestPB& request,
                                                  MemoryOperationResponsePB&      response) {
    response.set_disk_spill_capability_mask(disk_spill_cache_ ? static_cast<uint64_t>(disk_spill_cache_->blockSize())
                                                              : 0u);
    response.set_schema_hash(schema_hash_);
    if (!disk_spill_cache_) {
        response.set_success(false);
        response.set_error_type(disk_error::kCapabilityMismatch);
        RTP_LLM_LOG_WARNING("disk spill hello: local disk_spill_cache_ not initialised yet");
        return false;
    }
    if (request.schema_hash() != schema_hash_) {
        response.set_success(false);
        response.set_error_type(disk_error::kInitSchema);
        RTP_LLM_LOG_WARNING("disk spill hello: schema_hash mismatch req='%s' local='%s'",
                            request.schema_hash().c_str(),
                            schema_hash_.c_str());
        return false;
    }
    if (request.disk_spill_capability_mask() != static_cast<uint64_t>(disk_spill_cache_->blockSize())) {
        response.set_success(false);
        response.set_error_type(disk_error::kCapabilityMismatch);
        RTP_LLM_LOG_WARNING("disk spill hello: cap_mask mismatch req=%lu local=%zu",
                            request.disk_spill_capability_mask(),
                            disk_spill_cache_->blockSize());
        return false;
    }
    response.set_success(true);
    response.set_error_type("ok");
    return true;
}

bool KVCacheMemoryConnector::handleSpillWriteStatus(const MemoryOperationRequestPB& request,
                                                    MemoryOperationResponsePB&      response) {
    // Worker side: report pwrite status for each requested job. For MVP we keep
    // no separate tracker; we rely on the fact that worker putExternalSlot is
    // synchronous and either fully succeeded or returned false. If it succeeded,
    // the slot is in committed_index; we return SUCCESS. Otherwise UNKNOWN_JOB.
    for (int i = 0; i < request.copy_items_size(); ++i) {
        const auto& item   = request.copy_items(i);
        auto*       status = response.add_job_status();
        status->set_spill_job_id(item.spill_job_id());
        if (disk_spill_cache_ && disk_spill_cache_->contains(static_cast<CacheKeyType>(item.cache_key()))) {
            status->set_status(static_cast<int32_t>(SpillWriteStatus::SUCCESS));
        } else {
            status->set_status(static_cast<int32_t>(SpillWriteStatus::UNKNOWN_JOB));
        }
    }
    response.set_success(true);
    return true;
}

OpSequenceTracker& KVCacheMemoryConnector::trackerForIncomingRank(int peer_seq_id) {
    std::lock_guard<std::mutex> lock(incoming_op_sequence_mutex_);
    return incoming_op_sequence_[peer_seq_id];
}

bool KVCacheMemoryConnector::broadcastSpillToWorkers(SpillJobId job_id, const DiskSpillBlockCache::DiskItem& slot) {
    if (!broadcast_manager_ || broadcast_manager_->workerNum() == 0) {
        // TP=1: notify the coordinator that there are no workers to wait for
        if (commit_coordinator_) {
            commit_coordinator_->notifyWorkerStatus(job_id, 0, SpillWriteStatus::SUCCESS);
        }
        return true;
    }
    MemoryOperationRequestPB req;
    req.set_memory_op_protocol_version(kDiskSpillProtocolVersion);
    req.set_op_type(MemoryOperationRequestPB::SPILL_MEMORY_TO_DISK);
    req.set_op_sequence(outgoing_op_sequence_.next());
    req.set_copy_direction(MemoryOperationRequestPB::D2H);
    req.set_schema_hash(schema_hash_);
    auto* item = req.add_copy_items();
    item->set_cache_key(slot.cache_key);
    item->set_source_type(MemoryOperationRequestPB::DISK_SLOT);
    item->set_disk_id(slot.disk_id);
    item->set_disk_slot_id(slot.slot_id);
    item->set_generation(slot.gen.slot_gen);
    item->set_key_generation(slot.gen.key_gen);
    item->set_logical_bytes(slot.block_size);
    item->set_is_complete(slot.is_complete);
    item->set_spill_job_id(job_id);

    std::vector<FunctionRequestPB> requests;
    requests.reserve(broadcast_manager_->workerNum());
    for (size_t i = 0; i < broadcast_manager_->workerNum(); ++i) {
        FunctionRequestPB r;
        r.mutable_mem_request()->CopyFrom(req);
        requests.emplace_back(std::move(r));
    }
    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const FunctionRequestPB&                    rq,
                       grpc::CompletionQueue*                      cq) {
        return stub->AsyncExecuteFunction(context.get(), rq, cq);
    };
    const auto timeout = static_cast<int>(kv_cache_config_.memory_cache_disk_stage_ack_timeout_ms);
    auto       result  = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, timeout, rpc_call);
    if (!result || !result->waitDone(timeout) || !result->success()) {
        reportDiskError(disk_error::kTpBroadcast, "spill");
        return false;
    }
    for (const auto& resp : result->responses()) {
        if (!resp.has_mem_response() || !resp.mem_response().success()) {
            reportDiskError(disk_error::kRankWrite, "spill");
            return false;
        }
    }
    // All workers acked synchronously via the broadcast wait. Notify coordinator
    // so it can commit without waiting for the next poll cycle.
    if (commit_coordinator_) {
        for (int r = 0; r < static_cast<int>(broadcast_manager_->workerNum()); ++r) {
            commit_coordinator_->notifyWorkerStatus(job_id, r, SpillWriteStatus::SUCCESS);
        }
    }
    return true;
}

bool KVCacheMemoryConnector::broadcastDeleteToWorkers(const DiskSpillBlockCache::DiskItem& slot) {
    sendDeleteDiskSlot(slot);  // legacy helper already handles broadcast + wait
    return true;
}

SpillWriteStatus KVCacheMemoryConnector::pollWorkerSpillStatus(int /*worker_idx*/, SpillJobId /*job_id*/) {
    // MVP: we treat SPILL_MEMORY_TO_DISK as synchronous (worker side completes
    // the pwrite before returning). So if broadcastSpillToWorkers returned true,
    // every worker is already SUCCESS. Coordinator never reaches pollWorker
    // (worker_status[0..N-1] set to SUCCESS on broadcast ack via notifyWorkerStatus
    // — see below).
    return SpillWriteStatus::SUCCESS;
}

void KVCacheMemoryConnector::reportDiskMatchMetrics(bool    success,
                                                    int64_t latency_us,
                                                    int64_t input,
                                                    int64_t matched,
                                                    bool    contention) {
    if (!metrics_reporter_) {
        return;
    }
    RtpLLMMemoryDiskCacheMatchMetricsCollector c;
    c.failed          = !success;
    c.latency_us      = latency_us;
    c.input_token     = input;
    c.matched_token   = matched;
    c.take_contention = contention;
    metrics_reporter_->report<RtpLLMMemoryDiskCacheMetrics, RtpLLMMemoryDiskCacheMatchMetricsCollector>(nullptr, &c);
}

void KVCacheMemoryConnector::reportDiskWriteMetrics(bool success, int64_t latency_us, int64_t input, int64_t written) {
    if (!metrics_reporter_) {
        return;
    }
    RtpLLMMemoryDiskCacheWriteMetricsCollector c;
    c.failed      = !success;
    c.latency_us  = latency_us;
    c.input_token = input;
    c.write_token = written;
    metrics_reporter_->report<RtpLLMMemoryDiskCacheMetrics, RtpLLMMemoryDiskCacheWriteMetricsCollector>(nullptr, &c);
}

void KVCacheMemoryConnector::reportDiskReadMetrics(bool success, int64_t latency_us, int64_t input, int64_t read_token) {
    if (!metrics_reporter_) {
        return;
    }
    RtpLLMMemoryDiskCacheReadMetricsCollector c;
    c.failed      = !success;
    c.latency_us  = latency_us;
    c.input_token = input;
    c.read_token  = read_token;
    metrics_reporter_->report<RtpLLMMemoryDiskCacheMetrics, RtpLLMMemoryDiskCacheReadMetricsCollector>(nullptr, &c);
}

void KVCacheMemoryConnector::reportDiskCopyMetrics(bool                success,
                                                   int64_t             latency_us,
                                                   const std::string&  direction,
                                                   int                 disk_id) {
    if (!metrics_reporter_) {
        return;
    }
    RtpLLMMemoryDiskCacheCopyMetricsCollector c;
    c.failed         = !success;
    c.latency_us     = latency_us;
    c.copy_direction = direction;
    c.disk_id        = disk_id;
    metrics_reporter_->report<RtpLLMMemoryDiskCacheMetrics, RtpLLMMemoryDiskCacheCopyMetricsCollector>(nullptr, &c);
}

void KVCacheMemoryConnector::reportDiskError(const std::string& error_type, const std::string& op, int disk_id) {
    if (!metrics_reporter_) {
        return;
    }
    RtpLLMMemoryDiskCacheErrorMetricsCollector c;
    c.error_type = error_type;
    c.op         = op;
    c.disk_id    = disk_id;
    metrics_reporter_->report<RtpLLMMemoryDiskCacheMetrics, RtpLLMMemoryDiskCacheErrorMetricsCollector>(nullptr, &c);
}

}  // namespace rtp_llm
