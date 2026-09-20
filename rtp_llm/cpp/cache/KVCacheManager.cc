#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/events/KVCMPublisher.h"

#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/PrefillCacheHitMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#ifdef RTP_LLM_USE_REMOTE_KV_CACHE
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMStorageBackend.h"
#endif
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/KVCacheMetrics.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

class KVCacheAllocationWaitState {
public:
    std::atomic<uint64_t>   generation{0};
    std::atomic<bool>       stopped{false};
    std::mutex              mutex;
    std::condition_variable cv;
};

namespace {

void notifyAllocationChangeState(const std::shared_ptr<KVCacheAllocationWaitState>& state) {
    if (!state) {
        return;
    }
    {
        // Coordinate the predicate update with wait_for()'s unlock-and-wait
        // transition. Without this lock a notify can race between the final
        // predicate check and actually enqueueing the waiter.
        std::lock_guard<std::mutex> lock(state->mutex);
        state->generation.fetch_add(1, std::memory_order_release);
    }
    state->cv.notify_all();
}

std::string resolveKVCacheEventInstanceGroup(const std::string& event_group, const std::string& reco_group) {
    return event_group.empty() ? reco_group : event_group;
}

int64_t aggregateKVCacheEventSpecSizeBytes(const std::vector<int64_t>& group_sizes, int64_t tp_size) {
    return std::accumulate(group_sizes.begin(), group_sizes.end(), int64_t{0}) * std::max<int64_t>(tp_size, 1);
}

RtpLLMCacheMetricsCollector collectGlobalCacheMetrics(const CoordinatorCacheManagerPtr& coordinator_manager) {
    RtpLLMCacheMetricsCollector collector;
    const BlockTreeCachePtr     block_tree_cache = coordinator_manager->blockTreeCache();
    collector.kv_cache_item_num =
        block_tree_cache ? static_cast<int64_t>(block_tree_cache->getStats().tree_node_count) : 0;
    collector.kv_cache_left_seq = static_cast<int64_t>(coordinator_manager->availableTokensNum());
    collector.mr_cost_time_ms   = coordinator_manager->getMrCostTimeMs();

    return collector;
}

void logGlobalCacheMetrics(const RtpLLMCacheMetricsCollector& collector) {
    RTP_LLM_LOG_INFO("kvc raw global: tree_node_count=%ld available_tokens=%ld",
                     static_cast<long>(collector.kv_cache_item_num),
                     static_cast<long>(collector.kv_cache_left_seq));
}

void reportPoolCacheMetrics(const kmonitor::MetricsReporterPtr& metrics_reporter,
                            const CachePoolMetricsSnapshot&     pool_snapshot,
                            bool                                should_log) {
    if (should_log) {
        RTP_LLM_LOG_INFO("kvc raw pool[%s/%s]: block_size_bytes=%zu total=%zu free=%zu used=%zu available=%zu "
                         "active=%zu "
                         "reserve=%zu request_ref_blocks=%zu block_cache_ref_blocks=%zu "
                         "load_ref_blocks=%zu eviction_target_ref_blocks=%zu store_ref_blocks=%zu used_ratio=%.4f%%",
                         pool_snapshot.tier.c_str(),
                         pool_snapshot.pool_name.c_str(),
                         pool_snapshot.block_size_bytes,
                         pool_snapshot.total_blocks,
                         pool_snapshot.free_blocks,
                         pool_snapshot.used_blocks,
                         pool_snapshot.available_blocks,
                         pool_snapshot.active_blocks,
                         pool_snapshot.reserve_blocks,
                         pool_snapshot.request_ref_blocks,
                         pool_snapshot.block_cache_ref_blocks,
                         pool_snapshot.load_ref_blocks,
                         pool_snapshot.eviction_target_ref_blocks,
                         pool_snapshot.store_ref_blocks,
                         pool_snapshot.used_ratio);
    }

    RtpLLMCachePoolMetricsCollector pool_collector;
    pool_collector.block_size_bytes                  = static_cast<int64_t>(pool_snapshot.block_size_bytes);
    pool_collector.free_blocks                       = static_cast<int64_t>(pool_snapshot.free_blocks);
    pool_collector.used_blocks                       = static_cast<int64_t>(pool_snapshot.used_blocks);
    pool_collector.available_blocks                  = static_cast<int64_t>(pool_snapshot.available_blocks);
    pool_collector.active_blocks                     = static_cast<int64_t>(pool_snapshot.active_blocks);
    pool_collector.total_blocks                      = static_cast<int64_t>(pool_snapshot.total_blocks);
    pool_collector.reserve_blocks                    = static_cast<int64_t>(pool_snapshot.reserve_blocks);
    pool_collector.request_ref_blocks                = static_cast<int64_t>(pool_snapshot.request_ref_blocks);
    pool_collector.block_cache_ref_blocks            = static_cast<int64_t>(pool_snapshot.block_cache_ref_blocks);
    pool_collector.load_ref_blocks                   = static_cast<int64_t>(pool_snapshot.load_ref_blocks);
    pool_collector.eviction_target_ref_blocks        = static_cast<int64_t>(pool_snapshot.eviction_target_ref_blocks);
    pool_collector.report_eviction_target_ref_blocks = pool_snapshot.tier != tierName(Tier::DEVICE);
    pool_collector.store_ref_blocks                  = static_cast<int64_t>(pool_snapshot.store_ref_blocks);
    pool_collector.used_ratio                        = pool_snapshot.used_ratio;

    kmonitor::MetricsTags pool_tags("pool_name", pool_snapshot.pool_name);
    pool_tags.AddTag("tier", pool_snapshot.tier);
    metrics_reporter->report<RtpLLMCachePoolMetrics, RtpLLMCachePoolMetricsCollector>(&pool_tags, &pool_collector);
}

std::shared_ptr<const CacheTopology> projectTopology(const CacheTopology&       source,
                                                     const std::vector<size_t>& global_layer_ids) {
    std::vector<GroupBase> groups = source.groups();

    std::vector<LayerBase> layers;
    layers.reserve(global_layer_ids.size());
    for (size_t local_layer_id = 0; local_layer_id < global_layer_ids.size(); ++local_layer_id) {
        const auto& source_layer = source.layer(static_cast<int>(global_layer_ids[local_layer_id]));
        LayerBase   layer;
        layer.layer_id   = static_cast<int>(local_layer_id);
        layer.group_tags = source_layer.group_tags;
        layers.push_back(std::move(layer));
    }
    return CacheTopology::create(std::move(groups), std::move(layers));
}

GroupedCacheLayerLayout projectLayout(const GroupedCacheLayerLayout&       source,
                                      std::shared_ptr<const CacheTopology> target_topology,
                                      const std::vector<size_t>&           global_layer_ids) {
    RTP_LLM_CHECK_WITH_INFO(target_topology != nullptr, "cache layout projection requires a target topology");
    RTP_LLM_CHECK_WITH_INFO(target_topology->layers().size() == global_layer_ids.size(),
                            "cache layout projection topology layers=%zu mapping size=%zu",
                            target_topology->layers().size(),
                            global_layer_ids.size());

    GroupedCacheLayerLayout::GroupLayouts groups;
    for (const auto& target_group : target_topology->groups()) {
        std::vector<BlockBufferPtrInfo> layers(global_layer_ids.size());
        const auto&                     source_group = source.group(target_group.tag);
        for (int local_layer_id : target_topology->layerIdsForGroup(target_group.tag)) {
            RTP_LLM_CHECK_WITH_INFO(local_layer_id >= 0
                                        && static_cast<size_t>(local_layer_id) < global_layer_ids.size(),
                                    "cache layout projection tag=%s invalid local layer=%d",
                                    target_group.tag.c_str(),
                                    local_layer_id);
            const auto local  = static_cast<size_t>(local_layer_id);
            const auto global = global_layer_ids[local];
            if (source_group.hasLayer(global)) {
                layers[local] = source_group.at(global);
            }
        }
        groups.emplace(target_group.tag, CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(std::move(target_topology), std::move(groups));
}

bool cacheStatusSnapshotEnabled() {
    const char* env = std::getenv("RTP_LLM_CACHE_STATUS_SNAPSHOT");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

void reportCacheOperation(const kmonitor::MetricsReporterPtr&          metrics_reporter,
                          RtpLLMCacheOperationMetricsCollector::OpType operation_type,
                          int64_t                                      begin_time_us) {
    if (metrics_reporter == nullptr) {
        return;
    }
    RtpLLMCacheOperationMetricsCollector collector;
    collector.operation_type = operation_type;
    collector.latency_us     = currentTimeUs() - begin_time_us;
    metrics_reporter->report<RtpLLMCacheOperationMetrics, RtpLLMCacheOperationMetricsCollector>(nullptr, &collector);
}

}  // namespace

KVCacheManager::KVCacheManager(const CacheConfig&                 config,
                               bool                               warmup,
                               const kmonitor::MetricsReporterPtr metrics_reporter,
                               const KVCacheConfig&               kv_cache_config,
                               const ParallelismConfig&           parallelism_config,
                               const RuntimeConfig&               runtime_config,
                               const SpeculativeExecutionConfig&  sp_config,
                               const PDSepConfig&                 pd_sep_config,
                               const CacheStoreConfig& /*cache_store_config*/,
                               bool use_device_malloc_block_pool):
    config_(config),
    metrics_reporter_(metrics_reporter),
    kv_cache_config_(kv_cache_config),
    parallelism_config_(parallelism_config),
    runtime_config_(runtime_config),
    sp_config_(sp_config),
    pd_sep_config_(pd_sep_config),
    use_device_malloc_block_pool_(use_device_malloc_block_pool),
    warmup_(warmup),
    allocation_wait_state_(std::make_shared<KVCacheAllocationWaitState>()) {
    for (const auto& group : config_.topology().groups()) {
        RTP_LLM_CHECK_WITH_INFO(group.block_num > 0, "cache manager requires capacity-complete cache groups");
    }
    for (const auto& child : config_.mtp_sub_configs) {
        RTP_LLM_CHECK_WITH_INFO(child != nullptr, "null MTP cache configuration");
        for (const auto& group : child->topology().groups()) {
            RTP_LLM_CHECK_WITH_INFO(group.block_num > 0, "cache manager requires capacity-complete MTP cache groups");
        }
    }

    const auto& cp_cfg = parallelism_config_.prefill_cp_config;
    if (cp_cfg.kv_cache_sharded && parallelism_config_.tp_size > 1) {
        cp_slot_mapper_ = std::make_shared<CPSlotMapper>(static_cast<int>(parallelism_config_.tp_rank),
                                                         static_cast<int>(parallelism_config_.tp_size),
                                                         static_cast<int>(config_.seq_size_per_block));
        RTP_LLM_LOG_INFO("CP sharded KV cache enabled: cp_rank=%d, cp_size=%d, block_size=%zu, "
                         "virtual_block_size=%d",
                         (int)parallelism_config_.tp_rank,
                         (int)parallelism_config_.tp_size,
                         config_.seq_size_per_block,
                         cp_slot_mapper_->virtualBlockSize());
    }

    if (pd_sep_config_.role_type == RoleType::PREFILL) {
        if (PrefillCacheHitMetricsReporter::enabled()) {
            prefill_cache_hit_metrics_reporter_ = std::make_unique<PrefillCacheHitMetricsReporter>(metrics_reporter_);
        } else {
            RTP_LLM_LOG_INFO("prefill recent-cache-key metrics disabled by PREFILL_CACHE_HIT_METRIC_ENABLE");
        }
    }

    RTP_LLM_LOG_INFO("cache config: layer_num=%d, block_size=%dB, seq_size_per_block=%zu",
                     config_.layer_num,
                     config_.totalGroupBlockSizeBytes(),
                     config_.seq_size_per_block);
}

KVCacheManager::~KVCacheManager() {
    {
        std::lock_guard<std::mutex> lock(allocation_wait_state_->mutex);
        allocation_wait_state_->stopped.store(true, std::memory_order_release);
    }
    allocation_wait_state_->cv.notify_all();
    stopMetricsReporter();
    stopCacheEventPublisher();
}

void KVCacheManager::stopMetricsReporter() {
    stop_.store(true, std::memory_order_release);
    if (metrics_reporter_thread_.joinable()) {
        metrics_reporter_thread_.join();
    }
}

// 初始化和配置相关

bool KVCacheManager::init() {
    RTP_LLM_CHECK_WITH_INFO(!coordinator_manager_ && !block_tree_cache_ && !metrics_reporter_thread_.joinable(),
                            "KVCacheManager::init called more than once");
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() > 0, "cache specs must not be empty");
    if (kv_cache_config_.enable_remote_cache
        && (kv_cache_config_.kvcm_asyncwrapper_thread_num == 0 || kv_cache_config_.kvcm_asyncwrapper_queue_size == 0)) {
        RTP_LLM_LOG_ERROR("remote cache executor thread count and queue size must be positive, got %zu/%zu",
                          kv_cache_config_.kvcm_asyncwrapper_thread_num,
                          kv_cache_config_.kvcm_asyncwrapper_queue_size);
        return false;
    }

    coordinator_manager_ = std::make_shared<CoordinatorCacheManager>(config_,
                                                                     AllocationType::DEVICE,
                                                                     metrics_reporter_,
                                                                     kv_cache_config_.reserve_block_ratio,
                                                                     pd_sep_config_.role_type);

    if (use_device_malloc_block_pool_) {
        RTP_LLM_LOG_INFO("RDMA cache store enabled for PD role, use raw device malloc KV cache block-pool backing");
        coordinator_manager_->setUseDeviceMallocBlockPool(true);
    }

    coordinator_manager_->setCPSlotMapper(cp_slot_mapper_);
    RTP_LLM_CHECK_WITH_INFO(coordinator_manager_->init(), "CoordinatorCacheManager init failed");
    // Observe real pool capacity, including asynchronous eviction and lease release.
    const auto capacity_changed = allocationChangeCallback();
    for (const auto& pool : coordinator_manager_->groupBlockPools()) {
        pool->setCapacityChangeCallback(capacity_changed);
    }
    const bool requires_broadcast_manager = parallelism_config_.tp_size > 1 && parallelism_config_.tp_rank == 0
                                            && !runtime_config_.worker_grpc_addrs.empty();
    std::shared_ptr<BroadcastManager> broadcast_manager;
    if (requires_broadcast_manager) {
        broadcast_manager = createMultiRankBlockTransferManager();
        if (!broadcast_manager) {
            return false;
        }
    }

    std::shared_ptr<StorageBackend> storage_backend;
    if (kv_cache_config_.enable_remote_cache) {
#ifdef RTP_LLM_USE_REMOTE_KV_CACHE
        storage_backend = std::make_shared<KVCMStorageBackend>(
            config_, kv_cache_config_, runtime_config_, parallelism_config_, sp_config_, broadcast_manager);
#else
        RTP_LLM_LOG_ERROR("remote cache was requested, but this build does not include the KVCM client");
        return false;
#endif
    }

    block_tree_cache_ = createBlockTreeCache(config_,
                                             kv_cache_config_,
                                             coordinator_manager_,
                                             parallelism_config_,
                                             std::move(storage_backend),
                                             broadcast_manager,
                                             metrics_reporter_);
    if (!block_tree_cache_) {
        RTP_LLM_LOG_ERROR("KVCacheManager::init: failed to create BlockTreeCache");
        return false;
    }
    coordinator_manager_->attachBlockTreeCache(block_tree_cache_);
    initCacheEventPublisher();

    if (metrics_reporter_) {
        stop_.store(false, std::memory_order_relaxed);
        metrics_reporter_thread_ = std::thread(&KVCacheManager::reportMetricsLoop, this);
    }

    return true;
}

std::shared_ptr<BroadcastManager> KVCacheManager::createMultiRankBlockTransferManager() const {
    const size_t expected_worker_count = static_cast<size_t>(parallelism_config_.tp_size);
    if (runtime_config_.worker_grpc_addrs.size() != expected_worker_count) {
        RTP_LLM_LOG_ERROR("KVCacheManager: worker grpc address count mismatch, expected=%zu, actual=%zu",
                          expected_worker_count,
                          runtime_config_.worker_grpc_addrs.size());
        return nullptr;
    }

    auto broadcast_manager = std::make_shared<BroadcastManager>(runtime_config_.worker_grpc_addrs);
    if (!broadcast_manager->init()) {
        RTP_LLM_LOG_ERROR("KVCacheManager: failed to initialize BlockTreeCache BroadcastManager");
        return nullptr;
    }
    return broadcast_manager;
}

const CacheConfig& KVCacheManager::cacheConfig() const {
    return config_;
}

const CacheConfig& KVCacheManager::getMTPModuleCacheConfig(int mtp_module_id) const {
    RTP_LLM_CHECK_WITH_INFO(mtp_module_id >= 0 && static_cast<size_t>(mtp_module_id) < config_.mtp_sub_configs.size(),
                            "Invalid mtp_module_id: %d, must be in range [0, %zu)",
                            mtp_module_id,
                            config_.mtp_sub_configs.size());
    RTP_LLM_CHECK_WITH_INFO(
        config_.mtp_sub_configs[mtp_module_id] != nullptr, "mtp_sub_configs[%d] is null", mtp_module_id);
    return *config_.mtp_sub_configs[mtp_module_id];
}

// 显存管理和缓存分配

MallocResult KVCacheManager::malloc(const MallocInfo& malloc_info) {
    RTP_LLM_PROFILE_FUNCTION();
    const int64_t malloc_begin_time_us = currentTimeUs();
    RTP_LLM_CHECK(malloc_info.batch_kv_cache_resource && malloc_info.complete_token_ids);

    const int  seq_size_per_block = config_.seq_size_per_block;
    const bool is_first_malloc    = !malloc_info.batch_kv_cache_resource->curBlocksNum();
    // A first malloc that failed with MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED leaves the stream
    // WAITING and re-enters here having allocated nothing, so curBlocksNum() is still zero and
    // is_first_malloc is still true. Recomputing the keys is wasted work, and reporting the
    // prefill-cache-hit metric again would double-count the request. cacheKeysInitialized() is the
    // discriminator rather than hasCacheKeys(): it is cleared by resetBatchSize() /
    // resetAndReturnOldResources(), so a batch reshape between attempts correctly forces a recompute
    // even though the surviving batch rows still hold keys. Only the attempt that actually
    // initialises the keys owns the metric.
    const bool keys_already_initialized = malloc_info.batch_kv_cache_resource->cacheKeysInitialized();
    bool       keys_initialized_now     = false;
    if (is_first_malloc) {
        if (!keys_already_initialized) {
            initCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids, seq_size_per_block);
            keys_initialized_now = true;
        }
    } else {
        updateCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids, seq_size_per_block);
    }
    reportPrefillCacheHitMetrics(malloc_info, keys_initialized_now);

    MallocResult  result             = coordinator_manager_->malloc(malloc_info);
    const int64_t malloc_end_time_us = currentTimeUs();
    result.malloc_begin_time_us      = malloc_begin_time_us;
    if (result.load_attempted) {
        result.load_prepare_latency_us = std::max<int64_t>(malloc_end_time_us - result.match_end_time_us, 0);
    }
    reportCacheOperation(metrics_reporter_, RtpLLMCacheOperationMetricsCollector::OpType::MALLOC, malloc_begin_time_us);
    return result;
}

// is_first_malloc is passed as "this call is the one that initialised the cache keys", which is the
// first malloc minus its retries. Retried init-mallocs must not re-report, or the hit rate is
// double-counted for every request that waited on cache pressure.
void KVCacheManager::reportPrefillCacheHitMetrics(const MallocInfo& malloc_info, bool is_first_malloc) {
    if (!is_first_malloc || !prefill_cache_hit_metrics_reporter_ || !malloc_info.batch_kv_cache_resource
        || !malloc_info.complete_token_ids) {
        return;
    }
    prefill_cache_hit_metrics_reporter_->record(*malloc_info.batch_kv_cache_resource,
                                                cp_slot_mapper_,
                                                malloc_info.request_id,
                                                malloc_info.complete_token_ids->seqLength(),
                                                config_.seq_size_per_block);
}

void KVCacheManager::free(const FreeInfo& free_info) {
    RTP_LLM_PROFILE_FUNCTION();
    const int64_t begin_time_us = metrics_reporter_ == nullptr ? 0 : currentTimeUs();
    RTP_LLM_CHECK(free_info.batch_kv_cache_resource && free_info.complete_token_ids);
    coordinator_manager_->free(free_info);
    reportCacheOperation(metrics_reporter_, RtpLLMCacheOperationMetricsCollector::OpType::FREE, begin_time_us);
}

bool KVCacheManager::abortPendingLoad(const std::shared_ptr<AsyncContext>& context) {
    return coordinator_manager_ != nullptr && coordinator_manager_->abortPendingLoad(context);
}

uint64_t KVCacheManager::allocationGeneration() const {
    return allocation_wait_state_->generation.load(std::memory_order_acquire);
}

bool KVCacheManager::waitForAllocationChange(uint64_t observed_generation,
                                             int64_t  timeout_ms,
                                             int64_t  minimum_wait_ms) {
    const auto                   state = allocation_wait_state_;
    std::unique_lock<std::mutex> lock(state->mutex);
    const auto backoff_ms = std::min(std::max<int64_t>(minimum_wait_ms, 0), std::max<int64_t>(timeout_ms, 0));
    if (backoff_ms > 0) {
        // A failed match can release its own temporary prefix pins. That advances
        // generation without making the unavailable suffix any easier to allocate.
        // Bound repeated self-wakeups without discarding genuine release events.
        state->cv.wait_for(lock, std::chrono::milliseconds(backoff_ms), [state] {
            return state->stopped.load(std::memory_order_acquire);
        });
    }
    if (timeout_ms > backoff_ms) {
        state->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms - backoff_ms), [state, observed_generation] {
            return state->stopped.load(std::memory_order_acquire)
                   || state->generation.load(std::memory_order_acquire) != observed_generation;
        });
    }
    return state->generation.load(std::memory_order_acquire) != observed_generation;
}

std::function<void()> KVCacheManager::allocationChangeCallback() const {
    std::weak_ptr<KVCacheAllocationWaitState> weak_state = allocation_wait_state_;
    return [weak_state]() {
        if (const auto state = weak_state.lock()) {
            notifyAllocationChangeState(state);
        }
    };
}

void KVCacheManager::insertIntoCache(const InsertInfo& insert_info, size_t& resident_prefix_length) {
    RTP_LLM_PROFILE_FUNCTION();
    const int64_t begin_time_us = metrics_reporter_ == nullptr ? 0 : currentTimeUs();
    dropLastPartialBlock(insert_info.batch_kv_cache_resource);
    coordinator_manager_->insertIntoCache(insert_info, resident_prefix_length);
    reportCacheOperation(metrics_reporter_, RtpLLMCacheOperationMetricsCollector::OpType::INSERT, begin_time_us);
}

int KVCacheManager::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                          int                            seq_len,
                                          int                            reserve_step) const {
    RTP_LLM_CHECK_WITH_INFO(coordinator_manager_ != nullptr,
                            "singleBatchNeedBlocks called before KVCacheManager initialized");
    return coordinator_manager_->singleBatchNeedBlocks(batch_kv_cache_resource, seq_len, reserve_step);
}

int KVCacheManager::estimatePeakNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                           int                            seq_len,
                                           int                            common_seq_len,
                                           int                            remaining_tokens,
                                           int                            reserve_step,
                                           bool                           enable_reuse_cache,
                                           int                            target_batch_size) const {
    return coordinator_manager_->estimateBatchPeakNeedBlocks(batch_kv_cache_resource,
                                                             seq_len,
                                                             common_seq_len,
                                                             remaining_tokens,
                                                             reserve_step,
                                                             enable_reuse_cache,
                                                             target_batch_size);
}

// 块操作相关

void KVCacheManager::blockCopy(int src_block_index, int dest_block_index) {
    return coordinator_manager_->blockCopy(src_block_index, dest_block_index);
}

void KVCacheManager::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    return coordinator_manager_->blockBatchCopy(copy_mapping);
}

void KVCacheManager::blockBatchCopy(const torch::Tensor& copy_mapping) {
    return coordinator_manager_->blockBatchCopy(copy_mapping);
}

void KVCacheManager::blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end) {
    return coordinator_manager_->blockBatchCopy(copy_mapping_begin, copy_mapping_end);
}

void KVCacheManager::blockBatchCopyByGroup(const std::vector<TaggedBlockIdPair>& copy_mapping) {
    return coordinator_manager_->blockBatchCopyByGroup(copy_mapping);
}

bool KVCacheManager::updateKVBlock(const BatchKVCacheResourcePtr&  batch_kv_cache_resource,
                                   const std::vector<int>&         block_src_batch,
                                   bool                            copy_last_block,
                                   std::vector<TaggedBlockIdPair>& block_update_mapping) {
    RTP_LLM_PROFILE_FUNCTION();
    const bool updated = coordinator_manager_->updateKVBlock(
        batch_kv_cache_resource, block_src_batch, copy_last_block, block_update_mapping);
    return updated;
}

// 地址转换和缓冲区访问

BlockAddrInfo KVCacheManager::convertIndexToAddr(int block_index, int layer_id) const {
    return coordinator_manager_->convertIndexToAddr(layer_id, block_index);
}

std::vector<BlockInfo> KVCacheManager::convertIndexToBuffer(int block_index, int layer_id) const {
    return coordinator_manager_->convertIndexToBuffer(layer_id, block_index);
}

std::vector<BlockInfo>
KVCacheManager::convertIndexToBuffer(int block_index, int layer_id, int partition_count, int partition_id) const {
    return coordinator_manager_->convertIndexToBuffer(layer_id, block_index, partition_count, partition_id);
}

BlockAddrInfo KVCacheManager::convertIndexToAddr(int layer_id, const std::string& group_tag, int block_id) const {
    return coordinator_manager_->convertIndexToAddr(layer_id, group_tag, block_id);
}

std::vector<BlockInfo>
KVCacheManager::convertIndexToBuffer(int layer_id, const std::string& group_tag, int block_id) const {
    return coordinator_manager_->convertIndexToBuffer(layer_id, group_tag, block_id);
}

std::vector<BlockInfo> KVCacheManager::convertIndexToBuffer(
    int layer_id, const std::string& group_tag, int block_id, int partition_count, int partition_id) const {
    return coordinator_manager_->convertIndexToBuffer(layer_id, group_tag, block_id, partition_count, partition_id);
}

GroupedCacheLayerLayout KVCacheManager::allLayerCacheBase() const {
    return coordinator_manager_->allLayerCacheBase();
}

GroupedCacheLayerLayout KVCacheManager::getMainModelGroupedCacheLayerLayout() const {
    const auto          all_layout = coordinator_manager_->allLayerCacheBase();
    std::vector<size_t> global_layer_ids(config_.layer_num);
    std::iota(global_layer_ids.begin(), global_layer_ids.end(), 0);
    auto main_topology = projectTopology(all_layout.topology(), global_layer_ids);
    return projectLayout(all_layout, std::move(main_topology), global_layer_ids);
}

GroupedCacheLayerLayout KVCacheManager::getMTPModuleGroupedCacheLayerLayout(int mtp_module_id) const {
    RTP_LLM_CHECK_WITH_INFO(mtp_module_id >= 0 && static_cast<size_t>(mtp_module_id) < config_.mtp_sub_configs.size(),
                            "Invalid mtp_module_id: %d, must be in range [0, %zu)",
                            mtp_module_id,
                            config_.mtp_sub_configs.size());

    const auto& mtp_sub_config = config_.mtp_sub_configs[mtp_module_id];
    RTP_LLM_CHECK_WITH_INFO(mtp_sub_config != nullptr, "mtp_sub_configs[%d] is null", mtp_module_id);
    const uint32_t      mtp_layer_num = mtp_sub_config->layer_num;
    std::vector<size_t> global_layer_ids;
    global_layer_ids.reserve(mtp_layer_num);
    for (uint32_t local_layer_id = 0; local_layer_id < mtp_layer_num; ++local_layer_id) {
        const auto global_layer_id = CacheConfig::mtpGlobalLayerId(
            config_.layer_num, mtp_module_id, mtp_layer_num, static_cast<int>(local_layer_id));
        RTP_LLM_CHECK_WITH_INFO(global_layer_id != std::numeric_limits<uint32_t>::max(),
                                "invalid MTP global layer: main=%u module=%d module_layers=%u local=%u",
                                config_.layer_num,
                                mtp_module_id,
                                mtp_layer_num,
                                local_layer_id);
        global_layer_ids.push_back(global_layer_id);
    }
    return projectLayout(coordinator_manager_->allLayerCacheBase(), mtp_sub_config->topologyPtr(), global_layer_ids);
}

// 资源统计和信息查询

size_t KVCacheManager::freeBlocksNum() const {
    return coordinator_manager_->freeBlocksNum();
}

size_t KVCacheManager::availableBlocksNum() const {
    return coordinator_manager_->availableBlocksNum();
}

size_t KVCacheManager::reserveBlocksNum() const {
    return coordinator_manager_->reserveBlocksNum();
}

size_t KVCacheManager::availableTokensNum() const {
    return coordinator_manager_->availableTokensNum();
}

size_t KVCacheManager::totalBlocksNum() const {
    return coordinator_manager_->totalBlocksNum();
}

size_t KVCacheManager::maxAvailableTokensNum() const {
    return coordinator_manager_->maxAvailableTokensNum();
}

KVCacheInfo KVCacheManager::getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const {
    if (need_cache_keys && cacheStatusSnapshotEnabled()) {
        std::shared_ptr<const KVCacheInfo> snapshot;
        {
            std::lock_guard<std::mutex> lock(cache_status_snapshot_mutex_);
            snapshot = cache_status_snapshot_;
        }
        if (snapshot) {
            return *snapshot;
        }
    }
    return buildKVCacheInfo(latest_version, need_cache_keys);
}

void KVCacheManager::refreshKVCacheInfoSnapshot() {
    if (!coordinator_manager_ || !cacheStatusSnapshotEnabled()) {
        return;
    }
    auto snapshot = std::make_shared<KVCacheInfo>(buildKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/true));
    std::lock_guard<std::mutex> lock(cache_status_snapshot_mutex_);
    cache_status_snapshot_ = std::move(snapshot);
}

KVCacheInfo KVCacheManager::buildKVCacheInfo(int64_t latest_version, bool need_cache_keys) const {
    KVCacheInfo info;
    info.version = latest_version;

    if (!coordinator_manager_) {
        RTP_LLM_LOG_ERROR("getKVCacheInfo called before KVCacheManager initialized");
        return info;
    }

    if (need_cache_keys && block_tree_cache_) {
        BlockTreeKeySnapshot snapshot = block_tree_cache_->getKeySnapshot();
        info.version                  = snapshot.version;
        info.cached_keys              = std::move(snapshot.keys);
    }

    const size_t block_size_tokens = cp_slot_mapper_ && cp_slot_mapper_->isSharded() ?
                                         cp_slot_mapper_->virtualBlockSize() :
                                         config_.seq_size_per_block;

    const auto capacity     = coordinator_manager_->tokenCapacity(block_size_tokens);
    info.block_size         = block_size_tokens;
    info.total_kv_cache     = capacity.total_tokens;
    info.available_kv_cache = capacity.available_tokens;

    return info;
}

// 系统资源管理

void KVCacheManager::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    coordinator_manager_->regUserMr(model_id, std::move(cache_store));
}

void KVCacheManager::setCacheStore(std::shared_ptr<CacheStore> cache_store) {
    std::lock_guard<std::mutex> lock(cache_store_mutex_);
    cache_store_ = std::move(cache_store);
}

std::shared_ptr<CacheStore> KVCacheManager::getCacheStore() const {
    std::lock_guard<std::mutex> lock(cache_store_mutex_);
    return cache_store_;
}

// PD separation: increment KV cache reference count
std::shared_ptr<KVCacheResource>
KVCacheManager::incrKVCacheRef(const KVCacheResource& resource, const CacheKeysType& cache_keys, bool is_connector) {
    return coordinator_manager_->incrKVCacheRef(resource, cache_keys, is_connector);
}

bool KVCacheManager::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    if (request.has_remote_request()) {
#ifdef RTP_LLM_USE_REMOTE_KV_CACHE
        if (!block_tree_cache_) {
            RTP_LLM_LOG_WARNING("KVCacheManager::executeFunction: block tree cache is not initialized");
            return false;
        }
        auto backend = std::dynamic_pointer_cast<KVCMStorageBackend>(block_tree_cache_->storageBackend());
        if (!backend) {
            RTP_LLM_LOG_WARNING("KVCacheManager::executeFunction: KVCM storage backend is not initialized");
            return false;
        }
        return backend->execute(request.remote_request(), *response.mutable_remote_response());
#else
        RTP_LLM_LOG_WARNING("KVCacheManager::executeFunction: KVCM support is not compiled in");
        return false;
#endif
    }
    if (!request.has_mem_request()) {
        RTP_LLM_LOG_WARNING("KVCacheManager::executeFunction: unsupported request type");
        return false;
    }
    if (!block_tree_cache_) {
        RTP_LLM_LOG_WARNING("KVCacheManager::executeFunction: block tree cache is not initialized");
        return false;
    }

    MemoryOperationResponsePB* memory_response = response.mutable_mem_response();
    memory_response->set_code(MemoryOperationResponsePB::FAILED);
    std::vector<TransferDescriptor> descriptors;
    if (!BlockTransferRequestConverter::decodeTransfer(
            request.mem_request(), descriptors, block_tree_cache_->groupSets())) {
        RTP_LLM_LOG_WARNING("KVCacheManager::executeFunction: invalid grouped transfer request");
        return true;
    }

    const int64_t timeout_ms = request.mem_request().timeout_ms();
    if (timeout_ms > std::numeric_limits<int>::max()) {
        RTP_LLM_LOG_WARNING("KVCacheManager::executeFunction: transfer timeout exceeds supported range");
        return true;
    }
    const auto timeout =
        timeout_ms > 0 ? std::chrono::milliseconds(timeout_ms) : BlockTreeTaskPool::kDefaultQueueWaitTimeout;
    const bool transfer_success = block_tree_cache_->executeTransfer(TransferTask(std::move(descriptors), timeout));
    if (!transfer_success) {
        RTP_LLM_LOG_WARNING("KVCacheManager::executeFunction: grouped transfer failed");
        return true;
    }
    memory_response->set_code(MemoryOperationResponsePB::OK);
    return true;
}

bool KVCacheManager::hasTailSparseReuseGroup() const {
    const auto& groups = config_.groups();
    return std::any_of(groups.begin(), groups.end(), [](const GroupBase& group) {
        return group.policy.enable_prefix_reuse && group.policy.active_tail_blocks != 0;
    });
}

void KVCacheManager::initCacheEventPublisher() {
    try {
        const auto& publisher_type = kv_cache_config_.kv_cache_event_publisher_type;
        if (warmup_ || publisher_type.empty() || publisher_type == "none") {
            return;
        }
        if (publisher_type != "kvcm") {
            RTP_LLM_LOG_WARNING("unknown KV cache event publisher type=%s; publisher disabled", publisher_type.c_str());
            return;
        }
        if (!kv_cache_config_.reuse_cache || !kv_cache_config_.enable_device_cache) {
            RTP_LLM_LOG_WARNING("KV cache event publisher disabled because device cache reuse is disabled, type=%s "
                                "reuse_cache=%d enable_device_cache=%d",
                                publisher_type.c_str(),
                                kv_cache_config_.reuse_cache,
                                kv_cache_config_.enable_device_cache);
            return;
        }
        if (parallelism_config_.pp_size != 1 || parallelism_config_.tp_rank != 0) {
            RTP_LLM_LOG_WARNING("KV cache event publisher requires pp_size=1 and tp_rank=0, pp_size=%lld tp_rank=%lld",
                                static_cast<long long>(parallelism_config_.pp_size),
                                static_cast<long long>(parallelism_config_.tp_rank));
            return;
        }
        if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
            RTP_LLM_LOG_WARNING("KV cache event publisher disabled for CP-sharded KV cache");
            return;
        }

        // KVCM currently represents one complete prefix chain per key.  A
        // tail-sparse reuse group is still required by local reuse, but cannot
        // be represented in that contract; publishing only the FULL groups
        // would advertise keys that the local cache cannot actually reuse.
        if (hasTailSparseReuseGroup()) {
            RTP_LLM_LOG_WARNING("KV cache event publisher disabled because tail-sparse reuse groups are unsupported");
            return;
        }
        std::vector<int64_t>     group_block_size_bytes;
        std::vector<std::string> reuse_group_tags;
        for (const auto& group : config_.topology().groups()) {
            if (!cacheGroupPublishesPrefixChain(group.policy)) {
                continue;
            }
            if (group.policy.memory_placement != CacheMemoryPlacement::DEVICE) {
                RTP_LLM_LOG_WARNING(
                    "KV cache event publisher disabled because publishing non-DEVICE cache groups is unsupported");
                return;
            }
            reuse_group_tags.push_back(group.tag);
            group_block_size_bytes.push_back(static_cast<int64_t>(config_.blockSizeBytesForGroup(group.tag)));
        }
        if (reuse_group_tags.empty()) {
            RTP_LLM_LOG_ERROR("KV cache event publisher disabled because no cache group participates in prefix reuse");
            return;
        }

        if (!block_tree_cache_) {
            RTP_LLM_LOG_WARNING("KV cache event publisher disabled because BlockTreeCache is unavailable");
            cache_event_publisher_.reset();
            return;
        }

        KVCacheEventPublisherConfig publisher_config;
        publisher_config.manager_endpoint = kv_cache_config_.kv_cache_event_manager_endpoint;

        KVCacheEventPublisherContext publisher_context;
        publisher_context.instance_group = resolveKVCacheEventInstanceGroup(
            kv_cache_config_.kv_cache_event_instance_group, kv_cache_config_.kvcm_instance_group);
        publisher_context.instance_id       = kv_cache_config_.kv_cache_event_instance_id;
        publisher_context.host_ip_port      = kv_cache_config_.kv_cache_event_host_ip_port;
        publisher_context.model_name        = runtime_config_.model_name;
        publisher_context.dtype             = getDataTypeStr(config_.dtype);
        publisher_context.spec_name         = "rtp_llm_hbm_" + std::to_string(config_.seq_size_per_block);
        publisher_context.location_uri      = "rtp-llm://" + publisher_context.host_ip_port + "/hbm";
        publisher_context.block_size_tokens = static_cast<int32_t>(config_.seq_size_per_block);
        // Pipeline parallelism is rejected above because a unique PP owner is
        // not represented in ParallelismConfig yet.
        publisher_context.spec_size_bytes =
            aggregateKVCacheEventSpecSizeBytes(group_block_size_bytes, parallelism_config_.tp_size);
        publisher_context.tp_size = static_cast<int32_t>(parallelism_config_.tp_size);
        publisher_context.dp_size = static_cast<int32_t>(parallelism_config_.dp_size);
        publisher_context.pp_size = static_cast<int32_t>(parallelism_config_.pp_size);
        publisher_context.dp_rank = static_cast<int32_t>(parallelism_config_.dp_rank);
        publisher_context.use_mla = config_.use_mla;

        std::weak_ptr<BlockTreeCache> weak_shared_cache = block_tree_cache_;
        auto                          snapshot_provider = [weak_shared_cache]() {
            const auto shared_cache = weak_shared_cache.lock();
            if (!shared_cache) {
                throw std::runtime_error("BlockTreeCache is no longer available");
            }
            return shared_cache->logicalCacheSnapshot();
        };

        cache_event_publisher_ =
            std::make_shared<KVCMPublisher>(publisher_config, publisher_context, std::move(snapshot_provider));
        block_tree_cache_->setEventPublisher(cache_event_publisher_, reuse_group_tags);
        if (!cache_event_publisher_->start()) {
            RTP_LLM_LOG_WARNING("KV cache event publisher failed to start, type=%s; inference remains enabled",
                                publisher_type.c_str());
            stopCacheEventPublisher();
            cache_event_publisher_.reset();
            return;
        }

        RTP_LLM_LOG_INFO("KV cache event publisher started, type=%s instance_id=%s host=%s pp_size=%lld tp_rank=%lld "
                         "dp_rank=%lld",
                         publisher_type.c_str(),
                         publisher_context.instance_id.c_str(),
                         publisher_context.host_ip_port.c_str(),
                         static_cast<long long>(parallelism_config_.pp_size),
                         static_cast<long long>(parallelism_config_.tp_rank),
                         static_cast<long long>(parallelism_config_.dp_rank));
    } catch (const std::exception& e) {
        stopCacheEventPublisher();
        cache_event_publisher_.reset();
        RTP_LLM_LOG_WARNING("KV cache event publisher initialization failed; inference remains enabled: %s", e.what());
    } catch (...) {
        stopCacheEventPublisher();
        cache_event_publisher_.reset();
        RTP_LLM_LOG_WARNING(
            "KV cache event publisher initialization failed with unknown error; inference remains enabled");
    }
}

void KVCacheManager::stopCacheEventPublisher() {
    if (cache_event_publisher_) {
        cache_event_publisher_->stop();
    }
    if (block_tree_cache_) {
        block_tree_cache_->setEventPublisher(nullptr, {});
    }
    cache_event_publisher_.reset();
}

void KVCacheManager::recordCacheHitTokens(int64_t input_length, const RtpLLMCacheReuseMetricsCollector& metrics) {
    std::lock_guard<std::mutex> lock(cache_hit_mutex_);
    cache_hit_input_tokens_ += input_length;
    cache_hit_reuse_tokens_ += metrics.kv_cache_reuse_length;
    cache_hit_device_tokens_ += metrics.device_reuse_length;
    cache_hit_host_tokens_ += metrics.host_reuse_length;
    cache_hit_disk_tokens_ += metrics.disk_reuse_length;
}

bool KVCacheManager::collectCacheHitRates(std::chrono::steady_clock::time_point now,
                                          RtpLLMCacheReuseMetricsCollector&     metrics) {
    std::lock_guard<std::mutex> lock(cache_hit_mutex_);
    metrics.report_hit_rates = false;
    if (now - cache_hit_window_start_ < std::chrono::minutes(1)) {
        return false;
    }
    cache_hit_window_start_ = now;
    if (cache_hit_input_tokens_ > 0) {
        const double scale        = 100.0 / cache_hit_input_tokens_;
        metrics.kv_cache_hit_rate = static_cast<float>(cache_hit_reuse_tokens_ * scale);
        metrics.device_hit_rate   = static_cast<float>(cache_hit_device_tokens_ * scale);
        metrics.host_hit_rate     = static_cast<float>(cache_hit_host_tokens_ * scale);
        metrics.disk_hit_rate     = static_cast<float>(cache_hit_disk_tokens_ * scale);
        metrics.report_hit_rates  = true;
    }
    cache_hit_input_tokens_  = 0;
    cache_hit_reuse_tokens_  = 0;
    cache_hit_device_tokens_ = 0;
    cache_hit_host_tokens_   = 0;
    cache_hit_disk_tokens_   = 0;
    return metrics.report_hit_rates;
}

void KVCacheManager::reportMetricsLoop() {
    RTP_LLM_PROFILE_FUNCTION();
    kmonitor::MetricsTags tags;
    constexpr auto        kLogInterval  = std::chrono::minutes(1);
    auto                  last_log_time = std::chrono::steady_clock::now() - kLogInterval;
    while (!stop_.load(std::memory_order_acquire)) {
        if (!metrics_reporter_ || !coordinator_manager_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        RtpLLMCacheMetricsCollector global_metrics = collectGlobalCacheMetrics(coordinator_manager_);
        metrics_reporter_->report<RtpLLMCacheMetrics, RtpLLMCacheMetricsCollector>(&tags, &global_metrics);

        RtpLLMCacheReuseMetricsCollector hit_metrics;
        if (collectCacheHitRates(std::chrono::steady_clock::now(), hit_metrics)) {
            metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(&tags, &hit_metrics);
        }

        const auto now        = std::chrono::steady_clock::now();
        const bool should_log = (now - last_log_time) >= kLogInterval;
        if (should_log) {
            last_log_time = now;
            logGlobalCacheMetrics(global_metrics);
        }

        block_tree_cache_->reportMetrics();
        const std::vector<BlockTreePoolMetricsSnapshot> tree_pool_snapshots = block_tree_cache_->poolMetricsSnapshots();
        const std::vector<KVCachePoolMetricsSnapshot>   device_pool_snapshots =
            coordinator_manager_->poolMetricsSnapshots();
        const std::vector<CachePoolMetricsSnapshot> report_snapshots =
            mergeCachePoolMetricsSnapshots(device_pool_snapshots, tree_pool_snapshots);
        for (const CachePoolMetricsSnapshot& report_snapshot : report_snapshots) {
            reportPoolCacheMetrics(metrics_reporter_, report_snapshot, should_log);
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));  // 1s
    }
}

}  // namespace rtp_llm
