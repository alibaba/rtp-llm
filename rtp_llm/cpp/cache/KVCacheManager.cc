#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"
#include "rtp_llm/cpp/cache/events/KVCMPublisher.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <chrono>
#include <limits>
#include <numeric>
#include <optional>
#include <pthread.h>
#include <shared_mutex>
#include <sstream>
#include <unordered_set>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/PrefillCacheHitMetricsReporter.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#ifdef RTP_LLM_USE_REMOTE_KV_CACHE
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMStorageBackend.h"
#endif
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverterImpl.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/PipelineLayoutCodec.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageCompletionAdmission.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/metrics/CachePoolMetrics.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/metrics/RpcFailureMetrics.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

class KVCacheAllocationWaitState {
public:
    std::atomic<uint64_t>   generation{0};
    std::atomic<bool>       stopped{false};
    std::mutex              mutex;
    std::condition_variable cv;
};

// Storage work never owns the KVCacheManager. Only an active page scan borrows
// it; stopping the manager invalidates that borrow before tearing down pools.
class DSparkPublicationManagerAccess {
public:
    explicit DSparkPublicationManagerAccess(const KVCacheManager* manager): manager(manager) {}
    std::shared_mutex mutex;
    const KVCacheManager* manager;
};

namespace {

using DecodeFrozenPages = std::unordered_map<DeviceBlockPool*, std::unordered_set<BlockIdxType>>;

void freezeDecodePage(DecodeFrozenPages& frozen, const DeviceBlockPoolPtr& pool, BlockIdxType block) {
    // Ordinary request/endpoint pins still own these allocations. Failure to
    // acquire native custody must not unwind and release uncovered pages.
    if (pool && pool->isAllocated(block) && frozen[pool.get()].insert(block).second)
        pool->freezeBlock(block);
}

void freezeDecodeEndpointPages(DecodeFrozenPages& frozen,
                               const std::shared_ptr<const DecodeEndpoint>& endpoint) {
    if (!endpoint) return;
    for (const auto& [tag, pages] : endpoint->pages)
        for (const auto& page : pages)
            if (page && page->pin) freezeDecodePage(frozen, page->pool, page->block);
    for (const auto& entry : endpoint->full_prefixes) {
        for (auto segment = entry.second.get(); segment; segment = segment->previous.get())
            for (const auto& page : segment->suffix)
                freezeDecodePage(frozen, page->pool, page->block);
    }
}

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

std::string cacheEventSpecName(int32_t block_size_tokens, const std::string& tag, bool legacy_dsv41_specs) {
    if (tag == "global_kv_1" || (legacy_dsv41_specs && tag == "hca_kv")) {
        return "F0";
    }
    if (tag == "global_kv_2" || (legacy_dsv41_specs && tag == "csa_kv")) {
        return "F1";
    }
    if (tag == "indexer_kv") {
        return "F2";
    }
    return "rtp_llm_hbm_" + std::to_string(block_size_tokens) + "_" + tag;
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

std::string frozenBlockSummary(const std::vector<const IBlockPool*>& pools) {
    size_t             total = 0;
    std::ostringstream entries;
    for (size_t i = 0; i < pools.size(); ++i) {
        const auto*  pool  = pools[i];
        const size_t count = pool->referencedBlocksNum(BlockTreeRefType::FROZEN);
        total += count;
        entries << (i == 0 ? "" : ",") << pool->poolName() << "_" << i << ":" << count;
    }
    return "{total:" + std::to_string(total) + ",pools:{" + entries.str() + "}}";
}

void logFrozenCacheBlocks(const BlockTreeCache& cache) {
    std::array<std::vector<const IBlockPool*>, 3> pools;
    const auto                                    add_pool = [&pools](size_t tier, const IBlockPool* pool) {
        auto& tier_pools = pools[tier];
        if (pool && std::find(tier_pools.begin(), tier_pools.end(), pool) == tier_pools.end()) {
            tier_pools.push_back(pool);
        }
    };
    for (const auto& group_set : cache.groupSets()) {
        for (const auto& pool : group_set->devicePools()) {
            add_pool(0, pool.get());
        }
        add_pool(1, group_set->hostPool().get());
        add_pool(2, group_set->diskPool().get());
    }
    const auto device = frozenBlockSummary(pools[0]);
    const auto host   = frozenBlockSummary(pools[1]);
    const auto disk   = frozenBlockSummary(pools[2]);
    RTP_LLM_LOG_INFO("btc_leaked_blocks L1=%s L2=%s L3=%s", device.c_str(), host.c_str(), disk.c_str());
}

void reportPoolCacheMetrics(const kmonitor::MetricsReporterPtr& metrics_reporter,
                            const CachePoolMetrics&             pool_metrics,
                            bool                                report_frozen_ref_blocks) {
    RtpLLMCachePoolMetricsCollector pool_collector;
    pool_collector.block_size_bytes              = static_cast<int64_t>(pool_metrics.block_size_bytes);
    pool_collector.free_blocks                   = static_cast<int64_t>(pool_metrics.free_blocks);
    pool_collector.used_blocks                   = static_cast<int64_t>(pool_metrics.used_blocks);
    pool_collector.available_blocks              = static_cast<int64_t>(pool_metrics.available_blocks);
    pool_collector.active_blocks                 = static_cast<int64_t>(pool_metrics.active_blocks);
    pool_collector.total_blocks                  = static_cast<int64_t>(pool_metrics.total_blocks);
    pool_collector.reserve_blocks                = static_cast<int64_t>(pool_metrics.reserve_blocks);
    pool_collector.request_ref_blocks            = static_cast<int64_t>(pool_metrics.request_ref_blocks);
    pool_collector.remote_connector_ref_blocks   = static_cast<int64_t>(pool_metrics.remote_connector_ref_blocks);
    pool_collector.pd_connector_ref_blocks       = static_cast<int64_t>(pool_metrics.pd_connector_ref_blocks);
    pool_collector.block_cache_ref_blocks        = static_cast<int64_t>(pool_metrics.block_cache_ref_blocks);
    pool_collector.load_ref_blocks               = static_cast<int64_t>(pool_metrics.load_ref_blocks);
    pool_collector.eviction_ref_blocks           = static_cast<int64_t>(pool_metrics.eviction_ref_blocks);
    pool_collector.frozen_ref_blocks             = static_cast<int64_t>(pool_metrics.frozen_ref_blocks);
    pool_collector.report_frozen_ref_blocks      = report_frozen_ref_blocks;
    pool_collector.used_ratio                    = pool_metrics.used_ratio;
    pool_collector.read_bytes                    = pool_metrics.read_bytes;
    pool_collector.write_bytes                   = pool_metrics.write_bytes;
    pool_collector.read_bandwidth_bytes_per_sec  = pool_metrics.read_bandwidth_bytes_per_sec;
    pool_collector.write_bandwidth_bytes_per_sec = pool_metrics.write_bandwidth_bytes_per_sec;

    kmonitor::MetricsTags pool_tags("pool_name", pool_metrics.pool_name);
    pool_tags.AddTag("tier", tierName(pool_metrics.tier));
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

struct DSparkPublicationWork {
    struct Row {
        DeviceBlockPoolPtr pool;
        const BlockIndicesType* blocks;
    };
    // Destroy last so the budget covers both canceled REQUEST references and
    // any endpoint pins still owned by resource after planning fails.
    std::shared_ptr<StorageCompletionAdmission> admission;
    BatchKVCacheResourcePtr resource;
    std::vector<Row> rows;
    std::shared_ptr<KVCacheAllocationWaitState> allocation_wait_state;
    kmonitor::MetricsReporterPtr metrics_reporter;
    std::function<void()> cache_capacity_changed;
    std::atomic<bool> owns_request_pages{false};

    ~DSparkPublicationWork() {
        releaseRequestPages();
    }

    void releaseRequestPages() noexcept {
        if (!owns_request_pages.exchange(false)) return;
        const int64_t begin_time_us = metrics_reporter == nullptr ? 0 : currentTimeUs();
        bool capacity_grew = false;
        for (const auto& row : rows) {
            capacity_grew |= row.pool->releaseDeferredRequestReferences(*row.blocks);
        }
        if (capacity_grew) {
            notifyAllocationChangeState(allocation_wait_state);
            try { if (cache_capacity_changed) cache_capacity_changed(); } catch (...) {}
        }
        try {
            reportCacheOperation(metrics_reporter, RtpLLMCacheOperationMetricsCollector::OpType::FREE, begin_time_us);
        } catch (...) {
            // Metrics must not interfere with retiring the detached references.
        }
    }
};

}  // namespace

KVCacheManager::KVCacheManager(const CacheConfig&                 config,
                               bool                               warmup,
                               const kmonitor::MetricsReporterPtr metrics_reporter,
                               const KVCacheConfig&               kv_cache_config,
                               const ParallelismConfig&           parallelism_config,
                               const RuntimeConfig&               runtime_config,
                               const SpeculativeExecutionConfig&  sp_config,
                               const PDSepConfig&                 pd_sep_config,
                               const CacheStoreConfig&            cache_store_config,
                               bool                               use_device_malloc_block_pool,
                               BlockTreeCacheLayoutConstPtr       block_tree_layout):
    config_(config),
    metrics_reporter_(metrics_reporter),
    kv_cache_config_(kv_cache_config),
    parallelism_config_(parallelism_config),
    runtime_config_(runtime_config),
    sp_config_(sp_config),
    pd_sep_config_(pd_sep_config),
    cache_store_config_(cache_store_config),
    use_device_malloc_block_pool_(use_device_malloc_block_pool),
    warmup_(warmup),
    block_tree_layout_(std::move(block_tree_layout)),
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

    // Allocation, cache-store projection and PD loading share this local mapper.
    // The upstream Prefill shard count does not determine Decode's local layout.
    if (config_.local_kv_cache_sharded && parallelism_config_.tp_size > 1) {
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
        prefill_cache_hit_metrics_reporter_ = std::make_unique<PrefillCacheHitMetricsReporter>(metrics_reporter_);
    }

    RTP_LLM_LOG_INFO("cache config: layer_num=%d, block_size=%dB, seq_size_per_block=%zu",
                     config_.layer_num,
                     config_.totalGroupBlockSizeBytes(),
                     config_.seq_size_per_block);
}

KVCacheManager::~KVCacheManager() {
    // Wait only for scans already using the allocator. Queued storage work
    // releases its own pages even after the manager can no longer be borrowed.
    if (dspark_publication_access_) {
        std::unique_lock lock(dspark_publication_access_->mutex);
        dspark_publication_access_->manager = nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(allocation_wait_state_->mutex);
        allocation_wait_state_->stopped.store(true, std::memory_order_release);
    }
    allocation_wait_state_->cv.notify_all();
    stopP2PConnector();
    stopMetricsReporter();
    stopCacheEventPublisher();
}

void KVCacheManager::stopMetricsReporter() {
    stop_.store(true, std::memory_order_release);
    if (metrics_reporter_thread_.joinable()) {
        metrics_reporter_thread_.join();
    }
}

void KVCacheManager::stopP2PConnector() {
    if (p2p_connector_) {
        p2p_connector_->stop();
    }
}

// 初始化和配置相关

bool KVCacheManager::init() {
    RTP_LLM_CHECK_WITH_INFO(!coordinator_manager_ && !block_tree_cache_ && !metrics_reporter_thread_.joinable(),
                            "KVCacheManager::init called more than once");
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() > 0, "cache specs must not be empty");
    if (kv_cache_config_.enable_remote_cache && parallelism_config_.pp_size > 1) {
        RTP_LLM_LOG_ERROR("KVCM remote cache does not support pipeline parallelism (pp_size=%ld)",
                          parallelism_config_.pp_size);
        return false;
    }
    if (kv_cache_config_.block_tree_frozen_force_release_timeout_ms < 0
        || kv_cache_config_.block_tree_frozen_force_release_timeout_ms > std::numeric_limits<int>::max()) {
        RTP_LLM_LOG_ERROR("frozen force release timeout must be in [0, INT_MAX]");
        return false;
    }
    if (kv_cache_config_.enable_remote_cache && kv_cache_config_.kvcm_asyncwrapper_thread_num == 0) {
        RTP_LLM_LOG_ERROR("remote cache executor thread count must be positive");
        return false;
    }

    coordinator_manager_ = std::make_shared<CoordinatorCacheManager>(config_,
                                                                     AllocationType::DEVICE,
                                                                     metrics_reporter_,
                                                                     kv_cache_config_.reserve_block_ratio,
                                                                     pd_sep_config_.role_type);

    if (use_device_malloc_block_pool_) {
        RTP_LLM_LOG_INFO("use raw device malloc KV cache block-pool backing");
        coordinator_manager_->setUseDeviceMallocBlockPool(true);
    }

    coordinator_manager_->setCPSlotMapper(cp_slot_mapper_);
    coordinator_manager_->setSkipRemoteReusedWrite(pd_sep_config_.role_type == RoleType::PREFILL);
    RTP_LLM_CHECK_WITH_INFO(coordinator_manager_->init(), "CoordinatorCacheManager init failed");
    initLinearReplayPool();
    // Observe real pool capacity, including asynchronous eviction and lease release.
    const auto capacity_changed = allocationChangeCallback();
    for (const auto& pool : coordinator_manager_->groupBlockPools()) {
        pool->setCapacityChangeCallback(capacity_changed);
    }
    const bool is_first_stage_root        = parallelism_config_.pp_rank == 0 && parallelism_config_.tp_rank == 0;
    const bool requires_broadcast_manager = is_first_stage_root
                                            && (parallelism_config_.pp_size > 1 || parallelism_config_.tp_size > 1)
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
        const bool decode_publication =
            parallelism_config_.role_type == RoleType::DECODE && remoteCacheDecodePublicationEnabled();
        // Preserve the main branch's registration for existing read/write
        // traffic. A publishing Decode endpoint can explicitly select staging.
        const bool use_gdr_registration = !decode_publication || remoteCacheGdrEnabled();
        storage_backend = std::make_shared<KVCMStorageBackend>(config_,
                                                               kv_cache_config_,
                                                               runtime_config_,
                                                               parallelism_config_,
                                                               sp_config_,
                                                               broadcast_manager,
                                                               use_gdr_registration,
                                                               metrics_reporter_);
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
                                             metrics_reporter_,
                                             block_tree_layout_);
    if (!block_tree_cache_) {
        RTP_LLM_LOG_ERROR("KVCacheManager::init: failed to create BlockTreeCache");
        return false;
    }
    coordinator_manager_->attachBlockTreeCache(block_tree_cache_);
#ifdef RTP_LLM_USE_REMOTE_KV_CACHE
    if (kv_cache_config_.enable_remote_cache && remoteCacheDecodePublicationEnabled()
        && pd_sep_config_.role_type == RoleType::DECODE && isAllocatorOwner() && sp_config_.type == SP_TYPE_DSPARK) {
        initDSparkPublicationHandoff();
    }
#endif
    const auto notify_cache = block_tree_cache_->capacityChangeCallback();
    for (const auto& pool : coordinator_manager_->groupBlockPools()) {
        pool->setCapacityChangeCallback([capacity_changed, notify_cache] {
            capacity_changed();
            notify_cache();
        });
    }
    initCacheEventPublisher();

    if (!initP2PConnector()) {
        return false;
    }

    if (metrics_reporter_) {
        stop_.store(false, std::memory_order_relaxed);
        metrics_reporter_thread_ = std::thread(&KVCacheManager::reportMetricsLoop, this);
    }

    return true;
}

void KVCacheManager::initDSparkPublicationHandoff() {
    for (const auto& pool : coordinator_manager_->groupBlockPools()) {
        pool->prepareDeferredRequestRelease();
    }
    dspark_publication_access_ = std::make_shared<DSparkPublicationManagerAccess>(this);
}

std::shared_ptr<BroadcastManager> KVCacheManager::createMultiRankBlockTransferManager() const {
    const size_t tp_size               = static_cast<size_t>(parallelism_config_.tp_size);
    const size_t expected_worker_count = static_cast<size_t>(parallelism_config_.pp_size) * tp_size;
    if (runtime_config_.worker_grpc_addrs.size() != expected_worker_count) {
        RTP_LLM_LOG_ERROR("KVCacheManager: worker grpc address count mismatch, expected=%zu, actual=%zu",
                          expected_worker_count,
                          runtime_config_.worker_grpc_addrs.size());
        return nullptr;
    }

    /** Worker addresses already select this DP lane and are ordered by PP stage, then TP rank. */
    auto broadcast_manager = std::make_shared<BroadcastManager>(runtime_config_.worker_grpc_addrs,
                                                                makeConnectFailureReporter(metrics_reporter_));
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
            initCacheKeys(malloc_info.batch_kv_cache_resource,
                          malloc_info.complete_token_ids,
                          seq_size_per_block,
                          config_.cache_key_hash_seed);
            keys_initialized_now = true;
        }
    } else {
        updateCacheKeys(malloc_info.batch_kv_cache_resource,
                        malloc_info.complete_token_ids,
                        seq_size_per_block,
                        config_.cache_key_hash_seed);
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

bool KVCacheManager::detachSharedBlocksForRecompute(const BatchKVCacheResourcePtr& resource,
                                                    int                            device_ready_reuse_length) {
    return coordinator_manager_->detachSharedBlocksForRecompute(resource, device_ready_reuse_length);
}

void KVCacheManager::cancelP2PLoad(const std::shared_ptr<AsyncContext>& context) {
    if (p2p_connector_ && context) {
        // Runs on the request release/cleanup path. cancel() may rethrow an RPC
        // creation failure; keep it off that boundary, the Decode target lease is
        // retained until the checker confirms the cancellation.
        try {
            p2p_connector_->cancelRead(context);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("cancelP2PLoad failed, retaining Decode target resources, error: %s", e.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("cancelP2PLoad failed, retaining Decode target resources");
        }
    }
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

void KVCacheManager::reclaimLinearHistory(const BatchKVCacheResourcePtr& resource,
                                          int                            committed_tokens,
                                          bool                           reuse_cache) {
    if (config_.enable_linear_cache_reclaim && coordinator_manager_) {
        coordinator_manager_->reclaimLinearHistory(resource, committed_tokens, reuse_cache);
    }
}

bool KVCacheManager::freezeUnknownDecodeWriter(
    const BatchKVCacheResource& resource,
    const std::vector<std::shared_ptr<const DecodeEndpoint>>& unpublished) const noexcept {
    const auto retain_candidates = [&]() noexcept {
        if (resource.decode_endpoint) resource.decode_endpoint->retainOnFreezeFailure(resource.decode_endpoint);
        for (const auto& endpoint : resource.decode_provisional_endpoints)
            if (endpoint) endpoint->retainOnFreezeFailure(endpoint);
        for (const auto& endpoint : unpublished)
            if (endpoint) endpoint->retainOnFreezeFailure(endpoint);
    };
    if (!isAllocatorOwner() || !coordinator_manager_) {
        retain_candidates();
        return false;
    }
    try {
        DecodeFrozenPages frozen;
        const auto& pools = coordinator_manager_->groupBlockPools();
        const auto& tags = config_.groupTags();
        for (size_t group = 0; group < tags.size() && group < pools.size(); ++group) {
            const auto& pool = pools[group];
            if (!pool) continue;
            for (int batch = 0; batch < resource.batchSize(); ++batch) {
                for (auto block : resource.blocks(batch, tags[group])) {
                    // Own a frozen reference even if another operation already
                    // froze this page: its later recovery must not hide our hold.
                    freezeDecodePage(frozen, pool, block);
                }
            }
        }
        // Endpoint STATE/SWA pages may have left the current table. Their ordinary
        // connector pins must also be replaceable by native frozen references.
        freezeDecodeEndpointPages(frozen, resource.decode_endpoint);
        for (const auto& endpoint : resource.decode_provisional_endpoints)
            freezeDecodeEndpointPages(frozen, endpoint);
        for (const auto& endpoint : unpublished)
            freezeDecodeEndpointPages(frozen, endpoint);
        return true;
    } catch (...) {
        // Never drop the last pin of a candidate outside the REQUEST table.
        retain_candidates();
        // The caller keeps REQUEST/tensor custody when native freezing is partial.
        return false;
    }
}

size_t KVCacheManager::decodePublicationUnit() const {
    const auto& cp = parallelism_config_.prefill_cp_config;
    size_t unit = config_.seq_size_per_block * (cp.kv_cache_sharded ? cp.prefill_cp_size : 1);
    for (const auto& group : config_.topology().groups()) {
        if (!group.policy.enable_prefix_reuse || group.policy.group_type == CacheGroupType::FULL) continue;
        if (group.policy.group_type == CacheGroupType::LINEAR) return 0;
        const auto span = group.seqSizePerBlock();
        if (!unit || !span) return 0;
        const auto multiplier = span / std::gcd(unit, span);
        if (multiplier > std::numeric_limits<size_t>::max() / unit) return 0;
        unit *= multiplier;
    }
    return unit;
}

std::shared_ptr<const DecodeEndpoint> KVCacheManager::retainDecodeEndpoint(
    const torch::Tensor& physical_ids, const std::vector<std::string>& tags, size_t row, size_t end,
    const std::shared_ptr<const DecodeEndpoint>& previous, bool defer_writer_wait,
    const torch::Tensor& writer_kernel_ids) const {
    if (!physical_ids.defined() || physical_ids.dim() != 3 || !physical_ids.is_cpu() || physical_ids.scalar_type() != torch::kInt32
        || row >= static_cast<size_t>(physical_ids.size(1))
        || tags.size() != static_cast<size_t>(physical_ids.size(0))) return {};
    const bool exact_writer_row = writer_kernel_ids.defined();
    if (exact_writer_row && (!writer_kernel_ids.is_cpu() || writer_kernel_ids.scalar_type() != torch::kInt32
        || writer_kernel_ids.dim() != 3 || writer_kernel_ids.size(0) != physical_ids.size(0)
        || writer_kernel_ids.size(1) != physical_ids.size(1))) return {};
    return retainDecodeEndpointPages(end, previous, [&](const std::string& tag) {
        const auto it = std::find(tags.begin(), tags.end(), tag);
        const auto input = it == tags.end() ? torch::Tensor{} :
            physical_ids.select(0, it - tags.begin()).select(0, row);
        const auto kernel = exact_writer_row && it != tags.end() ?
            writer_kernel_ids.select(0, it - tags.begin()).select(0, row) : torch::Tensor{};
        return [input, kernel, exact_writer_row](size_t slot, bool full) -> BlockIdxType {
            if (!input.defined() || !input.numel()
                || ((full || exact_writer_row) && slot >= static_cast<size_t>(input.numel()))) {
                return NULL_BLOCK_IDX;
            }
            const auto source = input.accessor<int32_t, 1>()[slot % input.numel()];
            // These native STATE/SWA pages have identical physical and kernel IDs.
            // A missing column must never wrap to a page from an earlier endpoint.
            if (exact_writer_row && !full
                && (slot >= static_cast<size_t>(kernel.numel()) || kernel.accessor<int32_t, 1>()[slot] != source)) {
                return NULL_BLOCK_IDX;
            }
            return source;
        };
    }, defer_writer_wait, /*share_full_prefix=*/exact_writer_row);
}

std::shared_ptr<const DecodeEndpoint> KVCacheManager::retainDecodeEndpoint(
    const BatchKVCacheResource& resource, size_t end,
    const std::shared_ptr<const DecodeEndpoint>& previous) const {
    if (resource.batchSize() != 1) return {};
    return retainDecodeEndpointPages(end, previous, [&](const std::string& tag) {
        const auto* blocks = &resource.blocks(0, tag);
        return [blocks](size_t slot, bool) -> BlockIdxType {
            // Host BlockIds retain logical positions: removed old SWA slots
            // become NULL, not a compact ring that may wrap to a newer page.
            if (slot >= blocks->size()) return NULL_BLOCK_IDX;
            return (*blocks)[slot];
        };
    }, /*defer_writer_wait=*/true);
}

template<typename ReadGroup>
std::shared_ptr<const DecodeEndpoint> KVCacheManager::retainDecodeEndpointPages(
    size_t end, const std::shared_ptr<const DecodeEndpoint>& previous,
    ReadGroup&& read_group, bool defer_writer_wait, bool share_full_prefix,
    const std::unordered_map<std::string, std::vector<size_t>>* release_state_slots) const {
    // Decode owns the native pages locally; the existing D->P projection
    // performs the actual target-CP slicing at publication time.
    if (decode_capture_stopped_.load() || !isAllocatorOwner() || !coordinator_manager_ || !end) return {};
    const auto unit = decodePublicationUnit();
    if (!unit || end % unit) return {};
    auto endpoint = std::make_shared<DecodeEndpoint>();
    endpoint->end = end;
    endpoint->release_snapshot = release_state_slots != nullptr;
    const auto& pools = coordinator_manager_->groupBlockPools();
    const size_t canonical = config_.seq_size_per_block *
        (parallelism_config_.prefill_cp_config.kv_cache_sharded ? parallelism_config_.prefill_cp_config.prefill_cp_size : 1);
    for (size_t group_id = 0; group_id < config_.groupTags().size(); ++group_id) {
        const auto& tag = config_.groupTags()[group_id];
        const auto& group = config_.topology().group(tag);
        if (!group.policy.enable_prefix_reuse) continue;
        if (group.policy.group_type != CacheGroupType::FULL && group.policy.group_type != CacheGroupType::SWA)
            throw std::runtime_error("DSV4.1 endpoint requires native FULL/SWA pages");
        if (group_id >= pools.size()) return {};
        const auto& pool = pools[group_id];
        if (!pool || pool->where() != MemoryType::MEMORY_GPU) return {};
        const size_t count = end / group.seqSizePerBlock();
        if (!count) return {};
        const bool full = group.policy.group_type == CacheGroupType::FULL;
        const size_t first = full ? 0 : count - std::min(count, group.reuseBlockCount(end / canonical));
        const auto* selected_slots = release_state_slots && !full ? &release_state_slots->at(tag) : nullptr;
        if (share_full_prefix) {
            const size_t bytes = pool->blockSizeBytes();
            const size_t retained = selected_slots ? selected_slots->size() : count - first;
            if (bytes && retained > (std::numeric_limits<size_t>::max() - endpoint->retained_page_bytes) / bytes)
                return {};
            endpoint->retained_page_bytes += retained * bytes;
        }
        auto source_at = read_group(tag);
        auto pin = [&](BlockIdxType block) -> std::shared_ptr<const DecodeEndpoint::PageHold> {
            auto hold = std::make_shared<DecodeEndpoint::PageHold>();
            hold->pool = pool;
            hold->block = block;
            hold->pin = pool->tryPinForStorageIo(block);
            if (!hold->pin) return {};
            return hold;
        };
        if (share_full_prefix && full) {
            auto segment = std::make_shared<DecodeEndpoint::FullPrefix>();
            if (previous && previous->end <= end) {
                const auto prior = previous->full_prefixes.find(tag);
                if (prior != previous->full_prefixes.end()) {
                    segment->previous = prior->second;
                    segment->first = previous->end / group.seqSizePerBlock();
                }
            }
            // Share continuously held allocations only when the current
            // writer names the same pages at those absolute slots.
            for (auto prefix = segment->previous.get(); prefix; prefix = prefix->previous.get()) {
                bool matches = true;
                for (size_t i = 0; i < prefix->suffix.size(); ++i) {
                    const auto& held = prefix->suffix[i];
                    if (held->pool.get() != pool.get() || source_at(prefix->first + i, true) != held->block) {
                        matches = false;
                        break;
                    }
                }
                if (!matches) {
                    segment->previous.reset();
                    segment->first = 0;
                    break;
                }
            }
            segment->suffix.reserve(count - segment->first);
            for (size_t slot = segment->first; slot < count; ++slot) {
                const auto source = source_at(slot, true);
                if (isNullBlockIdx(source)) return {};
                auto held = pin(source);
                if (!held) return {};
                segment->suffix.push_back(std::move(held));
            }
            endpoint->full_prefixes.emplace(tag, std::move(segment));
            continue;
        }
        auto& blocks = endpoint->blocks[tag]; blocks.assign(count, NULL_BLOCK_IDX);
        auto& holds = endpoint->pages[tag];
        if (selected_slots) {
            holds.reserve(selected_slots->size());
            for (size_t slot : *selected_slots) {
                const auto source = source_at(slot, false);
                if (isNullBlockIdx(source)) return {};
                auto held = pin(source);
                if (!held) return {};
                blocks.at(slot) = source;
                holds.push_back(std::move(held));
            }
            continue;
        }
        holds.reserve(count - first);
        const DecodeEndpoint::PageHolds* prior = nullptr;
        if (group.policy.group_type == CacheGroupType::FULL && previous && previous->end <= end) {
            const auto found = previous->pages.find(tag);
            if (found != previous->pages.end()) prior = &found->second;
        }
        for (size_t slot = first; slot < count; ++slot) {
            const auto source = source_at(slot, group.policy.group_type == CacheGroupType::FULL);
            if (isNullBlockIdx(source)) return {};
            const auto* held = prior && slot < prior->size() ? (*prior)[slot].get() : nullptr;
            if (held && held->pool.get() == pool.get() && held->block == source
                && held->pin) {
                // This exact page's live connector pin already proves allocation.
                holds.push_back((*prior)[slot]); // Share this FULL page, never the old endpoint's owner.
            } else {
                auto held = pin(source);
                if (!held) return {};
                holds.push_back(std::move(held));
            }
            blocks[slot] = source; // Actual DSV4.1 STATE/SWA use position-based pages, not LINEAR accept swaps.
        }
    }
    try {
#if USING_CUDA
        // Production speculative captures remain pending until specUpdate,
        // after the original final draft_event host wait. Other callers keep
        // the synchronous capture contract; pin alone is not readiness.
        if (!defer_writer_wait)
            check_cuda_value(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream()));
#endif
    } catch (...) {
#if USING_CUDA
        if (cudaDeviceSynchronize() != cudaSuccess) {
            // Keep the original device-health stop. Native frozen references
            // protect every actual page after this temporary endpoint unwinds.
            decode_capture_stopped_.store(true);
            try {
                DecodeFrozenPages frozen;
                freezeDecodeEndpointPages(frozen, endpoint);
            } catch (...) {
                endpoint->retainOnFreezeFailure(endpoint);
            }
        }
#endif
        throw;
    }
    RTP_LLM_LOG_DEBUG("DSV41_NATIVE_ENDPOINT H=%zu state_copies=0 source=original_pages writer_wait_deferred=%d",
                      end, static_cast<int>(defer_writer_wait));
    return endpoint;
}

std::shared_ptr<const DecodeEndpoint> KVCacheManager::retainReleasedDSparkPages(
    const BatchKVCacheResource& resource, const torch::Tensor& physical_ids, const torch::Tensor& kernel_ids,
    bool all_resident_state, std::shared_ptr<StorageCompletionAdmission>* admission) const {
    const auto unit = resource.dspark_decode_publication_unit;
    const auto& tags = config_.groupTags();
    const auto valid_table = [&tags](const torch::Tensor& table) {
        return table.defined() && table.is_cpu() && table.scalar_type() == torch::kInt32 && table.dim() == 3
               && static_cast<size_t>(table.size(0)) == tags.size() && table.size(1) == 1;
    };
    if (!unit || unit != decodePublicationUnit() || !coordinator_manager_ || resource.batchSize() != 1
        || resource.dspark_release_provenance_lost
        || !valid_table(physical_ids) || !valid_table(kernel_ids)) return {};
    if (admission && !coordinator_manager_->canStartDecodePublication(resource)) return {};

    const auto physical = physical_ids.accessor<int32_t, 3>();
    const auto kernel = kernel_ids.accessor<int32_t, 3>();
    const auto source_at = [&](size_t group, size_t slot) -> BlockIdxType {
        const auto& row = *resource.groupBlockIds(0, tags[group]);
        const size_t stride = row.kernelBlocksPerKvBlock();
        if (slot >= row.blocksNum() || slot >= static_cast<size_t>(physical_ids.size(2))
            || slot >= static_cast<size_t>(kernel_ids.size(2)) / stride) return NULL_BLOCK_IDX;
        const auto block = row.blocks()[slot];
        if (block <= 0 || physical[group][0][slot] != block) return NULL_BLOCK_IDX;
        for (size_t at = slot * stride; at < (slot + 1) * stride; ++at)
            if (kernel[group][0][at] != row.kernelBlocks()[at]) return NULL_BLOCK_IDX;
        return block;
    };

    const size_t latest = resource.dspark_release_computed_end.value_or(0) / unit * unit;
    size_t end = latest;
    const size_t canonical = config_.seq_size_per_block * (parallelism_config_.prefill_cp_config.kv_cache_sharded
        ? parallelism_config_.prefill_cp_config.prefill_cp_size : 1);
    std::unordered_map<std::string, std::vector<size_t>> state_slots;
    std::vector<std::unordered_set<BlockIdxType>> aliased_state(tags.size());
    for (size_t group = 0; group < tags.size(); ++group) {
        const auto& spec = config_.topology().group(tags[group]);
        if (!spec.policy.enable_prefix_reuse) continue;
        if (spec.policy.group_type == CacheGroupType::FULL) {
            const auto span = spec.seqSizePerBlock();
            size_t complete = 0;
            while (complete < end / span && !isNullBlockIdx(source_at(group, complete))) ++complete;
            end = std::min(end, complete * span / unit * unit);
        } else if (spec.policy.group_type == CacheGroupType::SWA) {
            state_slots.try_emplace(tags[group]);
            std::unordered_set<BlockIdxType> seen;
            for (int64_t slot = 0; slot < physical_ids.size(2); ++slot) {
                const auto block = physical[group][0][slot];
                if (block > 0 && !seen.insert(block).second) aliased_state[group].insert(block);
            }
        } else {
            return {};
        }
    }
    if (!end || !canonical || end <= resource.dspark_decode_start) return {};

    std::vector<std::pair<size_t, size_t>> candidate;
    // A clipped prefix before the newly computed decode span does not prove a
    // source checkpoint. Every attached state slot must cover accepted rows.
    for (size_t h = end; h > resource.dspark_decode_start; h -= unit) {
        if (!all_resident_state && h != latest) break;
        candidate.clear();
        bool complete = true;
        for (size_t group = 0; group < tags.size() && complete; ++group) {
            const auto& tag = tags[group];
            if (!state_slots.count(tag)) continue;
            const auto& spec = config_.topology().group(tag);
            const auto span = spec.seqSizePerBlock();
            const size_t count = h / canonical;
            const size_t required = std::min(count, spec.reuseBlockCount(count));
            const auto backfills = resource.dspark_state_backfills.find(tag);
            for (size_t key = count - required; key < count; ++key) {
                const size_t checkpoint = (key + 1) * canonical;
                if (checkpoint % span) {
                    complete = false;
                    break;
                }
                const size_t slot = checkpoint / span - 1;
                const auto block = source_at(group, slot);
                if (isNullBlockIdx(block) || aliased_state[group].count(block)) {
                    complete = false;
                    break;
                }
                const size_t window = spec.policy.sliding_window_size > 0
                    ? std::min(span, static_cast<size_t>(spec.policy.sliding_window_size)) : span;
                if (checkpoint - window < resource.dspark_decode_start) {
                    const auto handoff = resource.prefillComputedEnd();
                    if (!handoff || static_cast<size_t>(*handoff) < resource.dspark_decode_start) {
                        complete = false;
                        break;
                    }
                }
                if (backfills != resource.dspark_state_backfills.end()) {
                    const auto replacement = backfills->second.find(slot);
                    if (replacement != backfills->second.end() && replacement->second > checkpoint - window) {
                        complete = false;
                        break;
                    }
                }
                candidate.emplace_back(group, slot);
            }
        }
        if (complete)
            for (const auto& [group, slot] : candidate) state_slots.at(tags[group]).push_back(slot);
        if (!all_resident_state) break;
    }
    for (auto& [tag, slots] : state_slots) {
        std::sort(slots.begin(), slots.end());
        slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
    }

    std::shared_ptr<StorageCompletionAdmission> reservation;
    if (admission) {
        if (!coordinator_manager_->canStartDecodePublication(resource)) return {};
        reservation = *admission;
        if (!reservation) {
            size_t bytes = 0;
            const auto& pools = coordinator_manager_->groupBlockPools();
            for (size_t group = 0; group < tags.size(); ++group) {
                const auto& spec = config_.topology().group(tags[group]);
                if (!spec.policy.enable_prefix_reuse) continue;
                const auto& pool = pools.at(group);
                if (!pool || pool->where() != MemoryType::MEMORY_GPU) return {};
                const auto count = spec.policy.group_type == CacheGroupType::FULL ? end / spec.seqSizePerBlock()
                    : state_slots.at(tags[group]).size();
                const auto size = pool->blockSizeBytes();
                if (size && count > (std::numeric_limits<size_t>::max() - bytes) / size) return {};
                bytes += count * size;
            }
            reservation = coordinator_manager_->reserveDecodePublication(bytes);
            if (!reservation) {
                RTP_LLM_LOG_DEBUG("DTOP_GPU release admission=rejected retained_bytes=%zu", bytes);
                return {};
            }
        }
    }
    auto endpoint = retainDecodeEndpointPages(end, {}, [&](const std::string& tag) {
        const auto group = std::distance(tags.begin(), std::find(tags.begin(), tags.end(), tag));
        return [&, group](size_t slot, bool) { return source_at(group, slot); };
    }, /*defer_writer_wait=*/true, /*share_full_prefix=*/true, &state_slots);
    if (endpoint && admission) *admission = std::move(reservation);
    return endpoint;
}

bool KVCacheManager::tryDeferDSparkPublication(const FreeInfo& free_info, bool append_only, bool all_resident_state) {
    auto& source = *free_info.batch_kv_cache_resource;
    const auto computed_end = source.dspark_release_computed_end.value_or(0);
    const auto unit = source.dspark_decode_publication_unit;
    if (!dspark_publication_access_ || !unit || source.batchSize() != 1
        || computed_end / unit <= source.dspark_decode_start / unit
        || source.dspark_release_provenance_lost
        || !source.dspark_release_writer_tables[0].defined() || !source.dspark_release_writer_tables[1].defined()) {
        return false;
    }
    const auto backend = block_tree_cache_->storageBackend();
    if (!backend || !backend->canStartDecodePublication(source)) return false;

    // Charge all still-owned slots, including draft pages and holes, without
    // walking any page IDs. The same budget applies in the storage queue.
    size_t queued_bytes = 0;
    const auto& tags = config_.groupTags();
    const auto& pools = coordinator_manager_->groupBlockPools();
    auto work = std::make_shared<DSparkPublicationWork>();
    work->rows.reserve(tags.size());
    for (size_t group = 0; group < tags.size(); ++group) {
        // Resolve tags before detaching. The row owners move with the resource
        // and cleanup can use these rows without allocating.
        const auto& row = source.blocks(0, tags[group]);
        work->rows.push_back({pools[group], &row});
        const auto bytes = pools[group]->blockSizeBytes();
        if (bytes && row.size() > (std::numeric_limits<size_t>::max() - queued_bytes) / bytes) return false;
        queued_bytes += row.size() * bytes;
    }
    work->admission = backend->reserveDecodePublication(queued_bytes);
    if (!work->admission) {
        RTP_LLM_LOG_DEBUG("DTOP_GPU preparation admission=rejected queued_bytes=%zu", queued_bytes);
        return false;
    }

    // CompleteTokenIds belongs to the live stream. Freeze only the CPU keys
    // and move the remaining request references and writer tables as-is.
    auto keys = std::make_shared<const CacheKeysType>(buildDecodePublicationKeys(
        *free_info.complete_token_ids, config_.seq_size_per_block, computed_end, config_.cache_key_hash_seed,
        append_only ? &source : nullptr));
    work->resource = std::make_shared<BatchKVCacheResource>();
    work->resource->resetBatchSize(1);
    work->resource->initGroups(config_.topologyPtr());
    work->cache_capacity_changed = block_tree_cache_->capacityChangeCallback();
    work->allocation_wait_state = allocation_wait_state_;
    work->metrics_reporter = metrics_reporter_;

    auto& detached = *work->resource;
    source.swap(detached);
    work->owns_request_pages.store(true);
    block_tree_cache_detail::ScopeRollback restore([&] {
        work->owns_request_pages.store(false);
        source.swap(detached);
    });
    const bool queued = backend->enqueueCompletedDecode(
        detached, computed_end,
        [access = dspark_publication_access_, work, keys = std::move(keys),
         queued_bytes, all_resident_state]() -> std::shared_ptr<const BatchKVCacheResource> {
            block_tree_cache_detail::ScopeRollback release([&] { work->releaseRequestPages(); });
            RTP_LLM_PROFILE_SCOPE("cache.dspark.prepare_publication");
            auto& resource = *work->resource;
            std::shared_ptr<const DecodeEndpoint> endpoint;
            {
                std::shared_lock lock(access->mutex);
                if (!access->manager) return {};
                const auto& tables = resource.dspark_release_writer_tables;
                resource.dspark_release_admission = work->admission;
                endpoint = access->manager->retainReleasedDSparkPages(
                    resource, tables[0], tables[1], all_resident_state, &resource.dspark_release_admission);
            }
            if (!endpoint) return {};
            if (endpoint->retained_page_bytes > queued_bytes) {
                RTP_LLM_LOG_ERROR("DSpark publication pinned more bytes than its preparation reservation");
                return {};
            }
            const auto pinned_bytes = endpoint->retained_page_bytes;
            resource.decode_endpoint = std::move(endpoint);
            resource.decode_publication_keys = keys;
            // Keep only the pinned endpoint for planning and I/O in this same
            // storage job. Return the rest before reducing the shared budget.
            release.run();
            work->admission->reduceTo(pinned_bytes);
            return work->resource;
        });
    if (queued) restore.dismiss();
    return queued;
}

bool KVCacheManager::captureCompletedDecode(const BatchKVCacheResource& resource,
                                            size_t                      computed_end,
                                            bool                        finished) const {
    return coordinator_manager_ && coordinator_manager_->captureCompletedDecode(resource, computed_end, finished);
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
                                           int                            target_batch_size,
                                           LinearCachePolicy              linear_cache_policy,
                                           const std::string&             pool_tag) const {
    return coordinator_manager_->estimateBatchPeakNeedBlocks(batch_kv_cache_resource,
                                                             seq_len,
                                                             common_seq_len,
                                                             remaining_tokens,
                                                             reserve_step,
                                                             enable_reuse_cache,
                                                             target_batch_size,
                                                             linear_cache_policy,
                                                             pool_tag);
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

void KVCacheManager::blockBatchCopyForForward(const torch::Tensor&            copy_mapping,
                                              const std::vector<std::string>& group_tags,
                                              const torch::Tensor&            valid_token_lengths) {
    coordinator_manager_->blockBatchCopyForForward(copy_mapping, group_tags, valid_token_lengths);
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

void KVCacheManager::initLinearReplayPool() {
    if (!config_.linear_replay.enabled()) {
        return;
    }
    const auto& plan = config_.linear_replay;
    RTP_LLM_CHECK(plan.slotCount() > 0 && plan.maxSteps() > 0);
    const auto layout = coordinator_manager_->allLayerCacheBase();
    const auto slots  = plan.slotCount();
    for (const int group_id : config_.linear_replay.group_ids) {
        const auto& group = config_.groups().at(static_cast<size_t>(group_id));
        auto& layers = linear_replay_by_tag_[group.tag];
        layers.resize(config_.layer_num);
        for (const int layer_id : config_.layerIdsForGroup(group.tag)) {
            if (layer_id < 0 || layer_id >= static_cast<int>(config_.layer_num)) {
                continue;
            }
            const auto& base = layout.at(group.tag, static_cast<size_t>(layer_id)).kv_addr;
            RTP_LLM_CHECK_WITH_INFO(base.defined(), "LINEAR replay layer %d tag=%s has no cache buffer",
                                    layer_id, group.tag.c_str());
            auto                              f32 = torch::TensorOptions().device(base.device()).dtype(torch::kFloat32);
            torch_ext::LinearReplayLayerCache replay;
            replay.k = torch::empty(c10::IntArrayRef(plan.k_shape.data(), plan.k_shape.size()), f32);
            replay.u = torch::empty(c10::IntArrayRef(plan.u_shape.data(), plan.u_shape.size()), f32);
            replay.g = torch::empty(c10::IntArrayRef(plan.g_shape.data(), plan.g_shape.size()), f32);
            replay.conv_inputs =
                torch::empty(c10::IntArrayRef(plan.conv_input_shape.data(), plan.conv_input_shape.size()),
                             f32.dtype(dataTypeToTorchType(plan.conv_dtype)));
            layers[static_cast<size_t>(layer_id)] = std::move(replay);
        }
    }
    linear_replay_slots_ = std::make_shared<LinearReplaySlotPool>(static_cast<size_t>(slots));
    linear_replay_retirement_ = std::make_shared<LinearReplayRetirementQueue>();
}

std::shared_ptr<LinearReplayLease> KVCacheManager::acquireLinearReplaySlot() {
    if (!linear_replay_slots_) {
        return nullptr;
    }
    linear_replay_retirement_->reap();
    return linear_replay_slots_->acquire();
}

std::shared_ptr<LinearReplayBlockHold> KVCacheManager::holdLinearReplayBlocks(const BatchKVCacheResourcePtr& resource,
                                                                              int state_token_position) {
    if (!resource || !config_.linear_replay.enabled()) {
        return nullptr;
    }
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIndicesType>> held;
    for (const int group_id : config_.linear_replay.group_ids) {
        const auto& group = config_.groups().at(static_cast<size_t>(group_id));
        const auto& pool = coordinator_manager_->groupBlockPools().at(static_cast<size_t>(group_id));
        std::unordered_set<BlockIdxType> unique;
        for (int batch = 0; batch < resource->batchSize(); ++batch) {
            const auto& blocks = resource->blocks(batch, group.tag);
            size_t begin = resource->cacheResource(batch).linearReplayActiveBegin(group.tag, blocks.size());
            size_t      end    = blocks.size();
            if (state_token_position >= 0) {
                begin = static_cast<size_t>(state_token_position) / group.seqSizePerBlock();
                RTP_LLM_CHECK_WITH_INFO(begin < end, "LINEAR replay state position exceeds its block table");
                end = begin + 1;
            }
            for (size_t position = begin; position < end; ++position) {
                if (blocks[position] > 0) {
                    unique.insert(blocks[position]);
                }
            }
        }
        BlockIndicesType ids(unique.begin(), unique.end());
        if (!ids.empty()) {
            pool->incRef(ids);
            held.emplace_back(pool, std::move(ids));
        }
    }
    auto* block_hold = new LinearReplayBlockHold;
    for (const auto& [pool, ids] : held) {
        block_hold->blocks_by_pool[pool.get()].insert(ids.begin(), ids.end());
    }
    return std::shared_ptr<LinearReplayBlockHold>(
        block_hold,
        [retirement = linear_replay_retirement_, held = std::move(held)](LinearReplayBlockHold* value) mutable {
            const auto event = value->lastUse();
            delete value;
            retirement->retire(event, [held = std::move(held)] {
                for (const auto& [pool, ids] : held) {
                    pool->decRef(ids);
                }
            });
        });
}

bool KVCacheManager::makeLinearReplayTailsPrivate(
    const BatchKVCacheResourcePtr&                             resource,
    int                                                        min_processed_length,
    const std::vector<std::shared_ptr<LinearReplayBlockHold>>& request_holds) {
    if (!resource || !config_.linear_replay.enabled()) {
        return true;
    }
    linear_replay_retirement_->reap();
    std::unordered_set<const LinearReplayBlockHold*> owned_holds;
    for (const auto& hold : request_holds) {
        if (hold) {
            owned_holds.insert(hold.get());
        }
    }
    struct Replacement {
        int batch;
        std::string tag;
        size_t position;
        BlockIdxType source;
        BlockIdxType destination;
        DeviceBlockPoolPtr pool;
    };
    std::vector<Replacement> replacements;
    struct ActiveBegin { int batch; std::string tag; size_t begin; };
    std::vector<ActiveBegin> active_begins;
    for (const int group_id : config_.linear_replay.group_ids) {
        const auto& group = config_.groups().at(static_cast<size_t>(group_id));
        const auto& pool = coordinator_manager_->groupBlockPools().at(static_cast<size_t>(group_id));
        for (int batch = 0; batch < resource->batchSize(); ++batch) {
            const auto& blocks = resource->blocks(batch, group.tag);
            size_t begin = blocks.size() > 2 ? blocks.size() - 2 : 0;
            if (min_processed_length >= 0) {
                // Capacity checks require two tail slots at seqLength (processed + 1).
                // Proposal slots and previous reservations must not move that tail forward.
                const size_t seq_slots =
                    (static_cast<size_t>(min_processed_length) + group.seqSizePerBlock()) / group.seqSizePerBlock();
                begin = std::min(begin, seq_slots > 2 ? seq_slots - 2 : 0);
            }
            active_begins.push_back({batch, group.tag, begin});
            for (size_t position = begin; position < blocks.size(); ++position) {
                const auto source = blocks[position];
                uint32_t   owned_references = 1;
                for (const auto* hold : owned_holds) {
                    owned_references += hold->contains(pool.get(), source);
                }
                // Verify and rebuild are ordered on the same compute stream.
                // Our own retained anchor is safe to update in place; external
                // request/tree references still require a private destination.
                if (source > 0 && pool->isExclusiveRequestBlock(source, owned_references)) {
                    continue;
                }
                auto destination = pool->malloc();
                if (!destination) {
                    coordinator_manager_->cacheGroups().at(static_cast<size_t>(group_id))->ensureFreeBlocks(1);
                    if (source > 0 && pool->isExclusiveRequestBlock(source, owned_references)) {
                        continue;
                    }
                    destination = pool->malloc();
                }
                if (!destination) {
                    RTP_LLM_LOG_WARNING("LINEAR replay tail copy failed tag=%s batch=%d position=%zu free=%zu "
                                        "total=%zu staged=%zu",
                                        group.tag.c_str(), batch, position, pool->freeBlocksNum(),
                                        pool->totalBlocksNum(), replacements.size());
                    for (const auto& item : replacements) {
                        item.pool->decRef(item.destination);
                    }
                    return false;
                }
                pool->incRef(*destination);
                replacements.push_back({batch, group.tag, position, source, *destination, pool});
            }
        }
    }
    for (const auto& item : replacements) {
        resource->mutableBlockIds(item.batch, item.tag).setAt(item.position, item.destination);
        if (item.source > 0) {
            item.pool->decRef(item.source);
        }
    }
    for (const auto& item : active_begins) {
        resource->cacheResource(item.batch).setLinearReplayActiveBegin(item.tag, item.begin);
        resource->cacheResource(item.batch).restrictLinearReplayPrefix(item.tag, item.begin);
    }
    return true;
}

void KVCacheManager::markLinearReplayStarted(const BatchKVCacheResourcePtr& resource, int processed_length) {
    if (!resource) {
        return;
    }
    for (const int group_id : config_.linear_replay.group_ids) {
        const auto& group = config_.groups().at(static_cast<size_t>(group_id));
        const size_t position = static_cast<size_t>(std::max(0, processed_length)) / group.seqSizePerBlock();
        for (int batch = 0; batch < resource->batchSize(); ++batch) {
            resource->cacheResource(batch).restrictLinearReplayPrefix(group.tag, position);
        }
    }
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
    KVCacheInfo info;
    info.version = latest_version;

    if (!coordinator_manager_) {
        RTP_LLM_LOG_ERROR("getKVCacheInfo called before KVCacheManager initialized");
        return info;
    }

    if (need_cache_keys && block_tree_cache_) {
        BlockTreeKeySnapshot snapshot = block_tree_cache_->getKeySnapshot(latest_version);
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

bool KVCacheManager::hasActiveConnectors() const {
    return p2p_connector_ != nullptr;
}

// PD separation: increment KV cache reference count
std::shared_ptr<KVCacheResource>
KVCacheManager::incrKVCacheRef(const KVCacheResource& resource, const CacheKeysType& cache_keys, bool is_connector) {
    return coordinator_manager_->incrKVCacheRef(resource, cache_keys, is_connector);
}

void KVCacheManager::cancelPrefillRequest(const std::string& unique_key, int64_t request_deadline_ms) {
    if (p2p_connector_) {
        if (auto store = p2p_connector_->streamStore())
            store->markCancelled(unique_key, request_deadline_ms);
    }
}

void KVCacheManager::expirePrefillRequest(const std::string& unique_key, int64_t request_deadline_ms) {
    if (p2p_connector_) {
        if (auto store = p2p_connector_->streamStore()) {
            store->markTerminal(unique_key, request_deadline_ms, P2PConnectorResourceStore::TerminalReason::TIMED_OUT);
        }
    }
}

int64_t KVCacheManager::prefillRequestDeadline(const std::string& unique_key, int64_t timeout_ms, ErrorInfo* error) {
    auto store = p2p_connector_ ? p2p_connector_->streamStore() : nullptr;
    return store ? store->requestDeadline(unique_key, timeout_ms, error) : 0;
}

bool KVCacheManager::hasP2PConnector() const {
    return p2p_connector_ != nullptr;
}

void KVCacheManager::publishPrefillPayload(const std::string&                    unique_key,
                                           int64_t                               deadline_ms,
                                           PrefillResultStore::SideChannelData&& data) {
    if (p2p_connector_ && pd_sep_config_.role_type == RoleType::PREFILL && parallelism_config_.tp_rank == 0) {
        auto stream_store = p2p_connector_->streamStore();
        if (stream_store) {
            stream_store->publishPrefillPayload(unique_key, deadline_ms, std::move(data));
        }
    }
}

void KVCacheManager::failP2PPublication(int64_t            request_id,
                                        const std::string& unique_key,
                                        int64_t            deadline_ms,
                                        const ErrorInfo&   error) {
    if (!p2p_connector_) {
        return;
    }
    p2p_connector_->rejectLayer(request_id, deadline_ms, 0, error);
    if (auto store = p2p_connector_->streamStore()) {
        store->markTerminal(unique_key, deadline_ms, P2PConnectorResourceStore::TerminalReason::FAILED);
    }
}

void KVCacheManager::finishP2PPublication(const torch::Tensor& final_prefill,
                                          const torch::Tensor& pd_separation,
                                          const torch::Tensor& request_ids,
                                          const torch::Tensor& deadlines,
                                          int64_t              clock_offset_ms) {
    if (!p2p_connector_ || !final_prefill.defined() || final_prefill.numel() == 0) {
        return;
    }
    const auto  pd             = pd_separation.cpu();
    const auto  ids            = request_ids.cpu();
    const auto  host_deadlines = deadlines.cpu();
    const auto* final_rows     = final_prefill.data_ptr<bool>();
    for (int64_t row = 0; row < final_prefill.numel(); ++row) {
        if (final_rows[row] && pd.data_ptr<bool>()[row]) {
            p2p_connector_->finishP2PPublication(
                ids.data_ptr<int64_t>()[row], host_deadlines.data_ptr<int64_t>()[row], clock_offset_ms);
        }
    }
}

bool KVCacheManager::writeP2PLayer(size_t                               model_id,
                                   int                                  local_layer_id,
                                   const std::string&                   tag,
                                   const std::vector<int64_t>&          cache_keys,
                                   const std::vector<int32_t>&          block_ids,
                                   int64_t                              request_id,
                                   const std::shared_ptr<torch::Event>& event,
                                   int64_t                              deadline_ms,
                                   int64_t                              deadline_clock_offset_ms,
                                   int                                  chunk_id) {
    if (!p2p_connector_) {
        return false;
    }
    const auto reject = [&](const std::string& reason) {
        return p2p_connector_->rejectLayer(
            request_id,
            deadline_ms,
            deadline_clock_offset_ms,
            ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                      reason + ", local_layer=" + std::to_string(local_layer_id) + " tag=" + tag));
    };
    if (chunk_id < 0) {
        return reject("P2P layer publication is missing chunk metadata");
    }
    if (cache_keys.size() != block_ids.size()) {
        RTP_LLM_LOG_WARNING(
            "writeP2PLayer rejected invalid key/block pairs, request_id=%ld model_id=%zu local_layer_id=%d "
            "tag=%s keys=%zu blocks=%zu",
            request_id,
            model_id,
            local_layer_id,
            tag.c_str(),
            cache_keys.size(),
            block_ids.size());
        return reject("P2P layer publication has mismatched key and block counts");
    }

    const uint32_t global_layer_id = coordinator_manager_->convertToGlobalLayerId(model_id, local_layer_id);
    if (global_layer_id == std::numeric_limits<uint32_t>::max()
        || global_layer_id >= config_.topology().layers().size()) {
        RTP_LLM_LOG_WARNING("writeP2PLayer failed to map layer, request_id=%ld model_id=%zu local_layer_id=%d tag=%s",
                            request_id,
                            model_id,
                            local_layer_id,
                            tag.c_str());
        return reject("P2P layer publication cannot map the local layer");
    }

    const auto& layer = config_.topology().layer(static_cast<int>(global_layer_id));
    if (std::find(layer.group_tags.begin(), layer.group_tags.end(), tag) == layer.group_tags.end()) {
        RTP_LLM_LOG_WARNING("writeP2PLayer rejected tag not owned by layer, request_id=%ld layer_id=%u tag=%s",
                            request_id,
                            global_layer_id,
                            tag.c_str());
        return reject("P2P layer publication tag does not belong to the layer");
    }

    auto layer_resource = std::make_shared<KVCacheResource>();
    layer_resource->initGroups(config_.topologyPtr());
    layer_resource->setCacheKeys(cache_keys);
    layer_resource->mutableBlockIdsForLayer(static_cast<int>(global_layer_id), tag).assign(block_ids);

    // Only rank 0 owns allocator bookkeeping. Keep its source blocks referenced
    // while an already-started sender is still copying after request cancellation.
    if (!cache_keys.empty() && isAllocatorOwner()) {
        for (size_t i = 0; i < block_ids.size(); ++i) {
            if (block_ids[i] <= 0) {
                RTP_LLM_LOG_WARNING("writeP2PLayer invalid block, request_id=%ld layer_id=%u tag=%s "
                                    "cache_key=%ld block_id=%d",
                                    request_id,
                                    global_layer_id,
                                    tag.c_str(),
                                    cache_keys[i],
                                    block_ids[i]);
                return reject("P2P layer publication contains an invalid KV block");
            }
        }
        layer_resource = coordinator_manager_->incrKVCacheRef(*layer_resource, cache_keys, true);
        if (!layer_resource || layer_resource->cacheKeys() != cache_keys
            || layer_resource->blocksForLayer(static_cast<int>(global_layer_id), tag) != block_ids) {
            RTP_LLM_LOG_WARNING("writeP2PLayer failed to hold exact source blocks, request_id=%ld layer_id=%u tag=%s",
                                request_id,
                                global_layer_id,
                                tag.c_str());
            return reject("P2P layer publication could not retain the source KV blocks");
        }
    }

    return p2p_connector_->writeByLayerTag(static_cast<int>(global_layer_id),
                                           tag,
                                           layer_resource,
                                           request_id,
                                           event,
                                           deadline_ms,
                                           deadline_clock_offset_ms,
                                           chunk_id);
}

// 异步连接器操作

std::shared_ptr<AsyncContext> KVCacheManager::asyncLoadCache(const KVCacheResource&       kv_cache_resource,
                                                             const std::shared_ptr<Meta>& meta,
                                                             size_t                       tree_covered_block_num) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!p2p_connector_ || !meta) {
        return nullptr;
    }

    if (pd_sep_config_.role_type == RoleType::PREFILL && parallelism_config_.tp_rank != 0) {
        return nullptr;
    }

    const auto routing     = meta ? meta->p2pRouting() : std::nullopt;
    const bool no_transfer = pd_sep_config_.role_type == RoleType::DECODE && routing && !routing->decode_needs_compute;
    if (no_transfer) {
        // Results still travel through StartLoad, but no Decode target KV is
        // referenced, projected or exposed to transfer workers.
        return p2p_connector_->asyncRead(std::make_shared<KVCacheResource>(), meta, 0, 0);
    }

    if (kv_cache_resource.cacheKeys().empty()) {
        return nullptr;
    }

    auto resource = coordinator_manager_->incrKVCacheRef(kv_cache_resource, kv_cache_resource.cacheKeys(), true);
    if (!resource) {
        RTP_LLM_LOG_WARNING("P2P async load failed, incr kvcache ref failed, resource: [%s]",
                            kv_cache_resource.debugString().c_str());
        return std::make_shared<CompletedAsyncContext>(ErrorInfo(
            ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, "P2P failed to hold Decode target blocks"));
    }

    if (pd_sep_config_.role_type == RoleType::PREFILL) {
        auto context = p2p_connector_->asyncRead(resource, meta, 0, 0);
        if (!context) {
            return std::make_shared<CompletedAsyncContext>(ErrorInfo(
                ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, "P2P Prefill resource registration failed"));
        }
        return context;
    }

    const int matched_blocks = static_cast<int>(resource->cacheKeys().size());
    const int p2p_start_block =
        static_cast<int>(std::min(tree_covered_block_num, static_cast<size_t>(std::max(matched_blocks, 0))));
    if (matched_blocks <= p2p_start_block) {
        // Even when Tree cache covers the full matched range, complete the
        // StartLoad handshake so Prefill can return side-channel data and
        // release the request-scoped KV resource. A zero-sized range is an
        // explicit P2P no-transfer request; Decode will not register buffers
        // or issue RDMA READs.
        return p2p_connector_->asyncRead(resource, meta, matched_blocks, 0);
    }
    return p2p_connector_->asyncRead(resource, meta, p2p_start_block, matched_blocks - p2p_start_block);
}

bool KVCacheManager::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    if (request.has_p2p_request()) {
        return p2p_connector_ && p2p_connector_->executeFunction(request, response);
    }
    const int rank = parallelism_config_.pp_rank * parallelism_config_.tp_size + parallelism_config_.tp_rank;
    return BlockTreeCache::executeFunction(block_tree_cache_.get(), rank, request, response,
                                           std::chrono::steady_clock::time_point::max(), parallelism_config_.pp_size > 1);
}

void KVCacheManager::initCacheEventPublisher(std::shared_ptr<KVCacheEventReporter> reporter) {
    try {
        const auto& publisher_type = kv_cache_config_.kv_cache_event_publisher_type;
        if (warmup_ || parallelism_config_.pp_size != 1 || parallelism_config_.tp_rank != 0 || !block_tree_cache_) {
            return;
        }

        KVCacheEventPublisherConfig publisher_config;
        publisher_config.manager_endpoint     = kv_cache_config_.kv_cache_event_manager_endpoint;
        publisher_config.snapshot_interval_ms = kv_cache_config_.kv_cache_event_snapshot_interval_ms;
        publisher_config.enable_vipserver     = kv_cache_config_.kvcm_enable_vipserver;
        publisher_config.vipserver_domain     = kv_cache_config_.kvcm_vipserver_domain;
        publisher_config.server_address       = kv_cache_config_.kvcm_server_address;
        std::optional<KVCacheEventPublisherContext> control_context;
        std::optional<bool>                         control_merge_full_specs;
#ifdef RTP_LLM_USE_REMOTE_KV_CACHE
        if (const auto backend = std::dynamic_pointer_cast<KVCMStorageBackend>(block_tree_cache_->storageBackend())) {
            if (const auto control = backend->controlConfig()) {
                control_context.emplace();
                control_merge_full_specs                   = control->merge_full_specs;
                control_context->instance_group            = control->instance_group;
                control_context->instance_id               = control->instance_id;
                control_context->register_instance_request = control->register_instance_request;
                publisher_config.enable_vipserver          = control->enable_vipserver;
                publisher_config.vipserver_domain          = control->vipserver_domain;
                publisher_config.server_addresses          = control->addresses;
            }
        }
#endif
        bool publish_cache_events = publisher_type == "kvcm";
        if (!publisher_type.empty() && publisher_type != "none" && !publish_cache_events) {
            RTP_LLM_LOG_WARNING("unknown HBM event publisher type=%s; cache events disabled", publisher_type.c_str());
        }
        if (!publish_cache_events && !control_context) {
            return;
        }
        if (!kv_cache_config_.reuse_cache || !kv_cache_config_.enable_device_cache) {
            publish_cache_events = false;
        }
        std::vector<std::string> reuse_group_tags;
        for (const auto& group : config_.topology().groups()) {
            if (!publish_cache_events || !cacheGroupPublishesEvents(group.policy)
                || group.policy.memory_placement != CacheMemoryPlacement::DEVICE || group.tag == "csa_state"
                || group.tag == "swa_kv") {
                continue;
            }
            reuse_group_tags.push_back(group.tag);
        }
        if (reuse_group_tags.empty()) {
            publish_cache_events = false;
        }

        // Match the runtime pool tags while keeping the published F/F0/F1/F2 schema stable.
        const bool legacy_dsv41_specs =
            runtime_config_.model_name == "DeepSeekV41" || runtime_config_.model_name == "DeepSeekV41DSpark";
        std::vector<std::string> full_group_tags{"global_kv_1", "global_kv_2", "indexer_kv"};
        if (legacy_dsv41_specs
            && std::find(reuse_group_tags.begin(), reuse_group_tags.end(), "hca_kv") != reuse_group_tags.end()) {
            full_group_tags[0] = "hca_kv";
        }
        if (legacy_dsv41_specs
            && std::find(reuse_group_tags.begin(), reuse_group_tags.end(), "csa_kv") != reuse_group_tags.end()) {
            full_group_tags[1] = "csa_kv";
        }
        const auto& group_sets = block_tree_cache_->groupSets();
        // Remote payload and HBM publication must use the same effective wire specs.
        const bool merge_full_specs =
            control_merge_full_specs.has_value() ?
                *control_merge_full_specs :
                std::any_of(group_sets.begin(), group_sets.end(), [&](const auto& group_set) {
                    if (group_set->groupType() != CacheGroupType::FULL) {
                        return false;
                    }
                    const auto& members = group_set->groupTags();
                    return std::all_of(full_group_tags.begin(), full_group_tags.end(), [&](const auto& tag) {
                        return std::find(reuse_group_tags.begin(), reuse_group_tags.end(), tag)
                                   != reuse_group_tags.end()
                               && std::find(members.begin(), members.end(), tag) != members.end();
                    });
                });
        if (merge_full_specs) {
            // One representative still requires every member of the shared FULL GroupSet to be resident.
            reuse_group_tags.erase(
                std::remove_if(reuse_group_tags.begin(),
                               reuse_group_tags.end(),
                               [&](const auto& tag) { return tag == full_group_tags[1] || tag == full_group_tags[2]; }),
                reuse_group_tags.end());
        }

        KVCacheEventPublisherContext publisher_context;
        publisher_context.instance_group = resolveKVCacheEventInstanceGroup(
            kv_cache_config_.kv_cache_event_instance_group,
            control_context ? control_context->instance_group : kv_cache_config_.kvcm_instance_group);
        publisher_context.instance_id  = kv_cache_config_.kv_cache_event_instance_id;
        publisher_context.host_ip_port = kv_cache_config_.kv_cache_event_host_ip_port;
        publisher_context.model_name   = runtime_config_.model_name;
        publisher_context.dtype        = getDataTypeStr(config_.dtype);
        // CoordinatorCacheManager inserts canonical CP keys into the tree at this token stride.
        publisher_context.block_size_tokens = cp_slot_mapper_ && cp_slot_mapper_->isSharded() ?
                                                  cp_slot_mapper_->virtualBlockSize() :
                                                  static_cast<int32_t>(config_.seq_size_per_block);
        // Pipeline parallelism is rejected above because a unique PP owner is
        // not represented in ParallelismConfig yet.
        for (const auto& tag : reuse_group_tags) {
            const auto name = merge_full_specs && tag == full_group_tags[0] ?
                                  "F" :
                                  cacheEventSpecName(publisher_context.block_size_tokens, tag, legacy_dsv41_specs);
            publisher_context.specs.push_back({name, "event_report://" + publisher_context.host_ip_port + "/hbm"});
        }
        publisher_context.tp_size = static_cast<int32_t>(parallelism_config_.tp_size);
        publisher_context.dp_size = static_cast<int32_t>(parallelism_config_.dp_size);
        publisher_context.pp_size = static_cast<int32_t>(parallelism_config_.pp_size);
        publisher_context.dp_rank = static_cast<int32_t>(parallelism_config_.dp_rank);
        publisher_context.use_mla = config_.use_mla;

        const bool valid_event_context =
            !publisher_context.instance_group.empty() && !publisher_context.instance_id.empty()
            && !publisher_context.host_ip_port.empty() && !publisher_context.model_name.empty()
            && !publisher_context.dtype.empty() && publisher_context.block_size_tokens > 0
            && !publisher_context.specs.empty() && publisher_config.snapshot_interval_ms > 0
            && (!control_context || publisher_context.instance_group == control_context->instance_group);
        if (!publish_cache_events || !valid_event_context) {
            if (!control_context) {
                return;
            }
            if (publisher_type == "kvcm") {
                RTP_LLM_LOG_WARNING("HBM event publishing unavailable; retaining KVCM control heartbeats");
            }
            publisher_context              = std::move(*control_context);
            publisher_context.host_ip_port = kv_cache_config_.kv_cache_event_host_ip_port;
            publisher_context.dp_rank      = static_cast<int32_t>(parallelism_config_.dp_rank);
            publish_cache_events           = false;
        }
        publisher_config.publish_cache_events = publish_cache_events;

        std::weak_ptr<BlockTreeCache> weak_shared_cache = block_tree_cache_;
        KVCacheSnapshotProvider       snapshot_provider;
        if (publish_cache_events) {
            snapshot_provider = [weak_shared_cache]() {
                const auto shared_cache = weak_shared_cache.lock();
                if (!shared_cache) {
                    throw std::runtime_error("BlockTreeCache is no longer available");
                }
                return shared_cache->logicalCacheSnapshot();
            };
        }

        std::weak_ptr<StorageBackend> weak_backend         = block_tree_cache_->storageBackend();
        auto                          remote_cache_control = [weak_backend,
                                     instance_group = publisher_context.instance_group,
                                     instance_id = publisher_context.instance_id](bool enabled) {
            if (const auto backend = weak_backend.lock()) {
                const bool previous = backend->remoteCacheEnabled();
                backend->setRemoteCacheEnabled(enabled);
                if (previous != enabled) {
                    RTP_LLM_LOG_INFO("KVCM remote cache control changed: instance_group=%s instance_id=%s "
                                                              "enabled=%d->%d",
                                     instance_group.c_str(),
                                     instance_id.c_str(),
                                     previous,
                                     enabled);
                }
            }
        };
        cache_event_publisher_ = std::make_shared<KVCMPublisher>(publisher_config,
                                                                 publisher_context,
                                                                 std::move(snapshot_provider),
                                                                 std::move(reporter),
                                                                 std::move(remote_cache_control));
        if (publish_cache_events) {
            block_tree_cache_->setEventPublisher(cache_event_publisher_, reuse_group_tags);
        } else {
            block_tree_cache_->setEventPublisher(nullptr, {});
        }
        if (!cache_event_publisher_->start()) {
            RTP_LLM_LOG_WARNING("KV cache event publisher failed to start, type=%s; inference remains enabled",
                                publisher_type.c_str());
            stopCacheEventPublisher();
            cache_event_publisher_.reset();
            return;
        }

        RTP_LLM_LOG_INFO("KVCM publisher started, type=%s cache_events=%d instance_id=%s host=%s "
                         "pp_size=%lld tp_rank=%lld "
                         "dp_rank=%lld",
                         publisher_type.c_str(),
                         publish_cache_events,
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

bool KVCacheManager::initP2PConnector() {
    const bool p2p_enabled =
        pd_sep_config_.role_type == RoleType::PREFILL || pd_sep_config_.role_type == RoleType::DECODE;
    if (!p2p_enabled) {
        return true;
    }

    const auto&       pc = parallelism_config_;
    P2PStageStartupPB startup;
    auto&             local_stage = *startup.mutable_layout();
    local_stage.set_main_layer_begin(config_.global_layer_begin);
    local_stage.set_main_layer_count(config_.layer_num);
    for (int lane = 0; lane < pc.tp_size; ++lane) {
        local_stage.add_worker_indices(pc.pp_rank * pc.tp_size + lane);
    }
    const auto append_layout = [&](const CacheConfig& cfg, uint32_t layers, P2PShardLayoutPB* out) {
        auto shards = ShardLayoutFactory::fromTopology(
            cfg.topology(), pc, pd_sep_config_.role_type, cfg.seq_size_per_block, cfg.local_kv_cache_sharded);
        shards.use_group_cache_transfer_policy = config_.usesGroupCacheTransferPolicy();
        shards.use_mla                         = cfg.use_mla;
        encodeP2PShardLayout(shards, out);
        for (uint32_t i = 0; i < layers; ++i) {
            auto* layer = local_stage.add_layers();
            for (const auto& tag : cfg.topology().layer(i).group_tags) {
                layer->add_tags(tag);
            }
        }
    };
    append_layout(config_, config_.layer_num, local_stage.mutable_main_shards());
    if (!config_.mtp_sub_configs.empty()) {
        // The active proposal module is shared across steps. Extra modules have no Prefill KV.
        const auto& draft = *config_.mtp_sub_configs.front();
        append_layout(draft, draft.layer_num, local_stage.mutable_draft_shards());
    }
    const bool decode = pd_sep_config_.role_type == RoleType::DECODE;
    if (decode) {
        for (const auto& group : config_.topology().groups()) {
            // Capacity includes all allocated KV and scales, including extra draft modules.
            const auto bytes = config_.blockSizeBytesForGroup(group.tag);
            if (bytes > 0) {
                (*startup.mutable_decode_block_bytes())[group.tag] = bytes;
            }
        }
    }
    P2PPipelineLayoutPB           pipeline;
    std::map<std::string, size_t> instance_block_bytes;
    pipeline.set_worker_count(pc.pp_size * pc.tp_size);
    if (pc.pp_size > 1) {
        for (const auto& payload : collectPPStagePayloads(startup.SerializeAsString(), pc.pp_size)) {
            P2PStageStartupPB stage;
            if (!stage.ParseFromString(payload) || !stage.has_layout()
                || (decode && !accumulateDecodeStageBlockBytes(stage, pc.tp_size, instance_block_bytes))) {
                RTP_LLM_LOG_ERROR("P2P connector init failed to read PP stage layout or KV capacity");
                return false;
            }
            *pipeline.add_stages() = std::move(*stage.mutable_layout());
        }
    } else {
        *pipeline.add_stages() = std::move(local_stage);
        // Preserve the existing per-rank capacity for non-PP deployments.
        if (decode && !accumulateDecodeStageBlockBytes(startup, 1, instance_block_bytes)) {
            RTP_LLM_LOG_ERROR("P2P connector init failed to read Decode KV capacity");
            return false;
        }
    }
    p2p_pipeline_layout_ = pipeline.SerializeAsString();

    RTP_LLM_LOG_INFO("KVCacheManager: initializing P2PConnector, role_type=%d, "
                     "pd_rdma_mode=%d, cache_store_rdma_mode=%d, listen_port=%ld",
                     static_cast<int>(pd_sep_config_.role_type),
                     pd_sep_config_.cache_store_rdma_mode ? 1 : 0,
                     cache_store_config_.cache_store_rdma_mode ? 1 : 0,
                     pd_sep_config_.cache_store_listen_port);

    auto p2p_config                                             = P2PConnectorConfig::create(runtime_config_,
                                                 cache_store_config_,
                                                 parallelism_config_,
                                                 pd_sep_config_,
                                                 static_cast<uint32_t>(config_.layer_all_num()),
                                                 config_.totalGroupBlockSizeBytes());
    p2p_config.scheduler_config.topology                        = config_.topologyPtr();
    p2p_config.scheduler_config.pipeline_layout                 = p2p_pipeline_layout_;
    p2p_config.scheduler_config.use_group_cache_transfer_policy = config_.usesGroupCacheTransferPolicy();
    p2p_config.scheduler_config.use_mla                         = config_.use_mla;
    p2p_config.scheduler_config.local_full_page_rr              = config_.local_kv_cache_sharded;
    p2p_config.worker_config.topology                           = config_.topologyPtr();
    p2p_config.worker_config.global_layer_begin                 = config_.global_layer_begin;
    auto& linear_specs                                          = p2p_config.worker_config.linear_specs_by_layer;
    linear_specs.resize(config_.topology().layers().size());
    for (const auto& layer : config_.topology().layers()) {
        for (const auto& tag : layer.group_tags) {
            const auto& spec = config_.physicalGroupForLayer(layer.layer_id, tag).spec;
            if (spec->tp_transfer.mode == TpCacheTransferMode::LINEAR_QKV) {
                linear_specs[layer.layer_id].emplace(tag, spec);
            }
        }
    }
    if (decode) {
        if (config_.mtp_sub_configs.size() > 1) {
            // Both MTP executors reuse module 0 for all proposal steps. The
            // extra allocated modules have no Prefill KV to transfer.
            p2p_config.scheduler_config.decode_transfer_layer_end =
                config_.layer_num + config_.mtp_sub_configs.front()->layer_num;
        }
        p2p_config.scheduler_config.physical_block_bytes_by_group = std::move(instance_block_bytes);
    }
    p2p_config.scheduler_config.cache_key_tokens_per_block = config_.seq_size_per_block;
    p2p_config.worker_config.cache_key_tokens_per_block    = config_.seq_size_per_block;
    const int cp_size =
        parallelism_config_.prefill_cp_config.kv_cache_sharded ? static_cast<int>(parallelism_config_.tp_size) : 1;
    p2p_config.scheduler_config.cp_size = cp_size;
    p2p_config.scheduler_config.cp_rank = static_cast<int>(parallelism_config_.tp_rank) % cp_size;
    p2p_config.worker_config.cp_size    = cp_size;

    auto layer_block_converter = std::make_shared<LayerBlockConverterImpl>(coordinator_manager_);
    auto p2p = std::make_shared<P2PConnector>(std::move(p2p_config), layer_block_converter, metrics_reporter_);
    if (!p2p->init()) {
        RTP_LLM_LOG_ERROR("KVCacheManager: P2PConnector init failed");
        return false;
    }
    p2p_connector_ = std::move(p2p);
    RTP_LLM_LOG_INFO("KVCacheManager: P2PConnector initialized without coordinator");
    return true;
}

void KVCacheManager::reportMetricsLoop() {
    pthread_setname_np(pthread_self(), "kv_cache_stats");
    RTP_LLM_PROFILE_FUNCTION();
    kmonitor::MetricsTags tags;
    constexpr auto        kFrozenLogInterval           = std::chrono::minutes(1);
    auto                  last_frozen_log_time         = std::chrono::steady_clock::now() - kFrozenLogInterval;
    uint64_t              previous_queue_full_count    = 0;
    uint64_t              previous_report_failed_count = 0;
    while (!stop_.load(std::memory_order_acquire)) {
        if (!metrics_reporter_ || !coordinator_manager_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        RtpLLMCacheMetricsCollector global_metrics = collectGlobalCacheMetrics(coordinator_manager_);
        metrics_reporter_->report<RtpLLMCacheMetrics, RtpLLMCacheMetricsCollector>(&tags, &global_metrics);
        if (parallelism_config_.tp_rank == 0 && block_tree_cache_->storageBackend()) {
            RtpLLMRemoteCacheControlMetricsCollector control_metrics;
            control_metrics.enabled = block_tree_cache_->storageBackend()->remoteCacheEnabled();
            metrics_reporter_->report<RtpLLMRemoteCacheControlMetrics, RtpLLMRemoteCacheControlMetricsCollector>(
                &tags, &control_metrics);
        }

        if (cache_event_publisher_) {
            const auto                           status = cache_event_publisher_->status();
            RtpLLMCacheEventDropMetricsCollector event_metrics;
            event_metrics.queue_full_count    = status.queue_full_count - previous_queue_full_count;
            event_metrics.report_failed_count = status.report_failed_count - previous_report_failed_count;
            previous_queue_full_count         = status.queue_full_count;
            previous_report_failed_count      = status.report_failed_count;
            metrics_reporter_->report<RtpLLMCacheEventDropMetrics, RtpLLMCacheEventDropMetricsCollector>(
                &tags, &event_metrics);
        }

        const auto now = std::chrono::steady_clock::now();
        if (isAllocatorOwner() && now - last_frozen_log_time >= kFrozenLogInterval) {
            last_frozen_log_time = now;
            logFrozenCacheBlocks(*block_tree_cache_);
        }

        block_tree_cache_->reportMetrics();
        const std::vector<CachePoolMetrics> tree_pool_metrics_list   = block_tree_cache_->poolMetrics();
        const std::vector<CachePoolMetrics> device_pool_metrics_list = coordinator_manager_->poolMetrics();
        const std::vector<CachePoolMetrics> report_metrics_list =
            mergeCachePoolMetrics(device_pool_metrics_list, tree_pool_metrics_list);
        for (const CachePoolMetrics& report_metrics : report_metrics_list) {
            reportPoolCacheMetrics(metrics_reporter_, report_metrics, isAllocatorOwner());
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));  // 1s
    }
}

void KVCacheManager::handleRead(const P2PConnectorStartLoadRequestPB& request,
                                P2PConnectorStartLoadResponsePB&      response,
                                std::function<bool()>                 is_cancelled) {
    if (p2p_connector_) {
        p2p_connector_->handleRead(request, response, std::move(is_cancelled));
    } else {
        RTP_LLM_LOG_WARNING("handleRead called but P2P connector is not initialized");
    }
}

}  // namespace rtp_llm
