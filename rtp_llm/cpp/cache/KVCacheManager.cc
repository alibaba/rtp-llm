#include "rtp_llm/cpp/cache/KVCacheManager.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <unordered_set>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

namespace {

struct GlobalCacheMetricsSnapshot {
    RtpLLMCacheMetricsCollector collector;
    size_t                      total_blocks         = 0;
    size_t                      available_blocks     = 0;
    size_t                      request_ref_blocks   = 0;
    size_t                      connector_ref_blocks = 0;
};

GlobalCacheMetricsSnapshot collectGlobalCacheMetrics(const KVCacheAllocatorPtr& allocator) {
    GlobalCacheMetricsSnapshot snapshot;
    auto                       shared_cache = allocator->sharedBlockCache();

    snapshot.total_blocks         = allocator->totalBlocksNum();
    snapshot.available_blocks     = allocator->availableBlocksNum();
    snapshot.request_ref_blocks   = allocator->requestRefBlocksNum();
    snapshot.connector_ref_blocks = allocator->connectorRefBlocksNum();

    auto& collector                         = snapshot.collector;
    collector.kv_cache_item_num             = shared_cache ? static_cast<int64_t>(shared_cache->size()) : 0;
    collector.kv_cache_left_seq             = static_cast<int64_t>(allocator->availableTokensNum());
    collector.kv_cache_available_blocks     = static_cast<int64_t>(snapshot.available_blocks);
    collector.kv_cache_request_ref_blocks   = static_cast<int64_t>(snapshot.request_ref_blocks);
    collector.kv_cache_connector_ref_blocks = static_cast<int64_t>(snapshot.connector_ref_blocks);
    collector.kv_cache_free_blocks          = static_cast<int64_t>(allocator->freeBlocksNum());
    collector.kv_cache_used_ratio           = (snapshot.total_blocks == 0) ?
                                                  0.0f :
                                                  static_cast<float>(100.0 * (snapshot.total_blocks - snapshot.available_blocks)
                                                           / static_cast<double>(snapshot.total_blocks));
    collector.mr_cost_time_ms               = allocator->getMrCostTimeMs();

    return snapshot;
}

void logGlobalCacheMetrics(const GlobalCacheMetricsSnapshot& snapshot) {
    RTP_LLM_LOG_INFO("kvc raw global: total=%zu avail=%zu req_ref=%zu con_ref=%zu free=%zu items=%ld ratio=%.4f%%",
                     snapshot.total_blocks,
                     snapshot.available_blocks,
                     snapshot.request_ref_blocks,
                     snapshot.connector_ref_blocks,
                     static_cast<size_t>(snapshot.collector.kv_cache_free_blocks),
                     static_cast<long>(snapshot.collector.kv_cache_item_num),
                     snapshot.collector.kv_cache_used_ratio);
}

void reportPoolCacheMetrics(const kmonitor::MetricsReporterPtr& metrics_reporter,
                            const KVCachePoolMetricsSnapshot&   pool_snapshot,
                            bool                                should_log) {
    if (should_log) {
        RTP_LLM_LOG_INFO("kvc raw pool[%s]: total=%zu avail=%zu req_ref=%zu con_ref=%zu free=%zu reserve=%zu "
                         "ratio=%.4f%%",
                         pool_snapshot.pool_name.c_str(),
                         pool_snapshot.total_blocks,
                         pool_snapshot.available_blocks,
                         pool_snapshot.request_ref_blocks,
                         pool_snapshot.connector_ref_blocks,
                         pool_snapshot.free_blocks,
                         pool_snapshot.reserve_blocks,
                         pool_snapshot.used_ratio);
    }

    RtpLLMCachePoolMetricsCollector pool_collector;
    pool_collector.free_blocks          = static_cast<int64_t>(pool_snapshot.free_blocks);
    pool_collector.available_blocks     = static_cast<int64_t>(pool_snapshot.available_blocks);
    pool_collector.request_ref_blocks   = static_cast<int64_t>(pool_snapshot.request_ref_blocks);
    pool_collector.connector_ref_blocks = static_cast<int64_t>(pool_snapshot.connector_ref_blocks);
    pool_collector.total_blocks         = static_cast<int64_t>(pool_snapshot.total_blocks);
    pool_collector.reserve_blocks       = static_cast<int64_t>(pool_snapshot.reserve_blocks);
    pool_collector.used_ratio           = pool_snapshot.used_ratio;

    kmonitor::MetricsTags pool_tags("pool_name", pool_snapshot.pool_name);
    metrics_reporter->report<RtpLLMCachePoolMetrics, RtpLLMCachePoolMetricsCollector>(&pool_tags, &pool_collector);
}

std::shared_ptr<const CacheTopology> projectTopology(const CacheTopology&       source,
                                                     const std::vector<size_t>& global_layer_ids) {
    std::vector<GroupBase> groups;
    groups.reserve(source.groups().size());
    std::unordered_map<std::string, GroupBase*> groups_by_tag;
    for (const auto& source_group : source.groups()) {
        groups.push_back(source_group);
        groups_by_tag.emplace(groups.back().tag, &groups.back());
    }
    for (auto& group : groups) {
        group.layer_ids.clear();
    }

    std::vector<LayerBase> layers;
    layers.reserve(global_layer_ids.size());
    for (size_t local_layer_id = 0; local_layer_id < global_layer_ids.size(); ++local_layer_id) {
        const auto& source_layer = source.layer(static_cast<int>(global_layer_ids[local_layer_id]));
        LayerBase   layer;
        layer.layer_id   = static_cast<int>(local_layer_id);
        layer.group_tags = source_layer.group_tags;
        for (const auto& tag : layer.group_tags) {
            groups_by_tag.at(tag)->layer_ids.push_back(static_cast<int>(local_layer_id));
        }
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
        for (int local_layer_id : target_group.layer_ids) {
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
                               bool                               use_cuda_malloc_block_pool):
    config_(config),
    metrics_reporter_(metrics_reporter),
    kv_cache_config_(kv_cache_config),
    parallelism_config_(parallelism_config),
    runtime_config_(runtime_config),
    sp_config_(sp_config),
    pd_sep_config_(pd_sep_config),
    cache_store_config_(cache_store_config),
    use_cuda_malloc_block_pool_(use_cuda_malloc_block_pool) {
    if (warmup) {
        config_.applyTokenCapacity(/*capacity_tokens=*/1);
    } else {
        allocateAndSync();
    }

    const auto& cp_cfg = parallelism_config_.prefill_cp_config;
    if (cp_cfg.kv_cache_sharded && parallelism_config_.tp_size > 1) {
        for (const auto& group : config_.topology().groups()) {
            auto mapper = std::make_shared<CPSlotMapper>(static_cast<int>(parallelism_config_.tp_rank),
                                                         static_cast<int>(parallelism_config_.tp_size),
                                                         static_cast<int>(group.seq_size_per_block));
            RTP_LLM_LOG_INFO("CP sharded KV cache tag=%s cp_rank=%d cp_size=%d physical_span=%zu "
                             "virtual_block_size=%d",
                             group.tag.c_str(),
                             (int)parallelism_config_.tp_rank,
                             (int)parallelism_config_.tp_size,
                             group.seq_size_per_block,
                             mapper->virtualBlockSize());
            cp_slot_mappers_.emplace(group.tag, std::move(mapper));
        }
    }

    RTP_LLM_LOG_INFO("cache config: layer_num=%d, token_capacity=%lu", config_.layer_num, config_.tokenCapacity());
}

KVCacheManager::~KVCacheManager() {
    stop_.store(true, std::memory_order_relaxed);
    if (metrics_reporter_thread_.joinable()) {
        metrics_reporter_thread_.join();
    }
    allocator_.reset();
    coordinator_.reset();
}

// 初始化和配置相关

bool KVCacheManager::init() {
    RTP_LLM_CHECK_WITH_INFO(!allocator_ && !coordinator_ && !metrics_reporter_thread_.joinable(),
                            "KVCacheManager::init called more than once");
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() > 0, "cache specs must not be empty");

    auto shared_cache = std::make_shared<SharedBlockCache>();
    shared_cache->setPrefixTreeEnabled(kv_cache_config_.enable_gpu_prefix_tree);
    const bool enable_independent_group_eviction = kv_cache_config_.enable_memory_cache
                                                   && kv_cache_config_.enable_prefix_tree_memory_cache
                                                   && kv_cache_config_.enable_independent_group_eviction;

    allocator_ = std::make_shared<rtp_llm::KVCacheAllocator>(config_,
                                                             AllocationType::DEVICE,
                                                             metrics_reporter_,
                                                             kv_cache_config_.reserve_block_ratio,
                                                             pd_sep_config_.role_type);

    if (use_cuda_malloc_block_pool_) {
        RTP_LLM_LOG_INFO("RDMA cache store enabled for PD role, use cudaMalloc KV cache block-pool backing");
        allocator_->setUseCudaMallocBlockPool(true);
    }

    allocator_->setCPSlotMappers(cp_slot_mappers_);
    allocator_->setSharedBlockCache(shared_cache);
    RTP_LLM_CHECK_WITH_INFO(allocator_->init(), "KVCacheAllocator init failed");
    shared_cache->setIndependentGroupEviction(enable_independent_group_eviction, allocator_->independentEvictionTags());

    if (metrics_reporter_) {
        stop_.store(false, std::memory_order_relaxed);
        metrics_reporter_thread_ = std::thread(&KVCacheManager::reportMetricsLoop, this);
    }

    initConnectorCoordinator();
    return true;
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
    RTP_LLM_CHECK(malloc_info.batch_kv_cache_resource && malloc_info.complete_token_ids);

    if (!malloc_info.batch_kv_cache_resource->curBlocksNum()) {
        initCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids);
    } else {
        updateCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids);
    }

    return allocator_->malloc(malloc_info);
}

void KVCacheManager::free(const FreeInfo& free_info) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK(free_info.batch_kv_cache_resource && free_info.complete_token_ids);
    allocator_->free(free_info);
}

void KVCacheManager::insertIntoCache(const InsertInfo& insert_info) {
    RTP_LLM_PROFILE_FUNCTION();
    dropLastPartialBlock(insert_info.batch_kv_cache_resource);
    allocator_->insertIntoCache(insert_info);
}

int KVCacheManager::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                          int                            seq_len,
                                          int                            reserve_step) const {
    RTP_LLM_CHECK_WITH_INFO(allocator_ != nullptr, "singleBatchNeedBlocks called before KVCacheManager initialized");
    return allocator_->singleBatchNeedBlocks(batch_kv_cache_resource, seq_len, reserve_step);
}

int KVCacheManager::estimatePeakNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                           int                            seq_len,
                                           int                            common_seq_len,
                                           int                            remaining_tokens,
                                           int                            reserve_step,
                                           bool                           enable_reuse_cache,
                                           int                            target_batch_size) const {
    return allocator_->estimateBatchPeakNeedBlocks(batch_kv_cache_resource,
                                                   seq_len,
                                                   common_seq_len,
                                                   remaining_tokens,
                                                   reserve_step,
                                                   enable_reuse_cache,
                                                   target_batch_size);
}

// 块操作相关

void KVCacheManager::blockBatchCopy(const std::vector<GroupBlockIdPair>& copy_mapping) {
    return allocator_->blockBatchCopy(copy_mapping);
}

bool KVCacheManager::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                   const std::vector<int>&        block_src_batch,
                                   bool                           copy_last_block,
                                   std::vector<GroupBlockIdPair>& block_update_mapping) {
    RTP_LLM_PROFILE_FUNCTION();
    return allocator_->updateKVBlock(batch_kv_cache_resource, block_src_batch, copy_last_block, block_update_mapping);
}

// 地址转换和缓冲区访问

BlockAddrInfo KVCacheManager::convertIndexToAddr(int block_index, int layer_id, const std::string& tag) const {
    return allocator_->convertIndexToAddr(layer_id, tag, block_index);
}

std::vector<BlockInfo>
KVCacheManager::convertIndexToBuffer(int block_index, int layer_id, const std::string& tag) const {
    return allocator_->convertIndexToBuffer(layer_id, tag, block_index);
}

std::vector<BlockInfo> KVCacheManager::convertIndexToBuffer(
    int block_index, int layer_id, const std::string& tag, int partition_count, int partition_id) const {
    return allocator_->convertIndexToBuffer(layer_id, tag, block_index, partition_count, partition_id);
}

GroupedCacheLayerLayout KVCacheManager::allLayerCacheBase() const {
    return allocator_->allLayerCacheBase();
}

GroupedCacheLayerLayout KVCacheManager::getMainModelGroupedCacheLayerLayout() const {
    const auto          all_layout = allocator_->allLayerCacheBase();
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
    return projectLayout(allocator_->allLayerCacheBase(), mtp_sub_config->topologyPtr(), global_layer_ids);
}

// 资源统计和信息查询

size_t KVCacheManager::freeBlocksNum() const {
    return allocator_->freeBlocksNum();
}

size_t KVCacheManager::availableBlocksNum() const {
    return allocator_->availableBlocksNum();
}

size_t KVCacheManager::reserveBlocksNum() const {
    return allocator_->reserveBlocksNum();
}

size_t KVCacheManager::notInUseBlocksNum() const {
    return allocator_->notInUseBlocksNum();
}

BatchKVCacheResourcePtr KVCacheManager::popBlocksFromCache(size_t min_blocks_to_free) {
    return allocator_->popBlocksFromCache(min_blocks_to_free);
}

void KVCacheManager::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    allocator_->blockCacheFree(batch_kv_cache_resource);
}

size_t KVCacheManager::availableTokensNum() const {
    return allocator_->availableTokensNum();
}

size_t KVCacheManager::totalBlocksNum() const {
    return allocator_->totalBlocksNum();
}

size_t KVCacheManager::maxAvailableTokensNum() const {
    return allocator_->maxAvailableTokensNum();
}

KVCacheInfo KVCacheManager::getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const {
    KVCacheInfo info;
    info.version = latest_version;

    if (!allocator_) {
        RTP_LLM_LOG_ERROR("getKVCacheInfo called before KVCacheManager initialized");
        return info;
    }

    if (need_cache_keys) {
        std::unordered_set<CacheKeyType> all_keys;
        // device cache keys
        std::vector<CacheKeyType> device_cache_keys;
        auto                      shared_cache = allocator_->sharedBlockCache();
        if (shared_cache) {
            device_cache_keys = shared_cache->allCacheKeys();
            all_keys.insert(device_cache_keys.begin(), device_cache_keys.end());
            info.version = shared_cache->version();
        }
        // memory cache keys
        RTP_LLM_CHECK_WITH_INFO(coordinator_ != nullptr,
                                "getKVCacheInfo called before KVCacheManager coordinator initialized");
        const auto mem_cache_keys = coordinator_->memoryCacheKeys();
        all_keys.insert(mem_cache_keys.begin(), mem_cache_keys.end());

        info.cached_keys.assign(all_keys.begin(), all_keys.end());
    }

    const size_t block_size_tokens =
        config_.groupNums() == 1 ? config_.topology().groups().front().seq_size_per_block : 0;
    const auto capacity     = allocator_->tokenCapacity();
    info.block_size         = block_size_tokens;
    info.total_kv_cache     = capacity.total_tokens;
    info.available_kv_cache = capacity.available_tokens;

    return info;
}

// 系统资源管理

void KVCacheManager::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    allocator_->regUserMr(model_id, std::move(cache_store));
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
    return coordinator_ && coordinator_->hasActiveConnectors();
}

// PD separation: increment KV cache reference count
std::shared_ptr<KVCacheResource> KVCacheManager::incrKVCacheRef(const KVCacheResource&  resource,
                                                                const CacheKeysByGroup& cache_keys_by_group,
                                                                bool                    is_connector) {
    return allocator_->incrKVCacheRef(resource, cache_keys_by_group, is_connector);
}

bool KVCacheManager::hasP2PConnector() const {
    return coordinator_ && coordinator_->hasP2PConnector();
}

// 异步连接器操作

std::shared_ptr<AsyncContext>
KVCacheManager::asyncLoadCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK_WITH_INFO(coordinator_ != nullptr, "asyncLoadCache called before KVCacheManager initialized");
    return coordinator_->asyncRead(connector_context);
}

std::shared_ptr<AsyncContext>
KVCacheManager::asyncStoreCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK_WITH_INFO(coordinator_ != nullptr, "asyncStoreCache called before KVCacheManager initialized");
    return coordinator_->asyncWrite(connector_context);
}

bool KVCacheManager::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    RTP_LLM_CHECK_WITH_INFO(coordinator_ != nullptr, "executeFunction called before KVCacheManager initialized");
    return coordinator_->executeFunction(request, response);
}

void KVCacheManager::initConnectorCoordinator() {
    RTP_LLM_LOG_INFO(
        "init connector coordinator, cache config: [%s], kv cache config: [%s], runtime config: [%s], parallelism config: [%s], sp config: [%s]",
        config_.debugString().c_str(),
        kv_cache_config_.to_string().c_str(),
        runtime_config_.to_string().c_str(),
        parallelism_config_.to_string().c_str(),
        sp_config_.to_string().c_str());
    coordinator_ = std::make_shared<KVCacheConnectorCoordinator>(config_,
                                                                 kv_cache_config_,
                                                                 runtime_config_,
                                                                 parallelism_config_,
                                                                 sp_config_,
                                                                 allocator_,
                                                                 metrics_reporter_,
                                                                 pd_sep_config_,
                                                                 cache_store_config_);
    RTP_LLM_CHECK_WITH_INFO(coordinator_->init(), "connector coordinator init failed");
}

void KVCacheManager::allocateAndSync() {
    uint64_t capacity_tokens = config_.tokenCapacity();
    RTP_LLM_LOG_INFO("allocateAndSync start, capacity_tokens=%lu", capacity_tokens);
    size_t world_size = parallelism_config_.tp_size * parallelism_config_.dp_size;
    if (world_size > 1) {
        size_t local_rank   = parallelism_config_.tp_size * parallelism_config_.dp_rank + parallelism_config_.tp_rank;
        auto   capacity_t   = torch::empty({(int64_t)world_size}, torch::kInt64).pin_memory();
        auto   capacity_ptr = capacity_t.data_ptr<int64_t>();
        RTP_LLM_CHECK_WITH_INFO(capacity_tokens <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                                "cache capacity exceeds collective int64 range");
        capacity_ptr[local_rank] = static_cast<int64_t>(capacity_tokens);
        execAllGather({{capacity_t}, ParallelMode::DP_AND_TP});
        execSyncCommunication(false);
        cudaSyncAndCheck();

        if (parallelism_config_.ffn_disaggregate_config.is_ffn_service()) {
            capacity_tokens = 1;
        } else {
            capacity_tokens = static_cast<uint64_t>(*std::min_element(capacity_ptr, capacity_ptr + world_size));
        }
    }
    config_.applyTokenCapacity(capacity_tokens);
    RTP_LLM_LOG_INFO("capacity_tokens is %lu after TP/DP sync", capacity_tokens);
}

void KVCacheManager::reportMetricsLoop() {
    RTP_LLM_PROFILE_FUNCTION();
    kmonitor::MetricsTags tags;
    constexpr auto        kLogInterval  = std::chrono::minutes(1);
    auto                  last_log_time = std::chrono::steady_clock::now() - kLogInterval;
    while (!stop_.load(std::memory_order_relaxed)) {
        if (!metrics_reporter_ || !allocator_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        auto global_metrics = collectGlobalCacheMetrics(allocator_);
        metrics_reporter_->report<RtpLLMCacheMetrics, RtpLLMCacheMetricsCollector>(&tags, &global_metrics.collector);

        const auto now        = std::chrono::steady_clock::now();
        const bool should_log = (now - last_log_time) >= kLogInterval;
        if (should_log) {
            last_log_time = now;
            logGlobalCacheMetrics(global_metrics);
        }

        for (const auto& pool_snapshot : allocator_->poolMetricsSnapshots()) {
            reportPoolCacheMetrics(metrics_reporter_, pool_snapshot, should_log);
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));  // 1s
    }
}

void KVCacheManager::handleRead(const P2PConnectorStartLoadRequestPB& request,
                                P2PConnectorStartLoadResponsePB&      response,
                                std::function<bool()>                 is_cancelled) {
    RTP_LLM_CHECK_WITH_INFO(coordinator_ != nullptr, "handleRead called before KVCacheManager initialized");
    coordinator_->handleRead(request, response, is_cancelled);
}

// Write one KV block (optionally per-layer) from host/device tensors for test
bool KVCacheManager::writeKVBlockForTest(int                  block_index,
                                         int                  layer_id,
                                         const std::string&   tag,
                                         const torch::Tensor& k_buffer,
                                         const torch::Tensor& v_buffer) {
    // Basic size/type validation to prevent out-of-bounds copy
    const auto& spec             = config_.specForGroup(tag);
    size_t      expected_k_bytes = spec->k_block_size_bytes();
    size_t      expected_v_bytes = spec->v_block_size_bytes();
    size_t      src_k_bytes      = k_buffer.nbytes();
    size_t      src_v_bytes      = v_buffer.nbytes();
    if (src_k_bytes < expected_k_bytes || src_v_bytes < expected_v_bytes) {
        RTP_LLM_LOG_ERROR("writeKVBlockForTest src bytes too small: k[%zu]<[%zu] or v[%zu]<[%zu]",
                          src_k_bytes,
                          expected_k_bytes,
                          src_v_bytes,
                          expected_v_bytes);
        return false;
    }

    auto dst = allocator_->convertIndexToBuffer(layer_id, tag, block_index);
    RTP_LLM_CHECK_WITH_INFO(
        !dst.empty(), "convertIndexToBuffer returned empty for layer %d, block %d", layer_id, block_index);
    if (!dst[0].addr) {
        RTP_LLM_LOG_ERROR("convertIndexToBuffer returned null for layer %d, block %d", layer_id, block_index);
        return false;
    }

    auto copyFunc = [&](const torch::Tensor& src_tensor,
                        const BlockInfo&     dst_block,
                        size_t               dst_byte_offset,
                        size_t               copy_bytes) -> bool {
        const size_t dst_bytes = dst_block.size_bytes;
        if (dst_bytes < dst_byte_offset + copy_bytes) {
            RTP_LLM_LOG_ERROR(
                "dst block bytes[%zu] < dst_offset[%zu] + copy bytes[%zu] in writeKVBlockForTest(layer=%d)",
                dst_bytes,
                dst_byte_offset,
                copy_bytes,
                layer_id);
            return false;
        }

        auto* dst_ptr    = static_cast<char*>(dst_block.addr) + dst_byte_offset;
        auto  dst_device = dst_block.is_cuda ? torch::kCUDA : torch::kCPU;
        auto  src_device = src_tensor.is_cuda() ? torch::kCUDA : torch::kCPU;
        auto  dst_t      = torch::from_blob(
            dst_ptr, {(int64_t)copy_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(dst_device));
        auto src_t = torch::from_blob(src_tensor.data_ptr(),
                                      {(int64_t)copy_bytes},
                                      torch::TensorOptions().dtype(torch::kUInt8).device(src_device));
        dst_t.copy_(src_t);
        return true;
    };

    if (!copyFunc(k_buffer, dst[0], 0, expected_k_bytes)) {
        return false;
    }

    if (!copyFunc(v_buffer, dst[0], expected_k_bytes, expected_v_bytes)) {
        return false;
    }

    cudaSyncAndCheck();
    return true;
}

bool KVCacheManager::writeKVBlockForTest(int                  block_index,
                                         const std::string&   tag,
                                         const torch::Tensor& k_buffer,
                                         const torch::Tensor& v_buffer) {
    const auto block_num = config_.blockNumForGroup(tag);
    if (block_index < 0 || static_cast<uint32_t>(block_index) >= block_num) {
        RTP_LLM_LOG_WARNING(
            "Invalid block_index: %d, valid range for tag=%s: [0, %u)", block_index, tag.c_str(), block_num);
        return false;
    }

    bool all_success = true;
    for (int layer_id : config_.layerIdsForGroup(tag)) {
        all_success = writeKVBlockForTest(block_index, layer_id, tag, k_buffer, v_buffer) && all_success;
    }
    return all_success;
}

}  // namespace rtp_llm
