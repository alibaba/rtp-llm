#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"

#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverterImpl.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/ZeroSwaCacheHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#ifdef USE_REMOTE_KV_CACHE
#include "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.h"
#endif

namespace rtp_llm {
namespace {

CacheGroupType groupTypeForConnector(const CacheConfig& cache_config, int group_id) {
    if (group_id >= 0 && static_cast<size_t>(group_id) < cache_config.group_types.size()) {
        return cache_config.group_types[static_cast<size_t>(group_id)];
    }
    return CacheGroupType::FULL;
}

bool selectedLastRankKeysAreAligned(const KVCacheResource& source, int cp_size) {
    if (source.lastBlockAligned()) {
        return true;
    }
    const auto& keys = source.cacheKeys();
    if (keys.empty() || cp_size <= 1) {
        return source.lastBlockAligned();
    }
    const int partial_key_pos = static_cast<int>(keys.size() - 1);
    const int last_rank       = cp_size - 1;
    return partial_key_pos % cp_size != last_rank;
}

KVCacheResource makeCpShardedConnectorResource(const KVCacheResource& source,
                                               const CacheConfig&     cache_config,
                                               const CacheKeysType&   selected_keys,
                                               int                    cp_size) {
    std::vector<CacheGroupType> group_types;
    group_types.reserve(static_cast<size_t>(source.groupNums()));
    for (int gid = 0; gid < source.groupNums(); ++gid) {
        group_types.push_back(groupTypeForConnector(cache_config, gid));
    }

    KVCacheResource selected = source;
    selected.initGroups(source.groupNums(),
                        static_cast<int>(cache_config.layer_all_num),
                        cache_config.layer_to_group_id,
                        cache_config.kernelBlocksPerKvBlock(),
                        group_types,
                        cache_config.layer_region_to_group_id);
    selected.cacheKeys()        = selected_keys;
    const bool selected_aligned = selectedLastRankKeysAreAligned(source, cp_size);
    selected.setLastBlockAligned(selected_aligned);

    // Memory connector intentionally drops the last key to avoid matching a
    // partial tail.  After CP Page-RR remap, a source partial can belong to a
    // non-last rank, making the selected last-rank key complete.  Append the
    // original partial key as a connector-only dummy tail so the drop-last
    // contract discards the dummy, not the usable selected key.
    if (!source.lastBlockAligned() && selected_aligned && !source.cacheKeys().empty()) {
        selected.cacheKeys().push_back(source.cacheKeys().back());
        selected.setLastBlockAligned(false);
    }

    for (int gid = 0; gid < source.groupNums(); ++gid) {
        const auto&      src_blocks = source.blocks(gid);
        BlockIndicesType dst_blocks;
        dst_blocks.reserve(selected_keys.size());

        if (group_types[static_cast<size_t>(gid)] == CacheGroupType::FULL) {
            // FULL groups are physically RR-sharded: blocks are rank-local and
            // compact, keyed by the canonical last-rank key sequence.
            for (size_t i = 0; i < selected_keys.size(); ++i) {
                dst_blocks.push_back(i < src_blocks.size() ? src_blocks[i] : NULL_BLOCK_IDX);
            }
        } else {
            // SWA/state groups keep the non-sharded logical coordinate system.
            // Select the block at the original logical key position instead of
            // reinterpreting the group as rank-local compact storage.
            for (size_t logical_pos = static_cast<size_t>(cp_size - 1); dst_blocks.size() < selected_keys.size();
                 logical_pos += static_cast<size_t>(cp_size)) {
                dst_blocks.push_back(logical_pos < src_blocks.size() ? src_blocks[logical_pos] : NULL_BLOCK_IDX);
            }
        }

        selected.mutableBlockIds(gid).assign(std::move(dst_blocks));
    }

    return selected;
}

}  // namespace

KVCacheConnectorCoordinator::KVCacheConnectorCoordinator(const CacheConfig&                       cache_config,
                                                         const KVCacheConfig&                     kv_cache_config,
                                                         const RuntimeConfig&                     runtime_config,
                                                         const ParallelismConfig&                 parallelism_config,
                                                         const SpeculativeExecutionConfig&        sp_config,
                                                         const std::shared_ptr<KVCacheAllocator>& allocator,
                                                         const kmonitor::MetricsReporterPtr&      metrics_reporter,
                                                         const PDSepConfig&                       pd_sep_config,
                                                         const CacheStoreConfig&                  cache_store_config):
    cache_config_(cache_config),
    kv_cache_config_(kv_cache_config),
    runtime_config_(runtime_config),
    parallelism_config_(parallelism_config),
    sp_config_(sp_config),
    allocator_(allocator),
    metrics_reporter_(metrics_reporter),
    pd_sep_config_(pd_sep_config),
    cache_store_config_(cache_store_config) {}

KVCacheConnectorCoordinator::~KVCacheConnectorCoordinator() {
    stop_.store(true);
    // release all connectors to make sure all async context done
    memory_connector_.reset();
    connectors_.clear();
    // connectors already released, all async context should be done
    autil::ScopedTime2 timer;
    while (true) {
        if (timer.done_ms() > update_interval_ms_ * 2) {
            RTP_LLM_LOG_WARNING(
                "coordinator destructor timeout, read or write list not empty, timeout: %d ms, read list size: %zu, write list size: %zu",
                timer.done_ms(),
                fused_async_read_context_list_.size(),
                fused_async_write_context_list_.size());
            break;
        }
        {
            std::lock_guard<std::mutex> lock(update_mutex_);
            if (fused_async_read_context_list_.empty() && fused_async_write_context_list_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (update_thread_) {
        update_thread_->stop();
        update_thread_.reset();
    }
}

bool KVCacheConnectorCoordinator::hasActiveConnectors() const {
    return !connectors_.empty();
}

bool KVCacheConnectorCoordinator::hasP2PConnector() const {
    return p2p_connector_ != nullptr;
}

bool KVCacheConnectorCoordinator::init() {
    RTP_LLM_LOG_INFO("connector coordinator init, cache config: [%s], kv cache config: [%s], runtime config: [%s]",
                     cache_config_.debugString().c_str(),
                     kv_cache_config_.to_string().c_str(),
                     runtime_config_.to_string().c_str());
    const bool zero_swa_remote_cache =
        isZeroSwaCachingEnabled(cache_config_) && kv_cache_config_.reuse_cache && kv_cache_config_.enable_remote_cache;
    RTP_LLM_CHECK_WITH_INFO(!zero_swa_remote_cache,
                            "DSV4 zero SWA caching does not support remote KV cache yet; disable remote cache or unset "
                            "DSV4_ZERO_SWA_CACHING");
    if (kv_cache_config_.reuse_cache && kv_cache_config_.enable_memory_cache) {
        // Zero-SWA memory/disk cache entries omit SWA_KV slots and are not layout-compatible
        // with full-SWA entries. The online memory cache is process-local, so rolling
        // deployments do not share these entries across old/new prefill processes. Keep
        // persistent/shared memory-disk namespaces isolated or cleared when toggling this flag.
        if (isZeroSwaCachingEnabled(cache_config_)) {
            RTP_LLM_LOG_WARNING("DSV4 zero SWA caching + memory cache: cache entries OMIT SWA_KV and are NOT "
                                "layout-compatible with full-SWA entries. This is safe for process-local memory cache; "
                                "persistent/shared namespaces must be isolated or cleared when toggling the flag.");
        }
        memory_connector_ = initMemoryConnector();
        connectors_.emplace_back(memory_connector_);
    }
#ifdef USE_REMOTE_KV_CACHE
    if (kv_cache_config_.reuse_cache && kv_cache_config_.enable_remote_cache) {
        remote_connector_ = initRemoteConnector();
        connectors_.emplace_back(remote_connector_);
    }
#endif
    if (!initP2PConnectorInternal()) {
        RTP_LLM_LOG_WARNING("init P2P connector failed, P2P path disabled — engine continues without it");
    }
    initUpdateThread();
    return true;
}

void KVCacheConnectorCoordinator::initUpdateThread() {
    update_thread_ = autil::LoopThread::createLoopThread(
        [this]() { updateOnce(); }, update_interval_ms_ * 1000, "CoordinatorUpdateThread");
    RTP_LLM_CHECK_WITH_INFO(update_thread_ != nullptr, "init update thread failed");
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncRead(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    RTP_LLM_PROFILE_FUNCTION();
    if (stop_.load()) {
        return nullptr;
    }
    if (!connector_context) {
        RTP_LLM_LOG_WARNING("async read failed, connector context is null");
        return nullptr;
    }
    const auto& kvcache_resource = connector_context->kvCacheResource();
    // empty cache keys will not handled by coordinator.
    if (kvcache_resource.cacheKeys().empty()) {
        return nullptr;
    }

    const int       cp_size      = cpSize();
    CacheKeysType   ref_keys     = kvcache_resource.cacheKeys();
    KVCacheResource ref_resource = kvcache_resource;
    if (cp_size > 1) {
        ref_keys = kvcache_resource.localCacheKeys(cp_size - 1, cp_size);
        // Short requests (< cp_size logical blocks) have no complete virtual
        // block, so the canonical last-rank-key namespace is empty by design.
        // Skip silently — connector activity for these is a no-op anyway.
        if (ref_keys.empty()) {
            return nullptr;
        }
        ref_resource = makeCpShardedConnectorResource(kvcache_resource, cache_config_, ref_keys, cp_size);
        ref_keys     = ref_resource.cacheKeys();
    }
    auto resource = allocator_->incrKVCacheRef(ref_resource, ref_keys, true);
    if (!resource) {
        RTP_LLM_LOG_WARNING("async read failed, incr kvcache ref failed, resource: [%s]",
                            kvcache_resource.debugString().c_str());
        return nullptr;
    }

    std::vector<std::shared_ptr<AsyncContext>> match_contexts(connectors_.size());
    for (int i = 0; i < connectors_.size(); i++) {
        match_contexts.at(i) = connectors_.at(i)->asyncMatch(resource, connector_context->meta());
    }

    auto fused_match_context = std::make_shared<FusedAsyncContext>(std::move(match_contexts));
    auto fused_read_context =
        std::make_shared<FusedAsyncReadContext>(fused_match_context, resource, connector_context->meta());
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        fused_async_read_context_list_.push_back(fused_read_context);
    }
    return fused_read_context;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncWrite(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    if (stop_.load()) {
        return nullptr;
    }
    if (!connector_context) {
        RTP_LLM_LOG_WARNING("async write failed, connector context is null");
        return nullptr;
    }
    const auto& kvcache_resource = connector_context->kvCacheResource();
    if (kvcache_resource.cacheKeys().empty()) {
        RTP_LLM_LOG_DEBUG("async write failed, kvcache resource cache keys is empty, resource: [%s]",
                          kvcache_resource.debugString().c_str());
        return nullptr;
    }

    const int       cp_size      = cpSize();
    CacheKeysType   ref_keys     = kvcache_resource.cacheKeys();
    KVCacheResource ref_resource = kvcache_resource;
    if (cp_size > 1) {
        ref_keys = kvcache_resource.localCacheKeys(cp_size - 1, cp_size);
        if (ref_keys.empty()) {
            return nullptr;  // request shorter than one virtual block — nothing to write
        }
        ref_resource = makeCpShardedConnectorResource(kvcache_resource, cache_config_, ref_keys, cp_size);
        ref_keys     = ref_resource.cacheKeys();
    }
    auto resource = allocator_->incrKVCacheRef(ref_resource, ref_keys, true);
    if (!resource) {
        RTP_LLM_LOG_WARNING("async write failed, incr kvcache ref failed, resource: [%s]",
                            kvcache_resource.debugString().c_str());
        return nullptr;
    }

    std::vector<std::shared_ptr<AsyncContext>> write_contexts(connectors_.size());
    for (int i = 0; i < connectors_.size(); i++) {
        write_contexts.at(i) = connectors_.at(i)->asyncWrite(resource, connector_context->meta());
    }

    auto fused_write_context = std::make_shared<FusedAsyncContext>(std::move(write_contexts));
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        fused_async_write_context_list_.push_back(fused_write_context);
    }
    return fused_write_context;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncWriteByLayer(int                                                  layer_id,
                                               const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) {
    if (!p2p_connector_) {
        return nullptr;
    }
    if (!layer_context) {
        RTP_LLM_LOG_WARNING("asyncWriteByLayer: layer_context is null, skip P2P write for layer %d", layer_id);
        return nullptr;
    }
    if (layer_id == 0) {
        RTP_LLM_LOG_INFO("asyncWriteByLayer [P2P]: dispatching layer_id=%d, request_id=%ld to P2PConnector",
                         layer_id,
                         layer_context->requestId());
    }
    return p2p_connector_->asyncWriteByLayer(layer_id, layer_context);
}

std::shared_ptr<KVCacheMemoryConnector> KVCacheConnectorCoordinator::initMemoryConnector() {
    auto memory_connector = std::make_shared<KVCacheMemoryConnector>(cache_config_,
                                                                     kv_cache_config_,
                                                                     parallelism_config_,
                                                                     allocator_,
                                                                     runtime_config_.worker_grpc_addrs,
                                                                     metrics_reporter_);
    RTP_LLM_CHECK_WITH_INFO(memory_connector->init(), "memory connector init failed");
    return memory_connector;
}

std::shared_ptr<RemoteConnector> KVCacheConnectorCoordinator::initRemoteConnector() {
#ifdef USE_REMOTE_KV_CACHE
    // TODO : get lora info map
    auto remote_connector_ = std::make_shared<RemoteConnector>(cache_config_,
                                                               kv_cache_config_,
                                                               runtime_config_,
                                                               parallelism_config_,
                                                               sp_config_,
                                                               allocator_->getBlockPool()->getBaseAddress(),
                                                               allocator_->getBlockPool()->getTotalSizeBytes(),
                                                               allocator_,
                                                               metrics_reporter_);

    RTP_LLM_CHECK_WITH_INFO(remote_connector_->init(), "remote connector init failed");
    return remote_connector_;
#else
    RTP_LLM_LOG_ERROR("not RemoteConnector");
    return nullptr;
#endif
}

int KVCacheConnectorCoordinator::cpSize() const {
    const auto& cp_cfg = parallelism_config_.prefill_cp_config;
    if (cp_cfg.kv_cache_sharded && parallelism_config_.tp_size > 1) {
        return static_cast<int>(parallelism_config_.tp_size);
    }
    return 1;
}

void KVCacheConnectorCoordinator::updateOnce() {
    RTP_LLM_PROFILE_FUNCTION();
    processReadContexts();
    processWriteContexts();
}

void KVCacheConnectorCoordinator::processReadContexts() {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(update_mutex_);
    for (auto it = fused_async_read_context_list_.begin(); it != fused_async_read_context_list_.end();) {
        auto fused_read_context = *it;
        if (fused_read_context->done()) {
            fused_read_context->notifyDone();
            it = fused_async_read_context_list_.erase(it);
            continue;
        }
        // 没有 done 但是有 read context, 或者 match context 还没 done, 或者 match 失败 (下一轮调度处理), 继续等待
        auto read_context  = fused_read_context->fusedReadContext();
        auto match_context = fused_read_context->fusedMatchContext();
        if (read_context || !match_context->done() || !match_context->success()) {
            it = std::next(it);
            continue;
        }
        // match success, start read
        asyncReadAfterMatch(fused_read_context);
        it = std::next(it);
    }
}

void KVCacheConnectorCoordinator::processWriteContexts() {
    std::lock_guard<std::mutex> lock(update_mutex_);
    for (auto it = fused_async_write_context_list_.begin(); it != fused_async_write_context_list_.end();) {
        auto fused_write_context = *it;
        if (fused_write_context->done()) {
            it = fused_async_write_context_list_.erase(it);
            continue;
        }
        it = std::next(it);
    }
}

// this function is called under lock
void KVCacheConnectorCoordinator::asyncReadAfterMatch(std::shared_ptr<FusedAsyncReadContext> fused_read_context) {
    RTP_LLM_PROFILE_FUNCTION();
    auto match_contexts = fused_read_context->fusedMatchContext()->contexts();
    RTP_LLM_CHECK_WITH_INFO(
        match_contexts.size() == connectors_.size(),
        "match contexts size is not equal to connectors size, match contexts size: [%d], connectors size: [%d]",
        match_contexts.size(),
        connectors_.size());

    int                                        already_reuse_num = fused_read_context->resource()->reuseBlockNum();
    std::vector<std::shared_ptr<AsyncContext>> connector_read_contexts;
    for (int i = 0; i < match_contexts.size(); i++) {
        auto match_context = std::dynamic_pointer_cast<AsyncMatchContext>(match_contexts.at(i));
        if (!match_context) {
            continue;
        }
        const auto matched_num = match_context->matchedBlockCount();
        if (matched_num <= already_reuse_num) {
            continue;
        }
        auto connector_read_context = connectors_.at(i)->asyncRead(fused_read_context->resource(),
                                                                   fused_read_context->meta(),
                                                                   match_context,
                                                                   already_reuse_num,
                                                                   matched_num - already_reuse_num);
        if (connector_read_context) {
            connector_read_contexts.emplace_back(connector_read_context);
            already_reuse_num = matched_num;
        }
    }
    fused_read_context->setFusedReadContext(std::make_shared<FusedAsyncContext>(connector_read_contexts));
}

void KVCacheConnectorCoordinator::handleRead(const P2PConnectorStartLoadRequestPB& request,
                                             P2PConnectorStartLoadResponsePB&      response,
                                             std::function<bool()>                 is_cancelled) {
    if (!p2p_connector_) {
        RTP_LLM_LOG_WARNING("handleRead called but P2P connector not initialized");
        return;
    }
    p2p_connector_->handleRead(request, response, std::move(is_cancelled));
}

bool KVCacheConnectorCoordinator::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    if (request.has_mem_request()) {
        RTP_LLM_CHECK(memory_connector_ != nullptr);
        return memory_connector_->copyCache(request.mem_request(), *(response.mutable_mem_response()));
    } else if (request.has_remote_request()) {
#ifdef USE_REMOTE_KV_CACHE
        RTP_LLM_CHECK(remote_connector_ != nullptr);
        return remote_connector_->copyCache(request.remote_request(), *(response.mutable_remote_response()));
#endif
        RTP_LLM_CHECK(false);
        return false;
    } else if (request.has_p2p_request()) {
        if (!p2p_connector_) {
            RTP_LLM_LOG_WARNING("executeFunction: p2p_request received but P2P connector not initialized");
            return false;
        }
        return p2p_connector_->executeFunction(request, response);
    } else {
        RTP_LLM_LOG_WARNING("execute function failed, request is invalid, request: [%s]",
                            request.DebugString().c_str());
        return false;
    }
}

bool KVCacheConnectorCoordinator::isPdInvertMode() const {
    return (pd_sep_config_.role_type == RoleType::PREFILL || pd_sep_config_.role_type == RoleType::DECODE)
           && pd_sep_config_.decode_entrance;
}

bool KVCacheConnectorCoordinator::initP2PConnectorInternal() {
    if (isZeroSwaCachingEnabled(cache_config_) && isPdInvertMode()) {
        RTP_LLM_LOG_WARNING("DSV4 zero SWA caching does not support P2P connector yet; P2P path disabled");
        return true;
    }
    // TODO: P2P connector initialization is disabled until the next PR enables
    // scheduler async load cache support. Change to `#if 1` to activate.
#if 0
    if (!isPdInvertMode()) {
        return true;
    }
    const uint32_t layer_all_num         = static_cast<uint32_t>(cache_config_.layer_all_num);
    auto           layer_block_converter = std::make_shared<LayerBlockConverterImpl>(allocator_);

    auto p2p_config = P2PConnectorConfig::create(
        runtime_config_, cache_store_config_, parallelism_config_, pd_sep_config_, layer_all_num);
    auto p2p = std::make_shared<P2PConnector>(std::move(p2p_config), layer_block_converter, metrics_reporter_);
    if (!p2p->init()) {
        RTP_LLM_LOG_ERROR("P2PConnector init failed");
        p2p.reset();  // 显式释放，避免半初始化状态的 P2PConnector 意外使用
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        p2p_connector_ = std::move(p2p);
        connectors_.emplace_back(p2p_connector_);
    }
    RTP_LLM_LOG_INFO("P2PConnector initialized successfully, total connectors: %zu", connectors_.size());
#endif
    return true;
}

std::vector<CacheKeyType> KVCacheConnectorCoordinator::memoryCacheKeys() const {
    if (!memory_connector_) {
        return {};
    }
    return memory_connector_->cacheKeys();
}

}  // namespace rtp_llm
