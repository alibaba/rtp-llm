#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"

#include <algorithm>
#include <utility>

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverterImpl.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#ifdef USE_REMOTE_KV_CACHE
#include "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.h"
#endif

namespace rtp_llm {

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

KVCacheResource KVCacheConnectorCoordinator::normalizeResourceForConnector(const KVCacheResource& resource,
                                                                           int                    layer_id) const {
    KVCacheResource normalized;
    normalized.initGroups(cache_config_.groupNums(),
                          static_cast<int>(cache_config_.layer_all_num),
                          cache_config_.layer_to_group_id,
                          cache_config_.kernelBlocksPerKvBlock(),
                          cache_config_.group_types);
    normalized.cacheKeys() = resource.cacheKeys();
    normalized.setDeviceReuseBlockNum(resource.deviceReuseBlockNum());
    normalized.setMemoryReuseBlockNum(resource.memoryReuseBlockNum());
    normalized.setRemoteReuseBlockNum(resource.remoteReuseBlockNum());
    normalized.setLastBlockAligned(resource.lastBlockAligned());

    bool        copied_blocks       = false;
    const auto& source_layer_blocks = resource.layerBlocks();
    if (layer_id >= 0 && static_cast<size_t>(layer_id) < source_layer_blocks.size()
        && static_cast<size_t>(layer_id) < cache_config_.layer_to_group_id.size()) {
        const auto& source_blocks = source_layer_blocks[static_cast<size_t>(layer_id)]->blocks();
        if (!source_blocks.empty()) {
            const int canonical_gid = cache_config_.layer_to_group_id[static_cast<size_t>(layer_id)];
            normalized.mutableBlockIds(canonical_gid).assign(source_blocks);
            copied_blocks = true;
        }
    } else {
        const auto max_layer_num = std::min(source_layer_blocks.size(), cache_config_.layer_to_group_id.size());
        for (size_t current_layer_id = 0; current_layer_id < max_layer_num; ++current_layer_id) {
            const auto& source_blocks = source_layer_blocks[current_layer_id]->blocks();
            if (source_blocks.empty()) {
                continue;
            }
            const int canonical_gid = cache_config_.layer_to_group_id[current_layer_id];
            auto&     dst_block_ids = normalized.mutableBlockIds(canonical_gid);
            if (dst_block_ids.blocks().empty()) {
                dst_block_ids.assign(source_blocks);
                copied_blocks = true;
                continue;
            }
            RTP_LLM_CHECK_WITH_INFO(dst_block_ids.blocks() == source_blocks,
                                    "conflicting connector block ids for canonical gid %d",
                                    canonical_gid);
        }
    }

    if (!copied_blocks) {
        const auto& source_groups  = resource.groupBlocks();
        const int   copy_group_num = std::min<int>(resource.groupNums(), normalized.groupNums());
        for (int gid = 0; gid < copy_group_num; ++gid) {
            const auto& source_blocks = source_groups[static_cast<size_t>(gid)]->blocks();
            if (!source_blocks.empty()) {
                normalized.mutableBlockIds(gid).assign(source_blocks);
            }
        }
    }
    return normalized;
}

bool KVCacheConnectorCoordinator::init() {
    RTP_LLM_LOG_INFO("connector coordinator init, cache config: [%s], kv cache config: [%s], runtime config: [%s]",
                     cache_config_.debugString().c_str(),
                     kv_cache_config_.to_string().c_str(),
                     runtime_config_.to_string().c_str());
    if (kv_cache_config_.reuse_cache && kv_cache_config_.enable_memory_cache) {
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
        RTP_LLM_LOG_ERROR("init P2P connector failed");
        return false;
    }
    if (isPdInvertMode() && !hasP2PConnector()) {
        RTP_LLM_LOG_ERROR("decode_entrance requires an initialized P2P connector");
        return false;
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

    auto resource = allocator_->incrKVCacheRef(kvcache_resource, kvcache_resource.cacheKeys(), true);
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

    auto resource = allocator_->incrKVCacheRef(kvcache_resource, kvcache_resource.cacheKeys(), true);
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
        RTP_LLM_LOG_DEBUG("asyncWriteByLayer [P2P]: dispatching layer_id=%d, request_id=%ld to P2PConnector",
                          layer_id,
                          layer_context->requestId());
    }
    return p2p_connector_->asyncWriteByLayer(layer_id, layer_context);
}

std::shared_ptr<KVCacheResource>
KVCacheConnectorCoordinator::holdKVCacheResourceForConnector(const KVCacheResource& resource, int layer_id) {
    if (resource.cacheKeys().empty()) {
        RTP_LLM_LOG_DEBUG("holdKVCacheResourceForConnector skipped, resource cache keys is empty");
        return nullptr;
    }
    auto normalized_resource = normalizeResourceForConnector(resource, layer_id);
    auto held_resource       = allocator_->incrKVCacheRef(normalized_resource, normalized_resource.cacheKeys(), true);
    if (!held_resource) {
        RTP_LLM_LOG_WARNING("holdKVCacheResourceForConnector failed, incr kvcache ref failed, resource: [%s]",
                            normalized_resource.debugString().c_str());
    }
    return held_resource;
}

std::shared_ptr<KVCacheMemoryConnector> KVCacheConnectorCoordinator::initMemoryConnector() {
    auto memory_connector = std::make_shared<KVCacheMemoryConnector>(
        cache_config_, kv_cache_config_, allocator_, runtime_config_.worker_grpc_addrs, metrics_reporter_);
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

void KVCacheConnectorCoordinator::updateOnce() {
    RTP_LLM_PROFILE_FUNCTION();
    processReadContexts();
    processWriteContexts();
}

void KVCacheConnectorCoordinator::processReadContexts() {
    RTP_LLM_PROFILE_FUNCTION();
    // Two-phase to keep the slow asyncReadAfterMatch path (per-connector gRPC ->
    // server_caller_->load + tp_broadcast_client_->broadcast, multi-second on
    // degraded networks) OUT OF update_mutex_. Holding the mutex across that
    // serializes the engine main thread's coordinator->asyncRead (which also
    // takes update_mutex_), causing scheduler deadlock. See incident: 1-hour
    // stuck StartLoad on prefill cascading into decode-side schedule freeze.
    //
    // Phase 1 (under lock): drain done contexts, collect contexts that need
    //   dispatch (match success && fusedReadContext not yet set), and snapshot
    //   connectors_ so the dispatch path is robust against destructor
    //   connectors_.clear(). Each FusedAsyncReadContext::setFusedReadContext is
    //   internally locked (read_ctx_mutex_), so writing it without holding
    //   update_mutex_ is safe.
    // Phase 2 (no lock): run asyncReadAfterMatch for each collected context.
    //
    // Duplicate-dispatch hazard analysis: processReadContexts runs in a single
    // LoopThread, so two iterations cannot interleave. Within a single iteration,
    // each context appears once. asyncReadAfterMatch sets fusedReadContext at
    // the end, so the next iteration's Phase 1 short-circuits on the
    // `read_context` check. No CAS in-flight flag needed.
    std::vector<std::shared_ptr<FusedAsyncReadContext>> to_dispatch;
    std::vector<std::shared_ptr<KVCacheConnector>>      connectors_snapshot;
    {
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
            // match success — defer dispatch to Phase 2
            to_dispatch.push_back(fused_read_context);
            it = std::next(it);
        }
        connectors_snapshot = connectors_;
    }

    if (to_dispatch.empty()) {
        return;
    }
    for (auto& fused_read_context : to_dispatch) {
        if (stop_.load()) {
            // Coordinator is shutting down; skip remaining dispatches. Connectors
            // captured in the snapshot may still be valid (kept alive by shared_ptr)
            // but issuing new RPCs at this point is wasted work.
            break;
        }
        asyncReadAfterMatch(fused_read_context, connectors_snapshot);
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

// Called WITHOUT update_mutex_ held (post two-phase refactor of processReadContexts).
// connectors_snapshot is a stable shared_ptr-holding copy taken under update_mutex_;
// each connector stays alive even if the destructor's connectors_.clear() races.
void KVCacheConnectorCoordinator::asyncReadAfterMatch(
    std::shared_ptr<FusedAsyncReadContext>                fused_read_context,
    const std::vector<std::shared_ptr<KVCacheConnector>>& connectors_snapshot) {
    RTP_LLM_PROFILE_FUNCTION();
    auto async_read_start_us = currentTimeUs();
    auto match_contexts      = fused_read_context->fusedMatchContext()->contexts();
    RTP_LLM_CHECK_WITH_INFO(
        match_contexts.size() == connectors_snapshot.size(),
        "match contexts size is not equal to connectors size, match contexts size: [%d], connectors size: [%d]",
        match_contexts.size(),
        connectors_snapshot.size());

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
        // [PD-DIAG] Per-connector asyncRead cost. Post two-phase refactor this
        // call no longer holds update_mutex_, so a slow gRPC inside here cannot
        // back-pressure the engine main thread's coordinator->asyncRead. The
        // WARN below still surfaces slow per-connector cost for visibility.
        const int64_t connector_async_read_start_us = currentTimeUs();
        auto          connector_read_context = connectors_snapshot.at(i)->asyncRead(fused_read_context->resource(),
                                                                                    fused_read_context->meta(),
                                                                                    match_context,
                                                                                    already_reuse_num,
                                                                                    matched_num - already_reuse_num);
        const int64_t connector_async_read_cost_us = currentTimeUs() - connector_async_read_start_us;
        if (connector_async_read_cost_us >= 100000) {
            RTP_LLM_LOG_WARNING("[PD-DIAG] asyncReadAfterMatch slow connector asyncRead, "
                                "connector_index=%d, cost_us=%ld",
                                i,
                                connector_async_read_cost_us);
        }
        if (connector_read_context) {
            connector_read_contexts.emplace_back(connector_read_context);
            already_reuse_num = matched_num;
        }
    }
    fused_read_context->setFusedReadContext(std::make_shared<FusedAsyncContext>(connector_read_contexts));
    // Only log slow paths (>500ms total). Healthy asyncReadAfterMatch finishes
    // in microseconds; the per-connector >100ms WARN above already captures
    // G-class slow connector calls. Demoted from unconditional INFO after
    // 5/22 audit (~1385 entries/4h with no diagnostic value at healthy speed).
    const int64_t async_read_total_us = currentTimeUs() - async_read_start_us;
    if (async_read_total_us >= 500000) {
        RTP_LLM_LOG_WARNING(
            "[PD-DIAG] asyncReadAfterMatch slow done, connectors_issued=%zu, already_reuse_num=%d, cost_us=%ld",
            connector_read_contexts.size(),
            already_reuse_num,
            async_read_total_us);
    }
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

void KVCacheConnectorCoordinator::notifySideChannelReady(const std::string&                                unique_key,
                                                         int64_t                                           deadline_ms,
                                                         const P2PConnectorResourceEntry::SideChannelData& data) {
    if (p2p_connector_) {
        p2p_connector_->streamStore()->notifySideChannelReady(unique_key, deadline_ms, data);
    }
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
    if (!isPdInvertMode()) {
        return true;
    }
    const uint32_t layer_all_num         = static_cast<uint32_t>(cache_config_.layer_all_num);
    const bool     is_mla                = cache_config_.use_mla || cache_config_.is_sparse;
    auto           layer_block_converter = std::make_shared<LayerBlockConverterImpl>(allocator_);

    RTP_LLM_LOG_INFO("initP2PConnectorInternal: pd_sep_config.cache_store_rdma_mode=%d, "
                     "cache_store_config.cache_store_rdma_mode=%d, "
                     "pd_sep_config.cache_store_listen_port=%ld, "
                     "pd_sep_config.role_type=%d",
                     pd_sep_config_.cache_store_rdma_mode ? 1 : 0,
                     cache_store_config_.cache_store_rdma_mode ? 1 : 0,
                     pd_sep_config_.cache_store_listen_port,
                     static_cast<int>(pd_sep_config_.role_type));

    auto p2p_config = P2PConnectorConfig::create(
        runtime_config_, cache_store_config_, parallelism_config_, pd_sep_config_, layer_all_num, is_mla);
    p2p_config.scheduler_config.layer_attn_types = cache_config_.layer_attn_types;
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
    return true;
}

std::vector<CacheKeyType> KVCacheConnectorCoordinator::memoryCacheKeys() const {
    if (!memory_connector_) {
        return {};
    }
    return memory_connector_->cacheKeys();
}

void KVCacheConnectorCoordinator::reportP2PCacheWriteFailure() {
    if (metrics_reporter_) {
        CacheWriteOpFailureMetricsCollector collector;
        metrics_reporter_->report<P2PConnectorMetrics, CacheWriteOpFailureMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llm
