#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"

#include <utility>

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"

namespace rtp_llm {

KVCacheConnectorCoordinator::KVCacheConnectorCoordinator(const CacheConfig&                       cache_config,
                                                         const KVCacheConfig&                     kv_cache_config,
                                                         const RuntimeConfig&                     runtime_config,
                                                         const std::shared_ptr<KVCacheAllocator>& allocator,
                                                         rtp_llm::DeviceBase*                     device,
                                                         const kmonitor::MetricsReporterPtr&      metrics_reporter):
    cache_config_(cache_config),
    kv_cache_config_(kv_cache_config),
    runtime_config_(runtime_config),
    allocator_(allocator),
    device_(device),
    metrics_reporter_(metrics_reporter) {}

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

bool KVCacheConnectorCoordinator::init() {
    RTP_LLM_LOG_INFO("connector coordinator init, cache config: [%s], kv cache config: [%s], runtime config: [%s]",
                     cache_config_.debugString().c_str(),
                     kv_cache_config_.to_string().c_str(),
                     runtime_config_.to_string().c_str());
    if (kv_cache_config_.reuse_cache && kv_cache_config_.enable_memory_cache) {
        memory_connector_ = initMemoryConnector();
        connectors_.emplace_back(memory_connector_);
    }
    initUpdateThread();
    return true;
}

std::shared_ptr<KVCacheMemoryConnector> KVCacheConnectorCoordinator::initMemoryConnector() {
    auto memory_connector = std::make_shared<KVCacheMemoryConnector>(
        cache_config_, kv_cache_config_, allocator_, device_, runtime_config_.worker_grpc_addrs, metrics_reporter_);
    RTP_LLM_CHECK_WITH_INFO(memory_connector->init(), "memory connector init failed");
    return memory_connector;
}

void KVCacheConnectorCoordinator::initUpdateThread() {
    update_thread_ = autil::LoopThread::createLoopThread(
        [self = shared_from_this()]() { self->updateOnce(); }, update_interval_ms_ * 1000, "CoordinatorUpdateThread");
    RTP_LLM_CHECK_WITH_INFO(update_thread_ != nullptr, "init update thread failed");
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncRead(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    if (stop_.load()) {
        return nullptr;
    }
    if (!connector_context) {
        RTP_LLM_LOG_WARNING("async read failed, connector context is null");
        return nullptr;
    }
    const auto& kvcache_resource = connector_context->kvCacheResource();
    if (kvcache_resource.cacheKeys().empty()) {
        RTP_LLM_LOG_DEBUG("async read failed, kvcache resource cache keys is empty, resource: [%s]",
                          kvcache_resource.debugString().c_str());
        return nullptr;
    }

    auto resource = allocator_->incrKVCacheRef(kvcache_resource, kvcache_resource.cacheKeys());
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

    auto resource = allocator_->incrKVCacheRef(kvcache_resource, kvcache_resource.cacheKeys());
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

std::shared_ptr<AsyncContext> KVCacheConnectorCoordinator::asyncWriteByLayer(
    int layer_id, const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    RTP_LLM_FAIL("async write by layer is not implemented");
}

void KVCacheConnectorCoordinator::updateOnce() {
    processReadContexts();
    processWriteContexts();
}

void KVCacheConnectorCoordinator::processReadContexts() {
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

bool KVCacheConnectorCoordinator::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    if (request.has_mem_request()) {
        RTP_LLM_CHECK(memory_connector_ != nullptr);
        return memory_connector_->copyCache(request.mem_request(), *(response.mutable_mem_response()));
    } else {
        RTP_LLM_LOG_WARNING("execute function failed, request is invalid, request: [%s]",
                            request.DebugString().c_str());
        return false;
    }
}

}  // namespace rtp_llm