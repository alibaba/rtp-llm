#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"

#include <utility>

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"

namespace rtp_llm {

// --------------------------------- AsyncReadMeta ---------------------------------

class AsyncReadMeta: public KVCacheConnector::Meta {
public:
    AsyncReadMeta(int start_block_index, int size): start_block_index_(start_block_index), size_(size) {}
    ~AsyncReadMeta() override = default;

public:
    std::pair<int, int> blockRange() const override {
        return {start_block_index_, size_};
    }

private:
    int start_block_index_;
    int size_;
};

// --------------------------------- KVCacheConnectorCoordinator ---------------------------------

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
    while (true) {
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
    connectors_.clear();
    memory_connector_.reset();
    remote_connector_.reset();
}

bool KVCacheConnectorCoordinator::init() {
    RTP_LLM_LOG_INFO("connector coordinator init, cache config: [%s], kv cache config: [%s], runtime config: [%s]",
                     cache_config_.debugString().c_str(),
                     kv_cache_config_.to_string().c_str(),
                     runtime_config_.to_string().c_str());

    if (kv_cache_config_.reuse_cache && kv_cache_config_.enable_memory_cache) {
        if (!initMemoryConnector()) {
            RTP_LLM_LOG_ERROR("init memory connector failed");
            return false;
        }
    }
    if (!initUpdateThread()) {
        RTP_LLM_LOG_ERROR("init update thread failed");
        return false;
    }
    return true;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncRead(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context,
                                       const std::shared_ptr<Meta>&                             meta) {
    if (stop_.load()) {
        return nullptr;
    }
    if (!allocator_) {
        RTP_LLM_LOG_WARNING("async read failed, allocator is null");
        return nullptr;
    }
    if (!connector_context) {
        RTP_LLM_LOG_WARNING("async read failed, connector context is null");
        return nullptr;
    }
    const auto& kvcache_resource = connector_context->kvCacheResource();
    if (kvcache_resource.cacheKeys().empty()) {
        RTP_LLM_LOG_WARNING("async read failed, kvcache resource cache keys is empty, resource: [%s]",
                            kvcache_resource.debugString().c_str());
        return nullptr;
    }

    auto resource = allocator_->incrKVCacheRef(kvcache_resource, kvcache_resource.cacheKeys());
    if (!resource) {
        RTP_LLM_LOG_WARNING("async read failed, incr kvcache ref failed, resource: [%s]",
                            kvcache_resource.debugString().c_str());
        return nullptr;
    }

    std::vector<std::shared_ptr<AsyncContext>> contexts;
    contexts.reserve(connectors_.size());
    for (const auto& [type, connector] : connectors_) {
        if (!connector) {
            continue;
        }
        if (type == KVCacheConnector::ConnectorType::Memory && connector_context->enableMemoryCache()) {
            auto match_context = connector->asyncMatch(resource, meta);
            if (match_context) {
                contexts.emplace_back(match_context);
            }
        }
    }
    if (contexts.empty()) {
        allocator_->decrKVCacheRef(*resource);
        return nullptr;
    }

    auto fused_match_context = std::make_shared<FusedAsyncContext>(contexts);
    auto deleter             = [allocator = allocator_, resource](FusedAsyncReadContext* context) {
        allocator->decrKVCacheRef(*resource);
        delete context;
    };
    std::shared_ptr<FusedAsyncReadContext> fused_read_context(new FusedAsyncReadContext(fused_match_context, resource),
                                                              deleter);
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        fused_async_read_context_list_.push_back(fused_read_context);
    }
    return fused_read_context;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncWrite(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context,
                                        const std::shared_ptr<Meta>&                             meta) {
    if (stop_.load()) {
        return nullptr;
    }
    if (!allocator_) {
        RTP_LLM_LOG_WARNING("async write failed, allocator is null");
        return nullptr;
    }
    if (!connector_context) {
        RTP_LLM_LOG_WARNING("async write failed, connector context is null");
        return nullptr;
    }
    const auto& kvcache_resource = connector_context->kvCacheResource();
    if (kvcache_resource.cacheKeys().empty()) {
        RTP_LLM_LOG_WARNING("async write failed, kvcache resource cache keys is empty, resource: [%s]",
                            kvcache_resource.debugString().c_str());
        return nullptr;
    }

    auto resource = allocator_->incrKVCacheRef(kvcache_resource, kvcache_resource.cacheKeys());
    if (!resource) {
        RTP_LLM_LOG_WARNING("async write failed, incr kvcache ref failed, resource: [%s]",
                            kvcache_resource.debugString().c_str());
        return nullptr;
    }

    std::vector<std::shared_ptr<AsyncContext>> write_contexts;
    for (const auto& [type, connector] : connectors_) {
        if (!connector) {
            continue;
        }
        if (type == KVCacheConnector::ConnectorType::Memory && connector_context->enableMemoryCache()) {
            auto write_context = connector->asyncWrite(resource, meta);
            if (write_context) {
                write_contexts.emplace_back(write_context);
            }
        }
    }
    if (write_contexts.empty()) {
        allocator_->decrKVCacheRef(*resource);
        return nullptr;
    }

    auto deleter = [allocator = allocator_, resource](FusedAsyncContext* context) {
        allocator->decrKVCacheRef(*resource);
        delete context;
    };
    std::shared_ptr<FusedAsyncContext> fused_write_context(new FusedAsyncContext(std::move(write_contexts)), deleter);
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        fused_async_write_context_list_.push_back(fused_write_context);
    }
    return fused_write_context;
}

std::shared_ptr<AsyncContext> KVCacheConnectorCoordinator::asyncWriteByLayer(
    int                                                      layer_id,
    const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context,
    const std::shared_ptr<Meta>&                             meta) {
    if (stop_.load()) {
        return nullptr;
    }
    if (!allocator_) {
        RTP_LLM_LOG_WARNING("async write by layer failed, allocator is null");
        return nullptr;
    }
    if (!connector_context) {
        RTP_LLM_LOG_WARNING("async write by layer failed, connector context is null");
        return nullptr;
    }
    const auto& kvcache_resource = connector_context->kvCacheResource();
    if (kvcache_resource.cacheKeys().empty()) {
        RTP_LLM_LOG_WARNING("async write by layer failed, kvcache resource cache keys is empty, resource: [%s]",
                            kvcache_resource.debugString().c_str());
        return nullptr;
    }
    auto resource = allocator_->incrKVCacheRef(kvcache_resource, kvcache_resource.cacheKeys());
    if (!resource) {
        RTP_LLM_LOG_WARNING("async write by layer failed, incr kvcache ref failed, resource: [%s]",
                            kvcache_resource.debugString().c_str());
        return nullptr;
    }

    std::vector<std::shared_ptr<AsyncContext>> write_contexts;
    for (const auto& [type, connector] : connectors_) {
        if (!connector) {
            continue;
        }
        if (type == KVCacheConnector::ConnectorType::P2P) {
            auto write_context = connector->asyncWriteByLayer(layer_id, resource, meta);
            if (write_context) {
                write_contexts.emplace_back(write_context);
            }
        }
    }
    if (write_contexts.empty()) {
        allocator_->decrKVCacheRef(*resource);
        return nullptr;
    }

    auto deleter = [allocator = allocator_, resource](FusedAsyncContext* context) {
        allocator->decrKVCacheRef(*resource);
        delete context;
    };
    std::shared_ptr<FusedAsyncContext> fused_write_context(new FusedAsyncContext(std::move(write_contexts)), deleter);
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        fused_async_write_context_list_.push_back(fused_write_context);
    }
    return fused_write_context;
}

bool KVCacheConnectorCoordinator::initMemoryConnector() {
    memory_connector_ = std::make_shared<KVCacheMemoryConnector>(
        cache_config_, kv_cache_config_, allocator_, device_, runtime_config_.worker_grpc_addrs, metrics_reporter_);
    if (!memory_connector_->init()) {
        RTP_LLM_LOG_ERROR("memory connector init failed");
        memory_connector_.reset();
        return false;
    }

    connectors_[KVCacheConnector::ConnectorType::Memory] = memory_connector_;
    return true;
}

bool KVCacheConnectorCoordinator::initUpdateThread() {
    update_thread_ = autil::LoopThread::createLoopThread(
        [self = shared_from_this()]() { self->updateOnce(); }, update_interval_ms_, "CoordinatorUpdateThread");
    return update_thread_ != nullptr;
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
            it = fused_async_read_context_list_.erase(it);
            continue;
        }
        auto read_context = fused_read_context->fusedReadContext();
        if (read_context) {
            // 有 read context 说明 match done 并且 success
            if (read_context->done()) {
                it = fused_async_read_context_list_.erase(it);
            } else {
                it = std::next(it);
            }
            continue;
        }
        auto match_context = fused_read_context->fusedMatchContext();
        if (!match_context) {
            it = fused_async_read_context_list_.erase(it);
            continue;
        }
        if (!match_context->done()) {
            it = std::next(it);
            continue;
        }
        if (!match_context->success()) {
            // match failed, cancel
            it = fused_async_read_context_list_.erase(it);
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
    int                                        reuse_num      = fused_read_context->resource()->reuseBlocksNum();
    auto                                       match_contexts = fused_read_context->fusedMatchContext()->contexts();
    std::vector<std::shared_ptr<AsyncContext>> connector_read_contexts;
    for (int i = 0; i < match_contexts.size(); i++) {
        auto match_context = std::dynamic_pointer_cast<KVCacheConnector::AsyncMatchContext>(match_contexts.at(i));
        if (!match_context) {
            continue;
        }
        if (match_context->matchedBlockCount() <= reuse_num) {
            continue;
        }
        auto read_meta = std::make_shared<AsyncReadMeta>(reuse_num, match_context->matchedBlockCount() - reuse_num);
        auto connector = connectors_.at(match_context->connectorType());
        auto connector_read_context = connector->asyncRead(fused_read_context->resource(), read_meta, match_context);
        if (connector_read_context) {
            connector_read_contexts.emplace_back(connector_read_context);
            reuse_num = match_context->matchedBlockCount();
        }
    }
    fused_read_context->setFusedReadContext(std::make_shared<FusedAsyncContext>(connector_read_contexts));
}

bool KVCacheConnectorCoordinator::broadcastTp(const BroadcastTpRequestPB& request, BroadcastTpResponsePB& response) {
    if (request.has_mem_request()) {
        if (!memory_connector_) {
            RTP_LLM_LOG_WARNING("broadcast tp failed, memory connector is null, request: [%s]",
                                request.DebugString().c_str());
            response.mutable_mem_response()->set_success(false);
            return false;
        }
        return memory_connector_->copyCache(request.mem_request(), *(response.mutable_mem_response()));
    } else {
        RTP_LLM_LOG_WARNING("broadcast tp failed, request is invalid, request: [%s]", request.DebugString().c_str());
        return false;
    }
}

void KVCacheConnectorCoordinator::clearMemoryCache() {
    if (!memory_connector_) {
        RTP_LLM_LOG_WARNING("clear memory cache failed, memory connector is null");
        return;
    }
    memory_connector_->clearCache();
}

}  // namespace rtp_llm