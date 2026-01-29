#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"

#include <utility>

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConvertorImpl.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

// --------------------------------- KVCacheConnectorCoordinator ---------------------------------

KVCacheConnectorCoordinator::KVCacheConnectorCoordinator(const CacheConfig&                       cache_config,
                                                         const KVCacheConfig&                     kv_cache_config,
                                                         const RuntimeConfig&                     runtime_config,
                                                         const CacheStoreConfig&                  cache_store_config,
                                                         const ParallelismConfig&                 parallelism_config,
                                                         const PDSepConfig&                       pd_sep_config,
                                                         const ModelConfig&                       model_config,
                                                         const std::shared_ptr<KVCacheAllocator>& allocator,
                                                         rtp_llm::DeviceBase*                     device,
                                                         const kmonitor::MetricsReporterPtr&      metrics_reporter):
    cache_config_(cache_config),
    kv_cache_config_(kv_cache_config),
    runtime_config_(runtime_config),
    cache_store_config_(cache_store_config),
    parallelism_config_(parallelism_config),
    pd_sep_config_(pd_sep_config),
    model_config_(model_config),
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
}

bool KVCacheConnectorCoordinator::init() {
    RTP_LLM_LOG_INFO("connector coordinator init, cache config: [%s], kv cache config: [%s], runtime config: [%s]",
                     cache_config_.debugString().c_str(),
                     kv_cache_config_.to_string().c_str(),
                     runtime_config_.to_string().c_str());

    if (kv_cache_config_.reuse_cache && kv_cache_config_.enable_memory_cache) {
        RTP_LLM_CHECK_WITH_INFO(initMemoryConnector(), "init memory connector failed");
    }
    if (pd_sep_config_.decode_entrance
        && (pd_sep_config_.role_type == RoleType::DECODE || pd_sep_config_.role_type == RoleType::PREFILL)) {
        RTP_LLM_CHECK_WITH_INFO(initP2PConnector(), "init p2p connector failed");
    }
    RTP_LLM_CHECK_WITH_INFO(initUpdateThread(), "init update thread failed");
    return true;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncRead(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context,
                                       const std::shared_ptr<KVCacheConnector::Meta>&           meta) {
    if (stop_.load()) {
        return nullptr;
    }

    if (!connector_context) {
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

    std::shared_ptr<KVCacheResource> resource_ptr(
        resource.get(), [allocator = allocator_, resource](KVCacheResource* res) { allocator->decrKVCacheRef(*res); });

    std::vector<std::shared_ptr<AsyncContext>> contexts;
    contexts.reserve(connectors_.size());
    for (const auto& [type, connector] : connectors_) {
        if (!connector) {
            continue;
        }
        if (type == KVCacheConnector::ConnectorType::Memory && connector_context->enableMemoryCache()) {
            auto match_context = connector->asyncMatch(resource_ptr, connector_context->meta());
            if (match_context) {
                contexts.emplace_back(match_context);
            }
        }
        if (type == KVCacheConnector::ConnectorType::P2P) {
            auto match_context = connector->asyncMatch(resource_ptr, connector_context->meta());
            if (match_context) {
                contexts.emplace_back(match_context);
            }
        }
    }
    if (contexts.empty()) {
        return nullptr;
    }

    auto fused_match_context = std::make_shared<FusedAsyncContext>(contexts);
    auto fused_read_context  = std::make_shared<FusedAsyncReadContext>(fused_match_context, resource_ptr);
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        fused_async_read_context_list_.push_back(std::make_pair(fused_read_context, connector_context->meta()));
    }
    return fused_read_context;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncWrite(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context,
                                        const std::shared_ptr<KVCacheConnector::Meta>&           meta) {
    if (stop_.load()) {
        return nullptr;
    }
    if (!connector_context) {
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
            auto write_context = connector->asyncWrite(resource, connector_context->meta());
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
    const std::shared_ptr<KVCacheConnector::Meta>&           meta) {
    if (stop_.load()) {
        return nullptr;
    }

    if (!connector_context) {
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
            auto write_context = connector->asyncWriteByLayer(layer_id, resource, connector_context->meta());
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

bool KVCacheConnectorCoordinator::initP2PConnector() {
    auto layer_block_convertor = std::make_shared<LayerBlockConvertorImpl>(allocator_);
    p2p_connector_             = std::make_shared<P2PConnector>(kv_cache_config_,
                                                    runtime_config_,
                                                    cache_store_config_,
                                                    parallelism_config_,
                                                    pd_sep_config_,
                                                    model_config_,
                                                    cache_config_.layer_all_num,
                                                    layer_block_convertor,
                                                    metrics_reporter_);
    if (!p2p_connector_->init()) {
        RTP_LLM_LOG_ERROR("p2p connector init failed");
        p2p_connector_.reset();
        return false;
    }

    connectors_[KVCacheConnector::ConnectorType::P2P] = p2p_connector_;
    return true;
}

bool KVCacheConnectorCoordinator::initUpdateThread() {
    update_thread_ =
        autil::LoopThread::createLoopThread([this]() { updateOnce(); }, update_interval_ms_, "CoordinatorUpdateThread");
    return update_thread_ != nullptr;
}

void KVCacheConnectorCoordinator::updateOnce() {
    processReadContexts();
    processWriteContexts();
}

void KVCacheConnectorCoordinator::processReadContexts() {
    std::lock_guard<std::mutex> lock(update_mutex_);
    for (auto it = fused_async_read_context_list_.begin(); it != fused_async_read_context_list_.end();) {
        auto [fused_read_context, meta] = *it;
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
        asyncReadAfterMatch(fused_read_context, meta);
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
void KVCacheConnectorCoordinator::asyncReadAfterMatch(std::shared_ptr<FusedAsyncReadContext>  fused_read_context,
                                                      std::shared_ptr<KVCacheConnector::Meta> meta) {
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

        RTP_LLM_LOG_DEBUG("async read after match, reuse_num: %d, matched_block_count: %d, resource: [%s]",
                          reuse_num,
                          match_context->matchedBlockCount(),
                          fused_read_context->resource()->debugString().c_str());
        auto read_meta               = std::make_shared<KVCacheConnector::Meta>(*meta);
        read_meta->start_block_index = reuse_num;
        read_meta->block_size        = match_context->matchedBlockCount() - reuse_num;

        auto connector              = connectors_.at(match_context->connectorType());
        auto connector_read_context = connector->asyncRead(fused_read_context->resource(), read_meta, match_context);
        if (connector_read_context) {
            connector_read_contexts.emplace_back(connector_read_context);
            reuse_num = match_context->matchedBlockCount();
        }
    }
    fused_read_context->setFusedReadContext(std::make_shared<FusedAsyncContext>(connector_read_contexts));
}

bool KVCacheConnectorCoordinator::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    if (request.has_mem_request()) {
        if (!memory_connector_) {
            RTP_LLM_LOG_WARNING("execute function failed, memory connector is null, request: [%s]",
                                request.DebugString().c_str());
            response.mutable_mem_response()->set_success(false);
            return false;
        }
        return memory_connector_->copyCache(request.mem_request(), *(response.mutable_mem_response()));
    } else if (request.has_p2p_request()) {
        if (!p2p_connector_) {
            RTP_LLM_LOG_WARNING("execute function failed, p2p connector is null, request: [%s]",
                                request.DebugString().c_str());
            auto* p2p_response = response.mutable_p2p_response();
            p2p_response->set_error_code(transErrorCodeToRPC(ErrorCode::UNKNOWN_ERROR));
            p2p_response->set_error_message("execute function failed, p2p connector is null");
            return false;
        }
        return p2p_connector_->executeFunction(request, response);
    } else {
        RTP_LLM_LOG_WARNING("execute function failed, request is invalid, request: [%s]",
                            request.DebugString().c_str());
        return false;
    }
}

void KVCacheConnectorCoordinator::handleRead(const P2PConnectorStartLoadRequestPB& request,
                                             P2PConnectorStartLoadResponsePB&      response,
                                             std::function<bool()>                 is_cancelled) {
    if (stop_.load()) {
        response.set_error_code(transErrorCodeToRPC(ErrorCode::UNKNOWN_ERROR));
        response.set_error_message("handleRead failed, coordinator is stopped");
        return;
    }

    if (!p2p_connector_) {
        RTP_LLM_LOG_WARNING("handleRead failed, p2p connector is null");
        response.set_error_code(transErrorCodeToRPC(ErrorCode::UNKNOWN_ERROR));
        response.set_error_message("handleRead failed, p2p connector is null");
        return;
    }

    p2p_connector_->handleRead(request, response, is_cancelled);
    return;
}

uint32_t KVCacheConnectorCoordinator::convertToGlobalLayerId(size_t model_id, int local_layer_id) const {
    return allocator_->convertToGlobalLayerId(model_id, local_layer_id);
}

}  // namespace rtp_llm