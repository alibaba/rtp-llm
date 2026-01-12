#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"

#include <utility>

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnector.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/LayerBlockConvertorImpl.h"

namespace rtp_llm {

// --------------------------------- FusedAsyncContext ---------------------------------

FusedAsyncContext::FusedAsyncContext(const std::vector<std::shared_ptr<AsyncContext>>& contexts): contexts_(contexts) {}

bool FusedAsyncContext::done() const {
    for (const auto& context : contexts_) {
        if (context && !context->done()) {
            return false;
        }
    }
    return true;
}

bool FusedAsyncContext::success() const {
    for (const auto& context : contexts_) {
        if (context && !context->success()) {
            return false;
        }
    }
    return true;
}

// --------------------------------- FusedAsyncReadContext ---------------------------------

FusedAsyncReadContext::FusedAsyncReadContext(const std::shared_ptr<FusedAsyncContext>&    fused_match_context,
                                             const std::shared_ptr<KVCacheResource>&      resource,
                                             const std::shared_ptr<KVCacheConnectorMeta>& meta):
    fused_match_context_(fused_match_context), resource_(resource), meta_(meta) {}

bool FusedAsyncReadContext::done() const {
    if (!fused_match_context_) {
        return true;
    }
    if (!fused_match_context_->done()) {
        return false;
    }
    if (!fused_match_context_->success()) {
        return true;
    }
    return fused_read_context_ && fused_read_context_->done();
}

bool FusedAsyncReadContext::success() const {
    return done() && (fused_match_context_ && fused_match_context_->success())
           && (!fused_read_context_ || fused_read_context_->success());
}

void FusedAsyncReadContext::setFusedReadContext(const std::shared_ptr<FusedAsyncContext>& fused_read_context) {
    fused_read_context_ = fused_read_context;
}

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
            if (match_context_list_.empty() && read_context_list_.empty() && write_context_list_.empty()) {
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
    p2p_connector_.reset();
}

bool KVCacheConnectorCoordinator::init() {
    bool inited = false;
    if (kv_cache_config_.memory_block_cache_size_mb > 0) {
        if (!initMemoryConnector()) {
            RTP_LLM_LOG_ERROR("init memory connector failed");
            return false;
        }
        inited = true;
    }

    if (pd_sep_config_.role_type == RoleType::PREFILL || pd_sep_config_.role_type == RoleType::DECODE) {
        if (!initP2PConnector()) {
            RTP_LLM_LOG_ERROR("init p2p connector failed");
            return false;
        }
        inited = true;
    }

    if (!inited) {
        RTP_LLM_LOG_INFO("connector coordinator is not initialized, role type: %d", pd_sep_config_.role_type);
        return true;
    }

    if (!initUpdateThread()) {
        return false;
    }
    RTP_LLM_LOG_INFO("connector coordinator initialized, role type: %d", pd_sep_config_.role_type);
    return true;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncRead(const KVCacheResource&                       resource,
                                       const std::shared_ptr<KVCacheConnectorMeta>& meta,
                                       const KVCacheConnectorControlParams&         control_params) {
    if (stop_.load()) {
        return nullptr;
    }

    auto incr_resource_ptr = allocator_->incrKVCacheRef(resource, resource.cacheKeys());
    if (!incr_resource_ptr) {
        return nullptr;
    }
    std::shared_ptr<KVCacheResource> resource_ptr(
        incr_resource_ptr.get(),
        [allocator = allocator_, incr_resource_ptr = incr_resource_ptr](KVCacheResource* resource) {
            allocator->decrKVCacheRef(*resource);
        });
    int reuse_block_num = meta->generate_stream ? meta->generate_stream->reuseBlockNum() : 0;
    RTP_LLM_LOG_INFO("generate_stream reuse block num: %d", reuse_block_num);

    std::vector<std::shared_ptr<AsyncContext>> contexts;
    contexts.reserve(connectors_.size());
    for (const auto& [type, connector] : connectors_) {
        if (!connector) {
            continue;
        }
        if (type == ConnectorType::Memory && control_params.enable_memory_cache) {
            auto match_context = connector->asyncMatch(resource_ptr, meta);
            if (match_context) {
                contexts.emplace_back(match_context);
            }
        }
        if (type == ConnectorType::P2P) {
            auto match_context = connector->asyncMatch(resource_ptr, meta);
            if (match_context) {
                contexts.emplace_back(match_context);
            }
        }
    }
    if (contexts.empty()) {
        return nullptr;
    }

    auto                                   fused_match_context = std::make_shared<FusedAsyncContext>(contexts);
    std::shared_ptr<FusedAsyncReadContext> fused_read_context(
        new FusedAsyncReadContext(fused_match_context, resource_ptr, meta));
    fused_read_context->setReuseBlockNum(reuse_block_num);

    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        match_context_list_.push_back(fused_read_context);  // 先放入 match 队列
    }
    return fused_read_context;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncWrite(const KVCacheResource&                       resource,
                                        const std::shared_ptr<KVCacheConnectorMeta>& meta,
                                        const KVCacheConnectorControlParams&         control_params) {
    if (stop_.load()) {
        return nullptr;
    }

    RTP_LLM_LOG_DEBUG("asyncWrite, resource: %p, meta: %p", &resource, meta.get());

    auto incr_resource_ptr = allocator_->incrKVCacheRef(resource, resource.cacheKeys());
    if (!incr_resource_ptr) {
        return nullptr;
    }
    std::shared_ptr<KVCacheResource> resource_ptr(
        incr_resource_ptr.get(), [allocator = allocator_, incr_resource_ptr](KVCacheResource* resource) {
            allocator->decrKVCacheRef(*resource);
        });

    std::vector<std::shared_ptr<AsyncContext>> write_contexts;
    for (const auto& [type, connector] : connectors_) {
        if (!connector) {
            continue;
        }
        if (type == ConnectorType::Memory && control_params.enable_memory_cache) {
            auto write_context = connector->asyncWrite(resource_ptr, meta);
            if (write_context) {
                write_contexts.emplace_back(write_context);
            }
        }
    }
    if (write_contexts.empty()) {
        return nullptr;
    }
    std::shared_ptr<FusedAsyncContext> fused_write_context(new FusedAsyncContext(std::move(write_contexts)));
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        write_context_list_.push_back(fused_write_context);  // 放入 write 队列
    }
    return fused_write_context;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncWriteByLayer(int                                          layer_id,
                                               const KVCacheResource&                       resource,
                                               const std::shared_ptr<KVCacheConnectorMeta>& meta,
                                               const KVCacheConnectorControlParams&         control_params) {
    if (stop_.load()) {
        return nullptr;
    }

    std::shared_ptr<KVCacheResource> resource_ptr = std::make_shared<KVCacheResource>(resource);

    std::vector<std::shared_ptr<AsyncContext>> write_contexts;
    for (const auto& [type, connector] : connectors_) {
        if (!connector) {
            continue;
        }
        if (type == ConnectorType::P2P) {
            auto write_context = connector->asyncWriteByLayer(layer_id, resource_ptr, meta);
            if (write_context) {
                write_contexts.emplace_back(write_context);
            }
        }
    }
    if (write_contexts.empty()) {
        return nullptr;
    }
    std::shared_ptr<FusedAsyncContext> fused_write_context(new FusedAsyncContext(std::move(write_contexts)));
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        write_context_list_.push_back(fused_write_context);  // 放入 write 队列
    }
    return fused_write_context;
}

bool KVCacheConnectorCoordinator::initMemoryConnector() {
    const auto memory_block_cache_size_mb         = kv_cache_config_.memory_block_cache_size_mb;
    const auto memory_block_cache_sync_timeout_ms = kv_cache_config_.memory_block_cache_sync_timeout_ms;
    if (memory_block_cache_size_mb <= 0 || memory_block_cache_sync_timeout_ms <= 0) {
        RTP_LLM_LOG_WARNING(
            "init memory connector failed, memory size or sync timeout is invalid, memory size: %ld MB, sync timeout: %ld ms",
            memory_block_cache_size_mb,
            memory_block_cache_sync_timeout_ms);
        return false;
    }

    // TODO(LXQ): init memory connector

    connectors_[ConnectorType::Memory] = memory_connector_;
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
                                                    layer_block_convertor,
                                                    metrics_reporter_);
    if (!p2p_connector_->init()) {
        RTP_LLM_LOG_ERROR("init p2p connector failed");
        return false;
    }
    connectors_[ConnectorType::P2P] = p2p_connector_;
    return true;
}

bool KVCacheConnectorCoordinator::initUpdateThread() {
    update_thread_ = autil::LoopThread::createLoopThread(
        [self = shared_from_this()]() { self->updateOnce(); }, update_interval_ms_, "CoordinatorUpdateThread");
    return update_thread_ != nullptr;
}

void KVCacheConnectorCoordinator::updateOnce() {
    std::lock_guard<std::mutex> lock(update_mutex_);

    // 1. 先处理 read_context 和 write_context，检查如果 done 就移除
    for (auto it = read_context_list_.begin(); it != read_context_list_.end();) {
        auto fused_read_context = *it;
        if (fused_read_context->done()) {
            it = read_context_list_.erase(it);
            RTP_LLM_LOG_DEBUG("read context done, remove from read context list, size: %zu", read_context_list_.size());
            continue;
        }
        it++;
    }

    for (auto it = write_context_list_.begin(); it != write_context_list_.end();) {
        auto fused_write_context = *it;
        if (fused_write_context->done()) {
            RTP_LLM_LOG_DEBUG("write context done, remove from write context list, context use_count: %zu, size: %zu",
                              fused_write_context.use_count(),
                              write_context_list_.size());
            it = write_context_list_.erase(it);
            continue;
        }
        it++;
    }

    // 2. 然后检查 match_context 队列，如果有 match done 就调用 asyncRead 然后放到 read 队列
    for (auto it = match_context_list_.begin(); it != match_context_list_.end();) {
        auto fused_read_context = *it;
        if (!fused_read_context->fusedMatchContext()) {
            // 无效的 match context，移除
            it = match_context_list_.erase(it);
            continue;
        }
        if (!fused_read_context->fusedMatchContext()->done()) {
            // match 还未完成，跳过
            it++;
            continue;
        }
        // match 已完成
        if (!fused_read_context->fusedMatchContext()->success()) {
            // match 失败，移除
            it = match_context_list_.erase(it);
            continue;
        }
        // match 成功，启动 read
        int  reuse_num      = fused_read_context->reuseBlockNum();
        auto match_contexts = fused_read_context->fusedMatchContext()->contexts();
        RTP_LLM_LOG_INFO(
            "read context match success, reuse_num: %d, match_contexts size: %zu", reuse_num, match_contexts.size());
        std::vector<std::shared_ptr<AsyncContext>> connector_read_contexts;
        for (size_t i = 0; i < match_contexts.size(); i++) {
            auto match_context = std::dynamic_pointer_cast<AsyncMatchContext>(match_contexts.at(i));
            if (!match_context) {
                continue;
            }
            if (match_context->matchedBlockCount() <= reuse_num) {
                continue;
            }
            auto connector = connectors_.at(match_context->connectorType());
            auto connector_read_context =
                connector->asyncRead(fused_read_context->resource(),
                                     fused_read_context->meta(),
                                     match_context,
                                     {reuse_num, match_context->matchedBlockCount() - reuse_num});
            if (connector_read_context) {
                connector_read_contexts.emplace_back(connector_read_context);
                if (match_context->connectorType() != ConnectorType::P2P) {
                    // 非 P2P，更新 reuse_num, p2p will always reuse all blocks
                    reuse_num = match_context->matchedBlockCount();
                }
            }
        }
        RTP_LLM_LOG_INFO("read context match success, reuse_num: %d, connector_read_contexts size: %zu",
                         reuse_num,
                         connector_read_contexts.size());

        // update reuse blocks num
        fused_read_context->setReuseBlockNum(reuse_num);
        RTP_LLM_LOG_INFO("read context update reuse blocks num, reuse_num: %d", reuse_num);

        fused_read_context->setFusedReadContext(std::make_shared<FusedAsyncContext>(connector_read_contexts));
        // 从 match 队列移除，放入 read 队列
        it = match_context_list_.erase(it);
        read_context_list_.push_back(fused_read_context);
    }
}

bool KVCacheConnectorCoordinator::broadcastTp(const BroadcastTpRequestPB& request, BroadcastTpResponsePB& response) {
    if (stop_.load()) {
        return false;
    }

    if (request.has_p2p_request()) {
        if (!p2p_connector_) {
            RTP_LLM_LOG_WARNING("broadcast tp failed, p2p connector is null, request: [%s]",
                                request.DebugString().c_str());
            response.mutable_p2p_response()->set_success(false);
            return false;
        }
        return p2p_connector_->handleTpBroadcast(request, response);
    }

    if (request.has_mem_request()) {
        if (!memory_connector_) {
            RTP_LLM_LOG_WARNING("broadcast tp failed, memory connector is null, request: [%s]",
                                request.DebugString().c_str());
            response.mutable_mem_response()->set_success(false);
            return false;
        }
        // TODO(LXQ): broadcast tp for memory connector
        return false;
    } else {
        return false;
    }
}

bool KVCacheConnectorCoordinator::handleRead(const P2PConnectorStartLoadRequestPB& request,
                                             P2PConnectorStartLoadResponsePB&      response) {
    if (stop_.load()) {
        response.set_success(false);
        return false;
    }

    if (!p2p_connector_) {
        RTP_LLM_LOG_WARNING("handleRead failed, p2p connector is null");
        response.set_success(false);
        return false;
    }

    auto ret = p2p_connector_->handleRead(request, response);
    return ret.ok();
}

ICompleteTokenIdImpl::ICompleteTokenIdImpl(const std::shared_ptr<CompleteTokenIds>& complete_token_ids):
    complete_token_ids_(complete_token_ids) {}

void ICompleteTokenIdImpl::appendTokenId(int batch_id, int token_id) {
    auto new_token = rtp_llm::Buffer(rtp_llm::MemoryType::MEMORY_CPU, rtp_llm::DataType::TYPE_INT32, {1}, &token_id);
    complete_token_ids_->appendTokens(batch_id, 0, new_token);
}

std::vector<int> ICompleteTokenIdImpl::currentExecuteTokens(int batch_id) {
    return complete_token_ids_->currentExecuteTokens(batch_id);
}
}  // namespace rtp_llm