#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"

#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorLayerContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorScheduler.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include <chrono>
#include <thread>

namespace rtp_llm {

P2PConnector::P2PConnector(P2PConnectorConfig                          config,
                           const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                           const kmonitor::MetricsReporterPtr&         metrics_reporter):
    config_(std::move(config)), layer_block_converter_(layer_block_converter), metrics_reporter_(metrics_reporter) {}

P2PConnector::~P2PConnector() = default;

bool P2PConnector::init() {
    if (config_.tp_rank == 0) {
        scheduler_             = std::make_shared<P2PConnectorScheduler>(config_.scheduler_config, metrics_reporter_);
        std::string process_id = autil::NetUtil::getBindIp() + "_pid_" + std::to_string(getpid()) + "_timestamp_"
                                 + std::to_string(currentTimeUs());
        if (!scheduler_->init(process_id)) {
            RTP_LLM_LOG_ERROR("init failed: scheduler init failed");
            return false;
        }
    }

    worker_ = std::make_shared<P2PConnectorWorker>(config_.worker_config, layer_block_converter_, metrics_reporter_);
    if (!worker_->init()) {
        RTP_LLM_LOG_ERROR("init failed: worker init failed");
        return false;
    }

    // 创建 stream store（用于管理 stream）
    stream_store_ = std::make_shared<P2PConnectorResourceStore>(
        metrics_reporter_, config_.scheduler_config.p2p_resource_store_timeout_check_interval_ms);
    if (!stream_store_->init()) {
        RTP_LLM_LOG_ERROR("init failed: stream_store init failed");
        return false;
    }

    return true;
}

std::shared_ptr<AsyncMatchContext> P2PConnector::asyncMatch(const KVCacheResourcePtr&    resource,
                                                            const std::shared_ptr<Meta>& meta) {
    if (!meta || !resource || !meta->generateStream()) {
        RTP_LLM_LOG_WARNING("asyncMatch failed, meta is null or resource is null or generate_stream is null");
        return nullptr;
    }

    if (config_.role_type == RoleType::PREFILL) {
        if (!stream_store_->addResource(meta->generateStream(), resource)) {
            RTP_LLM_LOG_WARNING("asyncMatch failed, stream_store add resource failed");
            return nullptr;
        }
        return nullptr;
    }

    if (config_.role_type == RoleType::DECODE) {
        return std::make_shared<P2PConnectorAsyncMatchContext>(resource);
    }
    RTP_LLM_LOG_WARNING("asyncMatch failed, unsupported role type %d", config_.role_type);
    return nullptr;
}

std::shared_ptr<AsyncContext> P2PConnector::asyncRead(const KVCacheResourcePtr&                 resource,
                                                      const std::shared_ptr<Meta>&              meta,
                                                      const std::shared_ptr<AsyncMatchContext>& match_context,
                                                      int                                       start_read_block_index,
                                                      int                                       read_block_num) {
    if (!meta || !resource || !meta->generateStream()) {
        RTP_LLM_LOG_WARNING("asyncRead failed, meta is null");
        return nullptr;
    }

    auto                generate_stream = meta->generateStream();
    std::pair<int, int> block_range{start_read_block_index, read_block_num};

    if (scheduler_ == nullptr) {
        RTP_LLM_LOG_WARNING("asyncRead failed, scheduler not ready (only tp_rank 0 has scheduler)");
        return nullptr;
    }

    if (config_.role_type == RoleType::DECODE) {
        auto result = scheduler_->asyncRead(resource, generate_stream, block_range);
        if (!result.ok()) {
            RTP_LLM_LOG_WARNING("asyncRead failed, unique_key: %s, error: %s",
                                generate_stream->uniqueKey().c_str(),
                                result.error_info.ToString().c_str());
            generate_stream->setStop(result.error_info.code(), result.error_info.ToString());
            return nullptr;
        }
        return result.context;
    }
    RTP_LLM_LOG_WARNING("asyncRead failed, unsupported role type %d", config_.role_type);
    return nullptr;
}

std::shared_ptr<AsyncContext> P2PConnector::asyncWrite(const KVCacheResourcePtr&    resource,
                                                       const std::shared_ptr<Meta>& meta) {
    // p2p connector not support async write
    return nullptr;
}

std::shared_ptr<AsyncContext>
P2PConnector::asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) {
    auto resource = std::make_shared<KVCacheResource>(layer_context->kvCacheResource());
    worker_->writeByLayer(layer_id, resource, layer_context->requestId(), layer_context->attentionEvent());
    return std::make_shared<P2PConnectorAsyncWriteByLayerContext>(resource);
}

void P2PConnector::handleRead(const P2PConnectorStartLoadRequestPB& request,
                              P2PConnectorStartLoadResponsePB&      response,
                              std::function<bool()>                 is_cancelled) {
    if (scheduler_ == nullptr) {
        RTP_LLM_LOG_WARNING("handleRead failed, scheduler not initialized (only tp_rank 0 has scheduler)");
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED));
        response.set_error_message("scheduler not initialized");
        return;
    }

    const std::string& unique_key  = request.unique_key();
    int64_t            deadline_ms = request.deadline_ms();

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    for (const auto& worker : request.workers()) {
        decode_transfer_servers.emplace_back(worker.ip(), worker.cache_store_port());
    }

    RTP_LLM_LOG_INFO("handleRead [P2P]: start unique_key=%s, deadline_ms=%ld, decode_servers=%zu",
                     unique_key.c_str(),
                     deadline_ms,
                     decode_transfer_servers.size());

    std::shared_ptr<P2PConnectorResourceEntry> resource_entry = nullptr;
    grpc::Status wait_status = waitForResourceEntry(unique_key, deadline_ms, is_cancelled, resource_entry);
    if (!wait_status.ok()) {
        RTP_LLM_LOG_WARNING("handleRead [P2P]: waitForResourceEntry failed, unique_key=%s, status=%s",
                            unique_key.c_str(),
                            wait_status.error_message().c_str());
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED));
        response.set_error_message("waitForResourceEntry failed: " + wait_status.error_message());
        return;
    }

    int64_t request_id = resource_entry->request_id;
    RTP_LLM_LOG_INFO("handleRead [P2P]: resource ready, unique_key=%s, request_id=%ld, sending KV cache",
                     unique_key.c_str(),
                     request_id);
    ErrorInfo error_info = scheduler_->sendKVCache(
        resource_entry->kv_cache_resource, unique_key, request_id, decode_transfer_servers, deadline_ms, is_cancelled);
    if (error_info.hasError()) {
        RTP_LLM_LOG_ERROR("handleRead failed: worker handleRead failed, unique_key: %s, error: %s",
                          unique_key.c_str(),
                          error_info.ToString().c_str());
        response.set_error_code(transErrorCodeToRPC(error_info.code()));
        response.set_error_message(error_info.ToString());
        return;
    }

    waitAndFillResponse(resource_entry, response);
}

void P2PConnector::waitAndFillResponse(const std::shared_ptr<P2PConnectorResourceEntry>& resource_entry,
                                       P2PConnectorStartLoadResponsePB&                  response) {
    resource_entry->generate_stream->waitForRemoteGenerate();

    grpc::Status fill_status = fillResponseWithStreamInfo(resource_entry, response);
    if (!fill_status.ok()) {
        RTP_LLM_LOG_WARNING("waitAndFillResponse failed, unique_key: %s, error: %s",
                            resource_entry->generate_stream->uniqueKey().c_str(),
                            fill_status.error_message().c_str());
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED));
        response.set_error_message("fillResponseWithStreamInfo failed: " + fill_status.error_message());
        return;
    }

    response.set_error_code(ErrorCodePB::NONE_ERROR);
}

namespace {

void setP2PResponse(FunctionResponsePB& response, const ErrorInfo& error_info) {
    auto* p2p_response = response.mutable_p2p_response();
    if (error_info.hasError()) {
        p2p_response->set_error_code(transErrorCodeToRPC(error_info.code()));
        p2p_response->set_error_message(error_info.ToString());
    } else {
        p2p_response->set_error_code(ErrorCodePB::NONE_ERROR);
        p2p_response->set_error_message("");
    }
}

void setP2PResponseOk(FunctionResponsePB& response) {
    auto* p2p_response = response.mutable_p2p_response();
    p2p_response->set_error_code(ErrorCodePB::NONE_ERROR);
    p2p_response->set_error_message("");
}

}  // namespace

bool P2PConnector::executeHandleRead(int64_t                                 request_id,
                                     const std::string&                      unique_key,
                                     int64_t                                 deadline_ms,
                                     const P2PConnectorBroadcastTpRequestPB& p2p_request,
                                     FunctionResponsePB&                     response) {
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    for (const auto& peer_worker : p2p_request.peer_workers()) {
        decode_transfer_servers.emplace_back(peer_worker.ip(), peer_worker.cache_store_port());
    }
    ErrorInfo error_info = worker_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
    if (error_info.hasError()) {
        RTP_LLM_LOG_WARNING("executeHandleRead failed, request_id: %ld, unique_key: %s, error: %s",
                            request_id,
                            unique_key.c_str(),
                            error_info.ToString().c_str());
    }
    setP2PResponse(response, error_info);
    return error_info.ok();
}

bool P2PConnector::executeRead(int64_t                                 request_id,
                               const std::string&                      unique_key,
                               int64_t                                 deadline_ms,
                               const P2PConnectorBroadcastTpRequestPB& p2p_request,
                               FunctionResponsePB&                     response) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    for (const auto& layer_block_pb : p2p_request.layer_blocks()) {
        auto layer_id           = layer_block_pb.layer_id();
        auto layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id);
        auto cache_keys         = layer_block_pb.cache_keys();
        auto block_ids          = layer_block_pb.block_ids();
        for (size_t i = 0; i < cache_keys.size(); i++) {
            layer_cache_buffer->addBlockId(cache_keys[i], block_ids[i]);
        }
        layer_cache_buffers.push_back(layer_cache_buffer);
    }
    int       remote_tp_size = (p2p_request.remote_tp_size() > 0) ? p2p_request.remote_tp_size() : 1;
    ErrorInfo error_info     = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers, remote_tp_size);
    if (error_info.hasError()) {
        RTP_LLM_LOG_WARNING("executeRead failed, request_id: %ld, unique_key: %s, error: %s",
                            request_id,
                            unique_key.c_str(),
                            error_info.ToString().c_str());
    }
    setP2PResponse(response, error_info);
    return error_info.ok();
}

bool P2PConnector::executeCancelRead(const std::string& unique_key, FunctionResponsePB& response) {
    bool ret = worker_->cancelRead(unique_key);
    setP2PResponseOk(response);
    return ret;
}

bool P2PConnector::executeCancelHandleRead(const std::string& unique_key, FunctionResponsePB& response) {
    bool ret = worker_->cancelSend(unique_key);
    setP2PResponseOk(response);
    return ret;
}

bool P2PConnector::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    if (worker_ == nullptr) {
        RTP_LLM_LOG_WARNING("executeFunction failed, worker not init");
        return false;
    }

    if (!request.has_p2p_request()) {
        RTP_LLM_LOG_WARNING("executeFunction failed, no p2p_request in FunctionRequestPB");
        return false;
    }

    const auto& p2p_request = request.p2p_request();
    int64_t     request_id  = p2p_request.request_id();
    std::string unique_key  = p2p_request.unique_key();
    int64_t     deadline_ms = p2p_request.deadline_ms();

    switch (p2p_request.type()) {
        case P2PConnectorBroadcastType::HANDLE_READ:
            return executeHandleRead(request_id, unique_key, deadline_ms, p2p_request, response);
        case P2PConnectorBroadcastType::READ:
            return executeRead(request_id, unique_key, deadline_ms, p2p_request, response);
        case P2PConnectorBroadcastType::CANCEL_READ:
            return executeCancelRead(unique_key, response);
        case P2PConnectorBroadcastType::CANCEL_HANDLE_READ:
            return executeCancelHandleRead(unique_key, response);
        default:
            RTP_LLM_LOG_WARNING("executeFunction failed, unsupported p2p_request type %d", p2p_request.type());
            auto* p2p_response = response.mutable_p2p_response();
            p2p_response->set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED));
            p2p_response->set_error_message("unsupported p2p_request type");
            return false;
    }
}

grpc::Status P2PConnector::waitForResourceEntry(const std::string&                          unique_key,
                                                int64_t                                     deadline_ms,
                                                std::function<bool()>                       is_cancelled,
                                                std::shared_ptr<P2PConnectorResourceEntry>& resource_entry) {
    resource_entry = stream_store_->waitAndStealResource(unique_key, deadline_ms, is_cancelled);
    if (resource_entry) {
        return grpc::Status::OK;
    }
    if (is_cancelled && is_cancelled()) {
        RTP_LLM_LOG_WARNING("waiting for resource cancelled, unique_key: %s", unique_key.c_str());
        return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
    }
    RTP_LLM_LOG_WARNING("resource not found, unique_key: %s", unique_key.c_str());
    return grpc::Status(grpc::StatusCode::INTERNAL, "resource not found");
}

grpc::Status P2PConnector::fillResponseWithStreamInfo(const std::shared_ptr<P2PConnectorResourceEntry>& resource_entry,
                                                      P2PConnectorStartLoadResponsePB&                  response) {
    // return first token id
    int  first_token = 0;
    auto all_tokens  = resource_entry->generate_stream->currentExecuteTokens(0);
    if (all_tokens.size() > 0) {
        first_token = all_tokens[all_tokens.size() - 1];
        response.set_first_generate_token_id(first_token);
        RTP_LLM_LOG_DEBUG("fill response: first token: %d", first_token);
    } else {
        RTP_LLM_LOG_WARNING("fill response failed: first token not found");
        return grpc::Status(grpc::StatusCode::INTERNAL, "first token not found");
    }

    // get context position ids from generate_stream
    auto position_ids = resource_entry->generate_stream->getContextPositionIdsPB();
    if (!position_ids.empty()) {
        response.mutable_position_ids()->CopyFrom({position_ids.begin(), position_ids.end()});
        RTP_LLM_LOG_DEBUG("fill response: position_ids: %s", vectorToString(position_ids).c_str());
    }

    auto [total_reuse_len, local_reuse_len, remote_reuse_len, memory_reuse_len] =
        resource_entry->generate_stream->getReuseLength();
    response.set_total_reuse_len(total_reuse_len);
    response.set_local_reuse_len(local_reuse_len);
    response.set_remote_reuse_len(remote_reuse_len);
    response.set_memory_reuse_len(memory_reuse_len);
    RTP_LLM_LOG_DEBUG("fill response: total: %d, local: %d, remote: %d, memory: %d",
                      total_reuse_len,
                      local_reuse_len,
                      remote_reuse_len,
                      memory_reuse_len);

    // get propose info from generate_stream
    auto sp_info_opt = resource_entry->generate_stream->getSPInfoPB();
    if (sp_info_opt.has_value()) {
        auto& [propose_tokens, propose_probs, propose_hidden] = sp_info_opt.value();
        response.mutable_propose_token_ids()->CopyFrom({propose_tokens.begin(), propose_tokens.end()});
        response.mutable_propose_probs()->CopyFrom(propose_probs);
        response.mutable_propose_hidden()->CopyFrom(propose_hidden);
        RTP_LLM_LOG_DEBUG("fill response: propose_tokens: %s", vectorToString(propose_tokens).c_str());
    }

    return grpc::Status::OK;
}

}  // namespace rtp_llm
