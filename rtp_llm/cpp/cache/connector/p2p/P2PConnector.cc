#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include <chrono>
#include <thread>

namespace rtp_llm {

P2PConnector::P2PConnector(const KVCacheConfig&                        cache_config,
                           const RuntimeConfig&                        runtime_config,
                           const CacheStoreConfig&                     cache_store_config,
                           const ParallelismConfig&                    parallelism_config,
                           const PDSepConfig&                          pd_sep_config,
                           const ModelConfig&                          model_config,
                           uint32_t                                    layer_all_num,
                           const std::shared_ptr<LayerBlockConvertor>& layer_block_convertor,
                           const kmonitor::MetricsReporterPtr&         metrics_reporter):
    cache_config_(cache_config),
    runtime_config_(runtime_config),
    cache_store_config_(cache_store_config),
    parallelism_config_(parallelism_config),
    pd_sep_config_(pd_sep_config),
    model_config_(model_config),
    layer_all_num_(layer_all_num),
    layer_block_convertor_(layer_block_convertor),
    metrics_reporter_(metrics_reporter) {}

P2PConnector::~P2PConnector() = default;

bool P2PConnector::init() {
    RTP_LLM_LOG_INFO("P2PConnector init start");

    // 只有 tp_rank == 0 才创建 scheduler（用于调度）
    if (parallelism_config_.tp_rank == 0) {
        scheduler_ = std::make_shared<P2PConnectorScheduler>(runtime_config_, cache_store_config_, metrics_reporter_);
        // 使用默认值初始化 scheduler（process_id 和 decode_polling_call_prefill_ms）
        std::string process_id = autil::NetUtil::getBindIp() + "_pid_" + std::to_string(getpid()) + "_timestamp_"
                                 + std::to_string(currentTimeUs());
        int64_t decode_polling_call_prefill_ms = pd_sep_config_.decode_polling_call_prefill_ms;
        if (!scheduler_->init(process_id, decode_polling_call_prefill_ms)) {
            RTP_LLM_LOG_ERROR("P2PConnector init failed: scheduler init failed");
            return false;
        }
    }

    // 每个 rank 都创建 worker（用于实际的数据传输）
    worker_ = std::make_shared<P2PConnectorWorker>(cache_config_,
                                                   cache_store_config_,
                                                   parallelism_config_,
                                                   pd_sep_config_,
                                                   model_config_,
                                                   layer_all_num_,
                                                   layer_block_convertor_,
                                                   metrics_reporter_);
    if (!worker_->init()) {
        RTP_LLM_LOG_ERROR("P2PConnector init failed: worker init failed");
        return false;
    }

    // 创建 stream store（用于管理 stream）
    stream_store_ = std::make_shared<P2PConnectorStreamStore>(metrics_reporter_);
    if (!stream_store_->init()) {
        RTP_LLM_LOG_ERROR("P2PConnector init failed: stream_store init failed");
        return false;
    }

    RTP_LLM_LOG_INFO("P2PConnector init success");
    return true;
}

std::shared_ptr<KVCacheConnector::AsyncMatchContext> P2PConnector::asyncMatch(const KVCacheResourcePtr&    resource,
                                                                              const std::shared_ptr<Meta>& meta) {
    if (meta->unique_key.empty()) {
        // no unique key, no need to call p2p connector
        RTP_LLM_LOG_DEBUG("P2PConnector asyncMatch failed, unique_key is empty");
        return nullptr;
    }
    if (!meta->generate_stream) {
        RTP_LLM_LOG_WARNING("P2PConnector asyncMatch failed, generate_stream is null");
        return nullptr;
    }

    // run p2p connector, prefill and decode side has different logic
    if (pd_sep_config_.role_type == RoleType::PREFILL) {
        // prefill side: save resource to stream_store, for decode side to read
        stream_store_->addResource(
            meta->unique_key, meta->request_id, meta->generate_stream, resource, meta->deadline_ms);
        return nullptr;
    }

    if (pd_sep_config_.role_type == RoleType::DECODE) {
        auto [prefill_ip, prefill_port] = meta->generate_stream->getPrefillAddr();
        if (prefill_ip.empty() || prefill_port == 0) {
            RTP_LLM_LOG_WARNING("P2PConnector asyncMatch failed, unique_key: %s, prefill_ip: %s, prefill_port: %d",
                                meta->unique_key.c_str(),
                                prefill_ip.c_str(),
                                prefill_port);
            return nullptr;
        }
        // decode side: P2PConnector 不需要 match，直接返回一个简单的 match context, 表示可以match全部数据
        return std::make_shared<P2PConnectorAsyncMatchContext>(resource);
    }
    RTP_LLM_LOG_WARNING("P2PConnector asyncMatch failed, unsupported role type %d", pd_sep_config_.role_type);
    return nullptr;
}

std::shared_ptr<AsyncContext> P2PConnector::asyncRead(const KVCacheResourcePtr&                 resource,
                                                      const std::shared_ptr<Meta>&              meta,
                                                      const std::shared_ptr<AsyncMatchContext>& match_context) {
    std::pair<int, int> block_range{meta->start_block_index, meta->block_size};
    RTP_LLM_LOG_DEBUG("P2PConnector::asyncRead start, unique_key: %s, block_range: %d-%d",
                      meta->unique_key.c_str(),
                      block_range.first,
                      block_range.second);
    if (scheduler_ == nullptr) {
        RTP_LLM_LOG_WARNING("P2PConnector::asyncRead failed, scheduler not ready (only tp_rank 0 has scheduler)");
        return nullptr;
    }

    if (pd_sep_config_.role_type == RoleType::DECODE) {
        return scheduler_->asyncRead(
            resource, meta->request_id, meta->unique_key, meta->deadline_ms, meta->generate_stream, block_range);
    }
    RTP_LLM_LOG_WARNING("P2PConnector::asyncRead failed, unsupported role type %d", pd_sep_config_.role_type);
    return nullptr;
}

std::shared_ptr<AsyncContext> P2PConnector::asyncWrite(const KVCacheResourcePtr&    resource,
                                                       const std::shared_ptr<Meta>& meta) {
    // p2p connector not support async write
    return nullptr;
}

// Prefill side: write by layer
std::shared_ptr<AsyncContext>
P2PConnector::asyncWriteByLayer(int layer_id, const KVCacheResourcePtr& resource, const std::shared_ptr<Meta>& meta) {
    RTP_LLM_LOG_DEBUG(
        "P2PConnector::asyncWriteByLayer start, layer_id: %d, unique_key: %s", layer_id, meta->unique_key.c_str());
    if (worker_ == nullptr) {
        RTP_LLM_LOG_WARNING("P2PConnector::asyncWriteByLayer failed, worker not init");
        return nullptr;
    }

    // writeByLayer is called by each rank
    worker_->writeByLayer(layer_id, resource, meta->request_id, meta->attention_event);
    return std::make_shared<P2PConnectorAsyncWriteByLayerContext>(resource);
}

// Prefill side: handle load request from decode side (StartLoad RPC)
void P2PConnector::handleRead(const P2PConnectorStartLoadRequestPB& request,
                              P2PConnectorStartLoadResponsePB&      response,
                              std::function<bool()>                 is_cancelled) {
    RTP_LLM_LOG_DEBUG("P2PConnector::handleRead start, unique_key: %s", request.unique_key().c_str());

    if (stream_store_ == nullptr) {
        RTP_LLM_LOG_WARNING("P2PConnector handleRead failed, stream_store not init");
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED));
        response.set_error_message("stream_store not init");
        return;
    }

    const std::string& unique_key  = request.unique_key();
    int64_t            deadline_ms = request.deadline_ms();

    // 构建 decode_transfer_servers
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    for (const auto& worker : request.workers()) {
        decode_transfer_servers.emplace_back(worker.ip(), worker.cache_store_port());
    }

    // wait for resource entry using condition variable, check is_cancelled every 1ms
    std::shared_ptr<P2PConnectorResourceEntry> resource_entry = nullptr;
    grpc::Status wait_status = waitForResourceEntry(unique_key, deadline_ms, is_cancelled, resource_entry);
    if (!wait_status.ok()) {
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED));
        response.set_error_message("waitForResourceEntry failed: " + wait_status.error_message());
        return;
    }

    if (resource_entry->generate_stream == nullptr || resource_entry->kv_cache_resource == nullptr) {
        // p2p connector prefill need stream for first token and kv cache resource
        RTP_LLM_LOG_WARNING(
            "P2PConnector::handleRead failed: generate_stream or kv_cache_resource is null, unique_key: %s",
            unique_key.c_str());
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED));
        response.set_error_message("generate_stream or kv_cache_resource is null");
        return;
    }

    // 执行 handleRead 操作 (发送 KV cache 到 decode 端)
    int64_t   request_id = resource_entry->request_id;
    ErrorInfo error_info = scheduler_->handleRead(
        resource_entry->kv_cache_resource, unique_key, request_id, decode_transfer_servers, deadline_ms, is_cancelled);
    if (error_info.hasError()) {
        RTP_LLM_LOG_ERROR("P2PConnector::handleRead failed: worker handleRead failed, unique_key: %s, error: %s",
                          unique_key.c_str(),
                          error_info.ToString().c_str());
        response.set_error_code(transErrorCodeToRPC(error_info.code()));
        response.set_error_message(error_info.ToString());
        return;
    }

    // wait for first call to update output
    resource_entry->generate_stream->waitForRemoteGenerate();

    // fill response with stream information
    grpc::Status fill_status = fillResponseWithStreamInfo(resource_entry, response);
    if (!fill_status.ok()) {
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED));
        response.set_error_message("fillResponseWithStreamInfo failed: " + fill_status.error_message());
        return;
    }

    response.set_error_code(ErrorCodePB::NONE_ERROR);
    return;
}

// worker side: handle tp broadcast request
bool P2PConnector::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    if (worker_ == nullptr) {
        RTP_LLM_LOG_WARNING("P2PConnector handleTpBroadcast failed, worker not init");
        return false;
    }

    if (!request.has_p2p_request()) {
        RTP_LLM_LOG_WARNING("P2PConnector handleTpBroadcast failed, no p2p_request in BroadcastTpRequestPB");
        return false;
    }

    const auto& p2p_request = request.p2p_request();
    int64_t     request_id  = p2p_request.request_id();
    std::string unique_key  = p2p_request.unique_key();
    int64_t     deadline_ms = p2p_request.deadline_ms();

    // Prefill 端: handleRead 请求
    if (p2p_request.type() == P2PConnectorBroadcastType::HANDLE_READ) {
        std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
        for (const auto& peer_worker : p2p_request.peer_workers()) {
            decode_transfer_servers.emplace_back(peer_worker.ip(), peer_worker.cache_store_port());
        }
        ErrorInfo error_info   = worker_->handleRead(request_id, unique_key, deadline_ms, decode_transfer_servers);
        auto*     p2p_response = response.mutable_p2p_response();
        if (error_info.hasError()) {
            p2p_response->set_error_code(transErrorCodeToRPC(error_info.code()));
            p2p_response->set_error_message(error_info.ToString());
        } else {
            p2p_response->set_error_code(ErrorCodePB::NONE_ERROR);
            p2p_response->set_error_message("");
        }
        return error_info.ok();
    }

    // Decode 端: read 请求
    if (p2p_request.type() == P2PConnectorBroadcastType::READ) {
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
        ErrorInfo error_info   = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);
        auto*     p2p_response = response.mutable_p2p_response();
        if (error_info.hasError()) {
            p2p_response->set_error_code(transErrorCodeToRPC(error_info.code()));
            p2p_response->set_error_message(error_info.ToString());
        } else {
            p2p_response->set_error_code(ErrorCodePB::NONE_ERROR);
            p2p_response->set_error_message("");
        }
        return error_info.ok();
    }

    // Decode 端: 取消 read 请求
    if (p2p_request.type() == P2PConnectorBroadcastType::CANCEL_READ) {
        bool  ret          = worker_->cancelRead(unique_key);
        auto* p2p_response = response.mutable_p2p_response();
        p2p_response->set_error_code(ErrorCodePB::NONE_ERROR);
        p2p_response->set_error_message("");
        return ret;
    }

    // Prefill 端: 取消 handleRead 请求
    if (p2p_request.type() == P2PConnectorBroadcastType::CANCEL_HANDLE_READ) {
        bool  ret          = worker_->cancelHandleRead(unique_key);
        auto* p2p_response = response.mutable_p2p_response();
        p2p_response->set_error_code(ErrorCodePB::NONE_ERROR);
        p2p_response->set_error_message("");
        return ret;
    }

    RTP_LLM_LOG_WARNING("P2PConnector handleTpBroadcast failed, unsupported p2p_request type %d", p2p_request.type());
    auto* p2p_response = response.mutable_p2p_response();
    p2p_response->set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED));
    p2p_response->set_error_message("unsupported p2p_request type");
    return false;
}

grpc::Status P2PConnector::waitForResourceEntry(const std::string&                          unique_key,
                                                int64_t                                     deadline_ms,
                                                std::function<bool()>                       is_cancelled,
                                                std::shared_ptr<P2PConnectorResourceEntry>& resource_entry) {
    resource_entry          = nullptr;
    int64_t current_time_ms = currentTimeMs();
    while (current_time_ms < deadline_ms) {
        // Check if cancelled
        if (is_cancelled && is_cancelled()) {
            RTP_LLM_LOG_WARNING(
                "P2PConnector::waitForResourceEntry cancelled while waiting for resource, unique_key: %s",
                unique_key.c_str());
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
        }

        // Try to get resource with short timeout (1ms)
        int64_t short_deadline_ms = std::min(current_time_ms + 1, deadline_ms);
        resource_entry            = stream_store_->waitAndStealResource(unique_key, short_deadline_ms);
        if (resource_entry) {
            return grpc::Status::OK;  // Resource found
        }

        // Sleep 1ms before next check
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        current_time_ms = currentTimeMs();
    }

    // Timeout: resource not found
    RTP_LLM_LOG_WARNING("P2PConnector::waitForResourceEntry failed: resource not found, unique_key: %s",
                        unique_key.c_str());
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
        RTP_LLM_LOG_DEBUG("P2PConnector::fillResponseWithStreamInfo: first token: %d", first_token);
    } else {
        RTP_LLM_LOG_WARNING("P2PConnector::fillResponseWithStreamInfo failed: first token not found");
        return grpc::Status(grpc::StatusCode::INTERNAL, "first token not found");
    }

    // get context position ids from generate_stream
    auto position_ids = resource_entry->generate_stream->getContextPositionIdsPB();
    if (!position_ids.empty()) {
        response.mutable_position_ids()->CopyFrom({position_ids.begin(), position_ids.end()});
        RTP_LLM_LOG_DEBUG("P2PConnector::fillResponseWithStreamInfo: position_ids: %s",
                          vectorToString(position_ids).c_str());
    }

    auto [total_reuse_len, local_reuse_len, remote_reuse_len] = resource_entry->generate_stream->getReuseLength();
    response.set_total_reuse_len(total_reuse_len);
    response.set_local_reuse_len(local_reuse_len);
    response.set_remote_reuse_len(remote_reuse_len);
    RTP_LLM_LOG_DEBUG(
        "P2PConnector::fillResponseWithStreamInfo: total_reuse_len: %d, local_reuse_len: %d, remote_reuse_len: %d",
        total_reuse_len,
        local_reuse_len,
        remote_reuse_len);

    // get propose info from generate_stream
    auto sp_info_opt = resource_entry->generate_stream->getSPInfoPB();
    if (sp_info_opt.has_value()) {
        auto& [propose_tokens, propose_probs, propose_hidden] = sp_info_opt.value();
        response.mutable_propose_token_ids()->CopyFrom({propose_tokens.begin(), propose_tokens.end()});
        response.mutable_propose_probs()->CopyFrom(propose_probs);
        response.mutable_propose_hidden()->CopyFrom(propose_hidden);
        RTP_LLM_LOG_DEBUG("P2PConnector::fillResponseWithStreamInfo: propose_tokens: %s",
                          vectorToString(propose_tokens).c_str());
    }

    return grpc::Status::OK;
}

}  // namespace rtp_llm
