#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "autil/NetUtil.h"

namespace rtp_llm {

P2PBroadcastClient::P2PBroadcastClient(const std::vector<std::string>& worker_addrs,
                                       int64_t                         cancel_broadcast_timeout_ms):
    worker_addrs_(worker_addrs), cancel_broadcast_timeout_ms_(cancel_broadcast_timeout_ms) {}

bool P2PBroadcastClient::init() {
    rpc_pool_             = std::make_shared<RPCPool>();
    tp_broadcast_manager_ = std::make_shared<BroadcastManager>(worker_addrs_);
    if (!tp_broadcast_manager_->init()) {
        RTP_LLM_LOG_ERROR("P2PBroadcastClient init failed: tp_broadcast_manager_ init failed");
        return false;
    }

    RTP_LLM_LOG_INFO("P2PBroadcastClient init success");
    return true;
}

std::shared_ptr<P2PBroadcastClient::Result>
P2PBroadcastClient::broadcast(int64_t                                               request_id,
                              const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                              const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
                              const std::string&                                    unique_key,
                              int64_t                                               deadline_ms,
                              P2PConnectorBroadcastType                             type,
                              int                                                   remote_tp_size) {
    // 构建 FunctionRequestPB
    std::vector<FunctionRequestPB> requests;
    size_t                         worker_num = tp_broadcast_manager_->workerNum();
    requests.reserve(worker_num);

    for (size_t i = 0; i < worker_num; ++i) {
        FunctionRequestPB request;
        genBroadcastRequest(request,
                            request_id,
                            layer_cache_buffers,
                            decode_transfer_servers,
                            unique_key,
                            deadline_ms,
                            type,
                            remote_tp_size);
        requests.push_back(std::move(request));
    }

    // gRPC 超时：在绝对 deadline_ms 前结束（worker 侧已按 D 提前返回，余量由 p2p_read_return_before_deadline_ms 承担）
    auto timeout_ms = deadline_ms - currentTimeMs();
    if (timeout_ms <= 0) {
        RTP_LLM_LOG_WARNING("broadcast timeout_ms: %ld <= 0, deadline_ms: %ld current_time_ms: %ld",
                            timeout_ms,
                            deadline_ms,
                            currentTimeMs());
        return nullptr;
    }

    // Define the RPC call lambda for ExecuteFunction
    auto rpc_call = [](std::shared_ptr<RpcService::Stub>&    stub,
                       std::shared_ptr<grpc::ClientContext>& client_context,
                       const FunctionRequestPB&              request,
                       grpc::CompletionQueue*                cq) {
        return stub->AsyncExecuteFunction(client_context.get(), request, cq);
    };

    auto result = tp_broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        requests, static_cast<int>(timeout_ms), rpc_call);
    if (!result) {
        RTP_LLM_LOG_WARNING("broadcast failed, cannot create broadcast result");
        return nullptr;
    }

    return std::make_shared<Result>(unique_key, result);
}

void P2PBroadcastClient::genBroadcastRequest(
    FunctionRequestPB&                                    request,
    int64_t                                               request_id,
    const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
    const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
    const std::string&                                    unique_key,
    int64_t                                               deadline_ms,
    P2PConnectorBroadcastType                             type,
    int                                                   remote_tp_size) {
    auto p2p_request = request.mutable_p2p_request();

    // 设置 layer_blocks
    for (const auto& layer_cache_buffer : layer_cache_buffers) {
        auto layer_block = p2p_request->add_layer_blocks();
        layer_block->set_layer_id(layer_cache_buffer->getLayerId());
        for (const auto& [key, block_id] : layer_cache_buffer->blockIdMap()) {
            layer_block->add_cache_keys(key);
            layer_block->add_block_ids(block_id);
        }
    }

    // 设置 peer_workers
    for (const auto& [ip, port] : decode_transfer_servers) {
        auto tp_worker = p2p_request->add_peer_workers();
        tp_worker->set_ip(ip);
        tp_worker->set_cache_store_port(port);
    }

    p2p_request->set_unique_key(unique_key);
    p2p_request->set_request_id(request_id);
    p2p_request->set_deadline_ms(deadline_ms);
    p2p_request->set_type(type);
    p2p_request->set_remote_tp_size(remote_tp_size);
}

bool P2PBroadcastClient::Result::success() const {
    if (!tp_broadcast_result_ || !tp_broadcast_result_->success()) {
        RTP_LLM_LOG_WARNING("P2PBroadcastClient::Result success is false, tp_broadcast_result failed");
        return false;
    }
    auto responses = tp_broadcast_result_->responses();
    for (const auto& response : responses) {
        if (!response.has_p2p_response()) {
            RTP_LLM_LOG_WARNING("P2PBroadcastClient::Result success is false, response has no p2p_response");
            return false;
        }
        const auto& p2p_response = response.p2p_response();
        if (p2p_response.error_code() != ErrorCodePB::NONE_ERROR) {
            RTP_LLM_LOG_WARNING("P2PBroadcastClient::Result success is false, p2p_response error code: %s",
                                ErrorCodeToString(transRPCErrorCode(p2p_response.error_code())).c_str());
            return false;
        }
    }
    return true;
}

void P2PBroadcastClient::Result::checkDone() {
    tp_broadcast_result_->waitDone(1);  // wait 1ms
    if (tp_broadcast_result_->done()) {
        total_cost_time_us_ = currentTimeUs() - start_time_us_;
    }
}

std::shared_ptr<P2PBroadcastClient::Result> P2PBroadcastClient::cancel(const std::string&        unique_key,
                                                                       P2PConnectorBroadcastType type) {
    RTP_LLM_LOG_DEBUG("P2PBroadcastClient cancel: unique_key: %s", unique_key.c_str());

    // 构建 FunctionRequestPB
    std::vector<FunctionRequestPB> requests;
    size_t                         worker_num = tp_broadcast_manager_->workerNum();
    requests.reserve(worker_num);

    for (size_t i = 0; i < worker_num; ++i) {
        FunctionRequestPB request;
        auto              p2p_request = request.mutable_p2p_request();
        p2p_request->set_unique_key(unique_key);
        p2p_request->set_type(type);
        requests.push_back(std::move(request));
    }

    int64_t timeout_ms = cancel_broadcast_timeout_ms_;

    // Define the RPC call lambda for ExecuteFunction
    auto rpc_call = [](std::shared_ptr<RpcService::Stub>&    stub,
                       std::shared_ptr<grpc::ClientContext>& client_context,
                       const FunctionRequestPB&              request,
                       grpc::CompletionQueue*                cq) {
        return stub->AsyncExecuteFunction(client_context.get(), request, cq);
    };

    auto result = tp_broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        requests, static_cast<int>(timeout_ms), rpc_call);
    if (!result) {
        RTP_LLM_LOG_WARNING("P2PBroadcastClient cancel: broadcast failed, unique_key: %s", unique_key.c_str());
        return nullptr;
    }

    // 不等待结果，异步发送取消请求即可
    RTP_LLM_LOG_DEBUG("P2PBroadcastClient cancel: broadcast sent, unique_key: %s", unique_key.c_str());
    return std::make_shared<Result>(unique_key, result);
}

ErrorCode P2PBroadcastClient::Result::errorCode() const {
    if (!tp_broadcast_result_ || !tp_broadcast_result_->done()) {
        return ErrorCode::UNKNOWN_ERROR;
    }
    auto responses = tp_broadcast_result_->responses();
    for (const auto& response : responses) {
        if (response.has_p2p_response()) {
            ErrorCodePB pb_error_code = response.p2p_response().error_code();
            if (pb_error_code != ErrorCodePB::NONE_ERROR) {
                return transRPCErrorCode(pb_error_code);
            }
        }
    }
    if (!tp_broadcast_result_->success()) {
        return ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED;
    }
    return ErrorCode::NONE_ERROR;
}

std::string P2PBroadcastClient::Result::errorMessage() const {
    if (!tp_broadcast_result_ || !tp_broadcast_result_->done()) {
        return "P2PBroadcastClient::Result not done yet";
    }
    auto responses = tp_broadcast_result_->responses();
    for (size_t rank = 0; rank < responses.size(); ++rank) {
        const auto& response = responses[rank];
        if (response.has_p2p_response()) {
            const auto& p2p_response = response.p2p_response();
            if (p2p_response.error_code() != ErrorCodePB::NONE_ERROR && !p2p_response.error_message().empty()) {
                return "RANK " + std::to_string(rank) + ": " + p2p_response.error_message();
            }
        }
    }
    if (!tp_broadcast_result_->success()) {
        return "P2PBroadcastClient broadcast failed";
    }
    return "";
}

}  // namespace rtp_llm
