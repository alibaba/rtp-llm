#include "rtp_llm/cpp/cache/connector/p2p/TPBroadcastClient.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "autil/NetUtil.h"

namespace rtp_llm {

TPBroadcastClient::TPBroadcastClient(const std::vector<std::string>& worker_addrs, int64_t extra_wait_time_ms):
    worker_addrs_(worker_addrs), extra_wait_time_ms_(extra_wait_time_ms) {}

bool TPBroadcastClient::init() {
    rpc_pool_ = std::make_shared<RPCPool>();
    if (!rpc_pool_) {
        RTP_LLM_LOG_ERROR("TPBroadcastClient init failed: rpc_pool_ is null");
        return false;
    }
    tp_broadcast_manager_ = std::make_shared<TpBroadcastManager>(worker_addrs_);
    if (!tp_broadcast_manager_->init()) {
        RTP_LLM_LOG_ERROR("TPBroadcastClient init failed: tp_broadcast_manager_ init failed");
        return false;
    }

    RTP_LLM_LOG_INFO("TPBroadcastClient init success");
    return true;
}

std::shared_ptr<TPBroadcastClient::Result>
TPBroadcastClient::broadcast(int64_t                                               request_id,
                             const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                             const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
                             const std::string&                                    unique_key,
                             int64_t                                               deadline_ms,
                             P2PConnectorBroadcastType                             type) {
    // 构建 FunctionRequestPB
    std::vector<FunctionRequestPB> requests;
    size_t                         worker_num = tp_broadcast_manager_->workerNum();
    requests.reserve(worker_num);

    for (size_t i = 0; i < worker_num; ++i) {
        FunctionRequestPB request;
        genBroadcastRequest(
            request, request_id, layer_cache_buffers, decode_transfer_servers, unique_key, deadline_ms, type);
        requests.push_back(std::move(request));
    }

    // 执行广播, broadcast 多等一会, 避免因为网络延迟导致超时而进程Abort
    auto timeout_ms = deadline_ms - currentTimeMs() + extra_wait_time_ms_;
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

void TPBroadcastClient::genBroadcastRequest(
    FunctionRequestPB&                                    request,
    int64_t                                               request_id,
    const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
    const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
    const std::string&                                    unique_key,
    int64_t                                               deadline_ms,
    P2PConnectorBroadcastType                             type) {
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
}

bool TPBroadcastClient::Result::success() const {
    if (!tp_broadcast_result_ || !tp_broadcast_result_->success()) {
        return false;
    }
    auto responses = tp_broadcast_result_->responses();
    for (const auto& response : responses) {
        if (!response.has_p2p_response() || !response.p2p_response().success()) {
            return false;
        }
    }
    return true;
}

void TPBroadcastClient::Result::checkDone() {
    tp_broadcast_result_->waitDone(1);  // wait 1ms
    if (tp_broadcast_result_->done()) {
        total_cost_time_us_ = currentTimeUs() - start_time_us_;
    }
}

std::shared_ptr<TPBroadcastClient::Result> TPBroadcastClient::cancel(const std::string&        unique_key,
                                                                     P2PConnectorBroadcastType type) {
    RTP_LLM_LOG_DEBUG("TPBroadcastClient cancel: unique_key: %s", unique_key.c_str());

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

    // 执行广播，使用短超时时间
    int64_t timeout_ms = 1000;  // 1秒超时

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
        RTP_LLM_LOG_WARNING("TPBroadcastClient cancel: broadcast failed, unique_key: %s", unique_key.c_str());
        return nullptr;
    }

    // 不等待结果，异步发送取消请求即可
    RTP_LLM_LOG_DEBUG("TPBroadcastClient cancel: broadcast sent, unique_key: %s", unique_key.c_str());
    return std::make_shared<Result>(unique_key, result);
}

void TPBroadcastClient::setExtraWaitTimeMs(int64_t extra_wait_time_ms) {
    extra_wait_time_ms_ = extra_wait_time_ms;
}

}  // namespace rtp_llm
