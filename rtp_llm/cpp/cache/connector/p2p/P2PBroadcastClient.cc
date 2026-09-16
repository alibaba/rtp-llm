#include <limits>
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

std::shared_ptr<P2PBroadcastClient::Result> P2PBroadcastClient::broadcast(BroadcastParams params) {
    const size_t worker_num = tp_broadcast_manager_->workerNum();
    if (!params.routes.empty() && params.routes.size() != worker_num) {
        RTP_LLM_LOG_WARNING("broadcast route count %zu does not match worker count %zu, unique_key=%s",
                            params.routes.size(),
                            worker_num,
                            params.unique_key.c_str());
        return nullptr;
    }

    const bool                     per_rank_plan = !params.routes.empty();
    std::vector<FunctionRequestPB> requests;
    requests.reserve(worker_num);
    for (size_t worker_rank = 0; worker_rank < worker_num; ++worker_rank) {
        FunctionRequestPB request;
        genBroadcastRequest(request, params, per_rank_plan ? &params.routes[worker_rank] : nullptr);
        requests.push_back(std::move(request));
    }

    return broadcastRequests(std::move(requests), params.unique_key, params.deadline_ms);
}

std::shared_ptr<P2PBroadcastClient::Result>
P2PBroadcastClient::broadcastRequests(std::vector<FunctionRequestPB> requests,
                                      const std::string&             unique_key,
                                      int64_t                        deadline_ms) {

    // gRPC 超时与物理传输共用绝对 deadline_ms（D）。
    for (const auto& request : requests) {
        const auto request_deadline_ms = request.p2p_request().request_deadline_ms();
        if (request_deadline_ms <= 0 || request_deadline_ms == std::numeric_limits<int64_t>::max()
            || deadline_ms > request_deadline_ms) {
            return nullptr;
        }
    }
    if (deadline_ms <= 0) {
        return nullptr;
    }
    auto timeout_ms = deadline_ms - currentTimeMs();
    if (timeout_ms <= 0) {
        RTP_LLM_LOG_WARNING("broadcast timeout_ms: %ld <= 0, deadline_ms: %ld current_time_ms: %ld",
                            timeout_ms,
                            deadline_ms,
                            currentTimeMs());
        return nullptr;
    }

    // Define the RPC call lambda for ExecuteFunction
    auto rpc_call = [deadline_ms](std::shared_ptr<RpcService::Stub>&    stub,
                       std::shared_ptr<grpc::ClientContext>& client_context,
                       const FunctionRequestPB&              request,
                       grpc::CompletionQueue*                cq) {
        client_context->set_deadline(
            std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms)));
        return stub->AsyncExecuteFunction(client_context.get(), request, cq);
    };

    auto result = tp_broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        std::move(requests), static_cast<int>(std::min<int64_t>(timeout_ms, std::numeric_limits<int>::max())), rpc_call);
    if (!result) {
        RTP_LLM_LOG_WARNING("broadcast failed, cannot create broadcast result");
        return nullptr;
    }

    return std::make_shared<Result>(unique_key, result);
}

void P2PBroadcastClient::genBroadcastRequest(FunctionRequestPB&                  request,
                                             const BroadcastParams&              params,
                                             std::vector<TransferRoutePB>*       routes_of_worker) {
    auto p2p_request = request.mutable_p2p_request();

    // 传输端点索引表：route 里的 peer_index 在 worker 侧解析成具体端点。
    for (const auto& [ip, port] : params.peer_workers) {
        auto tp_worker = p2p_request->add_peer_workers();
        tp_worker->set_ip(ip);
        tp_worker->set_cache_store_port(port);
    }

    p2p_request->set_unique_key(params.unique_key);
    p2p_request->set_request_id(params.request_id);
    p2p_request->set_deadline_ms(params.deadline_ms);
    p2p_request->set_request_deadline_ms(params.request_deadline_ms);
    p2p_request->set_type(params.type);

    // 本 worker 那一份传输计划。为空即「该 worker 无任务」—— 这取代了
    // allow_empty_projection 作为空投影的权威信号。
    if (routes_of_worker == nullptr) {
        return;
    }
    for (auto& route : *routes_of_worker) {
        p2p_request->add_routes()->Swap(&route);
    }
    p2p_request->set_plan_digest(params.plan_digest);
}

bool P2PBroadcastClient::Result::success() const {
    return locally_completed_ || (tp_broadcast_result_ && tp_broadcast_result_->success());
}

void P2PBroadcastClient::Result::checkDone() {
    (void)done();  // Completion is advanced by the shared CQ consumer.
}

std::shared_ptr<P2PBroadcastClient::Result> P2PBroadcastClient::cancel(const std::string&        unique_key,
                                                                       P2PConnectorBroadcastType type,
                                                                       int64_t request_deadline_ms,
                                                                       int64_t request_id,
                                                                       int64_t deadline_ms) {
    RTP_LLM_LOG_DEBUG("P2PBroadcastClient cancel: unique_key: %s", unique_key.c_str());

    // 构建 FunctionRequestPB
    std::vector<FunctionRequestPB> requests;
    size_t                         worker_num = tp_broadcast_manager_->workerNum();
    requests.reserve(worker_num);

    for (size_t i = 0; i < worker_num; ++i) {
        FunctionRequestPB request;
        auto              p2p_request = request.mutable_p2p_request();
        p2p_request->set_unique_key(unique_key);
        p2p_request->set_request_id(request_id);
        p2p_request->set_deadline_ms(deadline_ms);
        p2p_request->set_type(type);
        p2p_request->set_request_deadline_ms(request_deadline_ms);
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
        std::move(requests), static_cast<int>(timeout_ms), rpc_call);
    if (!result) {
        RTP_LLM_LOG_WARNING("P2PBroadcastClient cancel: broadcast failed, unique_key: %s", unique_key.c_str());
        return nullptr;
    }

    // The caller may return immediately. Retain this cleanup RPC until Finish;
    // otherwise BroadcastResult destruction cancels the cancellation itself.
    // complete() moves and releases this callback when all ranks finish.
    result->setDoneCallback([result]() {});

    // 不等待结果，异步发送取消请求即可
    RTP_LLM_LOG_DEBUG("P2PBroadcastClient cancel: broadcast sent, unique_key: %s", unique_key.c_str());
    return std::make_shared<Result>(unique_key, result);
}

ErrorCode P2PBroadcastClient::Result::errorCode() const {
    return firstError().error.code();
}

std::string P2PBroadcastClient::Result::errorMessage() const {
    return firstError().error.ToString();
}

P2PBroadcastClient::LeaseStatusResult P2PBroadcastClient::queryLeaseStatus(const std::string& unique_key,
                                                                             int64_t            poll_timeout_ms) {
    LeaseStatusResult result;

    const size_t worker_num = tp_broadcast_manager_->workerNum();

    std::vector<FunctionRequestPB> requests;
    requests.reserve(worker_num);
    for (size_t i = 0; i < worker_num; ++i) {
        FunctionRequestPB request;
        auto*             p2p_req = request.mutable_p2p_request();
        p2p_req->set_unique_key(unique_key);
        p2p_req->set_type(P2PConnectorBroadcastType::QUERY_LEASE_STATUS);
        requests.push_back(std::move(request));
    }

    auto rpc_call = [](std::shared_ptr<RpcService::Stub>&    stub,
                       std::shared_ptr<grpc::ClientContext>& client_context,
                       const FunctionRequestPB&              request,
                       grpc::CompletionQueue*                cq) {
        return stub->AsyncExecuteFunction(client_context.get(), request, cq);
    };

    auto tp_result = tp_broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        std::move(requests), static_cast<int>(poll_timeout_ms), rpc_call);
    if (!tp_result) {
        RTP_LLM_LOG_WARNING("queryLeaseStatus: broadcast failed to create result, unique_key=%s", unique_key.c_str());
        return result;
    }

    // Block until all workers respond (with the given timeout).
    const bool all_done = tp_result->waitDone(static_cast<int>(poll_timeout_ms));
    if (!all_done || !tp_result->success()) {
        RTP_LLM_LOG_WARNING("queryLeaseStatus: broadcast timed out or failed, unique_key=%s", unique_key.c_str());
        return result;
    }

    auto responses = tp_result->responses();
    result.ranks.reserve(responses.size());
    for (size_t rank = 0; rank < responses.size(); ++rank) {
        const auto& resp = responses[rank];
        LeaseStatusResult::RankStatus rs;
        if (resp.has_p2p_response() && resp.p2p_response().error_code() == ErrorCodePB::NONE_ERROR
            && resp.p2p_response().has_lease_status()) {
            const auto& ls = resp.p2p_response().lease_status();
            rs.valid       = true;
            rs.sealed      = ls.sealed();
            rs.started_ops = ls.started_ops();
            rs.finished_ops = ls.finished_ops();
            rs.stopped     = ls.stopped();
        } else {
            RTP_LLM_LOG_WARNING("queryLeaseStatus: rank %zu missing valid lease_status, unique_key=%s",
                                rank, unique_key.c_str());
        }
        result.ranks.push_back(rs);
    }
    result.success = true;
    return result;
}

}  // namespace rtp_llm
