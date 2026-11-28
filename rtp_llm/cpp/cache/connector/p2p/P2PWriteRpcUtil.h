#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/P2PNotification.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/RpcCompletionQueue.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm::p2p_internal {

inline int64_t writebackSteadyDeadline(int64_t wall_deadline_ms) {
    const auto remaining = std::max<int64_t>(0, wall_deadline_ms - currentTimeMs());
    const auto now       = P2PNotification::nowMs();
    return now + std::min(remaining, std::numeric_limits<int64_t>::max() - now);
}

inline int64_t writebackLocalDeadline(int64_t timeout_ms) {
    return timeout_ms > 0 && timeout_ms <= std::numeric_limits<int32_t>::max() ? currentTimeMs() + timeout_ms : 0;
}

// Returns call()'s pointer result; on an exception or a null result returns null and describes why in `failure`.
template<typename Call>
auto callForResult(Call&& call, std::string& failure) -> decltype(call()) {
    try {
        auto result = call();
        if (!result) {
            failure = "no result";
        }
        return result;
    } catch (const std::exception& error) {
        failure = error.what();
    } catch (...) {
        failure = "unknown exception";
    }
    return nullptr;
}

// Dispatches async unary RPCs to dynamic peer addresses using a shared RPCPool
// and RpcCompletionQueue::instance() (zero per-peer threads, matching DecodeLoadHelper).
// Returns nullptr if no rank ever started a call (e.g. connection failure, CQ stopping,
// or prepare_rpc returning nullptr on rank 0).
template<typename RequestPB, typename ResponsePB, typename PrepareRpc>
std::shared_ptr<BroadcastResult<RequestPB, ResponsePB>>
dispatchRpcToPeers(const std::shared_ptr<RPCPool>&                rpc_pool,
                   const std::vector<std::string>&                addresses,
                   std::vector<RequestPB>                         requests,
                   int                                            timeout_ms,
                   std::chrono::system_clock::time_point          deadline,
                   PrepareRpc&&                                   prepare_rpc,
                   const std::function<grpc::Status(RequestPB&)>& before_dispatch = {},
                   std::optional<int64_t>                         request_id      = std::nullopt,
                   std::string                                    direction       = "unknown") {
    if (!rpc_pool || addresses.empty() || requests.size() != addresses.size() || timeout_ms <= 0) {
        return nullptr;
    }

    using ResultT = BroadcastResult<RequestPB, ResponsePB>;
    using CtxT    = typename ResultT::WorkerRpcContext;

    const size_t                       worker_size = addresses.size();
    std::vector<std::shared_ptr<CtxT>> contexts(worker_size);
    for (size_t rank = 0; rank < worker_size; ++rank) {
        auto conn = rpc_pool->getConnection(addresses[rank]);
        if (!conn.ok() || !conn.value().stub) {
            RTP_LLM_LOG_WARNING("writeback peer connection failed, rank=%zu addr=%s err=%s",
                                rank,
                                addresses[rank].c_str(),
                                conn.ok() ? "null stub" : conn.status().ToString().c_str());
            return nullptr;
        }
        auto ctx            = std::make_shared<CtxT>();
        ctx->stub           = conn.value().stub;
        ctx->client_context = std::make_shared<grpc::ClientContext>();
        ctx->client_context->set_deadline(deadline);
        ctx->request        = std::move(requests[rank]);
        ctx->server_addr    = addresses[rank];
        ctx->timeout_ms     = timeout_ms;
        contexts[rank]      = std::move(ctx);
    }

    auto result      = std::make_shared<ResultT>(contexts, request_id, std::move(direction));
    bool any_started = false;
    for (size_t rank = 0; rank < worker_size; ++rank) {
        auto& ctx = contexts[rank];
        if (result->firstError().error.hasError()) {
            result->finishDispatchFailure(rank, "earlier rank failed");
            continue;
        }
        try {
            result->markDispatchStarted(rank);
            auto reader_holder = std::make_shared<std::unique_ptr<grpc::ClientAsyncResponseReader<ResponsePB>>>();
            grpc::Status before_status = grpc::Status::OK;
            const bool   submitted     = RpcCompletionQueue::instance().submit(
                ctx->client_context,
                [&](grpc::CompletionQueue* cq, void* tag) {
                    if (before_dispatch) {
                        before_status = before_dispatch(ctx->request);
                        if (!before_status.ok()) {
                            return false;
                        }
                    }
                    *reader_holder = prepare_rpc(ctx->stub, ctx->client_context, ctx->request, cq);
                    if (!*reader_holder) {
                        return false;
                    }
                    any_started = true;
                    (*reader_holder)->StartCall();
                    (*reader_holder)->Finish(&ctx->response, &ctx->status, tag);
                    return true;
                },
                [result, rank, reader_holder](bool ok) mutable {
                    result->finishRank(rank, ok);
                    reader_holder->reset();
                });
            if (submitted) {
                result->markSubmitted(rank);
                continue;
            }
            if (!before_status.ok()) {
                ctx->status = std::move(before_status);
                result->finishRank(rank, true);
                continue;
            }
            if (!any_started) {
                return nullptr;
            }
            result->finishDispatchFailure(rank, "RpcCompletionQueue submit failed");
        } catch (const std::exception& error) {
            if (!any_started) {
                RTP_LLM_LOG_WARNING("writeback peer dispatch threw before StartCall, rank=%zu addr=%s err=%s",
                                    rank,
                                    ctx->server_addr.c_str(),
                                    error.what());
                return nullptr;
            }
            result->finishDispatchFailure(rank, error.what());
        } catch (...) {
            if (!any_started) {
                RTP_LLM_LOG_WARNING("writeback peer dispatch threw unknown error before StartCall, rank=%zu addr=%s",
                                    rank,
                                    ctx->server_addr.c_str());
                return nullptr;
            }
            result->finishDispatchFailure(rank, "unknown dispatch error");
        }
        for (size_t rest = rank + 1; rest < worker_size; ++rest) {
            result->finishDispatchFailure(rest, "not submitted after an earlier dispatch failure");
        }
        break;
    }
    return result;
}

}  // namespace rtp_llm::p2p_internal
