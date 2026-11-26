#include "rtp_llm/cpp/cache/connector/p2p/DecodeWriteHelper.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWriteRpcUtil.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <chrono>
#include <limits>
#include <utility>

namespace rtp_llm {

std::shared_ptr<DecodeWriteHelper::Result> DecodeWriteHelper::write(const std::string&                  prefill_ip,
                                                                    uint32_t                            prefill_port,
                                                                    P2PConnectorStartWriteBackRequestPB request,
                                                                    int64_t                             deadline_ms) {
    const auto reject = [&](const char* reason, int64_t remaining_ms) {
        RTP_LLM_LOG_WARNING("StartWriteBack submission rejected, reason=%s request_id=%ld unique_key=%s "
                            "peer=%s port=%u deadline_ms=%ld remaining_ms=%ld",
                            reason,
                            request.request_id(),
                            request.unique_key().c_str(),
                            prefill_ip.c_str(),
                            prefill_port,
                            deadline_ms,
                            remaining_ms);
        return std::shared_ptr<Result>{};
    };
    const auto address      = formatGrpcHostPort(prefill_ip, prefill_port);
    const auto remaining_ms = deadline_ms - currentTimeMs();
    if (address.empty()) {
        return reject("invalid_peer_address", remaining_ms);
    }
    if (request.unique_key().empty()) {
        return reject("empty_unique_key", remaining_ms);
    }
    if (remaining_ms <= 0) {
        return reject("deadline_expired", remaining_ms);
    }
    if (remaining_ms > std::numeric_limits<int>::max()) {
        return reject("timeout_out_of_range", remaining_ms);
    }
    std::string host;
    int32_t     port = 0;
    if (!parseGrpcHostPort(address, host, port)) {
        return reject("invalid_peer_address", remaining_ms);
    }
    request.set_deadline_ms(deadline_ms);
    auto result =
        p2p_internal::dispatchRpcToPeers<P2PConnectorStartWriteBackRequestPB, P2PConnectorStartWriteBackResponsePB>(
            rpc_pool_,
            {address},
            {std::move(request)},
            static_cast<int>(remaining_ms),
            std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms)),
            [deadline_ms](std::shared_ptr<RpcService::Stub>&    stub,
                          std::shared_ptr<grpc::ClientContext>& context,
                          P2PConnectorStartWriteBackRequestPB&  req,
                          grpc::CompletionQueue*                queue) {
                const auto budget_ms = deadline_ms - currentTimeMs();
                if (budget_ms <= 0) {
                    return std::unique_ptr<grpc::ClientAsyncResponseReader<P2PConnectorStartWriteBackResponsePB>>{};
                }
                req.set_timeout_ms(budget_ms);
                return stub->PrepareAsyncStartWriteBack(context.get(), req, queue);
            });
    if (!result) {
        return reject("rpc_submission_failed", deadline_ms - currentTimeMs());
    }
    return result;
}

std::shared_ptr<DecodeWriteHelper::CancelResult> DecodeWriteHelper::cancel(const std::string& prefill_addr,
                                                                           const std::string& unique_key,
                                                                           int64_t            deadline_ms,
                                                                           int64_t            timeout_ms) {
    const auto reject = [&](const char* reason) {
        RTP_LLM_LOG_WARNING("CancelWriteBack submission rejected, reason=%s unique_key=%s peer=%s "
                            "deadline_ms=%ld timeout_ms=%ld",
                            reason,
                            unique_key.c_str(),
                            prefill_addr.c_str(),
                            deadline_ms,
                            timeout_ms);
        return std::shared_ptr<CancelResult>{};
    };
    if (unique_key.empty()) {
        return reject("empty_unique_key");
    }
    if (timeout_ms <= 0) {
        return reject("non_positive_timeout");
    }
    if (timeout_ms > std::numeric_limits<int>::max()) {
        return reject("timeout_out_of_range");
    }
    std::string host;
    int32_t     port = 0;
    if (!parseGrpcHostPort(prefill_addr, host, port)) {
        return reject("invalid_peer_address");
    }
    P2PConnectorCancelWriteBackRequestPB request;
    request.set_unique_key(unique_key);
    request.set_deadline_ms(deadline_ms);
    auto result =
        p2p_internal::dispatchRpcToPeers<P2PConnectorCancelWriteBackRequestPB, P2PConnectorStartWriteBackResponsePB>(
            rpc_pool_,
            {prefill_addr},
            {std::move(request)},
            static_cast<int>(timeout_ms),
            std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms),
            [](std::shared_ptr<RpcService::Stub>&    stub,
               std::shared_ptr<grpc::ClientContext>& context,
               P2PConnectorCancelWriteBackRequestPB& req,
               grpc::CompletionQueue*                queue) {
                return stub->PrepareAsyncCancelWriteBack(context.get(), req, queue);
            });
    if (!result) {
        return reject("rpc_submission_failed");
    }
    return result;
}

}  // namespace rtp_llm
