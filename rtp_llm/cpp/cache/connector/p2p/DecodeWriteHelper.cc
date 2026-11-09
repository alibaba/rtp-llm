#include "rtp_llm/cpp/cache/connector/p2p/DecodeWriteHelper.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <limits>

namespace rtp_llm {

std::shared_ptr<DecodeWriteHelper::Result>
DecodeWriteHelper::write(const std::string& prefill_ip,
                        uint32_t prefill_port,
                        P2PConnectorStartWriteBackRequestPB request,
                        int64_t deadline_ms) {
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
    auto peer = server(address);
    if (!peer) {
        return reject("peer_client_unavailable", deadline_ms - currentTimeMs());
    }
    const auto timeout_ms = deadline_ms - currentTimeMs();
    if (timeout_ms <= 0) {
        return reject("deadline_expired_after_peer_lookup", timeout_ms);
    }
    request.set_timeout_ms(timeout_ms);
    request.set_deadline_ms(deadline_ms);
    auto result = peer->broadcast<P2PConnectorStartWriteBackRequestPB, P2PConnectorStartWriteBackResponsePB>(
        {request},
        static_cast<int>(timeout_ms),
        [deadline_ms](std::shared_ptr<RpcService::Stub>&     stub,
                      std::shared_ptr<grpc::ClientContext>&  context,
                      const P2PConnectorStartWriteBackRequestPB& request,
                      grpc::CompletionQueue*                 queue) {
            context->set_deadline(std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms)));
            return stub->AsyncStartWriteBack(context.get(), request, queue);
        });
    if (!result) {
        return reject("rpc_submission_failed", deadline_ms - currentTimeMs());
    }
    return result;
}

std::shared_ptr<BroadcastManager> DecodeWriteHelper::server(const std::string& address) {
    std::string host;
    int32_t     port = 0;
    if (!parseGrpcHostPort(address, host, port)) {
        RTP_LLM_LOG_WARNING("writeback peer client rejected, reason=invalid_peer_address peer=%s", address.c_str());
        return nullptr;
    }
    return servers_.get(address, [&]() -> std::shared_ptr<BroadcastManager> {
        auto peer = std::make_shared<BroadcastManager>(std::vector<std::string>{address});
        if (!peer->init()) {
            RTP_LLM_LOG_WARNING("writeback peer client rejected, reason=client_init_failed peer=%s", address.c_str());
            return nullptr;
        }
        return peer;
    });
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
    auto peer = server(prefill_addr);
    if (!peer) {
        return reject("peer_client_unavailable");
    }
    P2PConnectorCancelWriteBackRequestPB request;
    request.set_unique_key(unique_key);
    request.set_deadline_ms(deadline_ms);
    auto result = peer->broadcast<P2PConnectorCancelWriteBackRequestPB, P2PConnectorStartWriteBackResponsePB>(
        {request}, static_cast<int>(timeout_ms), [](auto& stub, auto& context, const auto& request, auto* queue) {
            return stub->AsyncCancelWriteBack(context.get(), request, queue);
        });
    if (!result) {
        return reject("rpc_submission_failed");
    }
    return result;
}

}  // namespace rtp_llm
