#include "rtp_llm/cpp/cache/connector/p2p/DecodeWriteHelper.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <limits>

namespace rtp_llm {

std::shared_ptr<DecodeWriteHelper::Result>
DecodeWriteHelper::write(const std::string& prefill_ip,
                        uint32_t prefill_port,
                        P2PConnectorStartWriteRequestPB request,
                        int64_t deadline_ms) {
    const auto address      = formatGrpcHostPort(prefill_ip, prefill_port);
    const auto remaining_ms = deadline_ms - currentTimeMs();
    if (address.empty() || request.unique_key().empty() || remaining_ms <= 0
        || remaining_ms > std::numeric_limits<int>::max()) {
        return nullptr;
    }
    auto peer = server(address);
    if (!peer) {
        return nullptr;
    }
    const auto timeout_ms = deadline_ms - currentTimeMs();
    if (timeout_ms <= 0) {
        return nullptr;
    }
    request.set_timeout_ms(timeout_ms);
    request.set_deadline_ms(deadline_ms);
    return peer->broadcast<P2PConnectorStartWriteRequestPB, P2PConnectorStartWriteResponsePB>(
        {request},
        static_cast<int>(timeout_ms),
        [deadline_ms](std::shared_ptr<RpcService::Stub>&     stub,
                      std::shared_ptr<grpc::ClientContext>&  context,
                      const P2PConnectorStartWriteRequestPB& request,
                      grpc::CompletionQueue*                 queue) {
            context->set_deadline(std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms)));
            return stub->AsyncStartWrite(context.get(), request, queue);
        });
}

std::shared_ptr<BroadcastManager> DecodeWriteHelper::server(const std::string& address) {
    std::string host;
    int32_t     port = 0;
    if (!parseGrpcHostPort(address, host, port)) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = servers_.find(address);
    if (it != servers_.end()) {
        return it->second;
    }
    auto peer = std::make_shared<BroadcastManager>(std::vector<std::string>{address});
    if (!peer->init()) {
        return nullptr;
    }
    servers_.emplace(address, peer);
    return peer;
}

std::shared_ptr<DecodeWriteHelper::CancelResult> DecodeWriteHelper::cancel(const std::string& prefill_addr,
                                                                           const std::string& unique_key,
                                                                           int64_t            deadline_ms,
                                                                           int64_t            timeout_ms) {
    if (unique_key.empty() || timeout_ms <= 0 || timeout_ms > std::numeric_limits<int>::max()) {
        return nullptr;
    }
    auto peer = server(prefill_addr);
    if (!peer) {
        return nullptr;
    }
    P2PConnectorCancelWriteRequestPB request;
    request.set_unique_key(unique_key);
    request.set_deadline_ms(deadline_ms);
    return peer->broadcast<P2PConnectorCancelWriteRequestPB, P2PConnectorStartWriteResponsePB>(
        {request}, static_cast<int>(timeout_ms), [](auto& stub, auto& context, const auto& request, auto* queue) {
            return stub->AsyncCancelWrite(context.get(), request, queue);
        });
}

}  // namespace rtp_llm
