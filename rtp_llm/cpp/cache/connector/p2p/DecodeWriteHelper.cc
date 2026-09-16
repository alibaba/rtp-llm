#include "rtp_llm/cpp/cache/connector/p2p/DecodeWriteHelper.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <limits>

namespace rtp_llm {

std::shared_ptr<DecodeWriteHelper::Result> DecodeWriteHelper::write(const std::string&                     prefill_ip,
                                                                    uint32_t                               prefill_port,
                                                                    const P2PConnectorStartWriteRequestPB& request) {
    const auto address    = formatGrpcHostPort(prefill_ip, prefill_port);
    const auto timeout_ms = request.deadline_ms() - currentTimeMs();
    if (address.empty() || request.unique_key().empty() || timeout_ms <= 0
        || timeout_ms > std::numeric_limits<int>::max()) {
        return nullptr;
    }
    std::shared_ptr<BroadcastManager> server;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = servers_.find(address);
        if (it == servers_.end()) {
            server = std::make_shared<BroadcastManager>(std::vector<std::string>{address});
            if (!server->init()) {
                return nullptr;
            }
            servers_.emplace(address, server);
        } else {
            server = it->second;
        }
    }
    return server->broadcast<P2PConnectorStartWriteRequestPB, P2PConnectorStartWriteResponsePB>(
        {request},
        static_cast<int>(timeout_ms),
        [](std::shared_ptr<RpcService::Stub>&     stub,
           std::shared_ptr<grpc::ClientContext>&  context,
           const P2PConnectorStartWriteRequestPB& request,
           grpc::CompletionQueue* queue) { return stub->AsyncStartWrite(context.get(), request, queue); });
}

}  // namespace rtp_llm
