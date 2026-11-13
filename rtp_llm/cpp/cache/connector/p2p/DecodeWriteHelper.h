#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/P2PWriteRpcUtil.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include <memory>
#include <string>

namespace rtp_llm {

class DecodeWriteHelper {
public:
    using Result = BroadcastResult<P2PConnectorStartWriteBackRequestPB, P2PConnectorStartWriteBackResponsePB>;

    std::shared_ptr<Result>
    write(const std::string& prefill_ip,
          uint32_t prefill_port,
          P2PConnectorStartWriteBackRequestPB request,
          int64_t deadline_ms);

    using CancelResult = BroadcastResult<P2PConnectorCancelWriteBackRequestPB, P2PConnectorStartWriteBackResponsePB>;

    std::shared_ptr<CancelResult>
    cancel(const std::string& prefill_addr, const std::string& unique_key, int64_t deadline_ms, int64_t timeout_ms);

private:
    std::shared_ptr<BroadcastManager>                            server(const std::string& address);
    p2p_internal::PeerClientCache<std::string, BroadcastManager> servers_;
};

}  // namespace rtp_llm
