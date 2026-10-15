#pragma once

#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace rtp_llm {

class DecodeWriteHelper {
public:
    using Result = BroadcastResult<P2PConnectorStartWriteRequestPB, P2PConnectorStartWriteResponsePB>;

    std::shared_ptr<Result>
    write(const std::string& prefill_ip,
          uint32_t prefill_port,
          P2PConnectorStartWriteRequestPB request,
          int64_t deadline_ms);

    using ReadyResult  = BroadcastResult<P2PConnectorWriteReadyRequestPB, P2PConnectorWriteReadyResponsePB>;
    using CancelResult = BroadcastResult<P2PConnectorCancelWriteRequestPB, P2PConnectorStartWriteResponsePB>;

    std::shared_ptr<ReadyResult>
    ready(const std::string& decode_addr, P2PConnectorWriteReadyRequestPB request, int64_t deadline_ms);
    std::shared_ptr<CancelResult>
    cancel(const std::string& prefill_addr, const std::string& unique_key, int64_t deadline_ms, int64_t timeout_ms);

private:
    std::shared_ptr<BroadcastManager>                                  server(const std::string& address);
    std::mutex                                                         mutex_;
    std::unordered_map<std::string, std::shared_ptr<BroadcastManager>> servers_;
};

}  // namespace rtp_llm
