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
    write(const std::string& prefill_ip, uint32_t prefill_port, const P2PConnectorStartWriteRequestPB& request);

private:
    std::mutex                                                         mutex_;
    std::unordered_map<std::string, std::shared_ptr<BroadcastManager>> servers_;
};

}  // namespace rtp_llm
