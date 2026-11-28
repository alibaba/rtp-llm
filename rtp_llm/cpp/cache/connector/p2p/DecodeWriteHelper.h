#pragma once

#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include <memory>
#include <string>

namespace rtp_llm {

class DecodeWriteHelper {
public:
    using Result = BroadcastResult<P2PConnectorStartWriteBackRequestPB, P2PConnectorStartWriteBackResponsePB>;

    // Null means the RPC did not dispatch. An exception from dispatch leaves its outcome unknown.
    std::shared_ptr<Result> write(const std::string&                  prefill_ip,
                                  uint32_t                            prefill_port,
                                  P2PConnectorStartWriteBackRequestPB request,
                                  int64_t                             deadline_ms);

    using CancelResult = BroadcastResult<P2PConnectorCancelWriteBackRequestPB, P2PConnectorStartWriteBackResponsePB>;

    std::shared_ptr<CancelResult>
    cancel(const std::string& prefill_addr, const std::string& unique_key, int64_t deadline_ms, int64_t timeout_ms);

private:
    std::shared_ptr<RPCPool> rpc_pool_ = std::make_shared<RPCPool>();
};

}  // namespace rtp_llm
