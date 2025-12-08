#pragma once

#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/TPBroadcastClient.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include <grpc++/grpc++.h>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class P2PConnectorPrefillScheduler {
public:
    P2PConnectorPrefillScheduler(const GptInitParameter& gpt_init_parameter);
    ~P2PConnectorPrefillScheduler();

public:
    bool init();

    grpc::Status write(const std::shared_ptr<KVCacheResourceV1>&            resource,
                       const std::string&                                   unique_key,
                       int64_t                                              request_id,
                       const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                       int64_t                                              deadline_ms);

private:
    const GptInitParameter&            gpt_init_parameter_;
    std::shared_ptr<TPBroadcastClient> tp_broadcast_client_;
};

}  // namespace rtp_llm
