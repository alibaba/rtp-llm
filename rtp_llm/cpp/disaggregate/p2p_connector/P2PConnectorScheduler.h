#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/TPBroadcastClient.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorServerCaller.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorMetrics.h"
#include <grpc++/grpc++.h>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class P2PConnectorScheduler {
public:
    P2PConnectorScheduler(const RuntimeConfig& runtime_config, const kmonitor::MetricsReporterPtr& metrics_reporter);
    ~P2PConnectorScheduler();

public:
    bool init();

public:
    // Decode side: async read from prefill
    std::shared_ptr<P2PConnectorAsyncReadContext> asyncRead(const KVCacheResourcePtr&  resource,
                                                            int64_t                    request_id,
                                                            const std::string&         unique_key,
                                                            int64_t                    deadline_ms,
                                                            const IGenerateStreamPtr&  generate_stream,
                                                            const std::pair<int, int>& block_range);

    // Prefill side: handle read request from decode (sync)
    bool handleRead(const KVCacheResourcePtr&                            resource,
                    const std::string&                                   unique_key,
                    int64_t                                              request_id,
                    const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                    int64_t                                              deadline_ms);

private:
    const RuntimeConfig&                                 runtime_config_;
    kmonitor::MetricsReporterPtr                         metrics_reporter_;
    std::shared_ptr<TPBroadcastClient>                   tp_broadcast_client_;
    std::shared_ptr<P2PConnectorServerCaller>            server_caller_;
    std::shared_ptr<P2PConnectorAsyncReadContextChecker> checker_;
};

}  // namespace rtp_llm