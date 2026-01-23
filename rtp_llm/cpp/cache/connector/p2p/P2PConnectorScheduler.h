#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/TPBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorServerCaller.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class P2PConnectorScheduler {
public:
    P2PConnectorScheduler(const RuntimeConfig&                runtime_config,
                          const CacheStoreConfig&             cache_store_config,
                          const kmonitor::MetricsReporterPtr& metrics_reporter);
    ~P2PConnectorScheduler();

public:
    bool init();

    // Stop the background checker thread (for testing)
    void stopChecker();

public:
    // Decode side: async read from prefill
    std::shared_ptr<P2PConnectorAsyncReadContext> asyncRead(const KVCacheResourcePtr&  resource,
                                                            int64_t                    request_id,
                                                            const std::string&         unique_key,
                                                            int64_t                    deadline_ms,
                                                            const IGenerateStreamPtr&  generate_stream,
                                                            const std::pair<int, int>& block_range);

    // Prefill side: handle read request from decode (sync)
    // is_cancelled: optional callback to check if the request is cancelled by client
    bool handleRead(const KVCacheResourcePtr&                            resource,
                    const std::string&                                   unique_key,
                    int64_t                                              request_id,
                    const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                    int64_t                                              deadline_ms,
                    std::function<bool()>                                is_cancelled = nullptr);

private:
    const RuntimeConfig&                                 runtime_config_;
    const CacheStoreConfig&                              cache_store_config_;
    kmonitor::MetricsReporterPtr                         metrics_reporter_;
    std::shared_ptr<TPBroadcastClient>                   tp_broadcast_client_;
    std::shared_ptr<P2PConnectorServerCaller>            server_caller_;
    std::shared_ptr<P2PConnectorAsyncReadContextChecker> checker_;
};

}  // namespace rtp_llm