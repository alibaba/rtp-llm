#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/PrefillLoadCaller.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rtp_llm {

class P2PConnectorSchedulerDecode {
public:
    struct AsyncReadResult {
        std::shared_ptr<P2PConnectorAsyncReadContext> context;
        ErrorInfo                                     error_info;

        bool ok() const {
            return error_info.ok();
        }
    };

    P2PConnectorSchedulerDecode(P2PConnectorSchedulerConfig                config,
                                const kmonitor::MetricsReporterPtr&        metrics_reporter,
                                const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client);
    ~P2PConnectorSchedulerDecode();

public:
    bool init(const std::string& process_id);
    void stopChecker();

    AsyncReadResult asyncRead(const KVCacheResourcePtr&  resource,
                              const IGenerateStreamPtr&  generate_stream,
                              const std::pair<int, int>& block_range);

private:
    struct AsyncReadCallResults {
        std::shared_ptr<PrefillLoadCaller::Result>  server_call_result;
        std::shared_ptr<P2PBroadcastClient::Result> tp_sync_result;
    };

    std::optional<AsyncReadCallResults>
    startAsyncReadCalls(int64_t                                                 request_id,
                        const std::string&                                      prefill_ip,
                        uint32_t                                                prefill_port,
                        const std::string&                                      unique_key,
                        int64_t                                                 deadline_ms,
                        const std::vector<std::shared_ptr<LayerCacheBuffer>>&   layer_cache_buffers,
                        const IGenerateStreamPtr&                               generate_stream,
                        const std::shared_ptr<DecodeSchedulerMetricsCollector>& collector,
                        ErrorInfo&                                              out_error);

private:
    const P2PConnectorSchedulerConfig                    config_;
    kmonitor::MetricsReporterPtr                         metrics_reporter_;
    std::shared_ptr<P2PBroadcastClient>                  tp_broadcast_client_;
    std::shared_ptr<PrefillLoadCaller>                   server_caller_;
    std::shared_ptr<P2PConnectorAsyncReadContextChecker> checker_;
};

}  // namespace rtp_llm
