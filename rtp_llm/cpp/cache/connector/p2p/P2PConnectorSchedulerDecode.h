#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
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

namespace autil {
class LockFreeThreadPool;
}

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

    // asyncRead from Meta (extracts routing from Meta::p2pRouting())
    AsyncReadResult asyncRead(const KVCacheResourcePtr&       resource,
                              const std::shared_ptr<Meta>&    meta,
                              const std::pair<int, int>&      block_range,
                              bool                           no_transfer = false);
    void cancel(const std::shared_ptr<P2PConnectorAsyncReadContext>& context);

private:
    struct AsyncReadCallResults {
        std::shared_ptr<PrefillLoadCaller::Result>  server_call_result;
        std::shared_ptr<P2PBroadcastClient::Result> tp_sync_result;
    };

    std::optional<AsyncReadCallResults>
    startAsyncReadCalls(int64_t                                                  request_id,
                        const std::string&                                       prefill_ip,
                        uint32_t                                                 prefill_port,
                        const std::string&                                       unique_key,
                        int64_t                                                  request_deadline_ms,
                        int64_t                                                  transfer_deadline_ms,
                        const P2PBroadcastClient::RankLayerCacheBuffers&         rank_layer_cache_buffers,
                        const std::shared_ptr<DecodeSchedulerMetricsCollector>& collector,
                        ErrorInfo&                                               out_error,
                        int                                                      prefill_tp_size = 0,
                        bool                                                     no_transfer = false);

private:
    const P2PConnectorSchedulerConfig                    config_;
    kmonitor::MetricsReporterPtr                         metrics_reporter_;
    std::shared_ptr<P2PBroadcastClient>                  tp_broadcast_client_;
    std::shared_ptr<PrefillLoadCaller>                   server_caller_;
    std::shared_ptr<P2PConnectorAsyncReadContextChecker> checker_;
    std::shared_ptr<autil::LockFreeThreadPool>           async_read_pool_;
};

}  // namespace rtp_llm
