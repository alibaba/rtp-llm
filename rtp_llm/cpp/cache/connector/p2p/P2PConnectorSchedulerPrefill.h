#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class P2PConnectorSchedulerPrefill {
public:
    P2PConnectorSchedulerPrefill(P2PConnectorSchedulerConfig                config,
                                 const kmonitor::MetricsReporterPtr&        metrics_reporter,
                                 const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client);
    ~P2PConnectorSchedulerPrefill() = default;

public:
    ErrorInfo sendKVCache(const KVCacheResourcePtr&                            resource,
                          const std::string&                                   unique_key,
                          int64_t                                              request_id,
                          const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                          int64_t                                              deadline_ms,
                          std::function<bool()>                                is_cancelled = nullptr);

private:
    /// 轮询 result->done()；is_cancelled 或当前时间超过 deadline_ms 时发 CANCEL_HANDLE_READ。
    /// deadline_exceeded_out 若为非空，在因超时而发起 cancel 时置 true（与客户端 cancel 区分）。
    std::shared_ptr<P2PBroadcastClient::Result>
    waitForBroadcastCompletion(const std::shared_ptr<P2PBroadcastClient::Result>& result,
                               const std::string&                                 unique_key,
                               int64_t                                            request_id,
                               int64_t                                            deadline_ms,
                               std::function<bool()>                              is_cancelled,
                               bool*                                              deadline_exceeded_out = nullptr);

private:
    const P2PConnectorSchedulerConfig   config_;
    kmonitor::MetricsReporterPtr        metrics_reporter_;
    std::shared_ptr<P2PBroadcastClient> tp_broadcast_client_;
};

}  // namespace rtp_llm
