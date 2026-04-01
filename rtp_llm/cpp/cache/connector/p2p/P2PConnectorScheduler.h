#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerPrefill.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

/// @brief Facade: 统一外部接口，内部委托给 Prefill / Decode 子对象
class P2PConnectorScheduler {
public:
    using AsyncReadResult = P2PConnectorSchedulerDecode::AsyncReadResult;

    P2PConnectorScheduler(P2PConnectorSchedulerConfig config, const kmonitor::MetricsReporterPtr& metrics_reporter);
    ~P2PConnectorScheduler();

public:
    bool init(const std::string& process_id = "");

    void stopChecker();

public:
    AsyncReadResult asyncRead(const KVCacheResourcePtr&  resource,
                              const IGenerateStreamPtr&  generate_stream,
                              const std::pair<int, int>& block_range);

    ErrorInfo sendKVCache(const KVCacheResourcePtr&                            resource,
                          const std::string&                                   unique_key,
                          int64_t                                              request_id,
                          const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                          int64_t                                              deadline_ms,
                          std::function<bool()>                                is_cancelled = nullptr);

private:
    P2PConnectorSchedulerConfig         config_;
    kmonitor::MetricsReporterPtr        metrics_reporter_;
    std::shared_ptr<P2PBroadcastClient> tp_broadcast_client_;

    std::unique_ptr<P2PConnectorSchedulerPrefill> prefill_;
    std::unique_ptr<P2PConnectorSchedulerDecode>  decode_;
};

}  // namespace rtp_llm
