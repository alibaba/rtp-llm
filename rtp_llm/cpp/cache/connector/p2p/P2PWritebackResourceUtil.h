#pragma once

#include "rtp_llm/cpp/cache/SingleTypeCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include <map>
#include <mutex>
#include <vector>

namespace rtp_llm {

bool writebackKeepsKvReserve(const std::vector<SingleTypeCacheManagerPtr>& groups, double reserve_ratio);

bool writebackKeepsKvReserveAfterRelease(const std::vector<SingleTypeCacheManagerPtr>& groups,
                                         const KVCacheResource&              source,
                                         const KVCacheResource&              retained,
                                         double                              reserve_ratio);

class WritebackRetainedKv {
public:
    bool init(const std::shared_ptr<const CacheTopology>& topology,
              int                                         threshold_percent,
              bool                                        prefill,
              kmonitor::MetricsReporterPtr                metrics_reporter = nullptr);
    void check(const std::vector<KVCacheResourcePtr>& resources, WritebackStatusMetricsCollector status = {});

private:
    std::map<std::string, size_t> group_block_bytes_;
    size_t                       total_bytes_{0};
    int                          threshold_percent_{30};
    bool                         prefill_{false};
    std::mutex                   mutex_;
    int64_t                         last_log_ms_{0};
    kmonitor::MetricsReporterPtr    metrics_reporter_;
    int64_t                         last_report_ms_{-1};
    WritebackStatusMetricsCollector last_status_;
};

}  // namespace rtp_llm
