#pragma once

#include "rtp_llm/cpp/cache/SingleTypeCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include <map>
#include <mutex>
#include <vector>

namespace rtp_llm {

struct P2PConnectorSchedulerConfig;

// Direct connector users get a finite drain budget as well.
int64_t writebackShutdownDeadline(int64_t steady_deadline_ms);

// The Decode limit applies independently to each physical pool, including pools shared by cache groups.
class WritebackSourceBudget : public std::enable_shared_from_this<WritebackSourceBudget> {
public:
    class Reservation {
    public:
        Reservation() = default;
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;
        ~Reservation();

    private:
        friend class WritebackSourceBudget;
        std::shared_ptr<WritebackSourceBudget> budget_;
        std::map<DeviceBlockPool*, size_t>     blocks_;
    };

    std::shared_ptr<Reservation> tryReserve(const std::vector<SingleTypeCacheManagerPtr>& groups,
                                            const KVCacheResource&                       selected);

private:
    std::mutex                         mutex_;
    std::map<DeviceBlockPool*, size_t> held_blocks_;
};

class WritebackRetainedKv {
public:
    bool init(const P2PConnectorSchedulerConfig& config,
              bool                               prefill,
              kmonitor::MetricsReporterPtr       metrics_reporter = nullptr);
    void check(const std::vector<KVCacheResourcePtr>& resources, WritebackStatusMetricsCollector status = {});

private:
    std::map<std::string, size_t>   group_block_bytes_;
    size_t                          total_bytes_{0};
    bool                            prefill_{false};
    std::mutex                      mutex_;
    int64_t                         last_log_ms_{0};
    kmonitor::MetricsReporterPtr    metrics_reporter_;
    int64_t                         last_report_ms_{-1};
    WritebackStatusMetricsCollector last_status_;
};

}  // namespace rtp_llm
