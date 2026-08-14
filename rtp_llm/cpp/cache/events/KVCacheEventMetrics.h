#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "autil/Log.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"

namespace rtp_llm {

class RtpLLMKVCacheEventMetricsCollector final {
public:
    // Numeric code from PublisherState. Zero must remain DISABLED and
    // published values must never be renumbered; KVCacheEventPublisher.h
    // enforces that external metrics contract with static_asserts.
    int64_t publisher_state             = 0;
    int64_t queue_size                  = 0;
    int64_t queue_high_watermark        = 0;
    int64_t accepted_count              = 0;
    int64_t dropped_count               = 0;
    int64_t request_failure_count       = 0;
    int64_t overflow_recovery_count     = 0;
    int64_t snapshot_attempt_count      = 0;
    int64_t snapshot_commit_count       = 0;
    int64_t cache_insert_rejected_count = 0;
    // An intentionally disabled publisher still exports its categorical
    // state and cache-capacity signal, but suppresses inactive publisher
    // series. Defaulting to false makes a forgotten assignment over-report
    // harmless zeros instead of silently losing health data.
    bool suppress_publisher_details = false;
};

class RtpLLMKVCacheEventMetrics final: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMKVCacheEventMetricsCollector* collector);

private:
    kmonitor::MutableMetric* publisher_state_metric             = nullptr;
    kmonitor::MutableMetric* queue_size_metric                  = nullptr;
    kmonitor::MutableMetric* queue_high_watermark_metric        = nullptr;
    kmonitor::MutableMetric* accepted_count_metric              = nullptr;
    kmonitor::MutableMetric* dropped_count_metric               = nullptr;
    kmonitor::MutableMetric* request_failure_count_metric       = nullptr;
    kmonitor::MutableMetric* overflow_recovery_count_metric     = nullptr;
    kmonitor::MutableMetric* snapshot_attempt_count_metric      = nullptr;
    kmonitor::MutableMetric* snapshot_commit_count_metric       = nullptr;
    kmonitor::MutableMetric* cache_insert_rejected_count_metric = nullptr;

    AUTIL_LOG_DECLARE();
};

namespace detail {

template<typename Unsigned>
int64_t saturatingMetricValue(Unsigned value) noexcept {
    static_assert(std::is_unsigned_v<Unsigned>);
    if constexpr (std::numeric_limits<Unsigned>::digits <= std::numeric_limits<int64_t>::digits) {
        return static_cast<int64_t>(value);
    } else {
        constexpr auto kMetricMax = std::numeric_limits<int64_t>::max();
        return value > static_cast<Unsigned>(kMetricMax) ? kMetricMax : static_cast<int64_t>(value);
    }
}

}  // namespace detail

// Keep the PublisherStatus -> kmonitor boundary named and testable. Saturation
// prevents a practically unreachable unsigned counter wrap from appearing as
// a negative health metric.
inline RtpLLMKVCacheEventMetricsCollector
makeKVCacheEventMetricsCollector(const PublisherStatus& status, uint64_t cache_insert_rejected_count = 0) noexcept {
    RtpLLMKVCacheEventMetricsCollector collector;
    collector.publisher_state             = static_cast<int64_t>(status.state);
    collector.queue_size                  = detail::saturatingMetricValue(status.queue_size);
    collector.queue_high_watermark        = detail::saturatingMetricValue(status.queue_high_watermark);
    collector.accepted_count              = detail::saturatingMetricValue(status.accepted_count);
    collector.dropped_count               = detail::saturatingMetricValue(status.dropped_count);
    collector.request_failure_count       = detail::saturatingMetricValue(status.request_failure_count);
    collector.overflow_recovery_count     = detail::saturatingMetricValue(status.overflow_recovery_count);
    collector.snapshot_attempt_count      = detail::saturatingMetricValue(status.snapshot_attempt_count);
    collector.snapshot_commit_count       = detail::saturatingMetricValue(status.snapshot_commit_count);
    collector.cache_insert_rejected_count = detail::saturatingMetricValue(cache_insert_rejected_count);
    collector.suppress_publisher_details  = status.state == PublisherState::DISABLED;
    return collector;
}

}  // namespace rtp_llm
