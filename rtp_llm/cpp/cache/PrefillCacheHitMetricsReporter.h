#pragma once

#include <cstdint>
#include <memory>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/RecentCacheKeyWindow.h"
#include "rtp_llm/cpp/cache/Types.h"

namespace rtp_llm {

class PrefillCacheHitMetricsReporter {
public:
    explicit PrefillCacheHitMetricsReporter(kmonitor::MetricsReporterPtr metrics_reporter);
    ~PrefillCacheHitMetricsReporter();

    static bool enabled();

    void record(const BatchKVCacheResource&          resource,
                const std::shared_ptr<CPSlotMapper>& cp_mapper,
                int64_t                              request_id,
                int64_t                              token_num,
                int                                  seq_size_per_block);

private:
    struct TheoryHitStats;

    kmonitor::MetricsReporterPtr    metrics_reporter_;
    RecentCacheKeyWindow            recent_window_;
    std::unique_ptr<TheoryHitStats> theory_stats_;
};

CacheKeysType buildPrefillTheoryWindowKeys(const BatchKVCacheResource&          resource,
                                           const std::shared_ptr<CPSlotMapper>& cp_mapper);

}  // namespace rtp_llm
