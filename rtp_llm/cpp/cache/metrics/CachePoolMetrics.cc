#include "rtp_llm/cpp/cache/metrics/CachePoolMetrics.h"

#include <unordered_set>

#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::vector<CachePoolMetrics> mergeCachePoolMetrics(const std::vector<CachePoolMetrics>& allocator_metrics_list,
                                                    const std::vector<CachePoolMetrics>& tree_metrics_list) {
    std::vector<CachePoolMetrics> report_metrics_list;
    report_metrics_list.reserve(allocator_metrics_list.size() + tree_metrics_list.size());

    std::unordered_set<std::string> reported_device_pools;
    for (const CachePoolMetrics& metrics_data : allocator_metrics_list) {
        const bool inserted = reported_device_pools.insert(metrics_data.pool_name).second;
        if (!inserted) {
            RTP_LLM_LOG_WARNING("duplicate allocator pool name in metrics: %s", metrics_data.pool_name.c_str());
            continue;
        }
        CachePoolMetrics report_metrics = metrics_data;
        report_metrics.tier             = Tier::DEVICE;
        report_metrics_list.push_back(std::move(report_metrics));
    }

    for (const CachePoolMetrics& metrics_data : tree_metrics_list) {
        if (metrics_data.tier == Tier::DEVICE && reported_device_pools.count(metrics_data.pool_name) > 0) {
            continue;
        }
        CachePoolMetrics report_metrics = metrics_data;
        report_metrics.used_ratio =
            metrics_data.total_blocks == 0 ?
                0.0f :
                static_cast<float>(100.0 * (metrics_data.total_blocks - metrics_data.available_blocks)
                                   / static_cast<double>(metrics_data.total_blocks));
        report_metrics_list.push_back(std::move(report_metrics));
    }

    return report_metrics_list;
}

}  // namespace rtp_llm
