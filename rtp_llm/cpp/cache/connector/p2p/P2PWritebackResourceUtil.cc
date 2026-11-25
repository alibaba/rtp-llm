#include "rtp_llm/cpp/cache/connector/p2p/P2PWritebackResourceUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PNotification.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <map>
#include <set>

namespace rtp_llm {

int64_t writebackShutdownDeadline(int64_t steady_deadline_ms) {
    constexpr int64_t kDefaultWritebackShutdownTimeoutMs = 600 * 1000;
    return steady_deadline_ms > 0 ? steady_deadline_ms : P2PNotification::nowMs() + kDefaultWritebackShutdownTimeoutMs;
}

bool WritebackRetainedKv::init(const P2PConnectorSchedulerConfig& config,
                               bool                               prefill,
                               kmonitor::MetricsReporterPtr       metrics_reporter) {
    const auto& topology = config.topology;
    if (!topology) {
        return false;
    }
    group_block_bytes_.clear();
    total_bytes_ = 0;
    for (const auto& group : topology->groups()) {
        if (group.policy.memory_placement != CacheMemoryPlacement::DEVICE) {
            continue;
        }
        const auto   physical = config.physical_block_bytes_by_group.find(group.tag);
        const size_t bytes    = physical == config.physical_block_bytes_by_group.end() ?
                                    topology->blockSizeBytesForGroup(group.tag) :
                                    physical->second;
        group_block_bytes_.emplace(group.tag, bytes);
        total_bytes_ += bytes * group.block_num;
    }
    prefill_          = prefill;
    metrics_reporter_ = std::move(metrics_reporter);
    return total_bytes_ > 0;
}

void WritebackRetainedKv::check(const std::vector<KVCacheResourcePtr>& resources,
                                WritebackStatusMetricsCollector        status) {
    std::map<std::string, std::set<BlockIdxType>> held_blocks;
    for (const auto& resource : resources) {
        for (const auto& [tag, bytes] : group_block_bytes_) {
            for (const auto block : resource->blocks(tag)) {
                if (!isNullBlockIdx(block)) {
                    held_blocks[tag].insert(block);
                }
            }
        }
    }
    size_t held_bytes = 0;
    for (const auto& [tag, blocks] : held_blocks) {
        held_bytes += blocks.size() * group_block_bytes_.at(tag);
    }
    status.prefill                = prefill_;
    status.retained_context_count = resources.size();
    status.retained_kv_bytes      = held_bytes;
    status.retained_kv_percent    = total_bytes_ > 0 ? 100.0 * held_bytes / total_bytes_ : 0.0;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now_ms = P2PNotification::nowMs();
    if (last_report_ms_ < 0 || now_ms - last_report_ms_ >= 1000
        || status.inflight_context_count != last_status_.inflight_context_count
        || status.retained_kv_bytes != last_status_.retained_kv_bytes) {
        reportWritebackMetrics(metrics_reporter_, status);
        last_report_ms_ = now_ms;
        last_status_    = status;
    }
    if (held_bytes > 0 && (last_log_ms_ == 0 || now_ms - last_log_ms_ >= 30000)) {
        RTP_LLM_LOG_WARNING(
            "writeback retained KV: role=%s contexts=%zu held_bytes=%zu total_bytes=%zu retained_percent=%.2f",
            prefill_ ? "PREFILL" : "DECODE",
            resources.size(),
            held_bytes,
            total_bytes_,
            status.retained_kv_percent);
        last_log_ms_ = now_ms;
    } else if (held_bytes == 0 && last_log_ms_ != 0) {
        RTP_LLM_LOG_INFO("writeback retained KV released after physical stop, role=%s",
                         prefill_ ? "PREFILL" : "DECODE");
        last_log_ms_ = 0;
    }
}

}  // namespace rtp_llm
