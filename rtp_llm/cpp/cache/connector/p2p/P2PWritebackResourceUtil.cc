#include "rtp_llm/cpp/cache/connector/p2p/P2PWritebackResourceUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PNotification.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <cstdlib>
#include <map>
#include <set>

namespace rtp_llm {
namespace {

// Groups can share one physical pool; each pool is checked once against its own capacity.
using ReleasesByPool = std::map<DeviceBlockPool*, BlockIdList>;

ReleasesByPool distinctPools(const std::vector<SingleTypeCacheManagerPtr>& groups) {
    ReleasesByPool pools;
    for (const auto& group : groups) {
        pools[group->blockPool().get()];
    }
    return pools;
}

bool poolsKeepReserve(const ReleasesByPool& pools, double reserve_ratio) {
    for (const auto& [pool, releasing] : pools) {
        if (pool->availableBlocksAfterRelease(releasing)
            < static_cast<size_t>(reserve_ratio * pool->totalBlocksNum())) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool writebackKeepsKvReserve(const std::vector<SingleTypeCacheManagerPtr>& groups, double reserve_ratio) {
    return poolsKeepReserve(distinctPools(groups), reserve_ratio);
}

bool writebackKeepsKvReserveAfterRelease(const std::vector<SingleTypeCacheManagerPtr>& groups,
                                         const KVCacheResource&              source,
                                         const KVCacheResource&              retained,
                                         double                              reserve_ratio) {
    std::map<DeviceBlockPool*, std::set<BlockIdxType>> retained_by_pool;
    for (const auto& group : groups) {
        auto& kept = retained_by_pool[group->blockPool().get()];
        for (const auto block : retained.blocks(group->tag())) {
            if (!isNullBlockIdx(block)) {
                kept.insert(block);
            }
        }
    }
    auto pools = distinctPools(groups);
    for (const auto& group : groups) {
        const auto  pool      = group->blockPool().get();
        const auto& kept      = retained_by_pool[pool];
        auto&       releasing = pools[pool];
        for (const auto block : source.blocks(group->tag())) {
            if (!isNullBlockIdx(block) && !kept.count(block)) {
                releasing.push_back(block);
            }
        }
    }
    return poolsKeepReserve(pools, reserve_ratio);
}

bool WritebackRetainedKv::init(const std::shared_ptr<const CacheTopology>& topology,
                               int                                         threshold_percent,
                               bool                                        prefill,
                               kmonitor::MetricsReporterPtr                metrics_reporter) {
    if (!topology || threshold_percent < 1 || threshold_percent > 100) {
        return false;
    }
    group_block_bytes_.clear();
    total_bytes_ = 0;
    for (const auto& group : topology->groups()) {
        if (group.policy.memory_placement != CacheMemoryPlacement::DEVICE) {
            continue;
        }
        const size_t bytes = topology->blockSizeBytesForGroup(group.tag);
        group_block_bytes_.emplace(group.tag, bytes);
        total_bytes_ += bytes * group.block_num;
    }
    threshold_percent_ = threshold_percent;
    prefill_           = prefill;
    metrics_reporter_  = std::move(metrics_reporter);
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
    status.prefill                     = prefill_;
    status.retained_context_count      = resources.size();
    status.retained_kv_bytes           = held_bytes;
    status.retained_kv_percent         = total_bytes_ > 0 ? 100.0 * held_bytes / total_bytes_ : 0.0;
    const auto                  now_ms = P2PNotification::nowMs();
    std::lock_guard<std::mutex> lock(mutex_);
    if (last_report_ms_ < 0 || now_ms - last_report_ms_ >= 1000
        || status.inflight_context_count != last_status_.inflight_context_count
        || status.retained_kv_bytes != last_status_.retained_kv_bytes) {
        reportWritebackMetrics(metrics_reporter_, status);
        last_report_ms_ = now_ms;
        last_status_    = status;
    }
    if (held_bytes > 0 && (last_log_ms_ == 0 || now_ms - last_log_ms_ >= 30000)) {
        RTP_LLM_LOG_WARNING("writeback retained KV: role=%s contexts=%zu held_bytes=%zu total_bytes=%zu threshold=%d",
                            prefill_ ? "PREFILL" : "DECODE", resources.size(), held_bytes, total_bytes_, threshold_percent_);
        last_log_ms_ = now_ms;
    } else if (held_bytes == 0 && last_log_ms_ != 0) {
        RTP_LLM_LOG_INFO("writeback retained KV released after physical stop, role=%s", prefill_ ? "PREFILL" : "DECODE");
        last_log_ms_ = 0;
    }
    const size_t threshold_bytes = total_bytes_ / 100 * threshold_percent_
                                   + total_bytes_ % 100 * threshold_percent_ / 100;
    if (held_bytes > threshold_bytes) {
        RTP_LLM_LOG_ERROR("writeback retained KV exceeded exit threshold; aborting with KV held: "
                          "role=%s held_bytes=%zu total_bytes=%zu threshold=%d",
                          prefill_ ? "PREFILL" : "DECODE", held_bytes, total_bytes_, threshold_percent_);
        std::abort();
    }
}

}  // namespace rtp_llm
