#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"

namespace rtp_llm {

int64_t subZeroOrAbove(int64_t lhs, int64_t rhs) {
    if (lhs <= 0 || rhs <= 0 || lhs <= rhs) {
        return 0;
    }
    return lhs - rhs;
}

CacheStoreStoreMetricsCollector::CacheStoreStoreMetricsCollector(const kmonitor::MetricsReporterPtr& reporter,
                                                                 int64_t                             block_count,
                                                                 int64_t                             total_block_size):
    reporter_(reporter), start_time_us_(currentTimeUs()) {
    collector_.block_count      = block_count;
    collector_.total_block_size = total_block_size;
}

CacheStoreStoreMetricsCollector::~CacheStoreStoreMetricsCollector() {
    collector_.wait_task_run_latency_us   = subZeroOrAbove(task_run_time_us_, start_time_us_);
    collector_.wait_event_sync_latency_us = subZeroOrAbove(event_sync_done_time_us_, task_run_time_us_);
    collector_.latency_us                 = subZeroOrAbove(currentTimeUs(), start_time_us_);

    if (reporter_ != nullptr) {
        reporter_->report<RtpLLMCacheStoreMetrics, RtpLLMCacheStoreStoreMetricsCollector>(nullptr, &collector_);
    }
}

CacheStoreClientLoadMetricsCollector::CacheStoreClientLoadMetricsCollector(const kmonitor::MetricsReporterPtr& reporter,
                                                                           int64_t block_count,
                                                                           int64_t total_block_size):
    reporter_(reporter), start_time_us_(currentTimeUs()) {

    collector_.block_count      = block_count;
    collector_.total_block_size = total_block_size;
}

CacheStoreClientLoadMetricsCollector::~CacheStoreClientLoadMetricsCollector() {
    collector_.latency_us               = subZeroOrAbove(end_time_us_, start_time_us_);
    collector_.wait_task_run_latency_us = subZeroOrAbove(task_run_time_us_, start_time_us_);
    collector_.prepare_call_latency_us  = subZeroOrAbove(request_begin_call_time_us_, task_run_time_us_);
    collector_.server_call_latency_us   = subZeroOrAbove(request_call_done_time_us_, request_begin_call_time_us_);

    if (reporter_ != nullptr) {
        reporter_->report<RtpLLMCacheStoreMetrics, RtpLLMCacheStoreLoadClientMetricsCollector>(nullptr, &collector_);
    }
}

CacheStoreServerLoadMetricsCollector::CacheStoreServerLoadMetricsCollector(const kmonitor::MetricsReporterPtr& reporter,
                                                                           int64_t block_count,
                                                                           int64_t total_block_size,
                                                                           int64_t request_send_cost_us):
    reporter_(reporter), start_time_us_(currentTimeUs()) {
    collector_.block_count          = block_count;
    collector_.total_block_size     = total_block_size;
    collector_.request_send_cost_us = request_send_cost_us;
}

CacheStoreServerLoadMetricsCollector::~CacheStoreServerLoadMetricsCollector() {
    collector_.latency_us                   = subZeroOrAbove(end_time_us_, start_time_us_);
    collector_.first_block_ready_latency_us = subZeroOrAbove(first_block_ready_time_us_, start_time_us_);
    collector_.all_block_ready_latency_us   = subZeroOrAbove(all_block_ready_time_us_, start_time_us_);
    collector_.transfer_gap_latency_us      = subZeroOrAbove(end_time_us_, all_block_ready_time_us_);

    if (reporter_ != nullptr) {
        reporter_->report<RtpLLMCacheStoreMetrics, RtpLLMCacheStoreLoadServerMetricsCollector>(nullptr, &collector_);
    }
}

void CacheStoreServerLoadMetricsCollector::setWriteInfo(int64_t write_block_count,
                                                        int64_t write_total_block_size,
                                                        int64_t write_latency_us) {
    std::unique_lock<std::mutex> lock(write_mutex_);
    collector_.write_block_count.push_back(write_block_count);
    collector_.write_total_block_size.push_back(write_total_block_size);
    collector_.write_latency_us.push_back(write_latency_us);
}

CacheStoreRemoteStoreMetricsCollector::CacheStoreRemoteStoreMetricsCollector(
    const kmonitor::MetricsReporterPtr& reporter, int64_t block_count):
    reporter_(reporter), start_time_us_(currentTimeUs()) {
    collector_.block_count = block_count;
}

CacheStoreRemoteStoreMetricsCollector::~CacheStoreRemoteStoreMetricsCollector() {
    collector_.latency_us                   = subZeroOrAbove(end_time_us_, start_time_us_);
    collector_.first_block_ready_latency_us = subZeroOrAbove(first_block_ready_time_us_, start_time_us_);
    collector_.all_block_ready_latency_us   = subZeroOrAbove(all_block_ready_time_us_, start_time_us_);
    collector_.transfer_gap_latency_us      = subZeroOrAbove(end_time_us_, all_block_ready_time_us_);

    if (reporter_ != nullptr) {
        reporter_->report<RtpLLMCacheStoreMetrics, RtpLLMCacheStoreRemoteStoreMetricsCollector>(nullptr, &collector_);
    }
}

CacheStoreTransferMetricsCollector::CacheStoreTransferMetricsCollector(const kmonitor::MetricsReporterPtr& reporter,
                                                                       int64_t                             block_count,
                                                                       int64_t total_block_size):
    reporter_(reporter), start_time_us_(currentTimeUs()) {
    collector_.block_count      = block_count;
    collector_.total_block_size = total_block_size;
}

CacheStoreTransferMetricsCollector::~CacheStoreTransferMetricsCollector() {
    end_time_us_          = currentTimeUs();
    collector_.latency_us = subZeroOrAbove(end_time_us_, start_time_us_);
    if (reporter_ != nullptr) {
        reporter_->report<RtpLLMCacheStoreMetrics, RtpLLMCacheStoreTransferMetricsCollector>(nullptr, &collector_);
    }
}

}  // namespace rtp_llm
