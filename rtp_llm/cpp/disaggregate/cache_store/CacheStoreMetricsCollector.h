#pragma once

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <memory>

namespace rtp_llm {

class CacheStoreStoreMetricsCollector {

public:
    CacheStoreStoreMetricsCollector(const kmonitor::MetricsReporterPtr& reporter,
                                    int64_t                             block_count,
                                    int64_t                             total_block_size);
    ~CacheStoreStoreMetricsCollector();

public:
    void markTaskRun() {
        task_run_time_us_ = currentTimeUs();
    }
    void markEventSyncDone() {
        event_sync_done_time_us_ = currentTimeUs();
    }
    void markEnd(bool success) {
        collector_.success = success;
        end_time_us_       = currentTimeUs();
    }

private:
    kmonitor::MetricsReporterPtr          reporter_;
    RtpLLMCacheStoreStoreMetricsCollector collector_;
    int64_t                               start_time_us_           = 0;
    int64_t                               task_run_time_us_        = 0;
    int64_t                               event_sync_done_time_us_ = 0;
    int64_t                               end_time_us_             = 0;
};

class CacheStoreClientLoadMetricsCollector {

public:
    CacheStoreClientLoadMetricsCollector(const kmonitor::MetricsReporterPtr& reporter,
                                         int64_t                             block_count,
                                         int64_t                             total_block_size);
    ~CacheStoreClientLoadMetricsCollector();

public:
    void markTaskRun() {
        task_run_time_us_ = currentTimeUs();
    }
    void markRequestCallBegin() {
        request_begin_call_time_us_ = currentTimeUs();
    }
    void markRequestCallEnd(int64_t response_send_cost_us) {
        request_call_done_time_us_       = currentTimeUs();
        collector_.response_send_cost_us = response_send_cost_us;
    }
    void markEnd(bool success = true) {
        collector_.success = success;
        end_time_us_       = currentTimeUs();
    }

private:
    kmonitor::MetricsReporterPtr               reporter_;
    RtpLLMCacheStoreLoadClientMetricsCollector collector_;
    int64_t                                    start_time_us_              = 0;
    int64_t                                    task_run_time_us_           = 0;
    int64_t                                    request_begin_call_time_us_ = 0;
    int64_t                                    request_call_done_time_us_  = 0;
    int64_t                                    end_time_us_                = 0;
};

class CacheStoreServerLoadMetricsCollector {
public:
    CacheStoreServerLoadMetricsCollector(const kmonitor::MetricsReporterPtr& reporter,
                                         int64_t                             block_count,
                                         int64_t                             block_size,
                                         int64_t                             request_send_cost_us);
    ~CacheStoreServerLoadMetricsCollector();

public:
    void markFirstBlockReady() {
        first_block_ready_time_us_ = currentTimeUs();
    }
    void markAllBlocksReady() {
        all_block_ready_time_us_ = currentTimeUs();
    }
    void setWriteInfo(int64_t block_count, int64_t block_size, int64_t latency_us);
    void markEnd(bool success = true) {
        collector_.success = success;
        end_time_us_       = currentTimeUs();
    }

private:
    kmonitor::MetricsReporterPtr               reporter_;
    RtpLLMCacheStoreLoadServerMetricsCollector collector_;
    int64_t                                    start_time_us_             = 0;
    int64_t                                    first_block_ready_time_us_ = 0;
    int64_t                                    all_block_ready_time_us_   = 0;
    int64_t                                    end_time_us_               = 0;
    std::mutex                                 write_mutex_;
};

class CacheStoreRemoteStoreMetricsCollector {
public:
    CacheStoreRemoteStoreMetricsCollector(const kmonitor::MetricsReporterPtr& reporter,
                                          int64_t                             block_count);
    ~CacheStoreRemoteStoreMetricsCollector();
public:
    void markStart() {
        start_time_us_ = currentTimeUs();
    }
    void markEnd(bool success = true) {
        collector_.success = success;
        end_time_us_       = currentTimeUs();
    }
    void markFirstBlockReady() {
        first_block_ready_time_us_ = currentTimeUs();
    }
    void markAllBlocksReady() {
        all_block_ready_time_us_ = currentTimeUs();
    }   
    void setBlockSize(int total_block_size) {
        collector_.total_block_size = total_block_size;
    }

private:
    kmonitor::MetricsReporterPtr                reporter_;
    RtpLLMCacheStoreRemoteStoreMetricsCollector collector_;
    int64_t                                     start_time_us_             = 0;
    int64_t                                     first_block_ready_time_us_ = 0;
    int64_t                                     all_block_ready_time_us_   = 0;
    int64_t                                     end_time_us_               = 0;
    std::mutex                                  write_mutex_;
};

class CacheStoreTransferMetricsCollector {
public:
    CacheStoreTransferMetricsCollector(const kmonitor::MetricsReporterPtr& reporter,
                                       int64_t                             block_count,
                                       int64_t                             total_block_size);
    ~CacheStoreTransferMetricsCollector();
private:
    kmonitor::MetricsReporterPtr                reporter_;
    RtpLLMCacheStoreTransferMetricsCollector    collector_;
    int64_t                                     start_time_us_             = 0;
    int64_t                                     end_time_us_               = 0;
};


}  // namespace rtp_llm