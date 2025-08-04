#pragma once

#include "autil/TimeUtility.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm::threefs {

#define PROPERTY(PropertyType, property_name, MethodName, default_value)                                               \
public:                                                                                                                \
    static inline void set##MethodName(std::shared_ptr<DistKvCacheMetrics>& metrics, PropertyType value) {             \
        if (metrics) {                                                                                                 \
            metrics->property_name = value;                                                                            \
        }                                                                                                              \
    }                                                                                                                  \
    inline PropertyType MethodName() const {                                                                           \
        return property_name;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
private:                                                                                                               \
    PropertyType property_name{default_value};

#define TIME_FIELD(field_name, MethodName)                                                                             \
public:                                                                                                                \
    static inline void mark##MethodName(std::shared_ptr<DistKvCacheMetrics>& metrics) {                                \
        if (metrics) {                                                                                                 \
            metrics->field_name = autil::TimeUtility::currentTimeInMicroSeconds();                                     \
        }                                                                                                              \
    }                                                                                                                  \
    static inline void mark##MethodName(std::shared_ptr<DistKvCacheMetrics>& metrics, int64_t value) {                 \
        if (metrics) {                                                                                                 \
            metrics->field_name = value;                                                                               \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
private:                                                                                                               \
    int64_t field_name{0};

#define TIME_COST_GETTER(MethodName, begin_field_name, done_field_name)                                                \
public:                                                                                                                \
    inline int64_t MethodName() const {                                                                                \
        if (begin_field_name <= 0 || done_field_name < begin_field_name) {                                             \
            return -1;                                                                                                 \
        }                                                                                                              \
        return done_field_name - begin_field_name;                                                                     \
    }

class DistKvCacheMetricsReporter;

class DistKvCacheMetrics final {
public:
    // for single query
    TIME_FIELD(match_begin_us, MatchBeginUs);
    TIME_FIELD(match_done_us, MatchDoneUs);
    TIME_COST_GETTER(MatchCostUs, match_begin_us, match_done_us);
    PROPERTY(int64_t, cache_input_length, CacheInputLength, -1);
    PROPERTY(int64_t, cache_match_length, CacheMatchLength, -1);
    PROPERTY(int64_t, cache_get_length, CacheGetLength, -1);
    PROPERTY(int64_t, cache_put_length, CachePutLength, -1);
    PROPERTY(int64_t, cache_hit_rate, CacheHitRate, -1);
    PROPERTY(bool, match_qps, MatchQps, false);
    PROPERTY(bool, match_failed_qps, MatchFailedQps, false);
    PROPERTY(bool, get_cache_failed_qps, GetCacheFailedQps, false);
    PROPERTY(bool, put_cache_failed_qps, PutCacheFailedQps, false);

    // for all query
    PROPERTY(int64_t, total_cache_match_length, TotalCacheMatchLength, -1);
    PROPERTY(int64_t, total_cache_input_length, TotalCacheInputLength, -1);
    PROPERTY(float, total_cache_hit_rate, TotalCacheHitRate, -1);

    // for single rank
    TIME_FIELD(get_cache_begin_us, GetCacheBeginUs);
    TIME_FIELD(get_cache_done_us, GetCacheDoneUs);
    TIME_COST_GETTER(GetCacheCostUs, get_cache_begin_us, get_cache_done_us);
    TIME_FIELD(put_cache_begin_us, PutCacheBeginUs);
    TIME_FIELD(put_cache_done_us, PutCacheDoneUs);
    TIME_COST_GETTER(PutCacheCostUs, put_cache_begin_us, put_cache_done_us);

    // for all rank
    TIME_FIELD(total_get_cache_begin_us, TotalGetCacheBeginUs);
    TIME_FIELD(total_get_cache_done_us, TotalGetCacheDoneUs);
    TIME_COST_GETTER(TotalGetCacheCostUs, total_get_cache_begin_us, total_get_cache_done_us);
    TIME_FIELD(total_put_cache_begin_us, TotalPutCacheBeginUs);
    TIME_FIELD(total_put_cache_done_us, TotalPutCacheDoneUs);
    TIME_COST_GETTER(TotalPutCacheCostUs, total_put_cache_begin_us, total_put_cache_done_us);

    // read
    TIME_FIELD(total_read_begin_us, TotalReadBeginUs);
    TIME_FIELD(total_read_done_us, TotalReadDoneUs);
    TIME_COST_GETTER(TotalReadCostUs, total_read_begin_us, total_read_done_us);  // 3FS Read + CudaCopy + etc.
    PROPERTY(int64_t, total_read_len, TotalReadLen, -1);                         // byte
    PROPERTY(float, total_read_throughput, TotalReadThroughput, -1);             // MiB/s
    TIME_FIELD(read_block_begin_us, ReadBlockBeginUs);
    TIME_FIELD(read_block_done_us, ReadBlockDoneUs);
    TIME_COST_GETTER(ReadBlockCostUs, read_block_begin_us, read_block_done_us);  // only 3FS read
    PROPERTY(int64_t, read_block_len, ReadBlockLen, -1);                         // byte
    PROPERTY(float, read_block_throughput, ReadBlockThroughput, -1);             // MiB/s
    // TIME_FIELD(read_meta_begin_us, ReadMetaBeginUs);
    // TIME_FIELD(read_meta_done_us, ReadMetaDoneUs);
    // TIME_COST_GETTER(ReadMetaCostUs, read_meta_begin_us, read_meta_done_us);
    // PROPERTY(int64_t, read_meta_len, ReadMetaLen, -1);  // byte
    TIME_FIELD(read_cuda_copy_begin_us, ReadCudaCopyBeginUs);
    TIME_FIELD(read_cuda_copy_done_us, ReadCudaCopyDoneUs);
    TIME_COST_GETTER(ReadCudaCopyCostUs, read_cuda_copy_begin_us, read_cuda_copy_done_us);

    // write
    TIME_FIELD(total_write_begin_us, TotalWriteBeginUs);
    TIME_FIELD(total_write_done_us, TotalWriteDoneUs);
    TIME_COST_GETTER(TotalWriteCostUs, total_write_begin_us, total_write_done_us);  // 3FS Write + CudaCopy + etc.
    PROPERTY(int64_t, total_write_len, TotalWriteLen, -1);                          // byte
    PROPERTY(float, total_write_throughput, TotalWriteThroughput, -1);              // MiB/s
    TIME_FIELD(write_block_begin_us, WriteBlockBeginUs);
    TIME_FIELD(write_block_done_us, WriteBlockDoneUs);
    TIME_COST_GETTER(WriteBlockCostUs, write_block_begin_us, write_block_done_us);  // only 3FS write
    PROPERTY(float, write_block_throughput, WriteBlockThroughput, -1);              // MiB/s
    TIME_FIELD(write_cuda_copy_begin_us, WriteCudaCopyBeginUs);
    TIME_FIELD(write_cuda_copy_done_us, WriteCudaCopyDoneUs);
    TIME_COST_GETTER(WriteCudaCopyCostUs, write_cuda_copy_begin_us, write_cuda_copy_done_us);
    PROPERTY(int32_t, write_threadpool_workitem_count, WriteThreadPoolWorkItemCount, -1);

    // iov mempool
    PROPERTY(int64_t, read_iov_allocated_size, ReadIovAllocatedSize, -1);
    PROPERTY(int32_t, read_iov_allocated_count, ReadIovAllocatedCount, -1);
    PROPERTY(int64_t, read_iov_free_size, ReadIovFreeSize, -1);
    PROPERTY(int64_t, write_iov_allocated_size, WriteIovAllocatedSize, -1);
    PROPERTY(int32_t, write_iov_allocated_count, WriteIovAllocatedCount, -1);
    PROPERTY(int64_t, write_iov_free_size, WriteIovFreeSize, -1);

    // 3fs capacity
    PROPERTY(int64_t, fs_used_size, FSUsedSize, -1);
    PROPERTY(int64_t, fs_free_size, FSFreeSize, -1);
};

#define _GET_METRIC(metric_name) metric_name##_metric_
#define _METRIC_NAME(metric_name) "rtp_llm_3fs_" #metric_name
#define METRIC(metric_name) kmonitor::MutableMetric* metric_name##_metric_ = nullptr
#define REGISTER_METRIC(metric_name) REGISTER_GAUGE_MUTABLE_METRIC(_GET_METRIC(metric_name), _METRIC_NAME(metric_name))
#define REGISTER_QPS_METRIC(metric_name)                                                                               \
    REGISTER_QPS_MUTABLE_METRIC(_GET_METRIC(metric_name), _METRIC_NAME(metric_name))
#define REPORT_METRIC(metric_name, MethodName)                                                                         \
    if (metrics->MethodName() >= 0) {                                                                                  \
        REPORT_MUTABLE_METRIC(_GET_METRIC(metric_name), metrics->MethodName());                                        \
    }
#define REPORT_QPS_METRIC(metric_name, MethodName)                                                                     \
    if (metrics->MethodName()) {                                                                                       \
        REPORT_MUTABLE_QPS(_GET_METRIC(metric_name));                                                                  \
    }

class DistKvCacheMetricsReporter: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override {
        REGISTER_METRIC(match_cost_us);
        REGISTER_METRIC(cache_input_length);
        REGISTER_METRIC(cache_match_length);
        REGISTER_METRIC(cache_get_length);
        REGISTER_METRIC(cache_put_length);
        REGISTER_METRIC(cache_hit_rate);
        REGISTER_QPS_METRIC(match_qps);
        REGISTER_QPS_METRIC(match_failed_qps);
        REGISTER_QPS_METRIC(get_cache_failed_qps);
        REGISTER_QPS_METRIC(put_cache_failed_qps);

        REGISTER_METRIC(total_cache_match_length);
        REGISTER_METRIC(total_cache_input_length);
        REGISTER_METRIC(total_cache_hit_rate);

        REGISTER_METRIC(get_cache_cost_us);
        REGISTER_METRIC(put_cache_cost_us);

        REGISTER_METRIC(total_get_cache_cost_us);
        REGISTER_METRIC(total_put_cache_cost_us);

        REGISTER_METRIC(total_read_cost_us);
        REGISTER_METRIC(total_read_len);
        REGISTER_METRIC(total_read_throughput);
        REGISTER_METRIC(read_block_cost_us);
        REGISTER_METRIC(read_block_len);
        REGISTER_METRIC(read_block_throughput);
        // REGISTER_METRIC(read_meta_cost_us);
        // REGISTER_METRIC(read_meta_len);
        REGISTER_METRIC(read_cuda_copy_cost_us);

        REGISTER_METRIC(total_write_cost_us);
        REGISTER_METRIC(total_write_len);
        REGISTER_METRIC(total_write_throughput);
        REGISTER_METRIC(write_block_cost_us);
        REGISTER_METRIC(write_block_throughput);
        REGISTER_METRIC(write_cuda_copy_cost_us);
        REGISTER_METRIC(write_threadpool_workitem_count);

        REGISTER_METRIC(read_iov_allocated_size);
        REGISTER_METRIC(read_iov_allocated_count);
        REGISTER_METRIC(read_iov_free_size);
        REGISTER_METRIC(write_iov_allocated_size);
        REGISTER_METRIC(write_iov_allocated_count);
        REGISTER_METRIC(write_iov_free_size);

        REGISTER_METRIC(fs_used_size);
        REGISTER_METRIC(fs_free_size);
        return true;
    }
    void report(const kmonitor::MetricsTags* tags, DistKvCacheMetrics* metrics) {
        REPORT_METRIC(match_cost_us, MatchCostUs);
        REPORT_METRIC(cache_input_length, CacheInputLength);
        REPORT_METRIC(cache_match_length, CacheMatchLength);
        REPORT_METRIC(cache_get_length, CacheGetLength);
        REPORT_METRIC(cache_put_length, CachePutLength);
        REPORT_METRIC(cache_hit_rate, CacheHitRate);
        REPORT_QPS_METRIC(match_qps, MatchQps);
        REPORT_QPS_METRIC(match_failed_qps, MatchFailedQps);
        REPORT_QPS_METRIC(get_cache_failed_qps, GetCacheFailedQps);
        REPORT_QPS_METRIC(put_cache_failed_qps, PutCacheFailedQps);

        REPORT_METRIC(total_cache_match_length, TotalCacheMatchLength);
        REPORT_METRIC(total_cache_input_length, TotalCacheInputLength);
        REPORT_METRIC(total_cache_hit_rate, TotalCacheHitRate);

        REPORT_METRIC(get_cache_cost_us, GetCacheCostUs);
        REPORT_METRIC(put_cache_cost_us, PutCacheCostUs);

        REPORT_METRIC(total_get_cache_cost_us, TotalGetCacheCostUs);
        REPORT_METRIC(total_put_cache_cost_us, TotalPutCacheCostUs);

        REPORT_METRIC(total_read_cost_us, TotalReadCostUs);
        REPORT_METRIC(total_read_len, TotalReadLen);
        REPORT_METRIC(total_read_throughput, TotalReadThroughput);
        REPORT_METRIC(read_block_cost_us, ReadBlockCostUs);
        REPORT_METRIC(read_block_len, ReadBlockLen);
        REPORT_METRIC(read_block_throughput, ReadBlockThroughput);
        // REPORT_METRIC(read_meta_cost_us, ReadMetaCostUs);
        // REPORT_METRIC(read_meta_len, ReadMetaLen);
        REPORT_METRIC(read_cuda_copy_cost_us, ReadCudaCopyCostUs);

        REPORT_METRIC(total_write_cost_us, TotalWriteCostUs);
        REPORT_METRIC(total_write_len, TotalWriteLen);
        REPORT_METRIC(total_write_throughput, TotalWriteThroughput);
        REPORT_METRIC(write_block_cost_us, WriteBlockCostUs);
        REPORT_METRIC(write_block_throughput, WriteBlockThroughput);
        REPORT_METRIC(write_cuda_copy_cost_us, WriteCudaCopyCostUs);
        REPORT_METRIC(write_threadpool_workitem_count, WriteThreadPoolWorkItemCount);

        REPORT_METRIC(read_iov_allocated_size, ReadIovAllocatedSize);
        REPORT_METRIC(read_iov_allocated_count, ReadIovAllocatedCount);
        REPORT_METRIC(read_iov_free_size, ReadIovFreeSize);
        REPORT_METRIC(write_iov_allocated_size, WriteIovAllocatedSize);
        REPORT_METRIC(write_iov_allocated_count, WriteIovAllocatedCount);
        REPORT_METRIC(write_iov_free_size, WriteIovFreeSize);

        REPORT_METRIC(fs_used_size, FSUsedSize);
        REPORT_METRIC(fs_free_size, FSFreeSize);
    }

private:
    METRIC(match_cost_us);
    METRIC(cache_input_length);
    METRIC(cache_match_length);
    METRIC(cache_get_length);
    METRIC(cache_put_length);
    METRIC(cache_hit_rate);
    METRIC(get_cache_cost_us);
    METRIC(put_cache_cost_us);
    METRIC(match_qps);
    METRIC(match_failed_qps);
    METRIC(get_cache_failed_qps);
    METRIC(put_cache_failed_qps);

    METRIC(total_cache_hit_rate);
    METRIC(total_cache_match_length);
    METRIC(total_cache_input_length);
    METRIC(total_get_cache_cost_us);
    METRIC(total_put_cache_cost_us);

    METRIC(total_read_cost_us);
    METRIC(total_read_len);
    METRIC(total_read_throughput);
    METRIC(read_block_cost_us);
    METRIC(read_block_len);
    METRIC(read_block_throughput);
    // METRIC(read_meta_cost_us);
    // METRIC(read_meta_len);
    METRIC(read_cuda_copy_cost_us);

    METRIC(total_write_cost_us);
    METRIC(total_write_len);
    METRIC(total_write_throughput);
    METRIC(write_block_cost_us);
    METRIC(write_block_throughput);
    METRIC(write_cuda_copy_cost_us);
    METRIC(write_threadpool_workitem_count);

    METRIC(read_iov_allocated_size);
    METRIC(read_iov_allocated_count);
    METRIC(read_iov_free_size);
    METRIC(write_iov_allocated_size);
    METRIC(write_iov_allocated_count);
    METRIC(write_iov_free_size);

    METRIC(fs_used_size);
    METRIC(fs_free_size);
};

class DistKvCacheMetricsFactory final {
public:
    static std::shared_ptr<DistKvCacheMetrics> createMetrics(const kmonitor::MetricsReporterPtr& metrics_reporter) {
        auto deleter = [metrics_reporter](DistKvCacheMetrics* metrics) {
            if (metrics_reporter) {
                metrics_reporter->report<DistKvCacheMetricsReporter, DistKvCacheMetrics>(nullptr, metrics);
            }
            delete metrics;
        };
        return std::shared_ptr<DistKvCacheMetrics>(new DistKvCacheMetrics(), deleter);
    }
};

}  // namespace rtp_llm::threefs