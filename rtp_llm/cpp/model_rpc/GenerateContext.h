#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <thread>

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"
#include "rtp_llm/cpp/telemetry/RpcTraceHelper.h"

namespace rtp_llm {

class GenerateContext {
public:
    using RequestDeadline = std::optional<std::chrono::system_clock::time_point>;

    GenerateContext(int64_t                               request_id,
                    int64_t                               request_timeout_ms,
                    grpc::ServerContext*                  server_context,
                    kmonitor::MetricsReporterPtr&         metrics_reporter,
                    std::shared_ptr<RpcServerRuntimeMeta> meta,
                    bool                                  request_id_present = true):
        request_id(request_id),
        request_id_present(request_id_present),
        request_key(std::to_string(request_id)),
        server_context(server_context),
        metrics_reporter(metrics_reporter),
        meta(meta) {
        request_begin_time_us = currentTimeUs();
        request_begin_time_   = std::chrono::system_clock::now();
        setRequestTimeoutMs(request_timeout_ms);
    }
    virtual ~GenerateContext();
    virtual void                             reset();
    bool                                     ok() const;
    bool                                     hasError() const;
    bool                                     shouldRetry() const;
    void                                     setRetryable(bool retryable);
    void                                     setRequestTimeoutMs(int64_t request_timeout_ms);
    void                                     setRetryTimeoutMs(int64_t retry_timeout_ms);
    RequestDeadline                          streamRpcDeadline(int64_t relative_timeout_ms = 0) const;
    RequestDeadline                          effectiveDeadline(int64_t relative_timeout_ms = 0) const;
    bool                                     retryDeadlineExceeded() const;
    int64_t                                  cappedRetrySleepUs(int64_t retry_interval_ms) const;
    bool                                     cancelled() const;
    virtual bool                             isRequestCancelled() const;
    bool                                     requestDeadlineExceeded() const;
    ErrorInfo                                finalErrorInfo() const;
    int64_t                                  executeTimeMs();
    void                                     reportTime();
    void                                     collectBasicMetrics(RpcMetricsCollector& collector);
    void                                     reportMetrics(RpcMetricsCollector& collector);
    virtual void                             setStream(const std::shared_ptr<GenerateStream>& stream);
    virtual std::shared_ptr<GenerateStream>& getStream();

public:
    int64_t                               request_id;
    bool                                  request_id_present;
    std::string                           request_key;
    int64_t                               retry_times           = 0;
    int64_t                               retry_cost_time_ms    = 0;
    const std::atomic<size_t>*            onflight_requests     = nullptr;
    int64_t                               request_timeout_ms    = 0;
    bool                                  finished              = false;
    int64_t                               request_begin_time_us = 0;
    RequestDeadline                       request_deadline;
    RequestDeadline                       retry_deadline;
    ErrorInfo                             error_info;
    grpc::Status                          error_status = grpc::Status::OK;
    RequestInfo                           request_info;
    grpc::ServerContext*                  server_context;
    kmonitor::MetricsReporterPtr          metrics_reporter;
    std::shared_ptr<RpcServerRuntimeMeta> meta;

    // OTel SERVER span finish guard. Declared after `error_status` so it destructs
    // before it and can still read the final status; nullptr when telemetry is
    // disabled. Ends the span on every exit path (RAII).
    std::unique_ptr<telemetry::GrpcStatusSpanGuard> trace_span_guard;

protected:
    std::shared_ptr<GenerateStream>       stream_;
    bool                                  retryable_ = true;
    std::chrono::system_clock::time_point request_begin_time_;

protected:
    void stopStream();
};

#define CHECK_ERROR_STATUS(generate_context)                                                                           \
    if (generate_context.finished || generate_context.hasError()) {                                                    \
        return generate_context.error_status;                                                                          \
    }

#define CHECK_REQUEST_STOP(generate_context)                                                                           \
    CHECK_REQUEST_TIMEOUT(generate_context)                                                                            \
    CHECK_REQUEST_CANCELLED(generate_context)

#define CHECK_REQUEST_TIMEOUT(generate_context)                                                                        \
    {                                                                                                                  \
        auto request_cost_time_ms = (currentTimeUs() - generate_context.request_begin_time_us) / 1000;                 \
        if (generate_context.requestDeadlineExceeded()) {                                                              \
            generate_context.error_info = ErrorInfo(                                                                   \
                ErrorCode::GENERATE_TIMEOUT,                                                                           \
                "request cost time is " + std::to_string(request_cost_time_ms) + " ms" + ", request timeout is "       \
                    + std::to_string(generate_context.request_timeout_ms) + " ms");                                    \
            generate_context.error_status = serializeErrorMsg(                                                         \
                generate_context.request_key, generate_context.request_info, generate_context.error_info);             \
            return generate_context.error_status;                                                                      \
        }                                                                                                              \
    }

#define CHECK_REQUEST_CANCELLED(generate_context)                                                                      \
    if (generate_context.isRequestCancelled()) {                                                                       \
        generate_context.error_info   = ErrorInfo(ErrorCode::CANCELLED, "request is cancelled");                       \
        generate_context.error_status = serializeErrorMsg(                                                             \
            generate_context.request_key, generate_context.request_info, generate_context.error_info);                 \
        return generate_context.error_status;                                                                          \
    }

#define EXECUTE_STAGE_FUNC(func, generate_context)                                                                     \
    CHECK_REQUEST_STOP(generate_context)                                                                               \
    generate_context.stat_info.nextStage();                                                                            \
    func(generate_context);                                                                                            \
    generate_context.stat_info.finishStage();                                                                          \
    CHECK_ERROR_STATUS(generate_context)

// for prefill or decode retry
#define EXECUTE_WITH_RETRY(func, generate_context, max_retries, retry_timeout_ms, retry_interval_ms)                   \
    int64_t       begin_time_us  = currentTimeUs();                                                                    \
    const int64_t retry_attempts = std::max<int64_t>(max_retries, 0);                                                  \
    generate_context.setRetryTimeoutMs(retry_timeout_ms);                                                              \
    auto stage = generate_context.stat_info.saveStage();                                                               \
    for (int64_t attempt = 0; attempt <= retry_attempts; ++attempt) {                                                  \
        CHECK_REQUEST_STOP(generate_context)                                                                           \
        if (attempt > 0 && generate_context.retryDeadlineExceeded()) {                                                 \
            break;                                                                                                     \
        }                                                                                                              \
        generate_context.reset();                                                                                      \
        CHECK_REQUEST_STOP(generate_context)                                                                           \
        generate_context.stat_info.restoreStage(stage);                                                                \
        generate_context.retry_times++;                                                                                \
        func(generate_context);                                                                                        \
        if (generate_context.ok()) {                                                                                   \
            break;                                                                                                     \
        }                                                                                                              \
        auto cost_time_us                   = currentTimeUs() - begin_time_us;                                         \
        generate_context.retry_cost_time_ms = cost_time_us / 1000;                                                     \
        if (!generate_context.shouldRetry() || generate_context.retryDeadlineExceeded()                                \
            || attempt == retry_attempts) {                                                                            \
            break;                                                                                                     \
        }                                                                                                              \
        CHECK_REQUEST_STOP(generate_context)                                                                           \
        const int64_t retry_sleep_us = generate_context.cappedRetrySleepUs(retry_interval_ms);                         \
        if (retry_sleep_us > 0) {                                                                                      \
            std::this_thread::sleep_for(std::chrono::microseconds(retry_sleep_us));                                    \
        }                                                                                                              \
    }

}  // namespace rtp_llm
