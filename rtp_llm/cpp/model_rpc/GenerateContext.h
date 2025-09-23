#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"

namespace rtp_llm {

const int64_t MAX_GRPC_TIMEOUT_MS = 3600 * 1000;

class GenerateContext {
public:
    GenerateContext(int64_t                               request_id,
                    int64_t                               request_timeout_ms,
                    grpc::ServerContext*                  server_context,
                    kmonitor::MetricsReporterPtr&         metrics_reporter,
                    std::shared_ptr<RpcServerRuntimeMeta> meta):
        request_id(request_id),
        request_key(std::to_string(request_id)),
        request_timeout_ms(request_timeout_ms),
        server_context(server_context),
        metrics_reporter(metrics_reporter),
        meta(meta) {
        request_begin_time_us = currentTimeUs();
    }
    virtual ~GenerateContext();
    virtual void                             reset();
    bool                                     ok() const;
    bool                                     hasError() const;
    bool                                     cancelled() const;
    int64_t                                  executeTimeMs();
    void                                     reportTime();
    void                                     collectBasicMetrics(RpcMetricsCollector& collector);
    void                                     reportMetrics(RpcMetricsCollector& collector);
    virtual void                             setStream(const std::shared_ptr<GenerateStream>& stream);
    virtual std::shared_ptr<GenerateStream>& getStream();

public:
    int64_t                               request_id;
    std::string                           request_key;
    int64_t                               retry_times           = 0;
    int64_t                               retry_cost_time_ms    = 0;
    int64_t                               onflight_requests     = 0;
    int64_t                               request_timeout_ms    = 0;
    bool                                  finished              = false;
    int64_t                               request_begin_time_us = 0;
    ErrorInfo                             error_info;
    grpc::Status                          error_status = grpc::Status::OK;
    grpc::ServerContext*                  server_context;
    kmonitor::MetricsReporterPtr          metrics_reporter;
    std::shared_ptr<RpcServerRuntimeMeta> meta;

protected:
    std::shared_ptr<GenerateStream> stream_;

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
        if (generate_context.request_timeout_ms > 0 && request_cost_time_ms >= generate_context.request_timeout_ms) {  \
            generate_context.error_info = ErrorInfo(                                                                   \
                ErrorCode::GENERATE_TIMEOUT,                                                                           \
                "request cost time is " + std::to_string(request_cost_time_ms) + " ms" + ", request timeout is "       \
                    + std::to_string(generate_context.request_timeout_ms) + " ms");                                    \
            generate_context.error_status =                                                                            \
                serializeErrorMsg(generate_context.request_key, generate_context.error_info);                          \
            return generate_context.error_status;                                                                      \
        }                                                                                                              \
    }

#define CHECK_REQUEST_CANCELLED(generate_context)                                                                      \
    if (generate_context.server_context->IsCancelled()) {                                                              \
        generate_context.error_info   = ErrorInfo(ErrorCode::CANCELLED, "request is cancelled");                       \
        generate_context.error_status = serializeErrorMsg(generate_context.request_key, generate_context.error_info);  \
        return generate_context.error_status;                                                                          \
    }

#define EXECUTE_STAGE_FUNC(func, generate_context)                                                                     \
    CHECK_REQUEST_STOP(generate_context)                                                                               \
    generate_context.stat_info.nextStage();                                                                            \
    func(generate_context);                                                                                            \
    CHECK_ERROR_STATUS(generate_context)

// for prefill or decode retry
#define EXECUTE_WITH_RETRY(func, generate_context, max_retries, retry_timeout_ms)                                      \
    int64_t begin_time_us = currentTimeUs();                                                                           \
    auto    stage         = generate_context.stat_info.saveStage();                                                    \
    for (int attempt = 0; attempt <= max_retries; ++attempt) {                                                         \
        generate_context.reset();                                                                                      \
        generate_context.stat_info.restoreStage(stage);                                                                \
        generate_context.retry_times++;                                                                                \
        func(generate_context);                                                                                        \
        if (generate_context.ok()) {                                                                                   \
            break;                                                                                                     \
        }                                                                                                              \
        auto cost_time_us                   = currentTimeUs() - begin_time_us;                                         \
        generate_context.retry_cost_time_ms = cost_time_us / 1000;                                                     \
        if (retry_timeout_ms > 0 && cost_time_us >= retry_timeout_ms * 1000) {                                         \
            break;                                                                                                     \
        }                                                                                                              \
        CHECK_REQUEST_STOP(generate_context)                                                                           \
        usleep(1000 * 1);                                                                                              \
    }

}  // namespace rtp_llm
