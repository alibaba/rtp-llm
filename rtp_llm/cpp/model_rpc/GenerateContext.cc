#include "rtp_llm/cpp/model_rpc/GenerateContext.h"

#include <limits>

namespace rtp_llm {

GenerateContext::~GenerateContext() {
    stopStream();
    reportTime();
}

void GenerateContext::reset() {
    error_info   = ErrorInfo::OkStatus();
    error_status = grpc::Status::OK;
    retryable_   = true;
}

bool GenerateContext::ok() const {
    return error_status.ok();
}

bool GenerateContext::hasError() const {
    return !ok();
}

bool GenerateContext::shouldRetry() const {
    return retryable_;
}

void GenerateContext::setRetryable(bool retryable) {
    retryable_ = retryable;
}

void GenerateContext::setRequestTimeoutMs(int64_t timeout_ms) {
    request_timeout_ms = timeout_ms;
    if (timeout_ms > 0) {
        request_deadline = request_begin_time_ + std::chrono::milliseconds(timeout_ms);
    } else {
        request_deadline.reset();
    }
}

void GenerateContext::setRetryTimeoutMs(int64_t timeout_ms) {
    if (timeout_ms > 0) {
        retry_deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);
    } else {
        retry_deadline.reset();
    }
}

GenerateContext::RequestDeadline GenerateContext::streamRpcDeadline(int64_t relative_timeout_ms) const {
    auto deadline = request_deadline;
    if (relative_timeout_ms > 0) {
        const auto relative_deadline =
            std::chrono::system_clock::now() + std::chrono::milliseconds(relative_timeout_ms);
        if (!deadline.has_value() || relative_deadline < *deadline) {
            deadline = relative_deadline;
        }
    }
    return deadline;
}

GenerateContext::RequestDeadline GenerateContext::effectiveDeadline(int64_t relative_timeout_ms) const {
    auto deadline = streamRpcDeadline(relative_timeout_ms);
    if (retry_deadline.has_value() && (!deadline.has_value() || *retry_deadline < *deadline)) {
        deadline = retry_deadline;
    }
    return deadline;
}

bool GenerateContext::retryDeadlineExceeded() const {
    return retry_deadline.has_value() && std::chrono::system_clock::now() >= *retry_deadline;
}

int64_t GenerateContext::cappedRetrySleepUs(int64_t retry_interval_ms) const {
    if (retry_interval_ms <= 0) {
        return 0;
    }
    constexpr int64_t kMicrosecondsPerMillisecond = 1000;
    int64_t           sleep_us                    = std::numeric_limits<int64_t>::max();
    if (retry_interval_ms <= std::numeric_limits<int64_t>::max() / kMicrosecondsPerMillisecond) {
        sleep_us = retry_interval_ms * kMicrosecondsPerMillisecond;
    }
    const auto deadline = effectiveDeadline();
    if (deadline.has_value()) {
        const auto remaining_us =
            std::chrono::duration_cast<std::chrono::microseconds>(*deadline - std::chrono::system_clock::now()).count();
        sleep_us = std::min(sleep_us, std::max<int64_t>(remaining_us, 0));
    }
    return sleep_us;
}

bool GenerateContext::cancelled() const {
    return error_status.error_code() == grpc::StatusCode::CANCELLED;
}

bool GenerateContext::isRequestCancelled() const {
    return server_context && server_context->IsCancelled();
}

bool GenerateContext::requestDeadlineExceeded() const {
    return request_deadline.has_value() && std::chrono::system_clock::now() >= *request_deadline;
}

int64_t GenerateContext::executeTimeMs() {
    return (currentTimeUs() - request_begin_time_us) / 1000;
}

void GenerateContext::reportTime() {
    RpcMetricsCollector collector;
    collectBasicMetrics(collector);
    reportMetrics(collector);
}

void GenerateContext::collectBasicMetrics(RpcMetricsCollector& collector) {
    collector.qps        = true;
    collector.error_qps  = hasError();
    collector.cancel_qps = cancelled();
    if (error_info.hasError()) {
        collector.error_code = error_info.code();
    } else if (stream_ && stream_->hasError()) {
        collector.error_code = stream_->statusInfo().code();
    } else if (cancelled()) {
        collector.error_code = ErrorCode::CANCELLED;
    } else if (hasError()) {
        collector.error_code = ErrorCode::UNKNOWN_ERROR;
    }
    collector.onflight_request   = onflight_requests ? static_cast<int64_t>(onflight_requests->load()) : 0;
    collector.total_rt_us        = executeTimeMs() * 1000;
    collector.retry_times        = retry_times;
    collector.retry_cost_time_ms = retry_cost_time_ms;
}

void GenerateContext::reportMetrics(RpcMetricsCollector& collector) {
    if (metrics_reporter) {
        metrics_reporter->report<RpcMetrics, RpcMetricsCollector>(nullptr, &collector);
    }
}

void GenerateContext::setStream(const std::shared_ptr<GenerateStream>& stream) {
    stream_ = stream;
    if (stream) {
        meta->enqueue(request_id, stream_);
    }
}

void GenerateContext::stopStream() {
    if (stream_) {
        constexpr const char* kContextCleanupReason = "context cleanup before stream finished";
        const bool            context_has_error     = error_info.hasError();
        const bool            request_cancelled     = cancelled() || isRequestCancelled();
        if (stream_->getStatus() != StreamState::FINISHED && !stream_->hasError()) {
            if (context_has_error) {
                RTP_LLM_LOG_WARNING("request [%s] stopping stream with terminal source=context_error, code=%d, err=%s",
                                    request_key.c_str(),
                                    static_cast<int>(error_info.code()),
                                    error_info.ToString().c_str());
                stream_->reportError(error_info.code(), error_info.ToString());
            } else if (request_cancelled) {
                RTP_LLM_LOG_WARNING("request [%s] stopping stream with terminal source=client_cancel",
                                    request_key.c_str());
                stream_->reportError(ErrorCode::CANCELLED, "request cancelled by client");
            }
        }
        const char* cancel_reason = !context_has_error && !request_cancelled ? kContextCleanupReason : "cancel stream";
        if (!stream_->finishOrCancel(kStopStreamWaitTimeoutMs, cancel_reason)) {
            RTP_LLM_LOG_WARNING("stopStream timeout (%ld ms) waiting for Engine Loop for request [%d]",
                                kStopStreamWaitTimeoutMs,
                                stream_->generateInput()->request_id);
        }
        if (!context_has_error && !request_cancelled) {
            const auto stream_error = stream_->statusInfo();
            if (stream_error.code() == ErrorCode::CANCELLED
                && stream_error.ToString().rfind(kContextCleanupReason, 0) == 0) {
                RTP_LLM_LOG_WARNING("request [%s] stopped unfinished stream with terminal source=context_cleanup",
                                    request_key.c_str());
            }
        }
        // RuntimeMeta snapshots the stream's terminal status during dequeue.
        // Capture only after reportError/finishOrCancel have committed it so
        // FlexLB observes the real cancellation or context error code.
        meta->dequeue(request_id, stream_);
        stream_.reset();
    }
}

std::shared_ptr<GenerateStream>& GenerateContext::getStream() {
    return stream_;
}

}  // namespace rtp_llm
