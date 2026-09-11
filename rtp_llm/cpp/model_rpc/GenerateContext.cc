#include "rtp_llm/cpp/model_rpc/GenerateContext.h"

#include <limits>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

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

ErrorInfo GenerateContext::finalErrorInfo() const {
    if (error_info.hasError()) {
        return error_info;
    }
    if (stream_ && stream_->hasError()) {
        return stream_->statusInfo();
    }

    ErrorDetailsPB error_details;
    if (!error_status.error_details().empty() && error_details.ParseFromString(error_status.error_details())
        && error_details.error_code() != static_cast<int>(ErrorCode::NONE_ERROR)) {
        return ErrorInfo(static_cast<ErrorCode>(error_details.error_code()), error_details.error_message());
    }
    switch (error_status.error_code()) {
        case grpc::StatusCode::OK:
            return ErrorInfo::OkStatus();
        case grpc::StatusCode::CANCELLED:
            return ErrorInfo(ErrorCode::CANCELLED, error_status.error_message());
        case grpc::StatusCode::INVALID_ARGUMENT:
            return ErrorInfo(ErrorCode::INVALID_PARAMS, error_status.error_message());
        case grpc::StatusCode::DEADLINE_EXCEEDED:
            return ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, error_status.error_message());
        case grpc::StatusCode::RESOURCE_EXHAUSTED:
            return ErrorInfo(ErrorCode::MALLOC_FAILED, error_status.error_message());
        case grpc::StatusCode::INTERNAL:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error_status.error_message());
        default:
            return ErrorInfo(ErrorCode::UNKNOWN_ERROR, error_status.error_message());
    }
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
    const auto final_error_info  = finalErrorInfo();
    collector.qps                = true;
    collector.error_qps          = final_error_info.hasError();
    collector.cancel_qps         = final_error_info.code() == ErrorCode::CANCELLED;
    collector.error_code         = final_error_info.code();
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
        if (meta) {
            meta->enqueue(request_id, stream_);
        }
    }
}

void GenerateContext::stopStream() {
    if (stream_) {
        if (stream_->getStatus() != StreamState::FINISHED && !stream_->hasError()) {
            if (error_info.hasError()) {
                RTP_LLM_LOG_WARNING("request [%s] stopping stream with terminal source=context_error, code=%d, err=%s",
                                    request_key.c_str(),
                                    static_cast<int>(error_info.code()),
                                    error_info.ToString().c_str());
                stream_->reportError(error_info.code(), error_info.ToString());
            } else if (cancelled() || isRequestCancelled()) {
                RTP_LLM_LOG_WARNING("request [%s] stopping stream with terminal source=client_cancel",
                                    request_key.c_str());
                stream_->reportError(ErrorCode::CANCELLED, "request cancelled by client");
            } else {
                RTP_LLM_LOG_WARNING("request [%s] stopping unfinished stream with terminal source=context_cleanup",
                                    request_key.c_str());
                stream_->reportError(ErrorCode::CANCELLED, "context cleanup before stream finished");
            }
        }
        while (stream_->getStatus() == StreamState::RUNNING) {
            RTP_LLM_LOG_DEBUG("waiting stream [%d] running done to cancel", stream_->generateInput()->request_id);
            usleep(1000);
        }
        if (meta) {
            meta->dequeue(request_id, stream_);
        }
        stream_.reset();
    }
}

std::shared_ptr<GenerateStream>& GenerateContext::getStream() {
    return stream_;
}

}  // namespace rtp_llm
