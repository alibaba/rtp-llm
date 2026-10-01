#include "rtp_llm/cpp/model_rpc/GenerateContext.h"

namespace rtp_llm {

GenerateContext::~GenerateContext() {
    stopStream();
    reportTime();
}

void GenerateContext::reset() {
    error_status = grpc::Status::OK;
    error_info   = ErrorInfo();
}

bool GenerateContext::ok() const {
    return error_status.ok();
}

bool GenerateContext::hasError() const {
    return !ok();
}

bool GenerateContext::cancelled() const {
    return error_status.error_code() == grpc::StatusCode::CANCELLED;
}

void GenerateContext::setRequestTimeoutMs(int64_t timeout_ms) {
    request_timeout_ms = timeout_ms;
    if (timeout_ms > 0) {
        request_deadline = request_begin_time_ + std::chrono::milliseconds(timeout_ms);
    } else {
        request_deadline.reset();
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

bool GenerateContext::requestDeadlineExceeded() const {
    return request_deadline.has_value() && std::chrono::system_clock::now() >= *request_deadline;
}

bool GenerateContext::isRequestCancelled() const {
    return server_context && server_context->IsCancelled();
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
    collector.qps                = true;
    collector.error_qps          = hasError();
    collector.cancel_qps         = cancelled();
    if (error_info.hasError()) {
        collector.error_code = error_info.code();
    } else if (stream_ && stream_->hasError()) {
        collector.error_code = stream_->statusInfo().code();
    } else if (cancelled()) {
        collector.error_code = ErrorCode::CANCELLED;
    } else if (hasError()) {
        collector.error_code = ErrorCode::UNKNOWN_ERROR;
    }
    collector.onflight_request   = onflight_requests;
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
    if (stream_ && stream_ != stream) {
        stopStreamForRetry();
    }
    stream_ = stream;
    if (stream) {
        meta->enqueue(request_id, stream_);
    }
}

void GenerateContext::markRpcHandlingCompleted() {
    rpc_handling_completed_ = true;
}

void GenerateContext::cancelStreamOnTeardown() noexcept {
    if (!stream_ || stream_->getStatus() == StreamState::FINISHED || stream_->hasError()) {
        return;
    }
    if (rpc_handling_completed_ && !hasError() && !error_info.hasError() && !isRequestCancelled()) {
        return;
    }
    // Report the terminal cause before RuntimeMeta snapshots the stream.
    if (error_info.hasError()) {
        stream_->reportError(error_info.code(), error_info.ToString());
    } else {
        stream_->reportError(ErrorCode::CANCELLED, "RPC handling failed, was cancelled, or exited unexpectedly");
    }
}

void GenerateContext::stopStreamForRetry() {
    if (stream_) {
        if (stream_->getStatus() != StreamState::FINISHED && !stream_->hasError()) {
            stream_->reportError(ErrorCode::CANCELLED, "cancel stream");
        }
        if (meta) {
            meta->dequeue(request_id, stream_);
        }
        stream_.reset();
    }
}

void GenerateContext::stopStream() {
    cancelStreamOnTeardown();
    if (stream_) {
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
