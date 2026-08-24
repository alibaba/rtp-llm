#include "rtp_llm/cpp/model_rpc/GenerateContext.h"

namespace rtp_llm {

GenerateContext::~GenerateContext() {
    if (!rpc_handling_completed_) {
        RTP_LLM_LOG_ERROR("request [%s] GenerateContext destroyed before RPC handling completed, grpc code [%d], "
                          "grpc message [%s], finished [%d], has stream [%d]",
                          request_key.c_str(),
                          static_cast<int>(error_status.error_code()),
                          error_status.error_message().c_str(),
                          finished,
                          static_cast<bool>(stream_));
    }
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

bool GenerateContext::cancelled() const {
    return error_status.error_code() == grpc::StatusCode::CANCELLED;
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
    const bool request_cancelled = server_context && server_context->IsCancelled();
    if ((rpc_handling_completed_ && !hasError() && !request_cancelled) || !stream_
        || stream_->getStatus() == StreamState::FINISHED || stream_->hasError()) {
        return;
    }
    stream_->reportError(ErrorCode::CANCELLED, "RPC handling failed, was cancelled, or exited unexpectedly");
}

void GenerateContext::stopStreamForRetry() {
    if (!stream_) {
        return;
    }
    if (stream_->getStatus() != StreamState::FINISHED && !stream_->hasError()) {
        stream_->reportError(ErrorCode::CANCELLED, "cancel abandoned retry attempt");
    }
    if (meta) {
        meta->dequeue(request_id, stream_);
    }
    stream_.reset();
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
