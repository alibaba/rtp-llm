#include "rtp_llm/cpp/model_rpc/GenerateContext.h"

namespace rtp_llm {

GenerateContext::~GenerateContext() {
    if (stream_ && stream_->getStatus() != StreamState::FINISHED) {
        stream_->reportError(ErrorCode::CANCELLED, "cancel stream");
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
    stream_ = stream;
    if (stream) {
        meta->enqueue(request_id, stream_);
    }
}

void GenerateContext::stopStream() {
    if (stream_) {
        // if is waiting, cancel it
        meta->dequeue(request_id, stream_);
        // A request that did not finish on its own is being torn down by cancellation or abort. Read
        // this AFTER the dequeue, exactly where the reportError() condition below has always read it,
        // so the control flow is unchanged and the flag only drives logging.
        const bool cancelled_before_finish = (stream_->getStatus() != StreamState::FINISHED);
        const bool was_running             = (stream_->getStatus() == StreamState::RUNNING);
        if (cancelled_before_finish) {
            stream_->reportError(ErrorCode::CANCELLED, "cancel stream");
        }
        // if is running, waiting util done
        const int64_t drain_begin_us = currentTimeUs();
        while (stream_->getStatus() == StreamState::RUNNING) {
            RTP_LLM_LOG_DEBUG("waiting stream [%d] running done to cancel", stream_->generateInput()->request_id);
            usleep(1000);
        }
        if (cancelled_before_finish) {
            // Rare (cancellation/abort teardown only), so a WARNING per request is not spam, and this
            RTP_LLM_LOG_WARNING("request [%ld] cancelled: removed from the scheduler (was_running=%d), "
                                "drained in %ld us, releasing stream and KV blocks",
                                request_id,
                                (int)was_running,
                                (long)((currentTimeUs() - drain_begin_us)));
        }
        stream_.reset();
    }
}

std::shared_ptr<GenerateStream>& GenerateContext::getStream() {
    return stream_;
}

}  // namespace rtp_llm
