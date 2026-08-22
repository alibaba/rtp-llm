#include "rtp_llm/cpp/model_rpc/GenerateContext.h"

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

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
        if (stream_->getStatus() != StreamState::FINISHED) {
            stream_->reportError(ErrorCode::CANCELLED, "cancel stream");
        }
        // if is running, waiting util done
        while (stream_->getStatus() == StreamState::RUNNING) {
            RTP_LLM_LOG_DEBUG("waiting stream [%d] running done to cancel", stream_->generateInput()->request_id);
            usleep(1000);
        }
        stream_.reset();
    }
}

std::shared_ptr<GenerateStream>& GenerateContext::getStream() {
    return stream_;
}

}  // namespace rtp_llm
