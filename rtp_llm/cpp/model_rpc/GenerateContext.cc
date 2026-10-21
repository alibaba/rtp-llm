#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"

#include <limits>

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

void GenerateContext::synthesizeTracePhases(telemetry::PhaseRole role, bool exception_unwinding) {
    if (!trace_span_guard || !trace_span_guard->valid()) {
        return;
    }
    auto& stream = getStream();
    if (!stream) {
        return;
    }
    const auto             time_info = stream->getTimeInfo();
    telemetry::PhaseTiming phase_timing;
    phase_timing.begin_time_us           = time_info.begin_time_us;
    phase_timing.running_started         = time_info.running_started;
    phase_timing.running_started_time_us = time_info.running_started_time_us;
    phase_timing.first_token_committed   = time_info.first_token_committed;
    phase_timing.first_token_time_us     = time_info.first_token_time_us;
    phase_timing.generation_done         = time_info.generation_done;
    phase_timing.generation_done_time_us = time_info.generation_done_time_us;
    phase_timing.synthesis_end_time_us   = currentTimeUs();
    phase_timing.request_id              = request_id;
    const bool request_ok                = error_status.ok() && !exception_unwinding;
    if (request_ok && time_info.generation_done) {
        // Beam rows are an internal search width; Fusion exposes one primary
        // sequence (the remaining candidates live in beam_responses). Only
        // ordinary multi-return Fusion requests aggregate all active rows. PD
        // retains its existing per-sequence, per-hop usage contract.
        const auto returned_sequence_count =
            role != telemetry::PhaseRole::Fusion ? 1 :
            stream->hasNumBeams() ? std::max(stream->numReturnSequences(), 1) : stream->currentBatchSize();
        telemetry::setUsageTokenAttributes(*trace_span_guard,
                                           (int64_t)stream->inputLength(),
                                           (int64_t)(stream->outputTokenLen() * returned_sequence_count));
    }
    const auto local_error = stream->firstError();
    const auto first_error = FirstError::earlier(local_error, dependency_failure);
    const bool dependency_selected = dependency_failure.error.hasError()
                                     && first_error.order == dependency_failure.order;
    const bool priority_preempted = first_error.error.code() == ErrorCode::PRIORITY_PREEMPTED
                                    || error_info.code() == ErrorCode::PRIORITY_PREEMPTED;
    phase_timing.error_type = request_ok ? nullptr :
                               priority_preempted ? "PRIORITY_PREEMPTED" :
                               dependency_selected ? "DependencyFailure" :
                               !error_status.ok() ? telemetry::grpcStatusCodeName(error_status.error_code()) :
                                                    "Exception";
    if (role == telemetry::PhaseRole::Decode && time_info.loading_cache_start_time_us > 0) {
        const bool load_finished = time_info.loading_cache_done_time_us > 0;
        if (load_finished || !request_ok) {
            const bool load_ok = load_finished && time_info.loading_cache_succeeded;
            const auto error = first_error.error.hasError() ? first_error.error :
                               error_info.hasError() ? error_info : errorInfoFromGrpcStatus(error_status);
            const auto reason = error.hasError() ? ErrorCodeToString(error.code()) : std::string{};
            telemetry::synthesizeKvLoadSpan(
                trace_span_guard->sharedSpan(), time_info.loading_cache_start_time_us,
                load_finished ? time_info.loading_cache_done_time_us : phase_timing.synthesis_end_time_us,
                request_id, load_ok, load_ok ? nullptr : phase_timing.error_type,
                !load_ok && error.hasError() ? static_cast<int64_t>(error.code()) : -1,
                !load_ok && !reason.empty() ? reason.c_str() : nullptr);
        }
    }
    telemetry::synthesizePhaseSpans(trace_span_guard->sharedSpan(), phase_timing, role, request_ok);
}

void GenerateContext::recordRpcFailure(const ErrorInfo& error) {
    if (!error_info.hasError()) {
        const auto first = stream_ ? stream_->firstError() : FirstError::Snapshot{};
        error_info = first.error.hasError() ? first.error : error;
    }
    error_status = grpcStatusFromErrorInfo(error_info);
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
        collector.error_code = errorInfoFromGrpcStatus(error_status).code();
    }
    collector.onflight_request   = onflight_requests ? static_cast<int64_t>(onflight_requests->load()) : 0;
    collector.loading_cache_request = meta ? meta->loadingCacheRequestCount() : 0;
    collector.total_rt_us        = executeTimeMs() * 1000;
    collector.retry_times        = retry_times;
    collector.retry_cost_time_ms = retry_cost_time_ms;
}

void GenerateContext::reportMetrics(RpcMetricsCollector& collector) {
    // Stage collectors also carry gauges; refresh them instead of publishing
    // their default zero over other active requests.
    collector.onflight_request = onflight_requests ? static_cast<int64_t>(onflight_requests->load()) : 0;
    collector.loading_cache_request = meta ? meta->loadingCacheRequestCount() : 0;
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
    // Preserve the terminal cause before RuntimeMeta snapshots the stream.
    if (error_info.hasError()) {
        stream_->reportError(error_info.code(), error_info.ToString());
    } else {
        stream_->reportError(ErrorCode::CANCELLED, "RPC handling failed, was cancelled, or exited unexpectedly");
    }
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
