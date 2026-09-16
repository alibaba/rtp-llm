#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
#include "autil/EnvUtil.h"

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
    collector.onflight_request   = onflight_requests ? static_cast<int64_t>(onflight_requests->load()) : 0;
    collector.total_rt_us        = executeTimeMs() * 1000;
    collector.retry_times        = retry_times;
    collector.retry_cost_time_ms = retry_cost_time_ms;
}

void GenerateContext::reportMetrics(RpcMetricsCollector& collector) {
    if (metrics_reporter) {
        static const bool trace_phases = autil::EnvUtil::getEnv("DSV41_PD_PERF_TRACE", false);
        if (trace_phases) {
            // These are inclusive RPC phases; cache loading overlaps prefill.
            RTP_LLM_LOG_INFO(
                "PD phase timings: request_id=%ld request_key=%s error_code=%d total_rt_us=%ld "
                "retry_times=%ld get_rpc_connection_rt_us=%ld remote_allocate_resource_rt_us=%ld "
                "enqueue_request_rt_us=%ld remote_load_cache_wait_stream_rt_us=%ld "
                "remote_load_cache_write_request_rt_us=%ld poll_local_output_rt_us=%ld "
                "remote_load_cache_end_rt_us=%ld remote_generate_rt_us=%ld poll_remote_output_rt_us=%ld "
                "prepare_generate_context_rt_us=%ld allocate_resource_rt_us=%ld "
                "load_cache_from_prefill_rt_us=%ld local_generate_rt_us=%ld "
                "load_cache_min_rt_us=%ld load_cache_max_rt_us=%ld load_cache_polling_cost_us=%ld",
                request_id,
                request_key.c_str(),
                static_cast<int>(collector.error_code),
                collector.total_rt_us,
                collector.retry_times,
                collector.get_rpc_connection_rt_us,
                collector.remote_allocate_resource_rt_us,
                collector.enqueue_request_rt_us,
                collector.remote_load_cache_wait_stream_rt_us,
                collector.remote_load_cache_write_request_rt_us,
                collector.poll_local_output_rt_us,
                collector.remote_load_cache_end_rt_us,
                collector.remote_generate_rt_us,
                collector.poll_remote_output_rt_us,
                collector.prepare_generate_context_rt_us,
                collector.allocate_resource_rt_us,
                collector.load_cache_from_prefill_rt_us,
                collector.local_generate_rt_us,
                collector.load_cache_min_rt_us,
                collector.load_cache_max_rt_us,
                collector.load_cache_polling_cost_us);
        }
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
