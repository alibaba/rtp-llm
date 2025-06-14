#include "rtp_llm/cpp/model_rpc/GenerateContext.h"

namespace rtp_llm {

GenerateContext::~GenerateContext() {
    if (stream_ && !stream_->finished() && !stream_->stopped()) {
        stream_->cancel();
    }
    reportTime();
}

void GenerateContext::reset() {
    error_status = grpc::Status::OK;
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

int64_t GenerateContext::executeTimeMs() {
    return (currentTimeUs() - request_begin_time_us) / 1000;
}

void GenerateContext::reportTime() {
    RpcMetricsCollector collector;
    collectBasicMetrics(collector);
    reportMetrics(collector);
}

void GenerateContext::collectBasicMetrics(RpcMetricsCollector& collector) {
    collector.qps                               = true;
    collector.error_qps                         = hasError();
    collector.cancel_qps                        = cancelled();
    collector.onflight_request                  = onflight_requests;
    collector.total_rt_us                       = executeTimeMs() * 1000;
}

void GenerateContext::reportMetrics(RpcMetricsCollector& collector) {
    if (metrics_reporter) {
        metrics_reporter->report<RpcMetrics, RpcMetricsCollector>(nullptr, &collector);
    }
}

void GenerateContext::setStream(const std::shared_ptr<GenerateStream>& stream) {
    stream_ = stream;
}

std::shared_ptr<GenerateStream>& GenerateContext::getStream() {
    return stream_;
}

}  // namespace rtp_llm
