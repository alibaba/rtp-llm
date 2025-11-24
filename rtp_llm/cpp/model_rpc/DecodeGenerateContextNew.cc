#include "rtp_llm/cpp/model_rpc/DecodeGenerateContextNew.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"

namespace rtp_llm {

DecodeGenerateContextNew::~DecodeGenerateContextNew() {
    if (stream_ != nullptr && stream_->running()) {
        stream_->setStop(error_info.code(), error_info.ToString());
    }
    reportTime();
}

ErrorInfo DecodeGenerateContextNew::init(const std::shared_ptr<EngineBase>& engine) {
    RTP_LLM_LOG_DEBUG("request [%s] start to prepare generate context", request_key.c_str());

    generate_input = QueryConverter::transQuery(request);

    stream_            = engine->makeStream(generate_input);
    request_timeout_ms = stream_->getTimeoutMs();

    auto status = stream_->initKVBlock();
    if (!status.ok()) {
        RTP_LLM_LOG_WARNING("request [%s] init kv block failed, malloc kv cache block failed", request_key.c_str());
        error_info = ErrorInfo(ErrorCode::MALLOC_FAILED, "malloc kv cache block failed at decode node");
        return error_info;
    }

    prepare_generate_context_done_time_us = currentTimeUs();
    RTP_LLM_LOG_DEBUG("request [%s] prepare generate context done", request_key.c_str());
    return ErrorInfo::OkStatus();
}

void DecodeGenerateContextNew::reportTime() {
    RpcMetricsCollector collector;
    collectBasicMetrics(collector);
    collector.retry_times                    = retry_times;
    collector.prepare_generate_context_rt_us = prepare_generate_context_done_time_us - request_begin_time_us;
    collector.load_cache_from_prefill_rt_us =
        load_cache_from_prefill_done_time_us - prepare_generate_context_done_time_us;
    collector.local_generate_rt_us = local_generate_done_time_us - load_cache_from_prefill_done_time_us;

    reportMetrics(collector);
    metrics_reporter.reset();  // avoid to report metrics in base class
}

}  // namespace rtp_llm