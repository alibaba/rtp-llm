#include "rtp_llm/cpp/model_rpc/DecodeGenerateContext.h"

namespace rtp_llm {

DecodeStatInfo::ExecuteStage DecodeStatInfo::saveStage() const {
    return stage;
}

void DecodeStatInfo::restoreStage(DecodeStatInfo::ExecuteStage stage_) {
    stage = stage_;
}

void DecodeStatInfo::nextStage() {
    finishStage();
    stage      = static_cast<DecodeStatInfo::ExecuteStage>(static_cast<int>(stage) + 1);
    begin_time = currentTimeUs();
}

void DecodeStatInfo::finishStage() {
    if (begin_time == 0) {
        return;
    }
    const auto cost_time_us = currentTimeUs() - begin_time;
    begin_time              = 0;
    switch (stage) {
        case prepareGenerateContext:
            prepare_generate_context_rt_us += cost_time_us;
            break;
        case allocateResource:
            allocate_resource_rt_us += cost_time_us;
            break;
        case loadCacheFromPrefill:
            load_cache_from_prefill_rt_us += cost_time_us;
            break;
        case localGenerate:
            local_generate_rt_us += cost_time_us;
            break;
        case start:
        case finish:
            break;
        default:
            RTP_LLM_CHECK_WITH_INFO(false, "error stage");
    }
}

DecodeGenerateContext::~DecodeGenerateContext() {
    reportTime();
}

void DecodeGenerateContext::TimeInfo::updateRequestBegineTime() {
    request_begin_time_us = currentTimeUs();
}
void DecodeGenerateContext::TimeInfo::updateLoadBeginTime() {
    load_begin_time_us = currentTimeUs();
}
void DecodeGenerateContext::TimeInfo::updateLoadEndTime() {
    load_end_time_us = currentTimeUs();
}
void DecodeGenerateContext::TimeInfo::updateGenerateBeginTime() {
    generate_begin_time_us = currentTimeUs();
}
void DecodeGenerateContext::TimeInfo::updateGenerateEndTime() {
    generate_end_time_us = currentTimeUs();
}
int64_t DecodeGenerateContext::TimeInfo::loadCacheTimeMs() const {
    return (load_end_time_us - load_begin_time_us) / 1000;
}

void DecodeGenerateContext::reportTime() {
    RpcMetricsCollector collector;

    collectBasicMetrics(collector);

    collector.loading_cache_request          = loading_cache_requests;
    collector.prepare_generate_context_rt_us = stat_info.prepare_generate_context_rt_us;
    collector.allocate_resource_rt_us        = stat_info.allocate_resource_rt_us;
    collector.load_cache_from_prefill_rt_us  = stat_info.load_cache_from_prefill_rt_us;
    collector.local_generate_rt_us           = stat_info.local_generate_rt_us;

    // for tp
    collector.load_cache_min_rt_us       = stat_info.load_cache_min_rt_us;
    collector.load_cache_max_rt_us       = stat_info.load_cache_max_rt_us;
    collector.load_cache_polling_cost_us = stat_info.load_cache_polling_cost_us;

    reportMetrics(collector);
    metrics_reporter.reset();  // avoid to report metrics in base class
}

}  // namespace rtp_llm
