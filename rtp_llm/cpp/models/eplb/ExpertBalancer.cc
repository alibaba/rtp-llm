#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/models/eplb/ExpertLoadMonitor.h"
#include <c10/core/InferenceMode.h>
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {
ExpertBalancer::ExpertBalancer(const std::optional<MoeConfigs>& moe, const GptModelInitParams& params):
    config_(params.eplb_config),
    runtime_requested_(config_.enable_eplb()),
    ep_rank_(params.parallelism_config.ep_rank),
    ep_size_(params.parallelism_config.ep_size),
    model_id_(params.model_id),
    is_normal_engine_(params.parallelism_config.role_type == RoleType::PDFUSION),
    sync_prompt_phase_(is_normal_engine_ && params.parallelism_config.dp_size > 1
                       && params.sp_config.type != SP_TYPE_MTP && params.sp_config.type != SP_TYPE_EAGLE
                       && params.sp_config.type != SP_TYPE_DSPARK),
    metrics_reporter_(params.metrics_reporter) {
    // NormalEngine::isMTPEagle uses the same three types: its fake streams and
    // MtpExecutor's fixed prefill/decode sequence already align their DP phases.
    c10::InferenceMode guard(true);
    const auto         layers   = params.weights.layers.size();
    const auto         physical = moe ? moe->expert_num + moe->extra_expert_num : 0;
    RTP_LLM_CHECK_WITH_INFO(config_.eplb_metrics_report_interval >= 0,
                            "EPLB_METRICS_REPORT_INTERVAL must be non-negative");
    if (runtime_requested_) {
        const auto& parallel = params.parallelism_config;
        RTP_LLM_CHECK_WITH_INFO(moe && layers > 0 && ep_size_ > 0 && physical % ep_size_ == 0,
                                "EPLB requires MoE layers and equally sized physical expert shards");
        RTP_LLM_CHECK_WITH_INFO(parallel.ep_size == parallel.world_size && parallel.pp_size == 1
                                    && !parallel.ffn_disaggregate_config.enable_ffn_disaggregate,
                                "EPLB requires complete experts with EP=WORLD, PP=1 and no FFN disaggregation");
        RTP_LLM_CHECK_WITH_INFO(config_.eplb_update_time >= 0, "EPLB update interval must be non-negative");
        RTP_LLM_CHECK_WITH_INFO(config_.eplb_stats_window_size >= 0
                                    && (config_.eplb_update_time == 0
                                        || (config_.eplb_stats_window_size > 0
                                            && config_.eplb_stats_window_size <= config_.eplb_update_time)),
                                "EPLB stats window must be non-negative and between 1 and the enabled update interval");
        RTP_LLM_CHECK_WITH_INFO(config_.eplb_rearrange_p2p_groups > 0, "EPLB P2P group count must be positive");
    }
    if (config_.eplb_metrics_report_interval > 0) {
        load_monitor_ = ExpertLoadMonitor::create(layers,
                                                  physical,
                                                  params.parallelism_config,
                                                  config_,
                                                  model_id_,
                                                  metrics_reporter_,
                                                  model_id_ == 0 ? "target" : "draft");
    }
    if (load_monitor_) {
        if (moe && (ep_size_ > 1 || params.parallelism_config.tp_rank == 0)) {
            expert_token_counts_ = load_monitor_->stepCountsBuffer();
        }
    } else if (runtime_requested_) {
        expert_token_counts_ = torch::zeros({static_cast<int64_t>(layers), static_cast<int64_t>(physical / ep_size_)},
                                            torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    }
}

ExpertBalancer::~ExpertBalancer() = default;

void ExpertBalancer::initialize(const py::object& model) {
    if (runtime_requested_) {
        python_.initialize(model, config_, ep_rank_, ep_size_, model_id_, is_normal_engine_, sync_prompt_phase_);
    }
}

void ExpertBalancer::step(bool warmup, bool is_prompt) {
    if (!load_monitor_ && !runtime_requested_) {
        return;
    }
    RTP_LLM_PROFILE_SCOPE("py_model.eplb_step");
    const auto start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    bool       changed  = false;
    if (!warmup && runtime_requested_) {
        changed = python_.step(expert_token_counts_, is_prompt);
    }
    if (load_monitor_) {
        if (warmup) {
            load_monitor_->clearRecordBuffer();
        } else {
            load_monitor_->recordStep();
        }
    } else if (expert_token_counts_.defined()) {
        expert_token_counts_.zero_();
    }
    if (warmup || !runtime_requested_) {
        return;
    }
    if (changed && load_monitor_) {
        load_monitor_->beforeMappingChange();
    }
    if (metrics_reporter_) {
        RtpLLmEplbMetricsCollector collector{};
        collector.step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_us;
        metrics_reporter_->report<RtpLLmEplbMetrics, RtpLLmEplbMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llm
