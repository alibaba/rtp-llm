#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/models/eplb/ExpertLoadMonitor.h"
#include <c10/core/InferenceMode.h>
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
void EplbController::init(const EPLBConfig& config, size_t world_size) {
    requested_  = config;
    world_size_ = world_size;
    interval_   = config.eplb_control_step;
    if (world_size_ > 1) {
        host_   = torch::empty({2}, torch::TensorOptions(torch::kInt64).pinned_memory(true));
        device_ = torch::empty({2}, torch::TensorOptions(torch::kInt64).device(torch::kCUDA));
    }
}

void EplbController::setData(const EPLBConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    requested_.eplb_mode = config.eplb_mode;
    if (config.eplb_update_time >= 0) {
        requested_.eplb_update_time = config.eplb_update_time;
    }
}

std::optional<EPLBConfig> EplbController::step() {
    if (++steps_ < interval_) {
        return std::nullopt;
    }
    steps_ = 0;
    EPLBConfig current;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current = requested_;
    }
    if (world_size_ > 1) {
        auto* data = host_.data_ptr<int64_t>();
        data[0]    = static_cast<int64_t>(current.eplb_mode);
        data[1]    = current.eplb_update_time;
        device_.copy_(host_, true);
        execBroadcast({{device_}, 0, ParallelMode::DP_AND_TP});
        host_.copy_(device_);
        current.eplb_mode        = static_cast<EplbMode>(data[0]);
        current.eplb_update_time = data[1];
    }
    return current;
}

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
        RTP_LLM_CHECK_WITH_INFO(config_.eplb_update_time >= 0 && config_.eplb_control_step > 0,
                                "EPLB update interval must be non-negative and control interval positive");
        RTP_LLM_CHECK_WITH_INFO(config_.eplb_stats_window_size >= 0
                                    && (config_.eplb_update_time == 0
                                        || (config_.eplb_stats_window_size > 0
                                            && config_.eplb_stats_window_size <= config_.eplb_update_time)),
                                "EPLB stats window must be non-negative and between 1 and the enabled update interval");
        RTP_LLM_CHECK_WITH_INFO(config_.eplb_rearrange_p2p_groups > 0, "EPLB P2P group count must be positive");
        controller_.init(config_, parallel.world_size);
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
        if (auto current = controller_.step();
            current
            && (current->eplb_mode != config_.eplb_mode || current->eplb_update_time != config_.eplb_update_time)) {
            config_.eplb_mode        = current->eplb_mode;
            config_.eplb_update_time = current->eplb_update_time;
            python_.updateConfig(config_);
        }
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

bool ExpertBalancer::updateEplbConfig(const EPLBConfig& config) {
    if (!runtime_requested_) {
        return config.eplb_mode == EplbMode::NONE;
    }
    if (config.eplb_update_time < -1 || config.eplb_mode < EplbMode::NONE || config.eplb_mode > EplbMode::ALL
        || (config.eplb_update_time > 0
            && (config_.eplb_stats_window_size == 0 || config_.eplb_stats_window_size > config.eplb_update_time))) {
        return false;
    }
    controller_.setData(config);
    return true;
}
}  // namespace rtp_llm
