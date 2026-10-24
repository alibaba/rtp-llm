#include "rtp_llm/cpp/models/eplb/ExpertLoadMonitor.h"
#include "rtp_llm/cpp/models/eplb/ExpertMetricUtils.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <utility>

namespace rtp_llm {
struct ExpertLoadMonitor::Impl {
    size_t                              layer_count;
    size_t                              ep_size;
    size_t                              world_size;
    size_t                              world_rank;
    size_t                              model_id;
    std::string                         model_type;
    size_t                              capacity;
    size_t                              samples       = 0;
    uint64_t                            forward_id    = 0;
    uint64_t                            mapping_epoch = 0;
    torch::Tensor                       step_expert_token_counts;
    torch::Tensor                       window;
    torch::Tensor                       gathered_window;
    torch::Tensor                       host_moe_lb;
    kmonitor::MetricsReporterPtr        reporter;
    std::unique_ptr<ExpertLoadSnapshot> snapshot;

    void report(double moe_lb) {
        RTP_LLM_LOG_INFO("[EPLB] model_type=%s model_id=%zu moe_lb %.4f", model_type.c_str(), model_id, moe_lb);
        if (reporter) {
            kmonitor::MetricsTags tags("model_id", std::to_string(model_id));
            tags.AddTag("model_type", model_type);
            RtpLLmEplbMetricsCollector collector;
            collector.moe_lb = moe_lb;
            reporter->report<RtpLLmEplbMetrics, RtpLLmEplbMetricsCollector>(&tags, &collector);
        }
    }

    void consumeWindow() {
        RTP_LLM_PROFILE_SCOPE("expert_load_monitor.consume_window");
        // Dense models have no writers on any rank. Preserve the empty-window
        // metric without a collective, GPU reduction or device-to-host copy.
        if (step_expert_token_counts.size(1) == 0) {
            if (world_rank == 0) {
                report(1.0);
                if (snapshot) {
                    snapshot->dump(torch::zeros({static_cast<int64_t>(world_size),
                                                 static_cast<int64_t>(capacity),
                                                 static_cast<int64_t>(layer_count)},
                                                torch::TensorOptions(torch::kInt64).device(torch::kCPU)),
                                   forward_id,
                                   mapping_epoch);
                }
            }
            return;
        }
        auto gpu_window = window;
        if (world_size > 1) {
            execAllGather({{gathered_window}, ParallelMode::DP_AND_TP, {window}, false});
            if (world_rank != 0) {
                return;
            }
            gpu_window = gathered_window;
        }
        gpu_window = gpu_window.view(
            {static_cast<int64_t>(world_size), static_cast<int64_t>(capacity), static_cast<int64_t>(layer_count)});
        // Copy only the GPU-reduced scalar to the preallocated pinned buffer.
        host_moe_lb.copy_(reduceExpertLoadWindow(gpu_window, ep_size), false);
        report(host_moe_lb.item<double>());
        if (snapshot) {
            snapshot->dump(gpu_window, forward_id, mapping_epoch);
        }
    }
};

std::unique_ptr<ExpertLoadMonitor> ExpertLoadMonitor::create(size_t                       layer_count,
                                                             size_t                       physical_expert_count,
                                                             const ParallelismConfig&     parallelism,
                                                             const EPLBConfig&            config,
                                                             size_t                       model_id,
                                                             kmonitor::MetricsReporterPtr metrics_reporter,
                                                             std::string                  model_type) {
    RTP_LLM_CHECK_WITH_INFO(layer_count > 0 && parallelism.ep_size > 0
                                && physical_expert_count % parallelism.ep_size == 0,
                            "invalid expert load monitoring dimensions");
    RTP_LLM_CHECK_WITH_INFO(config.eplb_metrics_report_interval > 0 && config.eplb_metrics_snapshot_max_files >= 0,
                            "expert load monitoring window must be positive and snapshot retention non-negative");
    const char* unsupported = nullptr;
#if !USING_CUDA && !USING_ROCM
    unsupported = "requires a CUDA or ROCm device";
#else
    if (parallelism.pp_size != 1 || parallelism.ffn_disaggregate_config.enable_ffn_disaggregate) {
        unsupported = "requires an unpartitioned model (PP=1, no FFN disaggregation)";
    } else if (physical_expert_count != 0 && parallelism.ep_size == 1 && parallelism.prefill_cp_config.is_enabled()) {
        unsupported = "requires EP for MoE context parallel inputs";
    } else if (parallelism.ep_size != parallelism.world_size
               && !(parallelism.ep_size == 1 && parallelism.dp_size == 1)) {
        unsupported = "requires EP=WORLD or replicated TP with EP=1";
    }
#endif
    if (unsupported) {
        RTP_LLM_LOG_WARNING("[EPLB] skipping expert load monitoring for model_type=%s model_id=%zu: %s",
                            model_type.c_str(),
                            model_id,
                            unsupported);
        return nullptr;
    }
    return std::unique_ptr<ExpertLoadMonitor>(new ExpertLoadMonitor(layer_count,
                                                                    physical_expert_count,
                                                                    parallelism,
                                                                    config,
                                                                    model_id,
                                                                    std::move(metrics_reporter),
                                                                    std::move(model_type)));
}

ExpertLoadMonitor::ExpertLoadMonitor(size_t                       layer_count,
                                     size_t                       physical_expert_count,
                                     const ParallelismConfig&     parallelism,
                                     const EPLBConfig&            config,
                                     size_t                       model_id,
                                     kmonitor::MetricsReporterPtr metrics_reporter,
                                     std::string                  model_type):
    impl_(std::make_unique<Impl>()) {
    auto& state        = *impl_;
    state.layer_count  = layer_count;
    state.ep_size      = parallelism.ep_size;
    state.world_size   = parallelism.world_size;
    state.world_rank   = parallelism.world_rank;
    state.model_id     = model_id;
    state.model_type   = std::move(model_type);
    state.capacity     = config.eplb_metrics_report_interval;
    const auto options = torch::TensorOptions(torch::kInt64).device(torch::kCUDA);
    const auto layers  = static_cast<int64_t>(layer_count);
    const auto experts = static_cast<int64_t>(physical_expert_count / parallelism.ep_size);
    // Backends accumulate this step at a fixed graph-captured address.
    state.step_expert_token_counts = torch::zeros({layers, experts}, options.dtype(torch::kInt32));
    state.window                   = torch::empty({static_cast<int64_t>(state.capacity), layers}, options);
    if (state.world_size > 1 && physical_expert_count != 0) {
        state.gathered_window =
            torch::empty({static_cast<int64_t>(state.world_size * state.capacity), layers}, options);
    }
    state.reporter          = std::move(metrics_reporter);
    if (physical_expert_count != 0) {
        const auto host_options = torch::TensorOptions(torch::kDouble).device(torch::kCPU).pinned_memory(true);
        state.host_moe_lb       = torch::empty({}, host_options);
    }
    if (config.eplb_metrics_snapshot_max_files != 0) {
        state.snapshot = std::make_unique<ExpertLoadSnapshot>(
            layer_count, physical_expert_count, parallelism, config, model_id, state.model_type);
    }
}

ExpertLoadMonitor::~ExpertLoadMonitor() = default;

const torch::Tensor& ExpertLoadMonitor::stepCountsBuffer() const {
    return impl_->step_expert_token_counts;
}

void ExpertLoadMonitor::clearRecordBuffer() {
    impl_->step_expert_token_counts.zero_();
}

void ExpertLoadMonitor::recordStep() {
    auto& state = *impl_;
    auto  row   = state.window.select(0, state.samples);
    torch::sum_out(row, state.step_expert_token_counts, {1}, false, torch::kInt64);
    ++state.forward_id;
    if (++state.samples == state.capacity) {
        state.consumeWindow();
        state.samples = 0;
    }
    clearRecordBuffer();
}

void ExpertLoadMonitor::beforeMappingChange() {
    impl_->samples = 0;
    ++impl_->mapping_epoch;
}

}  // namespace rtp_llm
