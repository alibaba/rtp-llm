#include "rtp_llm/cpp/models/eplb/ExpertLoadMonitor.h"
#include "rtp_llm/cpp/models/eplb/ExpertLoadMetrics.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA || USING_ROCM
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#endif

#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace rtp_llm {
namespace {
size_t checkedProduct(size_t a, size_t b) {
    if (b != 0 && a > static_cast<size_t>(std::numeric_limits<int64_t>::max()) / b) {
        throw std::invalid_argument("expert load monitor buffer dimensions overflow");
    }
    return a * b;
}

void writeArray(std::ostream& out, const int64_t* values, size_t count) {
    out << '[';
    for (size_t i = 0; i < count; ++i) {
        if (i != 0) {
            out << ',';
        }
        out << values[i];
    }
    out << ']';
}
}  // namespace

struct ExpertLoadMonitor::Impl {
    size_t                            layer_count;
    size_t                            expert_count;
    size_t                            ep_size;
    size_t                            world_size;
    size_t                            world_rank;
    size_t                            model_id;
    size_t                            capacity;
    size_t                            expert_elements;
    size_t                            rank_elements;
    size_t                            samples        = 0;
    uint64_t                          total_steps    = 0;
    uint64_t                          mapping_epoch  = 0;
    bool                              active         = false;
    bool                              warmup         = false;
    bool                              event_recorded = false;
    torch::Tensor                     step_counts;
    torch::Tensor                     expert_totals;
    torch::Tensor                     rank_history;
    torch::Tensor                     packed;
    torch::Tensor                     packed_host;
    kmonitor::MetricsReporterPtr      reporter;
    std::string                       snapshot_dir;
    size_t                            max_files;
    std::string                       instance_id;
    std::deque<std::filesystem::path> written_files;
#if USING_CUDA || USING_ROCM
    torch::Event completion = cuda_graph::makeGraphEvent();
#endif

    void waitPrevious() {
#if USING_CUDA || USING_ROCM
        if (event_recorded) {
            completion.block(cuda_graph::graphGetCurrentStream());
        }
#endif
    }

    void recordCompletion() {
#if USING_CUDA || USING_ROCM
        completion.record(cuda_graph::graphGetCurrentStream());
        event_recorded = true;
#endif
    }

    void report(const std::vector<ExpertLoadLayerMetrics>& layers) {
        if (!reporter) {
            return;
        }
        for (size_t layer = 0; layer < layer_count; ++layer) {
            const auto&           value = layers[layer];
            kmonitor::MetricsTags tags("layer", std::to_string(layer));
            tags.AddTag("model_id", std::to_string(model_id));
            const auto emit = [&](double number, const char* metric) {
                reporter->report(number, metric, kmonitor::MetricType::GAUGE, &tags, true);
            };
            emit(value.valid ? 1.0 : 0.0, "rtp_llm_moe_lb_valid");
            emit(static_cast<double>(samples), "rtp_llm_moe_lb_sample_count");
            emit(static_cast<double>(value.total_activations), "rtp_llm_moe_expert_activations");
            if (value.valid) {
                emit(value.moe_lb, "rtp_llm_moe_lb");
                emit(value.moe_lb_window, "rtp_llm_moe_lb_window");
            }
            for (size_t rank = 0; rank < ep_size; ++rank) {
                auto rank_tags = tags;
                rank_tags.AddTag("ep_rank", std::to_string(rank));
                reporter->report(static_cast<double>(value.rank_activations[rank]),
                                 "rtp_llm_moe_rank_activations",
                                 kmonitor::MetricType::GAUGE,
                                 &rank_tags,
                                 true);
            }
        }
    }

    void snapshot(const int64_t*                             expert_data,
                  const std::vector<int64_t>&                history,
                  const std::vector<ExpertLoadLayerMetrics>& layers) {
        if (snapshot_dir.empty()) {
            return;
        }
        std::ostringstream out;
        out << std::setprecision(17) << "{\"schema_version\":1,\"model_id\":" << model_id
            << ",\"world_rank\":" << world_rank << ",\"world_size\":" << world_size << ",\"ep_size\":" << ep_size
            << ",\"layer_count\":" << layer_count << ",\"physical_expert_count\":" << expert_count
            << ",\"mapping_epoch\":" << mapping_epoch << ",\"step_begin\":" << total_steps - samples + 1
            << ",\"step_end\":" << total_steps << ",\"sample_count\":" << samples << ",\"window_capacity\":" << capacity
            << ",\"physical_expert_activations\":[";
        for (size_t layer = 0; layer < layer_count; ++layer) {
            if (layer != 0)
                out << ',';
            writeArray(out, expert_data + layer * expert_count, expert_count);
        }
        out << "],\"rank_history\":[";
        for (size_t sample = 0; sample < samples; ++sample) {
            if (sample != 0)
                out << ',';
            out << '[';
            for (size_t layer = 0; layer < layer_count; ++layer) {
                if (layer != 0)
                    out << ',';
                writeArray(out, history.data() + sample * rank_elements + layer * ep_size, ep_size);
            }
            out << ']';
        }
        out << "],\"layers\":[";
        for (size_t layer = 0; layer < layer_count; ++layer) {
            if (layer != 0)
                out << ',';
            const auto& value = layers[layer];
            out << "{\"layer\":" << layer << ",\"valid\":" << (value.valid ? "true" : "false")
                << ",\"total_activations\":" << value.total_activations << ",\"rank_activations\":";
            writeArray(out, value.rank_activations.data(), ep_size);
            out << ",\"moe_lb\":";
            if (value.valid)
                out << value.moe_lb;
            else
                out << "null";
            out << ",\"moe_lb_window\":";
            if (value.valid)
                out << value.moe_lb_window;
            else
                out << "null";
            out << '}';
        }
        out << "]}\n";

        const auto directory = std::filesystem::path(snapshot_dir);
        std::filesystem::create_directories(directory);
        const auto name =
            "model_" + std::to_string(model_id) + "_" + instance_id + "_step_" + std::to_string(total_steps) + ".json";
        const auto destination = directory / name;
        const auto temporary   = directory / (name + ".tmp");
        try {
            std::ofstream file(temporary, std::ios::out | std::ios::trunc);
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file << out.str();
            file.close();
            std::filesystem::rename(temporary, destination);
        } catch (...) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            throw;
        }
        written_files.push_back(destination);
        while (written_files.size() > max_files) {
            std::filesystem::remove(written_files.front());
            written_files.pop_front();
        }
    }

    void flush() {
        if (samples == 0) {
            return;
        }
        try {
            flushWindow();
        } catch (...) {
            // A failed publication loses this window, not the model's forward
            // progress. Never leave samples == capacity for the next select.
            // Keep total_steps so missing windows remain visible in artifacts.
            samples = 0;
            try {
                expert_totals.zero_();
            } catch (...) {
                RTP_LLM_LOG_WARNING("expert load failed-window buffer reset failed");
            }
            // beforeMappingChange has no forward guard to join cleanup work
            // on the next stream, so repair the event chain here as well.
            try {
                recordCompletion();
            } catch (...) {
                RTP_LLM_LOG_WARNING("expert load failed-window event record failed");
            }
            throw;
        }
    }

    void flushWindow() {
        RTP_LLM_PROFILE_SCOPE("expert_load_monitor.flush");
        const size_t history_elements = samples * rank_elements;
        const size_t packed_elements  = expert_elements + history_elements;
        auto         packed_window    = packed.narrow(0, 0, packed_elements);
        packed_window.narrow(0, 0, expert_elements).copy_(expert_totals.flatten());
        packed_window.narrow(0, expert_elements, history_elements).copy_(rank_history.narrow(0, 0, samples).flatten());
        // SUM source contributions before computing a maximum or a ratio.
        // This callback uses standard torch.distributed for DP_AND_TP; the
        // floating-point custom TP AllReduce is not selected for these int64s.
        if (world_size > 1) {
            RTP_LLM_PROFILE_SCOPE("expert_load_monitor.all_reduce");
            execAllReduce({packed_window, ReduceOp::Sum, false, ParallelMode::DP_AND_TP, {}});
        }
        if (world_rank == 0) {
            auto host_window = packed_host.narrow(0, 0, packed_elements);
            // One packed D2H; this is the sole monitoring CPU wait per window.
            {
                RTP_LLM_PROFILE_SCOPE("expert_load_monitor.copy_to_host");
                host_window.copy_(packed_window, /*non_blocking=*/false);
            }
            const auto*          values = host_window.data_ptr<int64_t>();
            std::vector<int64_t> history(values + expert_elements, values + packed_elements);
            const auto           layers = reduceExpertLoadWindow(history, samples, layer_count, ep_size);
            // Telemetry publication must not turn a successful forward into
            // an inference error. Communication/count errors still propagate.
            try {
                report(layers);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("expert load metric reporting failed: %s", e.what());
            }
            try {
                snapshot(values, history, layers);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("expert load snapshot failed: %s", e.what());
            }
        }
        samples = 0;
        expert_totals.zero_();
    }
};

ExpertLoadMonitor::ExpertLoadMonitor(size_t                       layer_count,
                                     size_t                       physical_expert_count,
                                     const ParallelismConfig&     parallelism,
                                     const EPLBConfig&            config,
                                     size_t                       model_id,
                                     kmonitor::MetricsReporterPtr metrics_reporter):
    impl_(std::make_unique<Impl>()) {
#if !USING_CUDA && !USING_ROCM
    throw std::invalid_argument("expert load monitoring requires a CUDA or ROCm device");
#endif
    RTP_LLM_CHECK_WITH_INFO(layer_count > 0 && physical_expert_count > 0 && parallelism.ep_size > 0
                                && parallelism.world_size > 0 && parallelism.world_rank >= 0
                                && parallelism.world_rank < parallelism.world_size,
                            "invalid expert load monitoring dimensions or rank");
    RTP_LLM_CHECK_WITH_INFO(physical_expert_count % parallelism.ep_size == 0,
                            "physical expert slots must be divisible by EP size");
    RTP_LLM_CHECK_WITH_INFO(config.eplb_metrics_report_interval > 0 && config.eplb_metrics_snapshot_max_files > 0,
                            "expert load monitoring window and snapshot retention must be positive");
    auto& state                   = *impl_;
    state.layer_count             = layer_count;
    state.expert_count            = physical_expert_count;
    state.ep_size                 = parallelism.ep_size;
    state.world_size              = parallelism.world_size;
    state.world_rank              = parallelism.world_rank;
    state.model_id                = model_id;
    state.capacity                = config.eplb_metrics_report_interval;
    state.expert_elements         = checkedProduct(layer_count, physical_expert_count);
    state.rank_elements           = checkedProduct(layer_count, state.ep_size);
    const size_t history_elements = checkedProduct(state.capacity, state.rank_elements);
    RTP_LLM_CHECK_WITH_INFO(state.expert_elements <= std::numeric_limits<int64_t>::max() - history_elements,
                            "expert load packed buffer dimensions overflow");
    const auto    gpu_options = torch::TensorOptions(torch::kInt64).device(torch::kCUDA);
    const int64_t layers      = layer_count;
    const int64_t experts     = physical_expert_count;
    state.step_counts         = torch::zeros({layers, experts}, gpu_options);
    state.expert_totals       = torch::zeros({layers, experts}, gpu_options);
    state.rank_history = torch::empty({static_cast<int64_t>(state.capacity), layers, parallelism.ep_size}, gpu_options);
    state.packed       = torch::empty({static_cast<int64_t>(state.expert_elements + history_elements)}, gpu_options);
    if (state.world_rank == 0) {
        state.packed_host = torch::empty(state.packed.sizes(),
                                         torch::TensorOptions(torch::kInt64).device(torch::kCPU).pinned_memory(true));
    }
    state.reporter     = std::move(metrics_reporter);
    state.snapshot_dir = config.eplb_metrics_snapshot_dir;
    state.max_files    = config.eplb_metrics_snapshot_max_files;
    static std::atomic<uint64_t> sequence{0};
    const auto                   stamp = std::chrono::system_clock::now().time_since_epoch().count();
    state.instance_id                  = std::to_string(stamp) + "_" + std::to_string(sequence.fetch_add(1));
    state.recordCompletion();
}

ExpertLoadMonitor::~ExpertLoadMonitor() {
#if USING_CUDA || USING_ROCM
    // Persistent tensors may have been written on an async worker stream,
    // different from their allocation stream. Drain that last use before free.
    try {
        if (impl_->event_recorded) {
            impl_->completion.synchronize();
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("expert load monitor teardown wait failed: %s", e.what());
    }
#endif
}

const torch::Tensor& ExpertLoadMonitor::stepCounts() const {
    return impl_->step_counts;
}

void ExpertLoadMonitor::beginStep(bool warmup) {
    auto& state = *impl_;
    RTP_LLM_CHECK_WITH_INFO(!state.active, "overlapping forwards cannot share an expert load monitor");
    state.waitPrevious();
    state.step_counts.zero_();
    state.warmup = warmup;
    state.active = true;
}

void ExpertLoadMonitor::finishStep() {
    auto& state = *impl_;
    RTP_LLM_CHECK_WITH_INFO(state.active, "expert load finishStep requires beginStep");
    if (!state.warmup) {
        RTP_LLM_CHECK_WITH_INFO(state.samples < state.capacity, "expert load window capacity exceeded");
        // Allocate the rank reduction before committing this step to totals.
        // A recoverable allocation failure must not leave an unrecorded step
        // in the expert window while samples still points at the old row.
        auto rank_counts = state.step_counts
                               .view({static_cast<int64_t>(state.layer_count),
                                      static_cast<int64_t>(state.ep_size),
                                      static_cast<int64_t>(state.expert_count / state.ep_size)})
                               .sum(2);
        state.rank_history.select(0, state.samples).copy_(rank_counts);
        state.expert_totals.add_(state.step_counts);
        ++state.samples;
        ++state.total_steps;
        if (state.samples == state.capacity) {
            state.flush();
        }
    } else {
        state.step_counts.zero_();
    }
    state.recordCompletion();
    state.active = false;
}

void ExpertLoadMonitor::abortStep() noexcept {
    auto& state = *impl_;
    if (!state.active) {
        return;
    }
    state.active = false;
    try {
        state.step_counts.zero_();
        state.recordCompletion();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("expert load aborted-step cleanup failed: %s", e.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("expert load aborted-step cleanup failed");
    }
}

void ExpertLoadMonitor::beforeMappingChange() {
    auto& state = *impl_;
    RTP_LLM_CHECK_WITH_INFO(!state.active, "cannot remap experts during a monitored forward");
    state.waitPrevious();
    state.flush();
    ++state.mapping_epoch;
    state.recordCompletion();
}

}  // namespace rtp_llm
