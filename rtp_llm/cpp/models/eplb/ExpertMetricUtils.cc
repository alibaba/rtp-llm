#include "rtp_llm/cpp/models/eplb/ExpertMetricUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "autil/EnvUtil.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

namespace rtp_llm {
namespace {
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

void writeMetrics(std::ostream& out, const ExpertLoadMetrics& metrics) {
    out << "\"moe_lb\":" << metrics.moe_lb << ",\"total_activations\":" << metrics.total_activations
        << ",\"max_rank_activations\":" << metrics.max_rank_activations << ",\"rank_activations\":";
    writeArray(out, metrics.rank_activations.data(), metrics.rank_activations.size());
}
}  // namespace

torch::Tensor reduceExpertLoadWindow(const torch::Tensor& window, size_t ep_size) {
    const auto ranks   = ep_size == 1 ? window.sum(0, true) : window;
    const auto total   = ranks.sum();
    const auto maximum = ranks.amax(0).sum();
    return torch::where(total != 0, maximum.to(torch::kDouble) * ep_size / total, 1.0);
}

ExpertLoadSnapshot::ExpertLoadSnapshot(size_t                   layer_count,
                                       size_t                   physical_expert_count,
                                       const ParallelismConfig& parallelism,
                                       const EPLBConfig&        config,
                                       size_t                   model_id,
                                       std::string              model_type):
    layer_count_(layer_count),
    expert_count_(physical_expert_count),
    ep_size_(parallelism.ep_size),
    world_size_(parallelism.world_size),
    world_rank_(parallelism.world_rank),
    model_id_(model_id),
    model_type_(std::move(model_type)),
    capacity_(config.eplb_metrics_report_interval),
    max_files_(config.eplb_metrics_snapshot_max_files),
    directory_(std::filesystem::path(autil::EnvUtil::getEnv("LOG_PATH", std::string("logs"))) / "eplb_snapshot") {
    const auto layers       = static_cast<int64_t>(layer_count_);
    const auto host_options = torch::TensorOptions(torch::kInt64).device(torch::kCPU).pinned_memory(true);
    host_window_ =
        torch::empty({static_cast<int64_t>(world_size_), static_cast<int64_t>(capacity_), layers}, host_options);
    host_snapshot_counts_ = torch::empty({static_cast<int64_t>(ep_size_ + 2), layers + 1}, host_options);
    host_snapshot_ratios_ = torch::empty({layers + 1}, host_options.dtype(torch::kDouble));
    static std::atomic<uint64_t> sequence{0};
    const auto                   stamp = std::chrono::system_clock::now().time_since_epoch().count();
    instance_id_                       = std::to_string(stamp) + "_" + std::to_string(sequence.fetch_add(1));
}

ExpertLoadWindowMetrics ExpertLoadSnapshot::snapshotMetrics(const torch::Tensor& gpu_window) {
    const auto ranks        = ep_size_ == 1 ? gpu_window.sum(0, true) : gpu_window;
    const auto rank_totals  = ranks.sum(1);
    const auto maxima       = ranks.amax(0).sum(0);
    const auto layer_counts = torch::cat({rank_totals.sum(0, true), maxima.unsqueeze(0), rank_totals}, 0);
    const auto counts       = torch::cat({layer_counts, layer_counts.sum(1, true)}, 1);
    const auto totals       = counts.select(0, 0);
    const auto ratios = torch::where(totals != 0, counts.select(0, 1).to(torch::kDouble) * ep_size_ / totals, 1.0);
    host_snapshot_counts_.copy_(counts, false);
    host_snapshot_ratios_.copy_(ratios, false);
    const auto              values   = host_snapshot_counts_.accessor<int64_t, 2>();
    const auto              balances = host_snapshot_ratios_.accessor<double, 1>();
    ExpertLoadWindowMetrics metrics;
    metrics.layers.resize(layer_count_);
    for (size_t layer = 0; layer <= layer_count_; ++layer) {
        auto& result                = layer == layer_count_ ? metrics.aggregate : metrics.layers[layer];
        result.total_activations    = values[0][layer];
        result.max_rank_activations = values[1][layer];
        result.moe_lb               = balances[layer];
        result.rank_activations.resize(ep_size_);
        for (size_t rank = 0; rank < ep_size_; ++rank) {
            result.rank_activations[rank] = values[rank + 2][layer];
        }
    }
    return metrics;
}

void ExpertLoadSnapshot::dump(const torch::Tensor& gpu_window, uint64_t forward_id, uint64_t mapping_epoch) {
    host_window_.copy_(gpu_window, false);
    const auto         metrics = snapshotMetrics(gpu_window);
    const auto*        packed  = host_window_.data_ptr<int64_t>();
    std::ostringstream out;
    out << std::setprecision(17) << "{\"schema_version\":4,\"model_type\":\"" << model_type_
        << "\",\"model_id\":" << model_id_ << ",\"world_rank\":" << world_rank_ << ",\"world_size\":" << world_size_
        << ",\"ep_size\":" << ep_size_ << ",\"layer_count\":" << layer_count_
        << ",\"physical_expert_count\":" << expert_count_ << ",\"mapping_epoch\":" << mapping_epoch
        << ",\"step_begin\":" << forward_id - capacity_ + 1 << ",\"step_end\":" << forward_id
        << ",\"sample_count\":" << capacity_ << ",\"window_capacity\":" << capacity_
        << ",\"sample_layout\":\"layer_sums\",";
    writeMetrics(out, metrics.aggregate);
    out << ",\"rank_samples\":[";
    for (size_t rank = 0; rank < world_size_; ++rank) {
        if (rank != 0) {
            out << ',';
        }
        out << '[';
        for (size_t sample = 0; sample < capacity_; ++sample) {
            if (sample != 0) {
                out << ',';
            }
            writeArray(out, packed + (rank * capacity_ + sample) * layer_count_, layer_count_);
        }
        out << ']';
    }
    out << "],\"layers\":[";
    for (size_t layer = 0; layer < layer_count_; ++layer) {
        if (layer != 0) {
            out << ',';
        }
        out << "{\"layer_id\":" << layer << ',';
        writeMetrics(out, metrics.layers[layer]);
        out << '}';
    }
    out << "]}\n";

    const auto name = "model_" + std::to_string(model_id_) + "_" + model_type_ + "_" + instance_id_ + "_slot_"
                      + std::to_string(snapshot_slot_) + ".json";
    const auto destination = directory_ / name;
    const auto temporary   = directory_ / (name + ".tmp");
    try {
        std::filesystem::create_directories(directory_);
        std::ofstream file(temporary, std::ios::out | std::ios::trunc);
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file << out.str();
        file.close();
        std::filesystem::rename(temporary, destination);
        snapshot_slot_ = (snapshot_slot_ + 1) % max_files_;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("expert load snapshot write failed: %s", e.what());
    }
}

}  // namespace rtp_llm
