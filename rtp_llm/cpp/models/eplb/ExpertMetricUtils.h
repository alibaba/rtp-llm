#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <torch/types.h>

namespace rtp_llm {

struct ParallelismConfig;
struct EPLBConfig;

struct ExpertLoadMetrics {
    double               moe_lb               = 1.0;
    int64_t              total_activations    = 0;
    int64_t              max_rank_activations = 0;
    std::vector<int64_t> rank_activations;
};

struct ExpertLoadWindowMetrics {
    ExpertLoadMetrics              aggregate;
    std::vector<ExpertLoadMetrics> layers;
};

// CUDA int64 rank-major window: [world_size, sample_count, layer_count].
// DeepInfer's moe_lb is sum(step/layer rank maxima) / sum(step/layer rank means).
// Return a CUDA double scalar; zero-load windows yield 1.0.
// EP=1 folds replicated TP ranks into one expert-parallel owner.
torch::Tensor reduceExpertLoadWindow(const torch::Tensor& window, size_t ep_size);

class ExpertLoadSnapshot {
public:
    ExpertLoadSnapshot(size_t                   layer_count,
                       size_t                   physical_expert_count,
                       const ParallelismConfig& parallelism,
                       const EPLBConfig&        config,
                       size_t                   model_id,
                       std::string              model_type);

    void dump(const torch::Tensor& gpu_window, uint64_t forward_id, uint64_t mapping_epoch);

private:
    ExpertLoadWindowMetrics snapshotMetrics(const torch::Tensor& gpu_window);

    size_t                layer_count_;
    size_t                expert_count_;
    size_t                ep_size_;
    size_t                world_size_;
    size_t                world_rank_;
    size_t                model_id_;
    std::string           model_type_;
    size_t                capacity_;
    size_t                max_files_;
    std::filesystem::path directory_;
    std::string           instance_id_;
    size_t                snapshot_slot_ = 0;
    torch::Tensor         host_window_;
    torch::Tensor         host_snapshot_counts_;
    torch::Tensor         host_snapshot_ratios_;
};

}  // namespace rtp_llm
