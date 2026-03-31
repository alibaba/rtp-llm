#pragma once

#include <torch/extension.h>

namespace rtp_llm {

enum class EplbBalanceMethod {
    EQUAL,   // equally distribute the tokens to redudant experts
    RANDOM,  // randomly distribute the tokens to redudant experts
    GREEDY,  // greedy algorithm to balance the tokens, need to gather all expert ids
};

struct ExpertStatsParams {
    size_t layer_num;
    size_t ep_size;
    size_t log_exp_num;
    size_t phy_exp_num;
};

struct ExpertStatsBuffer {
    torch::Tensor log_stats_buf;  // [layer, log_exp_num], INT32, GPU
    torch::Tensor gpu_loads_buf;  // [layer, ep_size], INT32, GPU
};

struct OverallExpertStats {
    size_t layer_num   = 0;
    size_t ep_size     = 0;
    size_t log_exp_num = 0;
    size_t phy_exp_num = 0;

    // use contiguous memory to store all layers' expert counts
    ExpertStatsBuffer stats_buf;
};

struct ExpertStats {
    size_t layer_id;
    size_t ep_size;
    size_t log_exp_num;
    size_t phy_exp_num;

    // note: need to access the tensor with offset
    ExpertStatsBuffer stats_buf;

    int* getLayerLogStats() const {
        return stats_buf.log_stats_buf.data_ptr<int>() + layer_id * log_exp_num;
    }

    int* getLayerGpuLoads() const {
        return stats_buf.gpu_loads_buf.data_ptr<int>() + layer_id * ep_size;
    }
};

using OptionalExpertStats    = std::optional<ExpertStats>;
using OptionalExpertStatsRef = std::optional<std::reference_wrapper<ExpertStats>>;
}  // namespace rtp_llm