#pragma once
#include "rtp_llm/cpp/pybind/PyUtils.h"

namespace rtp_llm {
struct ElasticEPStats {
    int           ep_rank_;
    int           ep_size_;  // may be used in upscale
    bool          is_downscale_ = false;
    torch::Tensor active_ranks_tensor_cpu_;
    int           active_ranks_num_;
    bool          is_rank_active_ = true;
};
class ElasticEPManager {
private:
    void updateElasticEPStats();

    int            last_active_ranks_num_;
    ElasticEPStats elastic_ep_stats_;

public:
    ElasticEPManager(int ep_size, int ep_rank);

    void stepForward(ElasticEPStats& stats);
};
}  // namespace rtp_llm