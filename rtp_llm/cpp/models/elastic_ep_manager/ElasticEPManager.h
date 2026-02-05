#pragma once
#include "rtp_llm/cpp/pybind/PyUtils.h"

namespace rtp_llm {
struct ElasticEPStats {
    bool          is_downscale_ = false;
    torch::Tensor active_ranks_tensor_cpu_;
    int           active_ranks_num_;
};
class ElasticEPManager {
private:
    void updateElasticEPStats();

    int            last_active_ranks_num_;
    ElasticEPStats elastic_ep_stats_;

public:
    ElasticEPManager(size_t ep_size);

    void stepForward(ElasticEPStats& stats);
};
}  // namespace rtp_llm