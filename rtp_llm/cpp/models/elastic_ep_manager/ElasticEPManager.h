#pragma once
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {
class ElasticEPManager {
private:
    void   query_active_ranks();
    size_t ep_size_ = 1;

    std::vector<int> active_ranks_;
    std::vector<int> last_active_ranks_;

    torch::Tensor active_ranks_tensor_;

    int active_ranks_cnt_;
    int last_active_ranks_cnt_;

public:
    ElasticEPManager(size_t ep_size):
        ep_size_(ep_size),
        active_ranks_(ep_size, 1),
        last_active_ranks_(ep_size, 1),
        active_ranks_cnt_(ep_size),
        last_active_ranks_cnt_(ep_size) {};

    bool          is_active_ranks_decrease();
    torch::Tensor get_active_ranks_tensor() const;
    int           get_active_ranks_cnt() const;

    // todo: mask tp rank
    void update_deepep_mask_buffer();
};
}  // namespace rtp_llm