#pragma once
#include "rtp_llm/cpp/pybind/PyUtils.h"

namespace rtp_llm {
class ElasticEPManager {
private:
    size_t ep_size_ = 1;

    std::vector<int> active_ranks_;
    std::vector<int> last_active_ranks_;

    int active_ranks_cnt_;
    int last_active_ranks_cnt_;

public:
    ElasticEPManager(size_t ep_size):
        ep_size_(ep_size),
        active_ranks_(ep_size, 0),
        last_active_ranks_(ep_size, 0),
        active_ranks_cnt_(ep_size),
        last_active_ranks_cnt_(ep_size) {};

    void query_active_ranks();
    bool is_active_ranks_decrease();

    // todo: mask tp rank
    void update_deepep_mask_buffer();
};
}  // namespace rtp_llm