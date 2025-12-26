#pragma once
#include "rtp_llm/cpp/pybind/PyUtils.h"

namespace rtp_llm {
class ElasticEPManager {
private:
    size_t           ep_size_ = 1;
    std::vector<int> active_ranks_;

public:
    ElasticEPManager(size_t ep_size): ep_size_(ep_size), active_ranks_(ep_size, 0) {};
    void query_deepep_mask_buffer();
    bool is_active_decrease();
    void update_deepep_mask_buffer();
};
}  // namespace rtp_llm