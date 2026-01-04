#include "rtp_llm/cpp/models/elastic_ep_manager/ElasticEPManager.h"

namespace rtp_llm {
std::vector<int> maskDeepEPBufferToActiveRanks(const torch::Tensor& tensor, size_t size) {
    auto             tensor_cpu = tensor.device().is_cuda() ? tensor.cpu() : tensor;
    std::vector<int> vec(size, 0);
    for (size_t i = 0; i < tensor_cpu.size(0); ++i) {
        vec[i] = 1 - tensor_cpu[i].item<int>();
    }
    return vec;
}

void ElasticEPManager::query_active_ranks() {
    py::gil_scoped_acquire acquire;
    py::module             deepep         = py::module::import("rtp_llm.models_py.distributed.deepep_wrapper");
    py::object             deepep_wrapper = deepep.attr("get_deepep_wrapper")();

    torch::Tensor mask_buffer = deepep_wrapper.attr("query_mask_buffer")().cast<torch::Tensor>();
    last_active_ranks_        = active_ranks_;
    active_ranks_             = maskDeepEPBufferToActiveRanks(mask_buffer, ep_size_);
    last_active_ranks_cnt_    = active_ranks_cnt_;
    active_ranks_cnt_         = std::count(active_ranks_.begin(), active_ranks_.end(), 1);
    // for (size_t i = 0; i < active_ranks_.size(); ++i) {
    //     printf("ElasticEPManager: active_ranks_[%zu] = %d\n", i, active_ranks_[i]);
    // }
}

bool ElasticEPManager::is_active_ranks_decrease() {
    query_active_ranks();
    if (last_active_ranks_cnt_ > active_ranks_cnt_) {
        printf("ElasticEPManager: active ranks decrease from %d to %d\n", last_active_ranks_cnt_, active_ranks_cnt_);
        return true;
    }
    return false;
}

}  // namespace rtp_llm