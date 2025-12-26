#include "rtp_llm/cpp/models/elastic_ep_manager/ElasticEPManager.h"

namespace rtp_llm {
std::vector<int> tensorToVector(const torch::Tensor& tensor, size_t size) {
    auto             tensor_cpu = tensor.device().is_cuda() ? tensor.cpu() : tensor;
    std::vector<int> vec(size, 0);
    for (size_t i = 0; i < tensor_cpu.size(0); ++i) {
        vec[i] = tensor_cpu[i].item<int>();
    }
    return vec;
}

void ElasticEPManager::query_deepep_mask_buffer() {
    // printf("ElasticEPManager::query_deepep_buffer called...\n");
    py::gil_scoped_acquire acquire;
    py::module             deepep         = py::module::import("rtp_llm.models_py.distributed.deepep_wrapper");
    py::object             deepep_wrapper = deepep.attr("get_deepep_wrapper")();

    torch::Tensor mask_buffer = deepep_wrapper.attr("query_mask_buffer")().cast<torch::Tensor>();
    active_ranks_             = tensorToVector(mask_buffer, ep_size_);
    for (size_t i = 0; i < active_ranks_.size(); ++i) {
        printf("ElasticEPManager: active_ranks_[%zu] = %d\n", i, active_ranks_[i]);
    }
}

}  // namespace rtp_llm