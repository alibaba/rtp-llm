#pragma once
#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaGraphOpUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace cuda_graph {
using namespace rtp_llm;

class CudaGraphPrefillOp: public torch::jit::CustomClassHolder {
public:
    void init(py::object       py_instance,
              int64_t          max_context_batch_size,
              int64_t          hidden_size,
              int64_t          max_seq_len,
              int64_t          tokens_per_block,
              int64_t          max_prefill_cuda_graph_len,
              std::vector<int> prefill_capture_seq_lens);

    int getCurrentRealGraphSize();

    PyModelOutputs forward(PyModelInputs inputs) {
        return cuda_graph_runner_->forward(inputs);
    }

    ~CudaGraphPrefillOp() {
        RTP_LLM_CHECK_WITH_INFO(cuda_graph_runner_ != nullptr, "cuda_graph_runner_ can not be nullptr");
        delete cuda_graph_runner_;
    }

private:
    CudaGraphRunnerPtr cuda_graph_runner_;
};
}  // namespace cuda_graph
