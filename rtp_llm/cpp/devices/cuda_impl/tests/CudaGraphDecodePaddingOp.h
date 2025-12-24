#pragma once
#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaGraphOpUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace cuda_graph {
using namespace rtp_llm;

class CudaGraphDecodePaddingOp: public torch::jit::CustomClassHolder {
public:
    void init(py::object       py_instance,
              int64_t          hidden_size,
              int64_t          max_seq_len,
              int64_t          tokens_per_block,
              std::vector<int> decode_capture_batch_sizes);

    int getCurrentRealGraphSize();

    PyModelOutputs forward(PyModelInputs& inputs) {
        bool executed = false;
        return cuda_graph_runner_->forward(inputs, executed);
    }

    ~CudaGraphDecodePaddingOp() {
        RTP_LLM_CHECK_WITH_INFO(cuda_graph_runner_ != nullptr, "cuda_graph_runner_ can not be nullptr");
        delete cuda_graph_runner_;
    }

private:
    CudaGraphRunnerPtr cuda_graph_runner_;
};
}  // namespace cuda_graph
