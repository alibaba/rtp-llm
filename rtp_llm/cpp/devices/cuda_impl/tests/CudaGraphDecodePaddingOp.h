#pragma once
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include <pybind11/pybind11.h>
#include "rtp_llm/cpp/utils/PyUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace cuda_graph {
using namespace rtp_llm;
using CudaGraphRunnerPtr = std::shared_ptr<CudaGraphRunner>;
class CudaGraphDecodePaddingOp: public torch::jit::CustomClassHolder {
public:
    void init(py::object py_instance);

    int getCurrentRealGraphSize();

    CudaGraphRunnerPtr createCudaGraphRunner(py::object py_instance);

    PyModelOutputs forward(PyModelInputs inputs) {
        return cuda_graph_runner_->forward(inputs);
    }
    PyModelInputs
    buildInputs(int64_t batch_size, int64_t max_seq_len, int64_t num_tokens_per_bs, int64_t seq_size_per_block);

private:
    CudaGraphRunnerPtr cuda_graph_runner_;
};
}  // namespace cuda_graph
