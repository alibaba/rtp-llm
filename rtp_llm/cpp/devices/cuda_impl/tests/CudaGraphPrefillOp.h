#pragma once
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/BufferManager.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
// #include "rtp_llm/cpp/normal_engine/NormalEngine.h"
// #include "rtp_llm/cpp/models/GptModel.h"
#include <pybind11/pybind11.h>
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/OpDefsUtils.h"
namespace cuda_graph {
using namespace rtp_llm;
using CudaGraphRunnerPtr = CudaGraphRunner*;
class CudaGraphPrefillOp: public torch::jit::CustomClassHolder {
public:
    void init(py::object py_instance);

    int getCurrentRealGraphSize();

    CudaGraphRunnerPtr createCudaGraphRunner(py::object py_instance);

    void setCufmhaPadded(bool is_s_padded);

    PyModelOutputs forward(PyModelInputs inputs) {
        return cuda_graph_runner_->forward(inputs);
    }
    PyModelInputs buildInputs(int64_t batch_size,
                              int64_t max_seq_len,
                              int64_t num_tokens_per_bs,
                              int64_t seq_size_per_block,
                              bool    use_max_padded_mode = false);
    ~CudaGraphPrefillOp() {
        RTP_LLM_CHECK_WITH_INFO(cuda_graph_runner_ != nullptr, "cuda_graph_runner_ can not be nullptr");
        delete cuda_graph_runner_;
    }

private:
    CudaGraphRunnerPtr cuda_graph_runner_;
};
}  // namespace cuda_graph
