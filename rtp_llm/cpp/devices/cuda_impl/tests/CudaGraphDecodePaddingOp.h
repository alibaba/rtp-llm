#pragma once
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace cuda_graph {
using namespace rtp_llm;
using CudaGraphRunnerPtr = CudaGraphRunner*;
class CudaGraphDecodePaddingOp: public torch::jit::CustomClassHolder {
public:
    void init(py::object       py_instance,
              int64_t          hidden_size,
              int64_t          max_seq_len,
              int64_t          tokens_per_block,
              int64_t          kv_block_offset,
              std::vector<int> decode_capture_batch_sizes);

    int getCurrentRealGraphSize();

    CudaGraphRunnerPtr createCudaGraphRunner(py::object       py_instance,
                                             int64_t          hidden_size,
                                             int64_t          max_seq_len,
                                             int64_t          tokens_per_block,
                                             int64_t          kv_block_offset,
                                             std::vector<int> decode_capture_batch_sizes);

    PyModelOutputs forward(PyModelInputs inputs) {
        return cuda_graph_runner_->forward(inputs);
    }

    ~CudaGraphDecodePaddingOp() {
        RTP_LLM_CHECK_WITH_INFO(cuda_graph_runner_ != nullptr, "cuda_graph_runner_ can not be nullptr");
        delete cuda_graph_runner_;
    }

private:
    CudaGraphRunnerPtr cuda_graph_runner_;
};
}  // namespace cuda_graph
