#pragma once

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_py_model_inputs.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {

/// CUDA graph capture/replay: shared state, embedding/position hooks, and capture helpers.
/// Prefill vs decode behavior is implemented by CudaGraphPrefillRunner / CudaGraphDecodeRunner.
class CudaGraphRunnerBase {
public:
    virtual ~CudaGraphRunnerBase();

    virtual void           initCapture()                                                           = 0;
    virtual PyModelOutputs forward(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) = 0;
    virtual bool           canRun(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor)  = 0;
    /// Decode: captured batch key; prefill: captured seq_len key (for tests / diagnostics).
    virtual int getCurrentRealGraphSize(const BatchDescriptor& batch_descriptor) const = 0;

    void setPositionEncoding(torch::Tensor position_encoding);
    void setTokenTypeEmbedding(torch::Tensor token_type_embedding);
    void setInputEmbeddingScalar(float input_embedding_scalar);

    py::object py_instance_;

protected:
    CudaGraphRunnerBase(py::object py_instance, GraphParams graph_params, size_t max_bs, bool is_prefill_capture);

    GraphParams                               graph_params_;
    cuda_graph::GraphStream                   capture_stream_;
    size_t                                    max_bs_{1};
    int                                       max_num_token_{1};
    std::unordered_map<int, GraphInstance>    graph_instances_;
    cuda_graph::CudaGraphCapturePyModelInputs capture_py_model_inputs_;
    torch::Tensor                             position_encoding_;
    torch::Tensor                             token_type_embedding_;
    float                                     input_embedding_scalar_{};
    cuda_graph::GraphPoolHandle               shared_graph_pool_{};
    torch::Event                              forward_event_ = cuda_graph::makeGraphEvent();

    py::object py_forward_method_;
    py::object py_attn_pyobj_method_;

    const bool is_prefill_capture_;

    void              captureOneGraphInstance(int key, const char* key_type);
    void              replayGraph(int key);
    void              replayAndSyncCheck(int key, const char* key_type);
    void              prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens);
    CaptureMemoryHold createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count);
    void              logCudaGraphPoolMemory(const char* phase);

    /// When enable_cuda_graph: pool, dispatcher, template buffers, first forward, allocate outputs, log
    /// "before_capture".
    void initCapturePreamble();

    virtual void buildCaptureDispatcher() = 0;
};

}  // namespace rtp_llm
