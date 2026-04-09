#pragma once

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_dispatcher.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_py_model_inputs.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {

/// Shared capture/replay state and helpers for prefill vs decode CUDA graph runners.
class CudaGraphRunnerShared {
protected:
    CudaGraphRunnerShared(CudaGraphRunnerBase& owner, GraphParams graph_params, size_t max_bs, bool is_prefill_capture);

    CudaGraphRunnerBase&                      owner_;
    GraphParams                               graph_params_;
    cuda_graph::GraphStream                   capture_stream_;
    size_t                                    max_bs_{1};
    int                                       max_num_token_{1};
    std::unordered_map<int, GraphInstance>    graph_instances_;
    cuda_graph::CudaGraphCaptureDispatcher    capture_dispatcher_;
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
};

class CudaGraphPrefillRunner: public CudaGraphRunnerBase, private CudaGraphRunnerShared {
public:
    explicit CudaGraphPrefillRunner(GraphParams graph_params, py::object py_instance);

    ~CudaGraphPrefillRunner() override {
        RTP_LLM_LOG_INFO("Release CudaGraphPrefillRunner .....");
        py::gil_scoped_acquire gil;
        py_instance_.release();
        RTP_LLM_LOG_INFO("Release CudaGraphPrefillRunner Successfully");
    }

    void           capturePrefill();
    void           capturePrefillOneSeqLen(int seq_len);
    void           replayPrefill(int seq_len);
    void           prepareInputs(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor);
    bool           canRun(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) override;
    PyModelOutputs forward(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) override;
    void           initCapture() override;
    int            getCurrentRealGraphSize(const BatchDescriptor& batch_descriptor) const override;

    void setPositionEncoding(torch::Tensor position_encoding) override;
    void setTokenTypeEmbedding(torch::Tensor token_type_embedding) override;
    void setInputEmbeddingScalar(float input_embedding_scalar) override;
};

class CudaGraphDecodeRunner: public CudaGraphRunnerBase, private CudaGraphRunnerShared {
public:
    explicit CudaGraphDecodeRunner(GraphParams graph_params, py::object py_instance);

    ~CudaGraphDecodeRunner() override {
        RTP_LLM_LOG_INFO("Release CudaGraphDecodeRunner .....");
        py::gil_scoped_acquire gil;
        py_instance_.release();
        RTP_LLM_LOG_INFO("Release CudaGraphDecodeRunner Successfully");
    }

    void           captureDecode();
    void           captureDecodeOneBatchSize(int bs);
    void           replayDecode(int bs);
    void           prepareInputs(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor);
    bool           canRun(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) override;
    PyModelOutputs forward(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) override;
    void           initCapture() override;
    int            getCurrentRealGraphSize(const BatchDescriptor& batch_descriptor) const override;

    void setPositionEncoding(torch::Tensor position_encoding) override;
    void setTokenTypeEmbedding(torch::Tensor token_type_embedding) override;
    void setInputEmbeddingScalar(float input_embedding_scalar) override;
};

/// Factory namespace for tests and internal use (no instances).
struct CudaGraphRunner {
    static CudaGraphRunnerBase* createForPrefill(py::object py_instance, GraphParams params);
    static CudaGraphRunnerBase* createForDecode(py::object py_instance, GraphParams params);
};

}  // namespace rtp_llm
