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

class CudaGraphRunner: public CudaGraphRunnerBase {
public:
    CudaGraphRunner(const GraphParams& graph_params, py::object py_instance):
        CudaGraphRunnerBase(std::move(py_instance)),
        graph_params_(graph_params),
        capture_stream_(cuda_graph::graphGetStreamFromPool(true)),
        max_bs_(graph_params.is_prefill_cuda_graph_mode ? graph_params.max_context_batch_size :
                                                          graph_params.concurrency_limit) {
        py::gil_scoped_acquire gil;
        if (!py_instance_ || py_instance_.is_none()) {
            throw std::runtime_error("CudaGraphRunner constructor: Python instance is null or none.");
        }
        if (graph_params_.kernel_tokens_per_block <= 0) {
            throw std::runtime_error("CudaGraphRunner constructor: kernel_tokens_per_block must be > 0.");
        }
        py_attn_pyobj_method_ = py_instance_.attr("prepare_fmha_impl");
        py_forward_method_    = py_instance_.attr("forward");
        RTP_LLM_LOG_INFO("Initialize CudaGraphRunner with parameters below: \n \
            enable_cuda_graph: %d, max_bs_: %zu, enable_cuda_graph_debug_mode: %d, max_seq_len: %d, kernel_tokens_per_block: %d, \
            hidden_size: %zu, num_tokens_per_bs: %d, is_prefill_cuda_graph_mode: %d, is_target_verify: %d",
                         graph_params_.enable_cuda_graph,
                         max_bs_,
                         graph_params_.enable_cuda_graph_debug_mode,
                         graph_params_.max_seq_len,
                         graph_params_.kernel_tokens_per_block,
                         graph_params_.hidden_size,
                         graph_params_.num_tokens_per_bs,
                         graph_params_.is_prefill_cuda_graph_mode,
                         graph_params_.is_target_verify);
    }

    ~CudaGraphRunner() {
        RTP_LLM_LOG_INFO("Release CudaGraphRunner .....");
        py::gil_scoped_acquire gil;
        py_instance_.release();
        RTP_LLM_LOG_INFO("Release CudaGraphRunner Successfully");
    }
    void           captureDecode();
    void           capturePrefill();
    void           captureDecodeOneBatchSize(int bs);
    void           capturePrefillOneSeqLen(int seq_len);
    void           prepareInputs(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor);
    bool           canRun(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) override;
    void           replayGraph(int key);
    void           replayDecode(int bs);
    void           replayPrefill(int seq_len);
    int            getCurrentRealGraphBs(const BatchDescriptor& batch_descriptor) const;
    PyModelOutputs forward(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) override;
    void           initCapture() override;

    // Factory methods for test: take GraphParams so callers can reuse the same struct
    static CudaGraphRunner* createForPrefill(py::object py_instance, GraphParams params);
    static CudaGraphRunner* createForDecode(py::object py_instance, GraphParams params);

private:
    // Common capture logic for both prefill and decode
    void captureOneGraphInstance(int key, const char* key_type);
    // Common replay and sync check logic
    void replayAndSyncCheck(int key, const char* key_type);
    // Common input preparation logic for capture
    void prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens);
    // Common memory hold creation logic
    CaptureMemoryHold createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count);
    void              logCudaGraphPoolMemory(const char* phase);
    void              setPositionEncoding(torch::Tensor position_encoding) override;
    void              setTokenTypeEmbedding(torch::Tensor token_type_embedding) override;
    void              setInputEmbeddingScalar(float input_embedding_scalar) override;

private:
    std::vector<int> getDecodeBatchSizesToCapture();
    std::vector<int> getPrefillSequenceLengthsToCapture();
    /// Select graph key for decode; false if no captured graph can serve current_batch_size (e.g. lower_bound hit end).
    bool tryGetRealGraphDecodeBatchSize(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor);
    /// Select graph key for prefill; false if capture_range_ empty or seq_len above max captured (lower_bound hit end).
    bool        tryGetRealGraphPrefillSeqLen(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor);
    py::object  py_forward_method_;
    py::object  py_attn_pyobj_method_;
    GraphParams graph_params_;
    cuda_graph::GraphStream capture_stream_;
    size_t                  max_bs_{1};
    int                     max_num_token_{1};
    std::vector<int>        capture_range_;
    // capture seqLen -> GraphInstance (prefill)
    // batch_size -> GraphInstance (decode)
    std::unordered_map<int, GraphInstance>    graph_instances_;
    cuda_graph::CudaGraphCapturePyModelInputs capture_py_model_inputs_;
    torch::Tensor                             position_encoding_;
    torch::Tensor                             token_type_embedding_;
    float                                     input_embedding_scalar_{};
    cuda_graph::GraphPoolHandle               shared_graph_pool_{};

    // event to record forward done
    torch::Event forward_event_ = cuda_graph::makeGraphEvent();
};

}  // namespace rtp_llm
