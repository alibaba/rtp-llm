#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include "ATen/core/TensorBody.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphUtils.h"
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include "rtp_llm/cpp/devices/GraphBase.h"

namespace py = pybind11;
namespace rtp_llm {
class CudaGraphRunner: public GraphBase {
public:
    CudaGraphRunner(const DeviceInitParams& params,
                    py::object              py_instance,
                    int                     kv_cache_block_offset,
                    DeviceBase*             device,
                    bool                    is_prefill_cuda_graph_mode = false):
        GraphBase(std::move(py_instance)),
        enable_cuda_graph_(params.hw_kernel_config.enable_cuda_graph),
        is_prefill_cuda_graph_mode_(is_prefill_cuda_graph_mode),
        capture_stream_(at::cuda::getStreamFromPool()),
        enable_cuda_graph_debug_mode_(params.hw_kernel_config.enable_cuda_graph_debug_mode),
        hidden_size_(params.hidden_size),
        max_seq_len_(params.max_seq_len),
        seq_size_per_block_(params.tokens_per_block),
        kv_cache_block_offset_(kv_cache_block_offset),
        device_(device) {
        py::gil_scoped_acquire gil;
        if (!py_instance_ || py_instance_.is_none()) {
            throw std::runtime_error("CudaGraphRunner constructor: Python instance is null or none.");
        }
        if (is_prefill_cuda_graph_mode) {
            concurrency_limit_ = params.fifo_scheduler_config.max_context_batch_size;
        } else {
            concurrency_limit_ = params.concurrency_config.concurrency_limit;
        }
        py_forward_method_     = py_instance_.attr("forward");
        py_fill_params_method_ = py_instance_.attr("fill_params");

        RTP_LLM_LOG_INFO("Initialize CudaGraphRunner with parameters below: \n \
            enable_cuda_graph_: %d, concurrency_limit_: %d, enable_cuda_graph_debug_mode_: %d, hidden_size_: %d, max_seq_len_: %d, seq_size_per_block_: %d, kv_cache_block_offset_: %d, is_prefill_cuda_graph_mode_: %d",
                         enable_cuda_graph_,
                         concurrency_limit_,
                         enable_cuda_graph_debug_mode_,
                         hidden_size_,
                         max_seq_len_,
                         seq_size_per_block_,
                         kv_cache_block_offset_,
                         is_prefill_cuda_graph_mode_);
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
    void           prepareInputs(PyModelInputs& inputs);
    bool           canRun(PyModelInputs& inputs);
    void           replayDecode(int bs);
    void           replayPrefill(int seq_len);
    void           initCapture() override;
    void           initKernelInternalMemory();
    int            getCurrentRealGraphBs();
    PyModelOutputs forward(PyModelInputs& inputs) override;
    py::object     normalForward(PyModelInputs& inputs);
    void           setPositionEncoding(torch::Tensor position_encoding) override;
    void           setTokenTypeEmbedding(torch::Tensor token_type_embedding) override;
    void           setInputEmbeddingScalar(float input_embedding_scalar) override;
    void           setModelDataType(caffe2::TypeMeta data_type) override;

private:
    void                 copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor);
    std::vector<int>     getDecodeBatchSizesToCapture(int concurrency_limit);
    std::vector<int>     getPrefillSequenceLengthsToCapture();
    bool                 tryGetRealGraphBatchSize(PyModelInputs& inputs);
    void                 extractValidHiddenStates(torch::Tensor& outputs,
                                                  torch::Tensor& inputs,
                                                  torch::Tensor& input_lengths,
                                                  int32_t        total_valid_tokens);
    void                 initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs);
    void                 initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token);
    py::object           py_forward_method_;
    py::object           py_fill_params_method_;
    bool                 enable_cuda_graph_{false};
    bool                 is_prefill_cuda_graph_mode_{false};
    int                  concurrency_limit_{32};
    at::cuda::CUDAStream capture_stream_;
    bool                 enable_cuda_graph_debug_mode_{false};
    int                  hidden_size_;
    size_t               max_bs_{1};
    int                  num_tokens_per_bs_{1};
    int                  max_num_token_{1};
    int                  current_batch_size_{1};
    // for decode
    int current_real_graph_bs_{1};
    // for prefill
    int              current_real_seq_len_{1};
    int              max_seq_len_{0};
    int              seq_size_per_block_{0};
    int              kv_cache_block_offset_{0};
    int              seq_len_sum_{0};
    std::vector<int> capture_range_;
    // capture seqLen -> GraphInstance (prefill)
    // batch_size -> GraphInstance (decode)
    std::unordered_map<int, GraphInstance> graph_instances_;
    CaptureMemoryHold                      capture_mem_hold_;
    torch::Tensor                          position_encoding_;
    torch::Tensor                          token_type_embedding_;
    float                                  input_embedding_scalar_;
    caffe2::TypeMeta                       model_data_type_;

public:
    DeviceBase* device_{nullptr};
};
}  // namespace rtp_llm
