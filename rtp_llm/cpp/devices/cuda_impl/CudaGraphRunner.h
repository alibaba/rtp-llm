#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include "ATen/core/TensorBody.h"
#include "c10/core/DeviceType.h"
#include "c10/core/TensorOptions.h"
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
                    c10::ScalarType         model_data_type,
                    int                     num_tokens_per_bs,
                    bool                    is_prefill_cuda_graph_mode = false):
        GraphBase(std::move(py_instance)),
        enable_cuda_graph_(params.hw_kernel_config.enable_cuda_graph),
        is_prefill_cuda_graph_mode_(is_prefill_cuda_graph_mode),
        capture_stream_(at::cuda::getStreamFromPool(true)),
        enable_cuda_graph_debug_mode_(params.hw_kernel_config.enable_cuda_graph_debug_mode),
        num_tokens_per_bs_(num_tokens_per_bs),
        max_seq_len_(params.max_seq_len),
        seq_size_per_block_(params.tokens_per_block),
        hidden_size_(params.hidden_size),
        prefill_capture_seq_lens_(params.hw_kernel_config.prefill_capture_seq_lens),
        decode_capture_batch_sizes_(params.hw_kernel_config.decode_capture_batch_sizes),
        model_data_type_(model_data_type),
        kv_cache_layer_to_group_(params.kv_cache_layer_to_group),
        kv_cache_group_num_(params.kv_cache_group_num) {
        py::gil_scoped_acquire gil;
        if (!py_instance_ || py_instance_.is_none()) {
            throw std::runtime_error("CudaGraphRunner constructor: Python instance is null or none.");
        }
        if (is_prefill_cuda_graph_mode) {
            max_bs_ = params.runtime_config.fifo_scheduler_config.max_context_batch_size;
        } else {
            max_bs_ = params.concurrency_config.concurrency_limit;
        }
        py_attn_pyobj_method_ = py_instance_.attr("prepare_fmha_impl");
        py_forward_method_    = py_instance_.attr("forward");
        options_cuda_int32_   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
        options_cpu_int32_    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
        options_cuda_float_   = torch::TensorOptions().dtype(model_data_type).device(torch::kCUDA).requires_grad(false);
        RTP_LLM_LOG_INFO("Initialize CudaGraphRunner with parameters below: \n \
            enable_cuda_graph_: %d, max_bs_: %d, enable_cuda_graph_debug_mode_: %d, max_seq_len_: %d, seq_size_per_block_: %d, \
            hidden_size_: %d, num_tokens_per_bs_: %d, is_prefill_cuda_graph_mode_: %d",
                         enable_cuda_graph_,
                         max_bs_,
                         enable_cuda_graph_debug_mode_,
                         max_seq_len_,
                         seq_size_per_block_,
                         hidden_size_,
                         num_tokens_per_bs_,
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
    void           replayGraph(int key);
    void           replayDecode(int bs);
    void           replayPrefill(int seq_len);
    void           setMaxPrefillCudaGraphLen(int max_prefill_cuda_graph_len);
    int            getCurrentRealGraphBs();
    PyModelOutputs forward(PyModelInputs& inputs) override;
    void           setModelDataType(caffe2::TypeMeta data_type) override;
    void           setPositionIdLenFactor(int position_id_len_factor) override;
    void           setNeedComboPositionIds(bool need_combo_position_ids) override;
    void           initCapture() override;

private:
    // Common capture logic for both prefill and decode
    void captureOneGraphInstance(int key, const char* key_type);
    // Common replay and sync check logic
    void replayAndSyncCheck(int key, const char* key_type);
    // Common input preparation logic for capture
    void prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens);
    // Common memory hold creation logic
    CaptureMemoryHold createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count);
    void              initKernelInternalMemory();
    void              setPositionEncoding(torch::Tensor position_encoding) override;
    void              setTokenTypeEmbedding(torch::Tensor token_type_embedding) override;
    void              setInputEmbeddingScalar(float input_embedding_scalar) override;

private:
    void                 copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor);
    std::vector<int>     getDecodeBatchSizesToCapture();
    std::vector<int>     getPrefillSequenceLengthsToCapture();
    void                 tryGetRealGraphDecodeBatchSize(PyModelInputs& inputs);
    void                 tryGetRealGraphPrefillSeqLen(PyModelInputs& inputs);
    void                 initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs);
    void                 initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token);
    void                 initCaptureAttentionInputsPost();
    py::object           py_forward_method_;
    py::object           py_attn_pyobj_method_;
    bool                 enable_cuda_graph_{false};
    int                  position_id_len_factor_{1};  // batch_size * position_id_len_factor_
    bool                 need_combo_position_ids_{false};
    bool                 is_prefill_cuda_graph_mode_{false};
    at::cuda::CUDAStream capture_stream_;
    bool                 enable_cuda_graph_debug_mode_{false};
    size_t               max_bs_{1};
    int                  num_tokens_per_bs_{1};
    int                  max_num_token_{1};
    int                  max_perfill_cuda_graph_len_{160};
    int                  max_seq_len_{0};
    int                  seq_size_per_block_{0};
    int                  hidden_size_{0};
    CudaGraphState       state_;
    std::vector<int>     capture_range_;
    std::vector<int>     prefill_capture_seq_lens_;    // Pre-configured sequence lengths from Python
    std::vector<int>     decode_capture_batch_sizes_;  // Pre-configured batch sizes from Python
    // capture seqLen -> GraphInstance (prefill)
    // batch_size -> GraphInstance (decode)
    std::unordered_map<int, GraphInstance> graph_instances_;
    CaptureMemoryHold                      capture_mem_hold_;
    torch::Tensor                          position_encoding_;
    torch::Tensor                          token_type_embedding_;
    float                                  input_embedding_scalar_;
    c10::ScalarType                        model_data_type_;
    at::TensorOptions                      options_cuda_int32_;
    at::TensorOptions                      options_cpu_int32_;
    at::TensorOptions                      options_cuda_float_;

    std::vector<int32_t> kv_cache_layer_to_group_;
    int32_t              kv_cache_group_num_ = 0;

    // event to record forward done
    torch::Event forward_event_ = torch::Event(torch::kCUDA);
};
}  // namespace rtp_llm
