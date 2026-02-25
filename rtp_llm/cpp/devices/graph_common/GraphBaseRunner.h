#pragma once

#include <functional>
#include <unordered_map>
#include <vector>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "c10/core/TensorOptions.h"
#include "rtp_llm/cpp/devices/DeviceData.h"
#include "rtp_llm/cpp/devices/graph_common/GraphCommonTypes.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {

struct GraphBackendCallbacks {
    c10::DeviceType event_device_type{c10::DeviceType::CUDA};
    const char*     debug_file_prefix{"graph_tokens"};

    std::function<void(const torch::Tensor&, torch::Tensor&, size_t)> memcpy_async;
    std::function<void()>                                             device_synchronize;
    std::function<void(torch::Event&)>                                record_forward_event;
    std::function<void()>                                             synchronize_forward_stream;
    std::function<void(const std::function<void()>&)>                 with_capture_stream;

    std::function<bool(py::object, bool)>             should_skip_decode_capture;
    std::function<void(py::object, int, const char*)> before_capture_stream;
    std::function<void(py::object)>                   enter_capture;
    std::function<void(py::object)>                   exit_capture;

    std::function<int(int, int)>                                kv_block_cols;
    std::function<torch::Tensor(int, const at::TensorOptions&)> sequence_lengths_plus_one_tensor;
};

class GraphBaseRunner {
public:
    GraphBaseRunner(const DeviceInitParams& params,
                    py::object              py_instance,
                    c10::ScalarType         model_data_type,
                    int                     num_tokens_per_bs,
                    bool                    is_prefill_graph_mode,
                    GraphBackendCallbacks   backend_callbacks);

    void           initCapture();
    PyModelOutputs forward(PyModelInputs& inputs);
    bool           canRun(PyModelInputs& inputs);
    py::object     normalForward(PyModelInputs& inputs);
    void           captureDecode();
    void           capturePrefill();
    void           captureDecodeOneBatchSize(int bs);
    void           capturePrefillOneSeqLen(int seq_len);
    void           prepareInputs(PyModelInputs& inputs);
    void           replayGraph(int key);
    void           replayDecode(int bs);
    void           replayPrefill(int seq_len);
    void           replayAndSyncCheck(int key, const char* key_type);

    void setPositionEncoding(torch::Tensor position_encoding);
    void setTokenTypeEmbedding(torch::Tensor token_type_embedding);
    void setInputEmbeddingScalar(float input_embedding_scalar);
    void setMaxPrefillGraphLen(int max_prefill_graph_len);
    int  getCurrentRealGraphBs() const;

private:
    void              captureOneGraphInstance(int key, const char* key_type);
    void              prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens);
    CaptureMemoryHold createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count);
    void              initKernelInternalMemory();
    void              copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor);
    std::vector<int>  getDecodeBatchSizesToCapture() const;
    std::vector<int>  getPrefillSequenceLengthsToCapture() const;
    void              tryGetRealGraphDecodeBatchSize(PyModelInputs& inputs);
    void              tryGetRealGraphPrefillSeqLen(PyModelInputs& inputs);
    void              initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs);
    void              initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token);
    void              initCaptureAttentionInputsPost();

private:
    GraphBackendCallbacks                  backend_callbacks_;
    py::object                             py_instance_;
    py::object                             py_forward_method_;
    py::object                             py_attn_pyobj_method_;
    bool                                   enable_graph_{false};
    bool                                   is_prefill_graph_mode_{false};
    bool                                   enable_graph_debug_mode_{false};
    size_t                                 max_bs_{1};
    int                                    num_tokens_per_bs_{1};
    int                                    max_num_token_{1};
    int                                    max_prefill_graph_len_{160};
    int                                    max_seq_len_{0};
    int                                    seq_size_per_block_{0};
    int                                    hidden_size_{0};
    GraphExecutionState                    state_;
    std::vector<int>                       capture_range_;
    std::vector<int>                       prefill_capture_seq_lens_;
    std::vector<int>                       decode_capture_batch_sizes_;
    std::unordered_map<int, GraphInstance> graph_instances_;
    CaptureMemoryHold                      capture_mem_hold_;
    torch::Tensor                          position_encoding_;
    torch::Tensor                          token_type_embedding_;
    float                                  input_embedding_scalar_{1.0f};
    c10::ScalarType                        model_data_type_;
    at::TensorOptions                      options_device_int32_;
    at::TensorOptions                      options_cpu_int32_;
    at::TensorOptions                      options_device_float_;
    torch::Event                           forward_event_;
};

}  // namespace rtp_llm
