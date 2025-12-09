#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <unordered_map>
#include "ATen/core/TensorBody.h"
#include "c10/core/DeviceType.h"
#include "c10/core/TensorOptions.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/devices/GraphUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
// Include graph headers based on platform
#if USING_ROCM
#include <ATen/hip/HIPGraph.h>
#else
#include <ATen/cuda/CUDAGraph.h>
#endif

namespace py = pybind11;
namespace rtp_llm {

using namespace torch_ext;

class GraphBase {
public:
    GraphBase(py::object py_instance): py_instance_(std::move(py_instance)) {}
    virtual ~GraphBase() {}

    // Public interface methods (implemented in GraphBase.cc)
    void           initCapture();
    PyModelOutputs forward(PyModelInputs& inputs);
    void           setPositionEncoding(torch::Tensor position_encoding);
    void           setTokenTypeEmbedding(torch::Tensor token_type_embedding);
    void           setInputEmbeddingScalar(float input_embedding_scalar);
    void           setModelDataType(caffe2::TypeMeta data_type);
    py::object     normalForward(PyModelInputs& inputs);

    // Decode methods (implemented in GraphBaseDecode.cc)
    void             captureDecode();
    void             captureDecodeOneBatchSize(int bs);
    std::vector<int> getDecodeBatchSizesToCapture();
    void             replayDecode(int bs);
    void             tryGetRealGraphDecodeBatchSize(PyModelInputs& inputs);

    // Prefill methods (implemented in GraphBasePrefill.cc)
    void             capturePrefill();
    void             capturePrefillOneSeqLen(int seq_len);
    std::vector<int> getPrefillSequenceLengthsToCapture();
    void             replayPrefill(int seq_len);
    void             tryGetRealGraphPrefillSeqLen(PyModelInputs& inputs);

    // Common methods (implemented in GraphBase.cc)
    void              prepareInputs(PyModelInputs& inputs);
    bool              canRun(PyModelInputs& inputs);
    void              prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens);
    CaptureMemoryHold createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count);
    void              initKernelInternalMemory();
    void              copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor);
    void              initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs);
    void              initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token);
    void              initCaptureAttentionInputsPost();
    void              replayAndSyncCheck(int key, const char* key_type);
    int               getCurrentRealGraphBs();
    void              setMaxPrefillGraphLen(int max_prefill_graph_len);

    // GPU memory copy and sync methods (must be implemented by subclass)
    virtual void optimizedCopy(const torch::Tensor& src, torch::Tensor& dst, size_t size) = 0;
    virtual void syncDevice()                                                             = 0;

protected:
    // Capture methods (implemented in GraphBase.cc)
    void captureOneGraphInstance(int key, const char* key_type);
    void warmupForCapture(int key, const char* key_type);

    // Platform-specific methods (must be implemented by subclass)
    virtual void replayGraphImpl(int key)                           = 0;
    virtual void performGraphCapture(int key, const char* key_type) = 0;

    // Python interface
    py::object py_instance_;
    py::object py_forward_method_;
    py::object py_fill_params_method_;

    // Configuration flags
    bool enable_graph_{false};
    bool is_prefill_graph_mode_{false};
    bool enable_graph_debug_mode_{false};

    // Size configuration
    size_t max_bs_{1};
    int    num_tokens_per_bs_{1};
    int    max_num_token_{1};
    int    max_prefill_graph_len_{160};
    int    max_seq_len_{0};
    int    seq_size_per_block_{0};
    int    kv_cache_block_offset_{0};

    // State management
    GraphState       state_;
    std::vector<int> capture_range_;
    std::vector<int> prefill_capture_seq_lens_;    // Pre-configured sequence lengths from Python
    std::vector<int> decode_capture_batch_sizes_;  // Pre-configured batch sizes from Python

    // Memory management
    CaptureMemoryHold capture_mem_hold_;
    // key -> CaptureMemoryHold map (key is batch_size for decode, seq_len for prefill)
    std::unordered_map<int, CaptureMemoryHold> graph_mem_holds_;
    // key -> CUDAGraph map (works for both CUDA and HIP via PyTorch)
    std::unordered_map<int, at::cuda::CUDAGraph> cuda_graphs_;

    // Tensor options (to be initialized by subclass based on device type)
    at::TensorOptions options_device_int32_;  // Device-specific int32 options (CUDA/HIP)
    at::TensorOptions options_cpu_int32_;
    at::TensorOptions options_device_float_;  // Device-specific float options (CUDA/HIP)

    // Embedding related
    torch::Tensor    position_encoding_;
    torch::Tensor    token_type_embedding_;
    float            input_embedding_scalar_;
    caffe2::TypeMeta model_data_type_;

    // Capture stream (to be set by subclass as void pointer to support both CUDA and HIP)
    void* capture_stream_{nullptr};
};

}  // namespace rtp_llm
