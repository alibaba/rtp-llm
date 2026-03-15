#pragma once

#include "ATen/core/TensorBody.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

#if USING_ROCM
#include <ATen/hip/HIPGraph.h>
#include "rtp_llm/cpp/devices/GraphRunnerDeviceShims.h"
#else
#include <ATen/cuda/CUDAGraph.h>
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#include <string>

using namespace torch_ext;

namespace rtp_llm {

// Debug utilities for printing tensor information
void printTensorInfo(const std::string& name, const torch::Tensor& tensor, int max_print_size = 20);
void debugPrintPyModelInputs(const PyModelInputs& inputs);

}  // namespace rtp_llm

class CaptureMemoryHold {
public:
    void setHiddenStates(at::Tensor hidden_states) {
        decoder_layer_hidden_states_ = hidden_states;
    };

    CaptureMemoryHold() {}

    CaptureMemoryHold(at::Tensor hidden_states, PyModelInputs& inputs, bool is_embedding):
        decoder_layer_hidden_states_(hidden_states) {
        py_model_inputs_.attention_inputs.input_lengths    = inputs.attention_inputs.input_lengths;
        py_model_inputs_.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths;
        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device =
            inputs.attention_inputs.kv_cache_kernel_block_id_device;
        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host =
            inputs.attention_inputs.kv_cache_kernel_block_id_host;
        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group =
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group;
        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group =
            inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group;
        py_model_inputs_.attention_inputs.kv_cache_layer_to_group = inputs.attention_inputs.kv_cache_layer_to_group;
        py_model_inputs_.attention_inputs.prefix_lengths          = inputs.attention_inputs.prefix_lengths;
        py_model_inputs_.input_ids                                = inputs.input_ids;

        // for spec
        py_model_inputs_.input_hiddens                            = inputs.input_hiddens;
        py_model_inputs_.attention_inputs.cu_seqlens              = inputs.attention_inputs.cu_seqlens;
        py_model_inputs_.attention_inputs.cu_kv_seqlens           = inputs.attention_inputs.cu_kv_seqlens;
        py_model_inputs_.attention_inputs.padding_offset          = inputs.attention_inputs.padding_offset;
        py_model_inputs_.attention_inputs.is_prefill              = is_embedding;
        py_model_inputs_.attention_inputs.dtype                   = inputs.attention_inputs.dtype;
        py_model_inputs_.attention_inputs.context_total_kv_length = inputs.attention_inputs.context_total_kv_length;

        py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params =
            inputs.attention_inputs.prefill_cuda_graph_copy_params;
        py_model_inputs_.bert_embedding_inputs                      = inputs.bert_embedding_inputs;
        py_model_inputs_.attention_inputs.is_s_padded               = inputs.attention_inputs.is_s_padded;
        py_model_inputs_.attention_inputs.decode_cu_seqlens_d       = inputs.attention_inputs.decode_cu_seqlens_d;
        py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d = inputs.attention_inputs.sequence_lengths_plus_1_d;
    }

public:
    py::object    attn_pyobj_{py::none()};
    at::Tensor    decoder_layer_hidden_states_;
    PyModelInputs py_model_inputs_;
};

class GraphInstance {
public:
    at::cuda::CUDAGraph graph_;
    CaptureMemoryHold   mem_hold_;
};

#if USING_ROCM

class CudaGraphStreamLife {
public:
    CudaGraphStreamLife(at::hip::HIPStream capture_stream):
        origin_stream_(at::hip::getCurrentHIPStream(at::hip::current_device())) {
        at::hip::setCurrentHIPStream(capture_stream);
        RTP_LLM_LOG_INFO("Set HIP Stream: capture_stream -> %d, origin_stream -> %d",
                         capture_stream.stream(),
                         origin_stream_.stream());
    }
    ~CudaGraphStreamLife() {
        at::hip::setCurrentHIPStream(origin_stream_);
    }

    CudaGraphStreamLife(const CudaGraphStreamLife&)            = delete;
    CudaGraphStreamLife& operator=(const CudaGraphStreamLife&) = delete;
    CudaGraphStreamLife(CudaGraphStreamLife&&)                 = delete;
    CudaGraphStreamLife& operator=(CudaGraphStreamLife&&)      = delete;

private:
    at::hip::HIPStream origin_stream_;
};

#else

class CudaGraphStreamLife {
public:
    CudaGraphStreamLife(at::cuda::CUDAStream capture_stream):
        origin_stream_(at::cuda::getCurrentCUDAStream(at::cuda::current_device())) {
        at::cuda::setCurrentCUDAStream(capture_stream);
        RTP_LLM_LOG_INFO("Set Cuda Stream: capture_stream -> %d, origin_stream -> %d",
                         capture_stream.stream(),
                         origin_stream_.stream());
    }
    ~CudaGraphStreamLife() {
        at::cuda::setCurrentCUDAStream(origin_stream_);
    }

    CudaGraphStreamLife(const CudaGraphStreamLife&)            = delete;
    CudaGraphStreamLife& operator=(const CudaGraphStreamLife&) = delete;
    CudaGraphStreamLife(CudaGraphStreamLife&&)                 = delete;
    CudaGraphStreamLife& operator=(CudaGraphStreamLife&&)      = delete;

private:
    at::cuda::CUDAStream origin_stream_;
};

#endif

class CudaGraphCaptureGuard {
public:
#if USING_ROCM
    explicit CudaGraphCaptureGuard(rtp_llm::graph_runner::GraphNcclCaptureContext* ctx = nullptr): ctx_(ctx) {
        rtp_llm::graph_runner::enter_graph_capture(ctx_);
    }
#else
    CudaGraphCaptureGuard() {
        rtp_llm::CaptureCheck::in_cuda_graph_capture = true;
    }
#endif

    ~CudaGraphCaptureGuard() {
#if USING_ROCM
        rtp_llm::graph_runner::exit_graph_capture(ctx_);
#else
        rtp_llm::CaptureCheck::in_cuda_graph_capture = false;
#endif
    }

    CudaGraphCaptureGuard(const CudaGraphCaptureGuard&)            = delete;
    CudaGraphCaptureGuard& operator=(const CudaGraphCaptureGuard&) = delete;
    CudaGraphCaptureGuard(CudaGraphCaptureGuard&&)                 = delete;
    CudaGraphCaptureGuard& operator=(CudaGraphCaptureGuard&&)      = delete;

private:
#if USING_ROCM
    rtp_llm::graph_runner::GraphNcclCaptureContext* ctx_{nullptr};
#endif
};
