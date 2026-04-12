#pragma once

#include "ATen/core/TensorBody.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <torch/version.h>

#include <string>

namespace rtp_llm {

// Debug utilities for printing tensor information
void printTensorInfo(const std::string& name, const torch::Tensor& tensor, int max_print_size = 20);
void debugPrintPyModelInputs(const torch_ext::PyModelInputs& inputs);

}  // namespace rtp_llm

class CaptureMemoryHold {
public:
    void setHiddenStates(at::Tensor hidden_states) {
        decoder_layer_hidden_states_ = hidden_states;
    };

    CaptureMemoryHold() {}

    CaptureMemoryHold(at::Tensor hidden_states, torch_ext::PyModelInputs& inputs, bool is_embedding):
        decoder_layer_hidden_states_(hidden_states) {
        py_model_inputs_.attention_inputs.input_lengths    = inputs.attention_inputs.input_lengths;
        py_model_inputs_.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths;
        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device =
            inputs.attention_inputs.kv_cache_kernel_block_id_device;
        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host =
            inputs.attention_inputs.kv_cache_kernel_block_id_host;
        py_model_inputs_.attention_inputs.kv_cache_block_id_device = inputs.attention_inputs.kv_cache_block_id_device;
        py_model_inputs_.attention_inputs.kv_cache_block_id_host   = inputs.attention_inputs.kv_cache_block_id_host;
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
        py_model_inputs_.attention_inputs.is_prefill              = inputs.attention_inputs.is_prefill;
        py_model_inputs_.attention_inputs.is_target_verify        = inputs.attention_inputs.is_target_verify;
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
    py::object               attn_pyobj_{py::none()};
    at::Tensor               decoder_layer_hidden_states_;
    torch_ext::PyModelInputs py_model_inputs_;
};

class GraphInstance {
public:
#if (TORCH_VERSION_MAJOR > 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 8)
    explicit GraphInstance(bool keep_graph = false): graph_(keep_graph) {}
#else
    explicit GraphInstance(bool keep_graph = false): graph_() {
        (void)keep_graph;
    }
#endif
    at::cuda::CUDAGraph graph_;
    CaptureMemoryHold   mem_hold_;
};

class CudaGraphStreamLife {
public:
    explicit CudaGraphStreamLife(rtp_llm::cuda_graph::GraphStream capture_stream):
        origin_stream_(rtp_llm::cuda_graph::graphGetCurrentStream()) {
        rtp_llm::cuda_graph::graphSetCurrentStream(capture_stream);
        RTP_LLM_LOG_INFO("Set graph stream for capture. origin_stream=%p, capture_stream=%p",
                         reinterpret_cast<void*>(origin_stream_.stream()),
                         reinterpret_cast<void*>(capture_stream.stream()));
    }
    ~CudaGraphStreamLife() {
        rtp_llm::cuda_graph::graphSetCurrentStream(origin_stream_);
        RTP_LLM_LOG_INFO("Restore graph stream after capture. restored_stream=%p",
                         reinterpret_cast<void*>(origin_stream_.stream()));
    }

    CudaGraphStreamLife(const CudaGraphStreamLife&)            = delete;
    CudaGraphStreamLife& operator=(const CudaGraphStreamLife&) = delete;
    CudaGraphStreamLife(CudaGraphStreamLife&&)                 = delete;
    CudaGraphStreamLife& operator=(CudaGraphStreamLife&&)      = delete;

private:
    rtp_llm::cuda_graph::GraphStream origin_stream_;
};

class CudaGraphCaptureGuard {
public:
    explicit CudaGraphCaptureGuard(rtp_llm::cuda_graph::GraphNcclCaptureContext* ctx = nullptr): ctx_(ctx) {
        rtp_llm::cuda_graph::enter_graph_capture(ctx_);
    }

    ~CudaGraphCaptureGuard() {
        try {
            rtp_llm::cuda_graph::exit_graph_capture(ctx_);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("Exception in CudaGraphCaptureGuard destructor: %s", e.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("Unknown exception in CudaGraphCaptureGuard destructor");
        }
    }

    CudaGraphCaptureGuard(const CudaGraphCaptureGuard&)            = delete;
    CudaGraphCaptureGuard& operator=(const CudaGraphCaptureGuard&) = delete;
    CudaGraphCaptureGuard(CudaGraphCaptureGuard&&)                 = delete;
    CudaGraphCaptureGuard& operator=(CudaGraphCaptureGuard&&)      = delete;

private:
    rtp_llm::cuda_graph::GraphNcclCaptureContext* ctx_{nullptr};
};
