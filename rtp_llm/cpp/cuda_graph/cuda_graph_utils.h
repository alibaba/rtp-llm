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
        py_model_inputs_.attentionInputs().input_lengths    = inputs.attentionInputs().input_lengths;
        py_model_inputs_.attentionInputs().input_lengths_device   = inputs.attentionInputs().input_lengths_device;
        py_model_inputs_.attentionInputs().sequence_lengths       = inputs.attentionInputs().sequence_lengths;
        py_model_inputs_.attentionInputs().kv_cache_kernel_block_id_device =
            inputs.attentionInputs().kv_cache_kernel_block_id_device;
        py_model_inputs_.attentionInputs().kv_cache_kernel_block_id =
            inputs.attentionInputs().kv_cache_kernel_block_id;
        py_model_inputs_.attentionInputs().kv_cache_block_id_device = inputs.attentionInputs().kv_cache_block_id_device;
        py_model_inputs_.attentionInputs().kv_cache_block_id   = inputs.attentionInputs().kv_cache_block_id;
        py_model_inputs_.attentionInputs().kv_cache_layer_to_group = inputs.attentionInputs().kv_cache_layer_to_group;
        py_model_inputs_.attentionInputs().prefix_lengths          = inputs.attentionInputs().prefix_lengths;
        py_model_inputs_.attentionInputs().prefix_lengths_device   = inputs.attentionInputs().prefix_lengths_device;
        py_model_inputs_.attentionInputs().combo_position_ids      = inputs.attentionInputs().combo_position_ids;
        py_model_inputs_.input_ids                                = inputs.input_ids;
        py_model_inputs_.combo_position_ids                       = inputs.combo_position_ids;

        // for spec
        py_model_inputs_.input_hiddens                            = inputs.input_hiddens;
        py_model_inputs_.attentionInputs().cu_seqlens_device       = inputs.attentionInputs().cu_seqlens_device;
        py_model_inputs_.attentionInputs().cu_seqlens         = inputs.attentionInputs().cu_seqlens;
        py_model_inputs_.attentionInputs().cu_kv_seqlens_device    = inputs.attentionInputs().cu_kv_seqlens_device;
        py_model_inputs_.attentionInputs().padding_offset          = inputs.attentionInputs().padding_offset;
        py_model_inputs_.attentionInputs().is_prefill              = inputs.attentionInputs().is_prefill;
        py_model_inputs_.attentionInputs().is_target_verify        = inputs.attentionInputs().is_target_verify;
        py_model_inputs_.attentionInputs().dtype                   = inputs.attentionInputs().dtype;
        py_model_inputs_.attentionInputs().context_total_kv_length = inputs.attentionInputs().context_total_kv_length;

        py_model_inputs_.attentionInputs().prefill_cuda_graph_copy_params =
            inputs.attentionInputs().prefill_cuda_graph_copy_params;
        py_model_inputs_.bert_embedding_inputs                      = inputs.bert_embedding_inputs;
        py_model_inputs_.attentionInputs().is_s_padded               = inputs.attentionInputs().is_s_padded;
        py_model_inputs_.attentionInputs().decode_cu_seqlens_device  = inputs.attentionInputs().decode_cu_seqlens_device;
        py_model_inputs_.attentionInputs().decode_cu_seqlens    = inputs.attentionInputs().decode_cu_seqlens;
        py_model_inputs_.attentionInputs().sequence_lengths_plus_1_device = inputs.attentionInputs().sequence_lengths_plus_1_device;
        py_model_inputs_.attn_inputs_list = inputs.attn_inputs_list.empty() ?
            torch_ext::makePyAttentionInputsByGroup(py_model_inputs_.attentionInputs()) :
            inputs.attn_inputs_list;
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
