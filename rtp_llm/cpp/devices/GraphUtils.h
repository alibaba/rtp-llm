#pragma once
#include "ATen/core/TensorBody.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

using namespace torch_ext;

// GPU-agnostic capture check state (used by both CUDA and HIP)
struct CaptureCheck {
    static bool in_cuda_graph_capture;
};

// GPU-agnostic memory hold for graph capture
class CaptureMemoryHold {
public:
    void setHiddenStates(at::Tensor hidden_states) {
        decoder_layer_hidden_states_ = hidden_states;
    };

    CaptureMemoryHold() {}

    CaptureMemoryHold(at::Tensor hidden_states, PyModelInputs& inputs, int kv_cache_block_offset, bool is_embedding):
        decoder_layer_hidden_states_(hidden_states) {
        py_model_inputs_.attention_inputs.input_lengths            = inputs.attention_inputs.input_lengths;
        py_model_inputs_.attention_inputs.sequence_lengths         = inputs.attention_inputs.sequence_lengths;
        py_model_inputs_.attention_inputs.kv_cache_block_id_device = inputs.attention_inputs.kv_cache_block_id_device;
        py_model_inputs_.attention_inputs.kv_cache_block_id_host   = inputs.attention_inputs.kv_cache_block_id_host;
        py_model_inputs_.attention_inputs.prefix_lengths           = inputs.attention_inputs.prefix_lengths;
        py_model_inputs_.input_ids                                 = inputs.input_ids;
        py_model_inputs_.attention_inputs.cu_seqlens               = inputs.attention_inputs.cu_seqlens;
        py_model_inputs_.attention_inputs.padding_offset           = inputs.attention_inputs.padding_offset;
        py_model_inputs_.attention_inputs.is_prefill               = is_embedding;
        py_model_inputs_.attention_inputs.dtype                    = inputs.attention_inputs.dtype;
        py_model_inputs_.attention_inputs.kv_block_offset          = kv_cache_block_offset;
        py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params =
            inputs.attention_inputs.prefill_cuda_graph_copy_params;
        py_model_inputs_.bert_embedding_inputs        = inputs.bert_embedding_inputs;
        py_model_inputs_.attention_inputs.is_s_padded = inputs.attention_inputs.is_s_padded;
    }

public:
    // for attention params
    ParamsBasePtr params_ptr{nullptr};
    // for output
    at::Tensor decoder_layer_hidden_states_;
    // for input
    PyModelInputs py_model_inputs_;
};

// Base graph instance without GPU-specific graph object
// Subclasses will add GPU-specific graph (CUDAGraph, HIPGraph, etc.)
class GraphInstanceBase {
public:
    virtual ~GraphInstanceBase() = default;
    CaptureMemoryHold mem_hold_;
};

// Current state of graph execution (GPU-agnostic)
struct GraphState {
    int current_batch_size{1};
    int current_seq_len{1};
    // for decode
    int current_real_graph_bs{1};
    // for prefill
    int current_real_graph_seq_len{1};
    int seq_len_sum{0};
};

// RAII guard for graph capture state (GPU-agnostic)
class GraphCaptureGuard {
public:
    GraphCaptureGuard() {
        CaptureCheck::in_cuda_graph_capture = true;
    }

    ~GraphCaptureGuard() {
        CaptureCheck::in_cuda_graph_capture = false;
    }

    // Non-copyable, non-movable
    GraphCaptureGuard(const GraphCaptureGuard&)            = delete;
    GraphCaptureGuard& operator=(const GraphCaptureGuard&) = delete;
    GraphCaptureGuard(GraphCaptureGuard&&)                 = delete;
    GraphCaptureGuard& operator=(GraphCaptureGuard&&)      = delete;
};

}  // namespace rtp_llm
