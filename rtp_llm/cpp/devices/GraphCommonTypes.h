#pragma once

#include "ATen/core/TensorBody.h"
#if defined(USE_ROCM)
#include <ATen/hip/HIPGraph.h>
#else
#include <ATen/cuda/CUDAGraph.h>
#endif
#include "rtp_llm/models_py/bindings/OpDefs.h"

using namespace torch_ext;

class CaptureMemoryHold {
public:
    void setHiddenStates(at::Tensor hidden_states) {
        decoder_layer_hidden_states_ = hidden_states;
    };

    CaptureMemoryHold() {}

    CaptureMemoryHold(at::Tensor hidden_states, PyModelInputs& inputs, bool is_embedding):
        decoder_layer_hidden_states_(hidden_states) {
        py_model_inputs_.attention_inputs.input_lengths            = inputs.attention_inputs.input_lengths;
        py_model_inputs_.attention_inputs.sequence_lengths         = inputs.attention_inputs.sequence_lengths;
        py_model_inputs_.attention_inputs.kv_cache_block_id_device = inputs.attention_inputs.kv_cache_block_id_device;
        py_model_inputs_.attention_inputs.kv_cache_block_id_host   = inputs.attention_inputs.kv_cache_block_id_host;
        py_model_inputs_.attention_inputs.prefix_lengths           = inputs.attention_inputs.prefix_lengths;
        py_model_inputs_.input_ids                                 = inputs.input_ids;

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

namespace rtp_llm {

struct GraphExecutionState {
    int current_batch_size{1};
    int current_seq_len{1};
    int current_real_graph_bs{1};
    int current_real_graph_seq_len{1};
    int seq_len_sum{0};
};

}  // namespace rtp_llm
