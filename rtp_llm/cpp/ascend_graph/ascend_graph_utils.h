// AscendGraphInstance and capture guards.
//
// `AscendGraphInstance` mirrors `GraphInstance` in cuda_graph_utils.h but
// wraps `c10_npu::NPUGraph`; `AscendGraphMemHold` is the Ascend equivalent of
// `CaptureMemoryHold`. Kept local to ascend_graph because cuda_graph_utils.h
// does not compile on Ascend (CudaGraphStreamLife references `.stream()` on a
// `void*` GraphStream).

#pragma once

#include "rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {
namespace ascend_graph {

// Mirrors CaptureMemoryHold from cuda_graph_utils.h, but scoped to ascend_graph
// to avoid pulling in the CUDA-only CudaGraphStreamLife/CudaGraphCaptureGuard.
class AscendGraphMemHold {
public:
    AscendGraphMemHold() = default;

    AscendGraphMemHold(at::Tensor hidden_states, torch_ext::PyModelInputs& inputs, bool /*is_embedding*/):
        decoder_layer_hidden_states_(std::move(hidden_states)) {
        // Copy every attention/bert field by value so the captured tensors are
        // the persistent ones allocated in initCapture.
        py_model_inputs_.attention_inputs.input_lengths    = inputs.attention_inputs.input_lengths;
        py_model_inputs_.attention_inputs.input_lengths_device  = inputs.attention_inputs.input_lengths_device;
        py_model_inputs_.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths;
        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device =
            inputs.attention_inputs.kv_cache_kernel_block_id_device;
        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id =
            inputs.attention_inputs.kv_cache_kernel_block_id;
        py_model_inputs_.attention_inputs.kv_cache_block_id_device = inputs.attention_inputs.kv_cache_block_id_device;
        py_model_inputs_.attention_inputs.kv_cache_block_id   = inputs.attention_inputs.kv_cache_block_id;
        // Tag-based hybrid groups: copy the whole per-tag map (each group's
        // PyAttentionInputs holds its own 2-D kernel/physical tables).
        py_model_inputs_.attention_inputs_by_tag = inputs.attention_inputs_by_tag;
        py_model_inputs_.attention_inputs.prefix_lengths          = inputs.attention_inputs.prefix_lengths;
        py_model_inputs_.attention_inputs.prefix_lengths_device        = inputs.attention_inputs.prefix_lengths_device;
        py_model_inputs_.input_ids                                = inputs.input_ids;
        py_model_inputs_.input_hiddens                            = inputs.input_hiddens;
        py_model_inputs_.attention_inputs.cu_seqlens              = inputs.attention_inputs.cu_seqlens;
        py_model_inputs_.attention_inputs.cu_seqlens_device       = inputs.attention_inputs.cu_seqlens_device;
        py_model_inputs_.attention_inputs.combo_position_ids      = inputs.attention_inputs.combo_position_ids;
        py_model_inputs_.attention_inputs.cu_kv_seqlens_device           = inputs.attention_inputs.cu_kv_seqlens_device;
        py_model_inputs_.attention_inputs.padding_offset          = inputs.attention_inputs.padding_offset;
        py_model_inputs_.attention_inputs.is_prefill              = inputs.attention_inputs.is_prefill;
        py_model_inputs_.attention_inputs.is_target_verify        = inputs.attention_inputs.is_target_verify;
        py_model_inputs_.attention_inputs.dtype                   = inputs.attention_inputs.dtype;
        py_model_inputs_.attention_inputs.context_total_kv_length = inputs.attention_inputs.context_total_kv_length;
        py_model_inputs_.bert_embedding_inputs                    = inputs.bert_embedding_inputs;
        py_model_inputs_.attention_inputs.is_s_padded             = inputs.attention_inputs.is_s_padded;
        py_model_inputs_.attention_inputs.decode_cu_seqlens_device     = inputs.attention_inputs.decode_cu_seqlens_device;
        py_model_inputs_.attention_inputs.sequence_lengths_plus_1_device =
            inputs.attention_inputs.sequence_lengths_plus_1_device;
    }

    void setHiddenStates(at::Tensor hidden_states) {
        decoder_layer_hidden_states_ = std::move(hidden_states);
    }

    py::object                attn_pyobj_{py::none()};
    at::Tensor                decoder_layer_hidden_states_;
    torch_ext::PyModelInputs  py_model_inputs_;
};

// Graph instance keyed by decode batch_size.
class AscendGraphInstance {
public:
    AscendGraphInstance() = default;

#if USING_ASCEND
    c10_npu::NPUGraph graph_;
#endif
    AscendGraphMemHold mem_hold_;
};

// RAII guard that switches the current NPU stream to a capture stream for the
// lifetime of the guard.
class AscendGraphStreamLife {
public:
    explicit AscendGraphStreamLife(GraphStream capture_stream)
        : origin_stream_(graphGetCurrentStream()) {
        graphSetCurrentStream(capture_stream);
        RTP_LLM_LOG_INFO("Set Ascend graph stream for capture. capture_stream=%p",
                         reinterpret_cast<void*>(capture_stream.stream()));
    }
    ~AscendGraphStreamLife() {
        graphSetCurrentStream(origin_stream_);
        RTP_LLM_LOG_INFO("Restored Ascend graph stream after capture.");
    }

    AscendGraphStreamLife(const AscendGraphStreamLife&)            = delete;
    AscendGraphStreamLife& operator=(const AscendGraphStreamLife&) = delete;
    AscendGraphStreamLife(AscendGraphStreamLife&&)                 = delete;
    AscendGraphStreamLife& operator=(AscendGraphStreamLife&&)      = delete;

private:
    GraphStream origin_stream_;
};

// RAII guard that flips the in-graph-capture flag.
class AscendGraphCaptureGuard {
public:
    AscendGraphCaptureGuard() { setGraphCaptureEnabled(true); }
    ~AscendGraphCaptureGuard() {
        try {
            setGraphCaptureEnabled(false);
        } catch (...) {
            RTP_LLM_LOG_WARNING("Unknown exception in AscendGraphCaptureGuard destructor");
        }
    }

    AscendGraphCaptureGuard(const AscendGraphCaptureGuard&)            = delete;
    AscendGraphCaptureGuard& operator=(const AscendGraphCaptureGuard&) = delete;
    AscendGraphCaptureGuard(AscendGraphCaptureGuard&&)                 = delete;
    AscendGraphCaptureGuard& operator=(AscendGraphCaptureGuard&&)      = delete;
};

}  // namespace ascend_graph
}  // namespace rtp_llm
