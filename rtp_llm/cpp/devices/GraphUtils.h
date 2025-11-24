#pragma once
#include "ATen/core/TensorBody.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#if USING_ROCM
#include <ATen/hip/HIPGraph.h>
#elif USING_CUDA
#include <ATen/cuda/CUDAGraph.h>
#endif
using namespace torch_ext;

namespace rtp_llm {

#if USING_ROCM || USING_CUDA
using GraphCaptureType = at::cuda::CUDAGraph;
#else
class DummyGraph {
public:
    void replay() {}
    void capture_begin() {}
    void capture_end() {}
    void enable_debug_mode() {}
    void debug_dump(const char*) {}
};
using GraphCaptureType = DummyGraph;
#endif

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
        py_model_inputs_.bert_embedding_inputs                     = inputs.bert_embedding_inputs;
    }

public:
    // for attention params
    rtp_llm::ParamsBasePtr params_ptr{nullptr};
    // for output
    at::Tensor decoder_layer_hidden_states_;
    // for input
    PyModelInputs py_model_inputs_;
};

class GraphInstance {
public:
    GraphCaptureType  graph_;
    CaptureMemoryHold mem_hold_;
};

namespace GraphUtils {

inline std::vector<int> getBatchSizesToCapture(int concurrency_limit) {
    std::vector<int> capture_bs;
    int              max_generate_batch_size = concurrency_limit;
    RTP_LLM_LOG_INFO("max_generate_batch_size for graph: %d", max_generate_batch_size);
    // Add range 1 to 32 (inclusive)
    for (int i = 1; i <= std::min(32, max_generate_batch_size); i += 1) {
        capture_bs.push_back(i);
    }
    // Add range from 48 to max_generate_batch_size (exclusive), stepping by 16
    for (int i = 48; i <= max_generate_batch_size; i += 16) {
        capture_bs.push_back(i);
    }
    if (capture_bs[capture_bs.size() - 1] != max_generate_batch_size) {
        capture_bs.push_back(max_generate_batch_size);
    }
    return capture_bs;
}

// 将较小的tensor复制到较大的tensor中 - 两个实现完全相同
inline void copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor) {
    if (source_tensor.dim() != target_tensor.dim()) {
        throw std::runtime_error("Error: Source and target tensors must have the same number of dimensions.");
    }

    for (int i = 0; i < source_tensor.dim(); ++i) {
        if (source_tensor.size(i) > target_tensor.size(i)) {
            std::string error_msg =
                "Error: Target tensor dimension " + std::to_string(i) + " (" + std::to_string(target_tensor.size(i))
                + ")" + " is smaller than source tensor dimension " + std::to_string(i) + " ("
                + std::to_string(source_tensor.size(i)) + "). " + "This violates the function's guarantee.";
            throw std::runtime_error(error_msg);
        }
    }

    torch::Tensor target_slice = target_tensor;

    for (int i = 0; i < source_tensor.dim(); ++i) {
        target_slice = target_slice.slice(i, 0, source_tensor.size(i));
    }

    target_slice.copy_(source_tensor);
}

// 提取有效的hidden states - 两个实现完全相同
inline void extractValidHiddenStates(PyModelOutputs&      outputs,
                                     const PyModelInputs& inputs,
                                     int32_t              total_valid_tokens,
                                     const torch::Tensor& hidden_states,
                                     int                  current_batch_size,
                                     int                  num_tokens_per_bs) {
    auto input_lengths = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();

    // Verify if total_valid_tokens calculation is correct
    RTP_LLM_LOG_DEBUG("total_valid_tokens: %d, hidden_states.size(0): %d", total_valid_tokens, hidden_states.size(0));

    // Extract valid parts for each batch
    int32_t output_offset = 0;
    RTP_LLM_LOG_DEBUG("Extracting valid hidden states for embedding mode - batch_size: %d, total_valid_tokens: %d",
                      current_batch_size,
                      total_valid_tokens);

    for (int i = 0; i < current_batch_size; i++) {
        int32_t actual_length = input_lengths[i];       // actual valid length
        int32_t batch_start   = i * num_tokens_per_bs;  // start position in padded tensor

        RTP_LLM_LOG_DEBUG("Batch %d: actual_length=%d, batch_start=%d, output_offset=%d",
                          i,
                          actual_length,
                          batch_start,
                          output_offset);

        // Add boundary checks and validation
        if (actual_length <= 0) {
            RTP_LLM_LOG_ERROR("Batch %d: actual_length=%d <= 0, skipping", i, actual_length);
            continue;
        }

        if (batch_start >= hidden_states.size(0)) {
            RTP_LLM_LOG_ERROR(
                "Batch %d: batch_start=%d >= hidden_states.size(0)=%d", i, batch_start, hidden_states.size(0));
            continue;
        }

        if (batch_start + actual_length > hidden_states.size(0)) {
            RTP_LLM_LOG_ERROR("Batch %d: batch_start=%d + actual_length=%d > hidden_states.size(0)=%d",
                              i,
                              batch_start,
                              actual_length,
                              hidden_states.size(0));
            continue;
        }

        if (output_offset + actual_length > outputs.hidden_states.size(0)) {
            RTP_LLM_LOG_ERROR("Batch %d: output_offset=%d + actual_length=%d > outputs.hidden_states.size(0)=%d",
                              i,
                              output_offset,
                              actual_length,
                              outputs.hidden_states.size(0));
            continue;
        }

        // Extract valid parts from padded tensor
        outputs.hidden_states.slice(0, output_offset, output_offset + actual_length) =
            hidden_states.slice(0, batch_start, batch_start + actual_length);
        output_offset += actual_length;
    }

    // Resize output to contain only valid tokens
    outputs.hidden_states = hidden_states.slice(0, 0, total_valid_tokens);

    // Verify final result
    RTP_LLM_LOG_DEBUG("Final output_offset: %d, expected: %d", output_offset, total_valid_tokens);
}

}  // namespace GraphUtils

}  // namespace rtp_llm
