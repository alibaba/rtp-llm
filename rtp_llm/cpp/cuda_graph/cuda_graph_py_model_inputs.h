#pragma once

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_utils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <vector>

namespace rtp_llm {
namespace cuda_graph {

// Layout / max buffers for CUDA graph capture (attention side of PyModelInputs).
struct CaptureAttentionLayoutConfig {
    bool is_target_verify{false};
    bool is_prefill_cuda_graph_mode{false};
    /// Same as `GraphParams::num_tokens_per_bs` (input_lengths full, is_prefill, prefix in spec decode).
    int member_num_tokens_per_bs{1};
    /// Used with `initCaptureAttentionTensors` for `sequence_lengths` fill (see `passed_num_tokens_per_bs`).
    int                  passed_num_tokens_per_bs{1};
    int                  max_num_token{0};
    size_t               max_bs{1};
    int                  max_seq_len{0};
    int                  seq_size_per_block{0};
    int                  kernel_seq_size_per_block{1};
    int                  sp_steps{0};
    c10::ScalarType      model_data_type{c10::ScalarType::Float};
    std::vector<int32_t> kv_cache_layer_to_group;
    int32_t              kv_cache_group_num{0};
    at::TensorOptions    options_cuda_int32;
    at::TensorOptions    options_cpu_int32;
};

void copySmallerIntoLarger(const at::Tensor& source_tensor, at::Tensor& target_tensor);

void optimizedCopyAsync(const at::Tensor& src, at::Tensor& dst, size_t size);

// Attention-side max buffers only; does not allocate input_ids / input_hiddens / BERT fields.
void initCaptureAttentionTensors(torch_ext::PyModelInputs& inputs, const CaptureAttentionLayoutConfig& cfg);

// Full max-sized PyModelInputs for CUDA graph capture (token tensors, attention, BERT slots).
struct CaptureMaxPyModelInputsConfig {
    int64_t                      hidden_size{0};
    at::TensorOptions            options_cuda_float{};
    CaptureAttentionLayoutConfig attention{};
    int                          bert_combo_flat_len{0};
    at::Tensor                   position_encoding;
    at::Tensor                   token_type_embedding;
    float                        input_embedding_scalar{0.f};
};

void initMaxCapturePyModelInputs(torch_ext::PyModelInputs& inputs, const CaptureMaxPyModelInputsConfig& cfg);

CaptureMaxPyModelInputsConfig makeCaptureMaxPyModelInputsConfig(const GraphParams&       graph_params,
                                                                size_t                   max_bs,
                                                                int                      max_num_token,
                                                                const at::TensorOptions& options_cuda_int32,
                                                                const at::TensorOptions& options_cpu_int32,
                                                                const at::TensorOptions& options_cuda_float,
                                                                const at::Tensor&        position_encoding,
                                                                const at::Tensor&        token_type_embedding,
                                                                float                    input_embedding_scalar);

void initPrefillCudaGraphCopyParams(torch_ext::PyModelInputs& inputs,
                                    const at::TensorOptions&  options_cpu_int32,
                                    int                       max_seq_len,
                                    int                       max_bs);

/// Builds max-sized `PyModelInputs`, `cu_seqlens` / `cu_kv_seqlens`, and `CaptureMemoryHold` for CUDA graph capture
/// init.
class CudaGraphCapturePyModelInputs {
public:
    CudaGraphCapturePyModelInputs(const GraphParams&       graph_params,
                                  size_t                   max_bs,
                                  int                      max_num_token,
                                  const at::TensorOptions& options_cuda_int32,
                                  const at::TensorOptions& options_cpu_int32,
                                  const at::TensorOptions& options_cuda_float,
                                  const at::Tensor&        position_encoding,
                                  const at::Tensor&        token_type_embedding,
                                  float                    input_embedding_scalar);

    /// Max template tensors + pinned cu_seqlens; hidden states placeholder until
    /// `allocateHiddenStatesAndPrefillCopyParams`.
    CaptureMemoryHold makeCaptureMemoryHold() const;

    /// Allocates `all_layers_output_` and installs prefill CUDA-graph copy params on `hold.py_model_inputs_`.
    void allocateHiddenStatesAndPrefillCopyParams(CaptureMemoryHold& hold) const;

    /// Prefill-only: patch full template buffers for the extra probe `forward` before per-seq capture.
    void patchForPrefillProbeForward(CaptureMemoryHold& hold) const;

    /// Sliced `PyModelInputs` for that probe `forward` (batch 1 / short cu_seqlens views).
    torch_ext::PyModelInputs sliceForPrefillProbeForward(const CaptureMemoryHold& hold) const;

    /// Fills `cu_seqlens` / `cu_kv_seqlens` from `input_lengths` / `prefix_lengths` (length `max_bs + 1`, pinned).
    static void fillCuSeqlensForCapture(torch_ext::PyModelInputs& py_inputs, size_t max_bs);

private:
    GraphParams       graph_params_;
    size_t            max_bs_{1};
    int               max_num_token_{1};
    at::TensorOptions options_cuda_int32_;
    at::TensorOptions options_cpu_int32_;
    at::TensorOptions options_cuda_float_;
    at::Tensor        position_encoding_;
    at::Tensor        token_type_embedding_;
    float             input_embedding_scalar_{0.f};
};

// Slice the max capture template down to the batch / token span used for one graph key.
void sliceTemplatePyModelInputsForCapture(torch_ext::PyModelInputs&       dst,
                                          const torch_ext::PyModelInputs& cap_template,
                                          int                             batch_size,
                                          int                             seq_len_or_tokens,
                                          bool                            is_prefill_cuda_graph_mode,
                                          int                             member_num_tokens_per_bs,
                                          bool                            is_target_verify);

// Copy runtime request tensors into the padded capture buffers (replay path).
void copyRuntimePyModelIntoCaptureBuffers(const torch_ext::PyModelInputs& runtime,
                                          torch_ext::PyModelInputs&       cap,
                                          const BatchDescriptor&          batch_descriptor,
                                          bool                            is_prefill_cuda_graph_mode,
                                          pybind11::object                decode_attn_pyobj);

}  // namespace cuda_graph
}  // namespace rtp_llm
