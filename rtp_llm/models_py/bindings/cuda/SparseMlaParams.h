#pragma once

#include <torch/extension.h>

#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

class SparseMlaParams: public FlashInferMlaAttnParams {
private:
    // SparseMlaParams-specific buffers (separate from base class buf_h/buf_d)
    // Base class only manages int32 buffers; SparseMlaParams needs int64 for slot_mapping
    torch::Tensor buf_h_i32_;  // HOST buffer for SparseMlaParams int32 tensors
    torch::Tensor buf_d_i32_;  // DEVICE buffer for SparseMlaParams int32 tensors
    torch::Tensor buf_h_i64_;  // HOST buffer for SparseMlaParams int64 tensors (slot_mapping)
    torch::Tensor buf_d_i64_;  // DEVICE buffer for SparseMlaParams int64 tensors (slot_mapping)

    size_t max_i32_elements_ = 0;
    size_t max_i64_elements_ = 0;
    int    max_batch_size_   = 0;
    int    max_token_num_    = 0;
    int    max_seq_len_      = 0;

    // Only generate SparseMlaParams-specific tensors
    torch::Tensor expanded_seq_lens_h_;
    torch::Tensor topk_indices_offset_h_;
    torch::Tensor ks_h_;
    torch::Tensor ke_h_;
    torch::Tensor page_table_1_h_;
    torch::Tensor slot_mapping_h_;

    torch::Tensor expanded_seq_lens_d_;
    torch::Tensor topk_indices_offset_d_;
    torch::Tensor ks_d_;
    torch::Tensor ke_d_;
    torch::Tensor page_table_1_d_;
    torch::Tensor slot_mapping_d_;

    void ensureTensorSize(int batch_size, int token_num, int max_seq_len);
    void fillParamsInternal(bool                 is_prefill,
                            const torch::Tensor& input_lengths_cpu,
                            const torch::Tensor& prefix_lengths_cpu,
                            const torch::Tensor& sequence_lengths_cpu,
                            int                  batch_size,
                            int                  seq_size_per_block,
                            int64_t              total_tokens,
                            int64_t              max_seq_len,
                            const torch::Tensor& positions_h,
                            torch::Tensor&       slot_mapping_h);
    void refreshBuffer(int batch_size, int token_num, int max_seq_len, bool is_prefill);

public:
    void fillParams(torch_ext::PyAttentionInputs attn_inputs, int seq_size_per_block);

    // SparseMlaParams-specific outputs (6 parameters)
    torch::Tensor expanded_seq_lens;
    torch::Tensor page_table_1;
    torch::Tensor topk_indices_offset;
    torch::Tensor slot_mapping;
    torch::Tensor ks;
    torch::Tensor ke;

    // schedule_metadata for deep_gemm
    torch::Tensor schedule_metadata;
};

void registerPySparseMlaParams(pybind11::module& m);

}  // namespace rtp_llm
