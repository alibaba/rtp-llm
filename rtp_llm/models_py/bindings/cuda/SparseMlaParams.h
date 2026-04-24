#pragma once

#include <torch/extension.h>

#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

class SparseMlaParams: public FlashInferMlaAttnParams {
private:
    // SparseMlaParams-specific buffers (separate from base class buf_h/buf_d)
    torch::Tensor buf_h_i32_;  // HOST buffer for SparseMlaParams int32 tensors
    torch::Tensor buf_d_i32_;  // DEVICE buffer for SparseMlaParams int32 tensors

    size_t max_i32_elements_ = 0;
    int    max_batch_size_   = 0;
    int    max_token_num_    = 0;

    // Only generate SparseMlaParams-specific tensors
    torch::Tensor expanded_seq_lens_h_;
    torch::Tensor topk_indices_offset_h_;
    torch::Tensor ks_h_;
    torch::Tensor ke_h_;

    torch::Tensor expanded_seq_lens_d_;
    torch::Tensor topk_indices_offset_d_;
    torch::Tensor ks_d_;
    torch::Tensor ke_d_;

    void ensureTensorSize(int batch_size, int token_num, bool forbid_realloc = false);
    void fillParamsInternal(bool                 is_prefill,
                            const torch::Tensor& input_lengths_cpu,
                            const torch::Tensor& prefix_lengths_cpu,
                            const torch::Tensor& sequence_lengths_cpu,
                            int                  batch_size,
                            int                  seq_size_per_block,
                            int64_t              total_tokens,
                            const torch::Tensor& positions_h);
    void refreshBuffer(int batch_size, int token_num, bool is_prefill);

    // CP Plan buffers
    torch::Tensor cp_buf_h_i64_;   // pinned HOST buffer for int64 CP tensors
    torch::Tensor cp_buf_d_i64_;   // DEVICE buffer for int64 CP tensors
    torch::Tensor cp_buf_h_i32_2_; // pinned HOST buffer for int32 CP tensors (cu_kv_seqlens)
    torch::Tensor cp_buf_d_i32_2_; // DEVICE buffer for int32 CP tensors

    size_t cp_max_i64_elements_  = 0;
    size_t cp_max_i32_elements_  = 0;
    int    cp_max_idx_count_     = 0;
    int    cp_max_batch_size_cp_ = 0;

    // Host/device view pairs for CP plan outputs
    torch::Tensor cp_kv_restore_unpad_indices_h_;
    torch::Tensor cp_kv_restore_unpad_indices_d_;
    torch::Tensor cp_total_global_ids_h_;
    torch::Tensor cp_total_global_ids_d_;
    torch::Tensor cp_total_local_ids_h_;
    torch::Tensor cp_total_local_ids_d_;
    torch::Tensor cp_cu_kv_seqlens_global_h_;
    torch::Tensor cp_cu_kv_seqlens_global_d_;

    void ensureCpTensorSize(int max_idx_count, int batch_size);
    void refreshCpBuffer(int kv_restore_count, int total_ids_count, int batch_size);

public:
    void fillParams(torch_ext::PyAttentionInputs attn_inputs, int seq_size_per_block, bool forbid_realloc = false);

    // SparseMlaParams-specific outputs (5 parameters)
    torch::Tensor expanded_seq_lens;
    torch::Tensor topk_indices_offset;
    torch::Tensor ks;
    torch::Tensor ke;

    // schedule_metadata for deep_gemm
    torch::Tensor schedule_metadata;

    // CP Plan: compute CP indices on CPU, single cudaMemcpyAsync to device
    void fillCpPlanParams(const torch::Tensor&         padding_mask,
                          const torch::Tensor&         kv_restore_indices,
                          const std::vector<int64_t>&  q0_idx,
                          const std::vector<int64_t>&  q1_idx,
                          int                          cp_rank,
                          int                          local_tokens,
                          const torch::Tensor&         actual_input_lengths,
                          const torch::Tensor&         prefix_lengths);

    // CP Plan outputs (device tensors)
    torch::Tensor cp_kv_restore_unpad_indices;  // [n_valid], int64
    torch::Tensor cp_total_global_ids;          // [n_q0_valid + n_q1_valid], int64
    torch::Tensor cp_total_local_ids;           // [n_q0_valid + n_q1_valid], int64
    torch::Tensor cp_cu_kv_seqlens_global;      // [batch_size + 1], int32
    int           cp_total_kv_len = 0;          // cu_kv_seqlens_global[-1]
};

void registerPySparseMlaParams(pybind11::module& m);

}  // namespace rtp_llm
