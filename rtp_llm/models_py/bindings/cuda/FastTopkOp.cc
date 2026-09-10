#include "rtp_llm/models_py/bindings/cuda/FastTopkOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/fast_topk.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/minimax_decode_topk.h"

namespace torch_ext {

void fast_topk_v2(at::Tensor&                         score,
                  at::Tensor&                         indices,
                  at::Tensor&                         lengths,
                  const std::optional<torch::Tensor>& row_starts) {
    rtp_llm::fast_topk_v2(score, indices, lengths, row_starts);
}

void fast_topk_v2_variable(at::Tensor&                         score,
                           at::Tensor&                         indices,
                           at::Tensor&                         lengths,
                           const std::optional<torch::Tensor>& row_starts,
                           int64_t                             top_k) {
    rtp_llm::fast_topk_v2_variable(score, indices, lengths, row_starts, top_k);
}

void fast_topk_transform_fused(at::Tensor&                         score,
                               at::Tensor&                         lengths,
                               at::Tensor&                         dst_page_table,
                               const std::optional<torch::Tensor>& src_page_table,
                               at::Tensor&                         cu_seqlens_q,
                               const std::optional<torch::Tensor>& row_starts) {
    rtp_llm::fast_topk_transform_fused(score, lengths, dst_page_table, src_page_table, cu_seqlens_q, row_starts);
}

void fast_topk_transform_ragged_fused(at::Tensor&                         score,
                                      at::Tensor&                         lengths,
                                      at::Tensor&                         topk_indices_ragged,
                                      at::Tensor&                         topk_indices_offset,
                                      const std::optional<torch::Tensor>& row_starts) {
    rtp_llm::fast_topk_transform_ragged_fused(score, lengths, topk_indices_ragged, topk_indices_offset, row_starts);
}

void minimax_decode_topk(
    at::Tensor& score, at::Tensor& seq_lens, at::Tensor& topk_idx, int64_t block_size, int64_t topk) {
    rtp_llm::minimax_decode_topk(score, seq_lens, topk_idx, block_size, topk);
}

void persistent_topk(at::Tensor& logits,
                     at::Tensor& lengths,
                     at::Tensor& output,
                     at::Tensor& workspace,
                     int64_t     k,
                     int64_t     max_seq_len) {
    rtp_llm::persistent_topk(logits, lengths, output, workspace, k, max_seq_len);
}

}  // namespace torch_ext
