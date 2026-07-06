#pragma once
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/fast_topk.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/minimax_decode_topk.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/persistent_topk.h"

namespace torch_ext {

void fast_topk_v2(at::Tensor&                         score,
                  at::Tensor&                         indices,
                  at::Tensor&                         lengths,
                  const std::optional<torch::Tensor>& row_starts = std::nullopt);

void fast_topk_v2_variable(at::Tensor&                         score,
                           at::Tensor&                         indices,
                           at::Tensor&                         lengths,
                           const std::optional<torch::Tensor>& row_starts,
                           int64_t                             top_k);

void fast_topk_transform_fused(at::Tensor&                         score,
                               at::Tensor&                         lengths,
                               at::Tensor&                         dst_page_table,
                               const std::optional<torch::Tensor>& src_page_table,
                               at::Tensor&                         cu_seqlens_q,
                               const std::optional<torch::Tensor>& row_starts = std::nullopt);

void fast_topk_transform_ragged_fused(at::Tensor&                         score,
                                      at::Tensor&                         lengths,
                                      at::Tensor&                         topk_indices_ragged,
                                      at::Tensor&                         topk_indices_offset,
                                      const std::optional<torch::Tensor>& row_starts = std::nullopt);

void minimax_decode_topk(
    at::Tensor& score, at::Tensor& seq_lens, at::Tensor& topk_idx, int64_t block_size, int64_t topk);

void persistent_topk(
    at::Tensor& logits, at::Tensor& lengths, at::Tensor& output, at::Tensor& workspace, int64_t k, int64_t max_seq_len);

}  // namespace torch_ext
