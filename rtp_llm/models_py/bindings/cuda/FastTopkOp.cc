#include "rtp_llm/models_py/bindings/cuda/FastTopkOp.h"
#include "rtp_llm/cpp/kernels/fast_topk.h"

namespace torch_ext {

void fast_topk_v2(at::Tensor&                         score,
                  at::Tensor&                         indices,
                  at::Tensor&                         lengths,
                  const std::optional<torch::Tensor>& row_starts) {
    rtp_llm::fast_topk_v2(score, indices, lengths, row_starts);
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

}  // namespace torch_ext
