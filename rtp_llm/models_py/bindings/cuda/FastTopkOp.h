#pragma once
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/cpp/kernels/fast_topk.h"

namespace torch_ext {

void fast_topk_v2(at::Tensor&                         score,
                  at::Tensor&                         indices,
                  at::Tensor&                         lengths,
                  const std::optional<torch::Tensor>& row_starts = std::nullopt);

void fast_topk_transform_fused(at::Tensor&                         score,
                               at::Tensor&                         lengths,
                               at::Tensor&                         dst_page_table,
                               at::Tensor&                         src_page_table,
                               at::Tensor&                         cu_seqlens_q,
                               const std::optional<torch::Tensor>& row_starts = std::nullopt);

void fast_topk_transform_ragged_fused(at::Tensor&                         score,
                                      at::Tensor&                         lengths,
                                      at::Tensor&                         topk_indices_ragged,
                                      at::Tensor&                         topk_indices_offset,
                                      const std::optional<torch::Tensor>& row_starts = std::nullopt);

}  // namespace torch_ext
