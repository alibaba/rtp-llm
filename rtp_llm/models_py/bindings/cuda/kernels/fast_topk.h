// Adapted from https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/elementwise/topk.cu
// Licensed under the Apache License, Version 2.0
#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/all.h>
#include <optional>

namespace rtp_llm {

void fast_topk_v2(const at::Tensor&         score,
                  at::Tensor&               indices,
                  const at::Tensor&         lengths,
                  std::optional<at::Tensor> row_starts_opt = std::nullopt);

void fast_topk_transform_fused(const at::Tensor&         score,
                               const at::Tensor&         lengths,
                               at::Tensor&               dst_page_table,
                               std::optional<at::Tensor> src_page_table_opt,
                               const at::Tensor&         cu_seqlens_q,
                               std::optional<at::Tensor> row_starts_opt = std::nullopt);

void fast_topk_transform_ragged_fused(const at::Tensor&         score,
                                      const at::Tensor&         lengths,
                                      at::Tensor&               topk_indices_ragged,
                                      const at::Tensor&         topk_indices_offset,
                                      std::optional<at::Tensor> row_starts_opt = std::nullopt);

}  // namespace rtp_llm
