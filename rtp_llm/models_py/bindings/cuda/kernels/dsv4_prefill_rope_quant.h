#pragma once

#include <optional>

#include <torch/all.h>

namespace torch_ext {

// Fused RMSNorm + partial RoPE on the trailing ``rope_head_dim`` columns.
// Supports in-place aliasing (``out.data_ptr() == x.data_ptr()``), optional
// ``weight`` ([D] BF16/FP16/FP32), and optional ``position_ids`` ([N_freq]
// int32/int64). All tensors must be CUDA and contiguous.
void dsv4_fused_rmsnorm_rope(const torch::Tensor&                x,
                             const std::optional<torch::Tensor>& weight,
                             const torch::Tensor&                freqs_ri,
                             const std::optional<torch::Tensor>& position_ids,
                             torch::Tensor&                      out,
                             int64_t                             freq_stride_n,
                             int64_t                             rope_head_dim,
                             double                              eps,
                             bool                                inverse = false);

// Fused inverse-RoPE + per-(token, 128-col block) UE8M0 FP8 quantization.
// Produces TMA-aligned ``fp8_work`` ([G, M, hpg * D] float8_e4m3fn) and
// ``scale_work`` ([G, M, hpg] int32 with stride(1) == 1) while zero-padding
// rows [M, scale_zero_rows) in the scale buffer.
void dsv4_fused_inv_rope_fp8_quant(const torch::Tensor&                o,
                                   const torch::Tensor&                freqs_ri,
                                   const std::optional<torch::Tensor>& position_ids,
                                   torch::Tensor&                      fp8_work,
                                   torch::Tensor&                      scale_work,
                                   int64_t                             q_len_per_b,
                                   int64_t                             n_groups,
                                   int64_t                             heads_per_group,
                                   int64_t                             nope_dim,
                                   int64_t                             rope_head_dim,
                                   int64_t                             quant_group_size,
                                   double                              fp8_max,
                                   double                              eps,
                                   int64_t                             scale_zero_rows);

// Fused indexer Q RoPE + per-(token, head) FP8 quantization + weight scale fold.
// Reads ``q_bf16`` ([BSH, 128] bf16), ``weights`` ([BSH] bf16/fp32), and
// ``freqs_ri`` ([N_freq, RD/2, 2] fp32), and writes ``q_fp8`` ([BSH, 128]
// float8_e4m3fn) and ``w_fold`` ([BSH] fp32).
void dsv4_indexer_q_rope_fp8_fold(const torch::Tensor& q_bf16,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& freqs_ri,
                                  torch::Tensor&       q_fp8,
                                  torch::Tensor&       w_fold,
                                  int64_t              freq_stride_n,
                                  int64_t              rope_head_dim,
                                  double               fp8_max);

}  // namespace torch_ext
