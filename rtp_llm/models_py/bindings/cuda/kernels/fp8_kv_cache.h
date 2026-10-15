#pragma once

#include "rtp_llm/cpp/model_utils/RopeConfig.h"

#include <optional>
#include <torch/extension.h>

namespace rtp_llm {

at::Tensor fused_rope_quantize_and_write_fp8_kv_cache(const at::Tensor&                qkv,
                                                      at::Tensor&                      kv_cache,
                                                      at::Tensor&                      kv_scales,
                                                      const at::Tensor&                batch_indices,
                                                      const at::Tensor&                positions,
                                                      const at::Tensor&                page_indptr,
                                                      const at::Tensor&                page_indices,
                                                      int64_t                          num_q_heads,
                                                      int64_t                          num_kv_heads,
                                                      int64_t                          kernel_page_size,
                                                      const RopeConfig&                rope_config,
                                                      const std::optional<at::Tensor>& cos_sin_cache = std::nullopt);

// Quantize post-RoPE K/V one [H, D] row at a time and write it into a
// persistent paged FP8 cache. target_physical_page_ids and token_offsets are
// one-dimensional CUDA int32/int64 tensors with one unique entry per input token.
// NaN maps to zero; infinities saturate under the finite-value row scale.
void quantize_and_write_fp8_kv_cache(const at::Tensor& k,
                                     const at::Tensor& v,
                                     at::Tensor&       kv_cache,
                                     at::Tensor&       kv_scales,
                                     const at::Tensor& target_physical_page_ids,
                                     const at::Tensor& token_offsets,
                                     int64_t           physical_page_size,
                                     int64_t           kernel_page_size,
                                     int64_t           subdivision);

// Gather source kernel pages from a persistent FP8 cache and dequantize them
// into output [R, 2, H, kernel_page_size, D]. source_kernel_page_ids are
// physical_page_id * subdivision + subpage_id.
void gather_and_dequantize_fp8_kv_cache(const at::Tensor& kv_cache,
                                        const at::Tensor& kv_scales,
                                        const at::Tensor& source_kernel_page_ids,
                                        at::Tensor&       output,
                                        int64_t           physical_page_size,
                                        int64_t           kernel_page_size,
                                        int64_t           subdivision);

}  // namespace rtp_llm
