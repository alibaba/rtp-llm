#pragma once

#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

namespace torch_ext {

void dsv4_indexed_rmsnorm_rope(at::Tensor& input,
                               at::Tensor& weight,
                               at::Tensor& freqs_cis,
                               at::Tensor& position_ids,
                               at::Tensor& output,
                               int64_t     rope_head_dim,
                               double      eps,
                               bool        has_weight);

void dsv4_indexed_inv_rope_fp8_quant(at::Tensor& input,
                                     at::Tensor& freqs_cis,
                                     at::Tensor& position_ids,
                                     at::Tensor& output_q,
                                     at::Tensor& output_s,
                                     int64_t     n_groups,
                                     int64_t     heads_per_group,
                                     int64_t     nope_dim,
                                     int64_t     rope_head_dim,
                                     double      eps,
                                     double      fp8_max);

}  // namespace torch_ext
