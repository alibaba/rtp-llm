#pragma once

#include <torch/all.h>

namespace rtp_llm {

void dsv4_indexed_inv_rope_fp8_quant(torch::Tensor input,
                                     torch::Tensor freqs_cis,
                                     torch::Tensor position_ids,
                                     torch::Tensor output_q,
                                     torch::Tensor output_s,
                                     int64_t       n_groups,
                                     int64_t       heads_per_group,
                                     int64_t       nope_dim,
                                     int64_t       rope_head_dim,
                                     double        eps,
                                     double        fp8_max);

}  // namespace rtp_llm
