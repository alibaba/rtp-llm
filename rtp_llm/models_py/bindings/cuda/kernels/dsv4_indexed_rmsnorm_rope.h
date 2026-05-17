#pragma once

#include <torch/all.h>

namespace rtp_llm {

void dsv4_indexed_rmsnorm_rope(torch::Tensor input,
                               torch::Tensor weight,
                               torch::Tensor freqs_cis,
                               torch::Tensor position_ids,
                               torch::Tensor output,
                               int64_t       rope_head_dim,
                               double        eps,
                               bool          has_weight);

}  // namespace rtp_llm
