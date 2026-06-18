#pragma once

#include <torch/extension.h>
#include <torch/all.h>

namespace rtp_llm {

void fake_balance_expert_op(at::Tensor& expert_ids,
                            at::Tensor& expert_scales,
                            int64_t     dp_rank,
                            int64_t     dp_size,
                            int64_t     ep_size,
                            int64_t     expert_num,
                            int64_t     hip_stream);

}  // namespace rtp_llm
