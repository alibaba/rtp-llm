// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <c10/util/Optional.h>
#include <torch/all.h>
#include <tuple>
#include <vector>

namespace rtp_llm {

int64_t tp2ArHcMetaSize();
int64_t tp2ArHcBufferSize();

std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>>
tp2ArHcInjectNorm(const torch::Tensor&        hyper_x,
                  const torch::Tensor&        local_sub,
                  const torch::Tensor&        gates,
                  const torch::Tensor&        gamma,
                  const std::vector<int64_t>& buffers,
                  const std::vector<int64_t>& signals,
                  int64_t                     rank,
                  double                      eps,
                  bool                        packed_dual,
                  bool                        collect_amax,
                  bool                        enable_pdl);

torch::Tensor tp2ArHcInjectOnly(const torch::Tensor&        hyper_x,
                                const torch::Tensor&        local_sub,
                                const torch::Tensor&        gates,
                                const std::vector<int64_t>& buffers,
                                const std::vector<int64_t>& signals,
                                int64_t                     rank,
                                bool                        packed_dual,
                                bool                        enable_pdl);

}  // namespace rtp_llm
