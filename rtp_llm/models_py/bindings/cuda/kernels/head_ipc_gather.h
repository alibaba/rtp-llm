// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <torch/all.h>
#include <vector>
namespace rtp_llm {
int64_t       headIpcMetaSize();
torch::Tensor headIpcGather(const torch::Tensor&        input,
                            const std::vector<int64_t>& buffers,
                            const std::vector<int64_t>& signals,
                            int64_t                     rank);
void          headIpcClose(const std::vector<int64_t>& pointers, int64_t rank);
}  // namespace rtp_llm
