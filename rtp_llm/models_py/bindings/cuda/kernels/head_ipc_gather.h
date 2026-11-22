// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <torch/all.h>
#include <vector>
namespace rtp_llm {
int64_t       headIpcMetaSize();
int64_t       headIpcArgmaxMetaSize();
int64_t       headIpcArgmaxBufferSize();
torch::Tensor headIpcGather(const torch::Tensor&        input,
                            const std::vector<int64_t>& buffers,
                            const std::vector<int64_t>& signals,
                            int64_t                     rank);
torch::Tensor headIpcArgmax(const torch::Tensor&        input,
                            const std::vector<int64_t>& buffers,
                            const std::vector<int64_t>& signals,
                            int64_t                     rank,
                            int64_t                     valid_vocab_size);
void          headIpcClose(const std::vector<int64_t>& pointers, int64_t rank);
}  // namespace rtp_llm
