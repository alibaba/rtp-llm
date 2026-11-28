#pragma once

#include <torch/extension.h>

namespace rtp_llm {

bool tryFusedMultimodalCopy(torch::Tensor&                    dst,
                            const std::vector<torch::Tensor>& srcs,
                            const std::vector<int64_t>&       row_offsets);

void fusedMultimodalCopy(torch::Tensor&                    dst,
                         const std::vector<torch::Tensor>& srcs,
                         const std::vector<int64_t>&       row_offsets);

}  // namespace rtp_llm
