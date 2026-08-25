#pragma once

#include <torch/extension.h>

namespace rtp_llm {

void fusedMultimodalCopy(torch::Tensor&                    dst,
                         const std::vector<torch::Tensor>& srcs,
                         const std::vector<int64_t>&       row_offsets);

}  // namespace rtp_llm
