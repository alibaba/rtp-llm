#pragma once

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

// token_count < 0 skips range checks when the final token sequence is not known yet.
absl::Status validateAndNormalizeInputEmbeddings(std::vector<torch::Tensor>& embeddings,
                                                 const std::vector<int32_t>& embedding_locs,
                                                 int64_t                     token_count);

absl::Status validateInputEmbeddings(const std::vector<torch::Tensor>& embeddings,
                                     const std::vector<int32_t>&       embedding_locs,
                                     int64_t                           token_count);

absl::Status validateAndNormalizeInputEmbeddings(GenerateInput& input);

}  // namespace rtp_llm
