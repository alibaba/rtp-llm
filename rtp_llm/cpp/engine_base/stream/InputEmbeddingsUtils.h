#pragma once

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

// Immutable deployment policy; populated from the instantiated model and engine config.
struct InputEmbeddingsRuntimePolicy {
    int64_t hidden_size      = 0;
    bool    model_supported  = false;
    int64_t tp_size          = 1;
    bool    context_parallel = false;
    bool    speculative      = false;
    bool    ffn_disaggregate = false;
};

absl::Status validateInputEmbeddingsRuntimeSupport(const InputEmbeddingsRuntimePolicy& policy);
absl::Status validateInputEmbeddingsForRequest(const GenerateInput&                input,
                                               const InputEmbeddingsRuntimePolicy& policy,
                                               bool                                check_token_range = true);

// token_count < 0 skips range checks when the final token sequence is not known yet.
absl::Status validateAndNormalizeInputEmbeddings(std::vector<torch::Tensor>& embeddings,
                                                 const std::vector<int32_t>& embedding_locs,
                                                 int64_t                     token_count);

absl::Status validateInputEmbeddings(const std::vector<torch::Tensor>& embeddings,
                                     const std::vector<int32_t>&       embedding_locs,
                                     int64_t                           token_count);

absl::Status validateAndNormalizeInputEmbeddings(GenerateInput& input);

}  // namespace rtp_llm
