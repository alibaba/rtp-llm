#pragma once

#include <cstddef>
#include <torch/types.h>

namespace rtp_llm {

// Only DEFAULT MLA layers populate this metadata. KDA's SSM/conv blocks and
// typed indexer regions retain their existing GPU tensors and addressing.
struct MlaHostCacheInfo {
    torch::Tensor hbm_cache;
    torch::Tensor block_generations;
    size_t        hbm_tokens = 0;
};

}  // namespace rtp_llm
