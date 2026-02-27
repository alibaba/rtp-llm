#pragma once
#include <torch/all.h>
#include "rtp_llm/cpp/model_utils/RopeConfig.h"

namespace rtp_llm {

struct RopeCache {
    bool          used = false;
    int           dim  = -1;
    int           base = -1;
    torch::Tensor data;
};

/**
 * @brief Get the Rope Cache object
 *
 * @param rope_config
 * @param max_position_embeddings
 * @param interleave cos/sin cache format: true=interleaved [cos,sin,cos,sin,...], false=non-interleaved
 * [cos,cos,...,sin,sin,...]
 * @return torch::Tensor
 */
torch::Tensor getRopeCache(const RopeConfig& rope_config, const int max_position_embeddings, const bool interleave);

/**
 * @brief Get the Rope Cache object Once
 *
 * @param rope_config
 * @param max_position_embeddings
 * @param is_cuda
 * @param interleave cos/sin cache format: true=interleaved [cos,sin,cos,sin,...], false=non-interleaved
 * [cos,cos,...,sin,sin,...]
 * @return RopeCache
 */
RopeCache getRopeCacheOnce(const RopeConfig& rope_config,
                           const int         max_position_embeddings,
                           const bool        is_cuda    = true,
                           const bool        interleave = true);

/**
 * @brief
 *
 * @param rope_config
 * @param rope_cache
 * @return true
 * @return false
 */
bool checkRopeCache(const RopeConfig& rope_config, const RopeCache& rope_cache);

}  // namespace rtp_llm
