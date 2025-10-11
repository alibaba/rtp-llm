#pragma once
#include <torch/all.h>
#include "rtp_llm/cpp/model_utils/RopeConfig.h"

namespace rtp_llm {

torch::Tensor genNormalCosSin(int rope_dim, int rope_theta, float rope_scale, int max_position_embeddings);

/**
 * @brief Get the Rope Cos Sin object, TODO: move to python
 *
 * @param device
 * @param rope_style
 * @param rope_dim
 * @param rope_theta
 * @param rope_scale
 * @param max_position_embeddings
 * @return BufferPtr
 */
torch::Tensor
getRopeCosSin(RopeStyle rope_style, int rope_dim, int rope_theta, float rope_scale, int max_position_embeddings);

} // namespace rtp_llm
