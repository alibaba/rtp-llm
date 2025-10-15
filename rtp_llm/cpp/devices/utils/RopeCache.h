#pragma once
#include <torch/all.h>
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/model_utils/RopeConfig.h"

namespace rtp_llm {

torch::Tensor getRopeCache(const RopeConfig& rope_config, int max_position_embeddings);
} // namespace rtp_llm
