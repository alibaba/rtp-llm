#pragma once

#include "rtp_llm/cpp/models/models_weight/Weights.h"
#include "rtp_llm/cpp/models/ModelTypes.h"

namespace rtp_llm {

std::unique_ptr<const rtp_llm::Weights> loadWeightsFromDir(std::string dir_path);

}  // namespace rtp_llm
