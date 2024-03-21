#pragma once

#include "src/fastertransformer/devices/Weights.h"

using namespace fastertransformer;

namespace rtp_llm {

std::unique_ptr<const Weights> loadWeightsFromDir(std::string dir_path);

} // namespace rtp_llm

