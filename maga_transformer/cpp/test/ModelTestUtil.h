#pragma once

#include "src/fastertransformer/devices/Weights.h"

namespace ft = fastertransformer;

namespace rtp_llm {

std::unique_ptr<const ft::Weights> loadWeightsFromDir(std::string dir_path);

} // namespace rtp_llm

