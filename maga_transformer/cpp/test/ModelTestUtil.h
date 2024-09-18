#pragma once

#include "src/fastertransformer/devices/Weights.h"
#include "maga_transformer/cpp/models/GptModel.h"

namespace ft = fastertransformer;

namespace rtp_llm {

std::unique_ptr<const ft::Weights> loadWeightsFromDir(std::string dir_path);

std::unique_ptr<GptModel> createGptModel(const GptModelInitParams& params);

} // namespace rtp_llm

