#pragma once

#include "maga_transformer/cpp/devices/Weights.h"
#include "maga_transformer/cpp/models/GptModel.h"



namespace rtp_llm {

std::unique_ptr<const rtp_llm::Weights> loadWeightsFromDir(std::string dir_path);

std::unique_ptr<GptModel> createGptModel(const GptModelInitParams& params);

} // namespace rtp_llm

