#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"
#include "maga_transformer/cpp/models/GptModel.h"

namespace rtp_llm {

std::unique_ptr<GptModel> createGptModel(const GptModelInitParams& params);

} // namespace rtp_llm
